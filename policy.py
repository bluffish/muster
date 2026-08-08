"""Local hex-tile actor with a centralized spatial MAPPO critic."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from rl_env import LOCAL_FEATURE_SIZE, LocalState
from simulator import (
    TERRITORY_CELLS,
    TERRITORY_COORDINATES,
    TERRITORY_RADIUS,
    TERRITORY_WEIGHTS,
    territory_neighborhood,
)

ACTION_SIZE = 4
CHECKPOINT_VERSION = 11
ENTITY_OFFSET_SCALE = 8.0


class Policy(nn.Module):
    """Shared local actor and centralized per-soldier value function."""

    def __init__(
        self,
        hidden_size: int = 256,
        entity_size: int = 16,
        tile_size: int = 32,
        local_radius: int = 2,
        mode_count: int = 0,
        mode_size: int = 16,
        log_std_floor: float = -5.0,
    ):
        super().__init__()
        if mode_count < 0 or mode_size < 1:
            raise ValueError("mode_count must be non-negative and mode_size positive")
        if not -5.0 <= log_std_floor < 1.0:
            raise ValueError("log_std_floor must be in [-5, 1)")
        neighbors, offsets = territory_neighborhood(local_radius)
        self.hidden_size = hidden_size
        self.entity_size = entity_size
        self.tile_size = tile_size
        self.local_radius = local_radius
        self.mode_count = mode_count
        self.mode_size = mode_size
        self.log_std_floor = log_std_floor
        self.local_tiles = neighbors.shape[1]
        self.attention_heads = math.gcd(4, entity_size)
        self.attention_size = 2 * entity_size
        self.attention_head_size = self.attention_size // self.attention_heads
        self.register_buffer("neighbors", torch.from_numpy(neighbors).long(), persistent=False)
        self.register_buffer("offsets", torch.from_numpy(offsets), persistent=False)
        self.register_buffer(
            "enemy_feature_sign",
            torch.tensor(
                (-1, 1, -1, 1, -1, 1, 1, 1, -1, 1, 1), dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer("team_x", torch.tensor((1, -1)), persistent=False)
        coordinates = torch.from_numpy(TERRITORY_COORDINATES).float()
        self.register_buffer(
            "cell_coordinates",
            torch.stack(
                (
                    coordinates[:, 0] / TERRITORY_RADIUS,
                    (coordinates[:, 1] + 0.5 * coordinates[:, 0]) / TERRITORY_RADIUS,
                ),
                dim=-1,
            ),
            persistent=False,
        )
        weights = torch.from_numpy(TERRITORY_WEIGHTS).float()
        self.register_buffer(
            "cell_value",
            (weights - weights.min()) / (weights.max() - weights.min()).clamp_min(1),
            persistent=False,
        )

        self.entity_encoder = nn.Sequential(
            nn.Linear(LOCAL_FEATURE_SIZE, entity_size),
            nn.SiLU(),
            nn.Linear(entity_size, entity_size),
            nn.SiLU(),
        )
        self.entity_token = nn.Sequential(
            nn.Linear(entity_size + 3, entity_size),
            nn.SiLU(),
        )
        self.entity_query = nn.Linear(entity_size, self.attention_size, bias=False)
        self.entity_key = nn.Linear(entity_size, self.attention_size, bias=False)
        self.entity_value = nn.Linear(entity_size, self.attention_size, bias=False)
        self.entity_output = nn.Linear(self.attention_size, 2 * entity_size, bias=False)
        self.tile_encoder = nn.Sequential(
            nn.Linear(9, tile_size),
            nn.SiLU(),
            nn.Linear(tile_size, tile_size),
            nn.SiLU(),
        )
        self.tile_query = nn.Linear(entity_size, tile_size, bias=False)
        self.tile_key = nn.Linear(tile_size, tile_size, bias=False)
        conditioning = mode_size if mode_count else 0
        self.backbone = nn.Sequential(
            nn.Linear(3 * entity_size + tile_size + conditioning, hidden_size),
            nn.SiLU(),
        )
        self.policy_head = nn.Linear(hidden_size, ACTION_SIZE)
        self.global_tile_encoder = nn.Sequential(
            nn.Linear(2 * entity_size + 8, tile_size),
            nn.SiLU(),
            nn.Linear(tile_size, tile_size),
            nn.SiLU(),
        )
        self.global_value_encoder = nn.Sequential(
            nn.Linear(2 * tile_size + conditioning, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        if mode_count:
            self.mode_embedding = nn.Embedding(mode_count, mode_size)
            self.mode_policy_bias = nn.Linear(mode_size, ACTION_SIZE, bias=False)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -0.5))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, 2**0.5)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, 0.01)
        nn.init.orthogonal_(self.value_head.weight, 1.0)
        if mode_count:
            nn.init.orthogonal_(self.mode_embedding.weight)
            # Non-trivial gain keeps episode modes visibly shifting the action
            # mean from the first update, so optimization shapes the modes
            # instead of deciding whether they exist.
            nn.init.orthogonal_(self.mode_policy_bias.weight, 0.3)
        self.use_bf16 = True

    @property
    def model_kwargs(self) -> dict[str, int]:
        return {
            "hidden_size": self.hidden_size,
            "entity_size": self.entity_size,
            "tile_size": self.tile_size,
            "local_radius": self.local_radius,
            "mode_count": self.mode_count,
            "mode_size": self.mode_size,
            "log_std_floor": self.log_std_floor,
        }

    def _mode_embedding(self, state: LocalState) -> torch.Tensor | None:
        if not self.mode_count:
            return None
        if state.mode is None:
            raise ValueError("this policy requires state.mode episode latents")
        return self.mode_embedding(state.mode)

    def _compute(self, state: LocalState):
        if self.use_bf16 and state.features.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _encode(self, state: LocalState):
        features = state.features
        if not torch.is_autocast_enabled():
            features = features.to(self.entity_encoder[0].weight.dtype)
        enemy = features.flip(1) * self.enemy_feature_sign.to(features.dtype)
        related_features = torch.stack((features, enemy), dim=2)
        related_alive = torch.stack((state.alive, state.alive.flip(1)), dim=2).bool()
        related_cells = torch.stack((state.cells, state.cells.flip(1)), dim=2).long()
        related_cells = torch.where(related_alive, related_cells, 0)
        return (
            self.entity_encoder(related_features),
            related_features,
            related_cells,
            related_alive,
        )

    def _tile_aggregates(
        self,
        encoded: torch.Tensor,
        cells: torch.Tensor,
        alive: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, teams, relations, _, width = encoded.shape
        sums = encoded.new_zeros((batch, teams, relations, TERRITORY_CELLS, width))
        counts = encoded.new_zeros((batch, teams, relations, TERRITORY_CELLS, 1))
        index = cells.unsqueeze(-1)
        mask = alive.to(encoded.dtype).unsqueeze(-1)
        sums.scatter_add_(3, index.expand_as(encoded), encoded * mask)
        counts.scatter_add_(3, index, mask)
        return sums, counts

    @staticmethod
    def _gather_tiles(values: torch.Tensor, neighbors: torch.Tensor) -> torch.Tensor:
        batch, teams, relations, _, width = values.shape
        soldiers, local_tiles = neighbors.shape[-2:]
        padded = torch.cat(
            (values, values.new_zeros((batch, teams, relations, 1, width))), dim=3
        )
        index = neighbors.flatten(-2)[:, :, None, :, None].expand(
            -1, -1, relations, -1, width
        )
        gathered = torch.gather(padded, 3, index)
        return gathered.view(batch, teams, relations, soldiers, local_tiles, width)

    def _entity_context(
        self,
        state: LocalState,
        encoded: torch.Tensor,
        related_features: torch.Tensor,
    ) -> torch.Tensor:
        """Attend over each soldier's k nearest entities as individual tokens.

        Every token binds one entity's embedding to its exact egocentric
        offset and side, so targeting-grade geometry survives to the actor.
        """
        if state.neighbors is None:
            raise ValueError("this policy requires state.neighbors entity indices")
        batch, teams, relations, soldiers, width = encoded.shape
        index = state.neighbors.long()
        neighbor_count = index.shape[-1]
        valid = index >= 0
        safe = index.clamp_min(0)
        total = relations * soldiers
        gather = safe.reshape(batch, teams, soldiers * neighbor_count, 1)
        flat_encoded = encoded.reshape(batch, teams, total, width)
        embeddings = flat_encoded.gather(2, gather.expand(-1, -1, -1, width)).view(
            batch, teams, soldiers, neighbor_count, width
        )
        flat_positions = related_features[..., 8:10].reshape(batch, teams, total, 2)
        positions = flat_positions.gather(2, gather.expand(-1, -1, -1, 2)).view(
            batch, teams, soldiers, neighbor_count, 2
        )
        observer = state.features[..., 8:10].to(positions.dtype).unsqueeze(-2)
        offsets = (positions - observer) * ENTITY_OFFSET_SCALE
        enemy = (safe >= soldiers).to(embeddings.dtype).unsqueeze(-1)
        tokens = self.entity_token(
            torch.cat((embeddings, offsets.to(embeddings.dtype), enemy), dim=-1)
        )

        heads, head_width = self.attention_heads, self.attention_head_size
        query = self.entity_query(encoded[:, :, 0]).view(
            batch, teams, soldiers, 1, heads, head_width
        )
        keys = self.entity_key(tokens).view(
            batch, teams, soldiers, neighbor_count, heads, head_width
        )
        values = self.entity_value(tokens).view(
            batch, teams, soldiers, neighbor_count, heads, head_width
        )
        scores = (query * keys).sum(-1) / math.sqrt(head_width)
        mask = valid.unsqueeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-2).to(values.dtype)
        context = (weights.unsqueeze(-1) * values).sum(-3).flatten(-2)
        context = context * valid.any(-1, keepdim=True).to(context.dtype)
        return self.entity_output(context)

    def _backbone(
        self,
        state: LocalState,
        encoded: torch.Tensor,
        related_features: torch.Tensor,
        related_cells: torch.Tensor,
        related_alive: torch.Tensor,
        mode_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sums, counts = self._tile_aggregates(encoded, related_cells, related_alive)
        own_cells = torch.where(state.alive.bool(), state.cells.long(), 0)
        neighbors = self.neighbors[own_cells]
        gathered_count = self._gather_tiles(counts, neighbors)

        self_embedding = encoded[:, :, 0]
        center = encoded.new_zeros((1, 1, 1, self.local_tiles, 1))
        center[..., 0, :] = 1
        self_mask = state.alive.bool().to(encoded.dtype).unsqueeze(-1)
        ally_count = gathered_count[:, :, 0] - center * self_mask.unsqueeze(-2)
        enemy_count = gathered_count[:, :, 1]
        entity_context = self._entity_context(state, encoded, related_features)

        batch, teams, soldiers, local_tiles = neighbors.shape
        owner_pad = torch.cat(
            (state.owners, state.owners.new_full((batch, 1), -2)), dim=-1
        )
        owner_source = owner_pad[:, None, None].expand(-1, teams, soldiers, -1)
        owners = torch.gather(owner_source, 3, neighbors)
        team = torch.arange(teams, device=owners.device).view(1, teams, 1, 1)
        own_owner = (owners == team).to(encoded.dtype).unsqueeze(-1)
        enemy_owner = (owners == 1 - team).to(encoded.dtype).unsqueeze(-1)
        neutral_owner = (owners == -1).to(encoded.dtype).unsqueeze(-1)
        valid = (neighbors != TERRITORY_CELLS)
        value_pad = torch.cat((self.cell_value, self.cell_value.new_zeros(1)))
        cell_value = value_pad[neighbors].to(encoded.dtype).unsqueeze(-1)

        offset_x = self.offsets[:, 0].view(1, 1, 1, local_tiles)
        offset_x = offset_x * self.team_x.view(1, teams, 1, 1)
        offset_y = self.offsets[:, 1].view(1, 1, 1, local_tiles)
        offset_y = offset_y * state.mirror_y.view(batch, 1, 1, 1)
        offsets = torch.stack(
            (offset_x.expand(batch, -1, soldiers, -1), offset_y.expand(-1, teams, soldiers, -1)),
            dim=-1,
        ).to(encoded.dtype)
        count_scale = math.log1p(state.features.shape[2])
        tile_input = torch.cat(
            (
                torch.log1p(ally_count) / count_scale,
                torch.log1p(enemy_count) / count_scale,
                own_owner,
                enemy_owner,
                neutral_owner,
                cell_value,
                valid.to(encoded.dtype).unsqueeze(-1),
                offsets,
            ),
            dim=-1,
        )
        tiles = self.tile_encoder(tile_input)
        query = self.tile_query(self_embedding).unsqueeze(-2)
        score = (self.tile_key(tiles) * query).sum(-1) / math.sqrt(self.tile_size)
        score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
        weight = torch.softmax(score.float(), dim=-1).to(tiles.dtype)
        context = (weight.unsqueeze(-1) * tiles).sum(-2)
        inputs = [self_embedding, entity_context, context]
        if mode_embed is not None:
            inputs.append(
                mode_embed.to(context.dtype).unsqueeze(-2).expand(-1, -1, soldiers, -1)
            )
        hidden = self.backbone(torch.cat(inputs, dim=-1))
        return hidden, sums, counts

    def _policy_mean(
        self, hidden: torch.Tensor, mode_embed: torch.Tensor | None
    ) -> torch.Tensor:
        mean = self.policy_head(hidden)
        if mode_embed is not None:
            mean = mean + self.mode_policy_bias(mode_embed.to(mean.dtype)).unsqueeze(-2)
        return mean

    def _values(
        self,
        state: LocalState,
        hidden: torch.Tensor,
        sums: torch.Tensor,
        counts: torch.Tensor,
        mode_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ally_count = counts[:, :, 0]
        enemy_count = counts[:, :, 1]
        ally_mean = sums[:, :, 0] / ally_count.clamp_min(1)
        enemy_mean = sums[:, :, 1] / enemy_count.clamp_min(1)

        batch, teams, cells, _ = ally_mean.shape
        owners = state.owners[:, None].expand(-1, teams, -1)
        team = torch.arange(teams, device=owners.device).view(1, teams, 1)
        own_owner = (owners == team).to(ally_mean.dtype).unsqueeze(-1)
        enemy_owner = (owners == 1 - team).to(ally_mean.dtype).unsqueeze(-1)
        neutral_owner = (owners == -1).to(ally_mean.dtype).unsqueeze(-1)

        x = self.cell_coordinates[:, 0].view(1, 1, cells)
        x = x * self.team_x.view(1, teams, 1)
        y = self.cell_coordinates[:, 1].view(1, 1, cells)
        y = y * state.mirror_y.view(batch, 1, 1)
        coordinates = torch.stack(
            (x.expand(batch, -1, -1), y.expand(-1, teams, -1)), dim=-1
        ).to(ally_mean.dtype)
        cell_value = self.cell_value.view(1, 1, cells, 1).expand(
            batch, teams, -1, -1
        ).to(ally_mean.dtype)
        count_scale = math.log1p(state.features.shape[2])
        global_tiles = self.global_tile_encoder(
            torch.cat(
                (
                    ally_mean,
                    enemy_mean,
                    torch.log1p(ally_count) / count_scale,
                    torch.log1p(enemy_count) / count_scale,
                    own_owner,
                    enemy_owner,
                    neutral_owner,
                    cell_value,
                    coordinates,
                ),
                dim=-1,
            )
        )
        global_inputs = [global_tiles.mean(-2), global_tiles.square().mean(-2)]
        if mode_embed is not None:
            global_inputs.append(mode_embed.to(global_tiles.dtype))
        global_context = torch.cat(global_inputs, dim=-1)
        global_hidden = self.global_value_encoder(global_context).unsqueeze(-2)
        value = self.value_head(F.silu(hidden + global_hidden)).squeeze(-1).float()
        return value * state.alive.to(torch.float32)

    def _distribution(self, mean: torch.Tensor) -> Normal:
        std = self.log_std.clamp(self.log_std_floor, 1).exp().to(mean.dtype).expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _log_prob(distribution: Normal, raw: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        stable = Normal(distribution.loc.float(), distribution.scale.float())
        correction = torch.log(1 - action.float().square() + 1e-6)
        return (stable.log_prob(raw.float()) - correction).sum(-1)

    def act(
        self, state: LocalState, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._compute(state):
            mode_embed = self._mode_embedding(state)
            encoded, related, cells, alive = self._encode(state)
            hidden, sums, counts = self._backbone(
                state, encoded, related, cells, alive, mode_embed
            )
            distribution = self._distribution(self._policy_mean(hidden, mode_embed))
            raw = distribution.mean if deterministic else distribution.sample()
            actions = raw.float().tanh()
            mask = state.alive.bool().to(actions.dtype)
            return (
                actions * mask.unsqueeze(-1),
                self._log_prob(distribution, raw, actions) * mask,
                self._values(state, hidden, sums, counts, mode_embed),
            )

    def actor_actions(self, state: LocalState, deterministic: bool = False) -> torch.Tensor:
        """Run the shared backbone and policy head without the value head."""
        with self._compute(state):
            mode_embed = self._mode_embedding(state)
            encoded, related, cells, alive = self._encode(state)
            hidden, _, _ = self._backbone(
                state, encoded, related, cells, alive, mode_embed
            )
            distribution = self._distribution(self._policy_mean(hidden, mode_embed))
            raw = distribution.mean if deterministic else distribution.sample()
            return raw.float().tanh() * state.alive.bool().unsqueeze(-1)

    def evaluate_actions(
        self, state: LocalState, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._compute(state):
            mode_embed = self._mode_embedding(state)
            encoded, related, cells, alive = self._encode(state)
            hidden, sums, counts = self._backbone(
                state, encoded, related, cells, alive, mode_embed
            )
            distribution = self._distribution(self._policy_mean(hidden, mode_embed))
            bounded = actions.float().clamp(-1 + 1e-6, 1 - 1e-6)
            raw = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
            log_prob = self._log_prob(distribution, raw, bounded)
            entropy = Normal(distribution.loc.float(), distribution.scale.float()).entropy().sum(-1)
            mask = state.alive.bool().to(log_prob.dtype)
            return log_prob * mask, entropy * mask, self._values(
                state, hidden, sums, counts, mode_embed
            )

    def value(self, state: LocalState) -> torch.Tensor:
        with self._compute(state):
            mode_embed = self._mode_embedding(state)
            encoded, related, cells, alive = self._encode(state)
            hidden, sums, counts = self._backbone(
                state, encoded, related, cells, alive, mode_embed
            )
            return self._values(state, hidden, sums, counts, mode_embed)


__all__ = ["ACTION_SIZE", "CHECKPOINT_VERSION", "Policy"]
