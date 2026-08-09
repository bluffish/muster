"""GPU-resident self-play opponent snapshots."""

from __future__ import annotations

import copy

import torch

from muster.rl.env import LocalState
from muster.rl.policy import Policy
from muster.rl.rollout import select_state

class OpponentPool:
    """GPU-resident policy snapshots that never change during an episode."""

    def __init__(
        self,
        policy: Policy,
        size: int,
        latest_probability: float = 0.5,
        checkpoint: dict[str, object] | None = None,
    ) -> None:
        if size < 2:
            raise ValueError("opponent pool needs at least two snapshots")
        if not 0 <= latest_probability <= 1:
            raise ValueError("latest snapshot probability must be between zero and one")
        self.models = [copy.deepcopy(policy).requires_grad_(False).eval() for _ in range(size)]
        self.device = next(policy.parameters()).device
        self.latest_probability = latest_probability
        self.available = list(range(size))
        self.retiring: int | None = None
        self.pending: dict[str, torch.Tensor] | None = None
        self.next_slot = 0
        self.latest_slot = 0
        self.snapshots = 0
        if checkpoint is not None:
            states = checkpoint["models"]
            if len(states) != size:
                raise ValueError("checkpoint pool size does not match --opponent-pool-size")
            for model, state in zip(self.models, states, strict=True):
                model.load_state_dict(state)
            self.next_slot = int(checkpoint.get("next_slot", 0)) % size
            self.latest_slot = int(checkpoint.get("latest_slot", 0)) % size
            self.snapshots = int(checkpoint.get("snapshots", 0))
        self._refresh_available()

    def initial_slots(self, num_envs: int) -> torch.Tensor:
        return self._sample_slots((num_envs,))

    @torch.no_grad()
    def actions(
        self,
        state: LocalState,
        slots: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> torch.Tensor:
        output = torch.empty((*state.features.shape[:-1], 4), device=state.features.device)
        for slot, model in enumerate(self.models):
            selected = slots == slot
            slot_memory = memory[selected] if memory is not None else None
            actions, new_memory = model.actor_step(
                select_state(state, selected), slot_memory
            )
            output[selected] = actions
            if memory is not None:
                memory[selected] = new_memory
        return output

    def resample_finished(self, slots: torch.Tensor, finished: torch.Tensor) -> None:
        slots.copy_(torch.where(finished, self._sample_slots(slots.shape), slots))

    def _refresh_available(self) -> None:
        self.available_tensor = torch.tensor(self.available, device=self.device)
        historical = [slot for slot in self.available if slot != self.latest_slot]
        self.historical_tensor = torch.tensor(historical, device=self.device)

    def _sample_slots(self, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
        uniform = self.available_tensor[
            torch.randint(len(self.available), shape, device=self.device)
        ]
        if (
            self.snapshots == 0
            or self.latest_slot not in self.available
            or not len(self.historical_tensor)
        ):
            return uniform
        historical = self.historical_tensor[
            torch.randint(len(self.historical_tensor), shape, device=self.device)
        ]
        newest = torch.full(shape, self.latest_slot, device=self.device)
        choose_newest = torch.rand(shape, device=self.device) < self.latest_probability
        return torch.where(choose_newest, newest, historical)

    @torch.no_grad()
    def advance(
        self, policy: Policy, slots: torch.Tensor, request_snapshot: bool
    ) -> dict[str, int]:
        if self.retiring is not None and not bool((slots == self.retiring).any()):
            self.models[self.retiring].load_state_dict(self.pending)
            self.latest_slot = self.retiring
            self.available.append(self.retiring)
            self.available.sort()
            self.retiring = None
            self.pending = None
            self.snapshots += 1
            self._refresh_available()
        if request_snapshot and self.retiring is None:
            self.retiring = self.next_slot
            self.next_slot = (self.next_slot + 1) % len(self.models)
            self.pending = {
                name: value.detach().clone() for name, value in policy.state_dict().items()
            }
            self.available.remove(self.retiring)
            self._refresh_available()
        return {
            "opponent_snapshots": self.snapshots,
            "opponent_pool_active": len(self.available),
            "opponent_pool_draining": int(self.retiring is not None),
        }

    def checkpoint(self) -> dict[str, object]:
        states = [model.state_dict() for model in self.models]
        latest_slot = self.latest_slot
        snapshots = self.snapshots
        if self.retiring is not None:
            states[self.retiring] = self.pending
            latest_slot = self.retiring
            snapshots += 1
        return {
            "models": states,
            "next_slot": self.next_slot,
            "latest_slot": latest_slot,
            "snapshots": snapshots,
        }

    @property
    def latest(self) -> Policy:
        return self.models[self.latest_slot]
