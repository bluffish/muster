# Muster changelog

Version history of the training stack and game rules. Each run's exact flags
are preserved in its checkpoint (`train_args`) and in git history.

## v21 — assault scoring (rules 0.11, current)
The bases. Each team owns one radius-1 base deep in its half of the long
map — (-14,7) west / (14,-7) east, midline-centered mirror pair — and scores ONLY
through influence on the enemy's base (weight 250/tile, ~76% of a team's
attainable score; plain ground keeps weight 1; own base is worth 0 and
matters only as denial). Chess-inspired structural fix after the truce
history: the objective now lives behind the opposing army, so no peaceful
arrangement is an equilibrium — mutual camping scores zero and loses to
any raid, a base swap loses to recalling a few defenders. Scoring,
observation features (per-team cell values), the scripted opponent
(marches on the enemy base), the viewer (bases tinted in the scoring
team's color), and the tests are all per-team now.

## v20.2 — the long map (rules 0.10)
Elongated hexagonal arena: flat north/south walls tapering to east/west
points, 39 columns x 19 rows at the waist (561 cells, world 88.5 x 49.4,
long axis = attack axis, spawn separation ~44 units, strongpoints moved
to (0,-6),(0,0),(0,6)). Groundwork for the assault ("one base per side,
score only on the enemy's") game mode: longer raids make interception,
screening, and recall real decisions. Replays now ship the axial cell
list and tile size so the viewer renders any lattice; wall collision
generalized to per-pair apothems; CHECKPOINT_VERSION 13 (old-map
checkpoints are incompatible by geometry).

## v20.1 — strongpoint scarcity (rules 0.9)
`STRONGPOINT_WEIGHT` 30 -> 90 (strongpoints now ~78% of total score);
run `local-hex-v20-charger-selfplay-v3` restarts charger-seeded pure
self-play under the new rules. The v2 run proved the deeper problem was
the game, not the learning: seeded fully aggressive, self-play de-escalated
from 15.8k damage/episode to 61 in 41 updates — correctly, because a fair
perception-limited charger lost 26/32 head-to-heads against a dispersal
policy at weight 30. Coverage outpriced combat, so combat was unlearned.
A weight sweep (30/60/90/120/180) showed the charger's mean control
advantage rising to +0.25 and plateauing at 90; win rate ~47% with the
naive chaser being kiteable off its points, which leaves exactly the
skill margin RL should exploit. Now holding the contested center beats
out-spreading, and the only counter to a camper is assault.

## v20 — charger-seeded pure self-play
Fresh run (`local-hex-v20-charger-selfplay-v2`) warm-started from
`distilled_charger.pt`: a behavior clone of a perception-limited charger
(attack the nearest enemy within visual radius 5, otherwise march on the
nearest strongpoint; move-direction cosine 0.98 to the teacher). Pure pool
self-play, no scripted opponents. Rationale: the passive equilibrium was
locally stable under every payoff fix because gradient descent evaluates
combat under its own (in)competence — the v19 lineage beat 0/64 of the
distilled charger while it kept 99.8% HP. Initializing inside the
aggressive basin makes self-play refine fighting instead of avoiding it.
Critic and value pathways start untrained (BC touched only the actor), so
early updates run on noisy advantages; target-KL 0.02 bounds the damage.
The -v1 launch was scrapped after ~30 updates: its distillation had only
ever seen mode 0 (a probe-harness env defaulted mode_count=1), so under
the run's uniform 0-15 mode sampling the untrained mode embeddings cut
combat intensity 4.5x (rollout damage 2.4k vs 11.3k mode-0-pinned). -v2
re-distills with modes sampled across the full range.

## v19.2 — entropy control
Entropy coefficient 0.003 -> 0.0005 and log_std ceiling +1 -> 0. With the
task gradient nearly flat at the passive equilibrium (KL ~0.003/update),
the entropy bonus had pushed policy entropy from 3.9 to 5.5 nats
(pre-tanh std ~1.0, sampled play near-random) while scripted wins fell
12.7% -> 3.8%. Interpretability probes (see /home/ubuntu/mechinterp on the
training host) showed intact competence being buried by sampling noise
rather than unlearned.

## v19.1 — health-weighted influence (rules 0.8)
Wounded soldiers project quadratically less control: influence is now
`(h / H)^2 * exp(-0.5 (d / sigma)^2)`. Motivated by the measured
nonconfrontational self-play equilibrium (pool episodes ended with 0.5%
total HP lost while armies split the map): under all-or-nothing influence,
damage only paid at the kill threshold, so combat was gradient-flat. Now
every point of damage moves the dense advantage reward both teams already
optimize — combat pays continuously, with no new reward channel.

## v19 — score-aware presence game (rules 0.7)
Presence scoring restored (v17's kappa rule) after v18's full allocation
measurably subsidized huddling. Added score observability: features 11/12
give every soldier its team's current signed control advantage and the
banked integral fraction (own-signed, negated in the enemy view), so
trailing teams can learn desperation and leaders defense. Feature width 13,
`CHECKPOINT_VERSION` 12. Learning rate 1e-3 (and a fix: resume previously
restored the checkpointed LR, silently ignoring the flag).

## v18 — relative force projection (rules 0.6, superseded)
Control shares as the pure ratio of Gaussian influence fields (log-domain,
order-invariant fixed point). Made encirclement/annihilation directly
territorial, but let massed armies own their half without presence:
measured behavior was one skirmish then both blobs frozen for 25 of 45
seconds. Withdrawn same day.

## v17 — soft influence control (rules 0.5)
Gaussian influence fields with neutral mass kappa=0.125 replaced discrete
ownership: continuous control shares drive scoring, margin of force
matters, unreached ground stays neutral. Viewer learned to render the
shares as blue/red intensity fields (`control_u8` in replays).

## v16 — presence control + integral scoring (rules 0.4, superseded in hours)
First presence-based rules: nearest-living-soldier ownership, winner by
time-integrated advantage, per-step reward = held advantage level.
Motivated by v15's measured self-play peace treaty (zero contacts,
strongpoints split 9/9, armies parked 40+ units apart) — persistence
ownership made peaceful partition stable, as it earlier enabled
martyrdom-painting and endgame-only play.

## v15 — GRU memory + sequence PPO
Per-soldier 64-unit GRU between backbone and heads; recurrent PPO with
truncated BPTT over 15-decision windows ((window, env) chunk minibatches,
boundary hidden states stored in rollouts). Dormant per-entity message slot
reserved for the future comms run. At matched age, first run to resist the
avoidance valley: scripted wins from u106 (v13 needed ~2x longer, v14
never), anchor specialist modes at 50-100%.

## v14 — egocentric entity perception
Replaced pooled per-cell entity statistics with token cross-attention over
each soldier's 16 nearest entities (exact offsets, individual binding,
radius unchanged at 5 so information stays scarce for future comms).
Fused warp kNN kernel (in-register k-best) replaced torch cdist/topk.
Throughput work: max-autotune compile + 16 minibatches.

## v13 — mixed self-play
Pool self-play (auto-curriculum for combat) with a 25% scripted-charger
slice; honest bot (occupies strongpoints when no enemies remain — closed
v12's freeze exploit); log-std floor; warm starts. Produced the project's
first anchor wins and, at ~u3300, specialist mode 2 beating the charger 62%
deterministically with a measured feigned-withdrawal victory while
outnumbered 4:1.

## v12 — episode clock
`time_remaining` observation (fixed-horizon Markov repair, Pardo et al.).
Led to the martyrdom-painting exploit era and the discovery that the
scripted charger halted when all enemies died.

## v11 — MAVEN episode modes
Per-team discrete episode latent (16 modes) conditioning actor and critic;
MI diversity bonus with discriminator; anchor evaluation; combat metrics.
Escaped v10's passivity into kiting.

## v10 and earlier
Fixed nearest-charge opponent training (v10, 0 wins in 1452 updates —
entropy collapse into avoidance), pool self-play prototypes (v8/v9),
paint-persistence territory rules (rules 0.3).
