# Muster changelog

Version history of the training stack and game rules. Each run's exact flags
are preserved in its checkpoint (`train_args`) and in git history.

## v19 — score-aware presence game (rules 0.7, current)
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
