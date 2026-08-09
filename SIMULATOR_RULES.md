# Muster Simulator Rules

Status: normative specification for the current simulator implementation

Version: 0.9

Scope: world dynamics, collisions, combat, terrain, and episode termination

## 1. Design principles

Muster is a continuous-time-looking, fixed-timestep 2D battle simulator. Every
soldier is an equal-sized disc with velocity, health, and an independently
controlled attack direction.

The rules should produce tactics from local interactions rather than from
hand-written tactical bonuses:

- Movement is continuous in two dimensions.
- Movement direction and attack direction are independent.
- Soldiers have momentum but can actively steer toward a desired velocity.
- Discs cannot pass through one another or through blocked terrain.
- Enemy collisions bounce much more strongly than allied collisions.
- A ready soldier automatically strikes one touching enemy through a binary
  frontal attack arc.
- Every strike deals a base amount; the attacker's own velocity into the target
  can raise it to charge damage.
- Side and rear hits deal more damage than hits on the target's front.
- Three symmetric high-value strongpoints create focal objectives along the
  center of the battlefield.
- Terrain has no direct combat modifiers. Any advantage from a slope must arise
  from gravity changing physical velocity.
- All interactions within a physics substep are resolved simultaneously. Array
  order must not determine who attacks, dies, or wins.

The simulator is deterministic given the initial state, control inputs,
configuration, and random seed. Physics itself contains no randomness.

## 2. World and units

The battlefield is a regular point-up hexagon centered in the bounding box

```text
[0, world_width] x [0, world_height]
```

Its western and eastern sides are vertical, and
`world_height = 2 * world_width / sqrt(3)`. The six-fold-symmetric boundary is
the playable region; the rest of the bounding box is outside the world.

World `+x` points right and world `+y` points up. Positions, distances, and time
use arbitrary but internally consistent units.

Every soldier has the same:

- radius `r`;
- diameter `D = 2r`;
- mass, treated as `1`;
- movement capabilities;
- collision behavior for a given relationship type;
- initial maximum health.

A soldier's physical body has no rotational inertia. Its attack direction is a
gameplay state, not the orientation of a rigid body.

### 2.1 Soldier state

Each soldier stores:

```text
team                 integer team identifier
position             p = (x, y)
velocity             v = (vx, vy)
attack_angle         theta in radians
health               non-negative scalar
alive                boolean
attack_recovery      remaining physics ticks before another strike
```

Dead soldiers have zero velocity and do not move, collide, attack, receive
damage, or obstruct living soldiers.

### 2.2 Initial conditions

Initial placement and formations belong to the scenario, not to physics. The
simulator must accept explicit initial states. A default reset helper may place
equal mirrored formations facing one another. Formation slot and stable soldier
ID have no physical effect except deterministic tie-breaking where required.

Initial living soldiers should be inside the valid world region, outside static
obstacles, and non-overlapping. If a scenario violates this, the ordinary
collision solver must resolve it deterministically rather than failing.

## 3. Control inputs

One control input is held constant for an entire decision step. Each living
soldier receives:

```text
move       (mx, my), a desired world-space movement vector
aim        (ax, ay), a desired world-space attack direction
```

### 3.1 Movement input

Clamp each component of `move` to `[-1, 1]`, then project the vector onto the
unit disc:

```text
m = move / max(1, length(move))
```

The desired velocity is:

```text
v_desired = maximum_running_speed * m
```

Thus `length(m)` controls desired speed and diagonal movement is not faster than
cardinal movement. `move` is expressed in world coordinates and never changes
the attack direction.

### 3.2 Aim input

Clamp each component of `aim` to `[-1, 1]`.

- If `length(aim) > 1e-6`, its angle is the desired attack angle.
- Otherwise, the soldier retains its current attack angle.

Aim affects neither movement velocity nor the collision shape.

## 4. Time

The simulator uses fixed timesteps:

```text
decision_dt = 0.10 seconds
physics_substeps = 4
physics_dt = decision_dt / physics_substeps = 0.025 seconds
```

Control inputs update once per decision step. Motion, collision, damage, and
death update once per physics substep.

Variable timesteps are not permitted.

## 5. Normative update order

At the beginning of a decision step:

1. Sanitize and normalize all control inputs.

For each physics substep, in order:

1. Decrement living soldiers' attack-recovery timers.
2. Turn each attack direction toward its requested aim.
3. Apply ground drag to velocity.
4. Apply the bounded movement motor.
5. Apply terrain gravity.
6. Apply the physics safety speed limit.
7. Integrate positions.
8. Resolve world boundaries and static obstacles.
9. Detect soldier contacts.
10. Each ready soldier chooses at most one strike target. Compute all strikes
    and first-iteration collision effects from the same state, apply health
    simultaneously, and start recovery for soldiers that struck.
11. Resolve soldier collision velocities and penetrations iteratively.
12. Resolve static obstacles again after every position-correction iteration.

At the end of the decision step:

1. Update persistent territory ownership.
2. Increment the episode's decision-step counter.
3. Evaluate episode termination.

Damage calculations use the velocity and geometry captured before collision
impulses are applied in that substep.

## 6. Attack-direction turning

Let `theta_desired` be the desired aim angle. Compute the signed shortest angular
error in `[-pi, pi]`:

```text
error = atan2(sin(theta_desired - theta), cos(theta_desired - theta))
delta = clamp(error, -maximum_turn_rate * physics_dt,
                     +maximum_turn_rate * physics_dt)
theta = wrap_to_pi(theta + delta)
```

An absent aim input produces `delta = 0`. Dead soldiers do not turn. Attack
recovery does not prevent turning.

## 7. Movement dynamics

Movement is a desired-velocity motor, not instantaneous velocity assignment and
not a force tied to attack direction.

### 7.1 Ground drag

Apply isotropic exponential drag first:

```text
v = v * exp(-linear_drag * physics_dt)
```

This removes frictionless sliding without preferring any direction.

### 7.2 Movement motor

If the soldier is alive, move its velocity toward the desired velocity with a
bounded acceleration:

```text
velocity_error = v_desired - v
maximum_change = acceleration * physics_dt
v = v + limit_length(velocity_error, maximum_change)
```

Attack recovery does not suppress or otherwise alter movement.

### 7.3 Terrain gravity

For the initial planar slope, terrain elevation is:

```text
h(x, y) = slope_height * x / world_width
```

Its grade is `slope_height / world_width`, and its acceleration is:

```text
a_terrain = (-slope_gravity * slope_height / world_width, 0)
v = v + a_terrain * physics_dt
```

A positive slope rises toward `+x`, so gravity points toward `-x`.

There are no explicit uphill, downhill, high-ground, or damage bonuses. A
downhill soldier may exceed its movement motor's desired running speed because
gravity is a separate acceleration.

### 7.4 Safety speed limit and integration

After motor and gravity acceleration, limit physical speed:

```text
v = limit_length(v, physics_speed_limit)
p = p + v * physics_dt
```

`maximum_running_speed` is the motor's target speed. `physics_speed_limit` is a
larger numerical safety limit that also contains gravity and collision velocity.

## 8. Static collision rules

Static collision constraints apply to soldier centers after expanding the
obstacle by the soldier radius.

### 8.1 Battlefield walls

Let `c` be the arena center, `a = world_width / 2` its apothem, and let the six
outward unit wall normals point at angles `0, 60, ..., 300` degrees. The valid
center region is the hexagon inset by the soldier radius:

```text
dot(p - c, normal_k) <= a - r    for every wall k
```

If a soldier crosses a wall:

1. Project its center to the valid boundary.
2. Reflect the wall-normal velocity component using `wall_restitution`.
3. Leave the tangential velocity component unchanged.

### 8.2 River and bridge

When `river_width > 0`, an impassable vertical water band is centered at
`world_width / 2`. A passable bridge opening of `bridge_width` is centered at
`world_height / 2`.

Equivalently, water consists of two axis-aligned rectangles: one below the
bridge and one above it. Each water rectangle is expanded by `r` for collision
queries.

If a soldier center enters expanded water:

1. Project it to the closest rectangle edge.
2. Remove velocity directed into the water.
3. Preserve tangential velocity.

Water has no bounce, flow, slowing effect, or damage. The bridge is ordinary
terrain and has no invisible movement bonus.

## 9. Soldier contacts

For two distinct living soldiers `i` and `j`, define:

```text
delta_ij = p_j - p_i
distance_ij = length(delta_ij)
n_ij = delta_ij / distance_ij
```

`n_ij` points from `i` toward `j`. The pair is touching when:

```text
distance_ij < D
```

If centers coincide, use a deterministic antisymmetric fallback normal derived
from stable soldier IDs. For example, the lower ID uses `(1, 0)` toward the
higher ID and the higher ID uses `(-1, 0)`.

Broad-phase collision detection may use any exact method, such as a uniform
grid. It must not omit a touching pair, process a pair twice, or change rules
based on cell boundaries.

### 9.1 Closing speed

Pairwise closing speed is:

```text
closing_ij = dot(v_i - v_j, n_ij)
```

The pair is approaching when `closing_ij > 0`.

Closing speed controls collision response and serves as a small impact-damage
gate. It does not determine the amount of attack damage.

### 9.2 Pair impulse

For an approaching pair of equal masses, choose restitution by relationship:

```text
e = ally_restitution    if team_i == team_j
e = enemy_restitution  otherwise

impulse_magnitude = 0.5 * (1 + e) * closing_ij
delta_v_i = -impulse_magnitude * n_ij
delta_v_j = +impulse_magnitude * n_ij
```

The enemy restitution may be greater than `1`. This is an intentional game
rule that creates a pronounced enemy rebound; it is not energy-conserving
real-world physics.

All pair contributions in an iteration are computed from the same input
velocity and then applied simultaneously.

### 9.3 Crowded-contact aggregation

Unbounded summation lets a crowd impart an excessive instantaneous shove. Use
the following per-soldier aggregation rule in each collision iteration:

- Sum allied velocity changes and divide the result by
  `sqrt(max(1, allied_approaching_contact_count))`.
- Sum enemy velocity changes, then limit the magnitude of that vector to the
  largest individual enemy impulse affecting that soldier in the iteration.
- Add the allied and enemy aggregate changes, then apply the physics safety
  speed limit.

This rule intentionally sacrifices exact momentum conservation in crowded
contacts. Its gameplay purpose is to prevent a dense group from behaving like
one arbitrarily massive projectile while allowing allies to push and slide as
a formation.

### 9.4 Penetration correction

For every touching pair:

```text
overlap = D - distance_ij
raw_position_change_i = -0.5 * overlap * n_ij
raw_position_change_j = +0.5 * overlap * n_ij
```

Within one solver iteration, compute all raw changes from the same positions.
For each soldier:

1. Sum its changes.
2. Divide by `sqrt(max(1, touching_contact_count))`.
3. Limit the resulting change length to `0.5r`.
4. Multiply by `collision_relaxation`.
5. Apply it simultaneously with all other soldiers.

Repeat contact detection and resolution for `collision_iterations`. Static
constraints are reapplied after every iteration.

Small residual overlap is acceptable because this is an iterative solver. With
the default iteration count, the final center distance for an isolated pair
should be at least `0.99D`.

## 10. Attack geometry

The attack surface is the frontal `120` degrees of the soldier's circumference,
one third of the full disc.

Define the attacker's unit facing vector and arc threshold:

```text
f_i = (cos(theta_i), sin(theta_i))
c_arc = cos(damaging_arc_degrees / 2)
```

For the default 120-degree arc, `c_arc = 0.5`.

Target `j` is inside attacker `i`'s attack arc when:

```text
alignment_i = dot(f_i, n_ij)
alignment_i >= c_arc
```

The arc is binary: every eligible point of the 120-degree surface has full
strength, including its boundary. There is no alignment-based damage falloff.

### 10.1 Target-facing vulnerability

The direction from target `j` toward attacker `i` is `-n_ij`. The contact hits
the target's protected front when:

```text
dot(f_j, -n_ij) >= c_arc
```

Set:

```text
target_vulnerability = 1                         for a frontal hit
target_vulnerability = flank_damage_multiplier  otherwise
```

With the default multiplier, all side and rear strikes deal twice the damage of
a frontal strike.

## 11. Strike target selection

A living soldier may strike when `attack_recovery == 0`. Its eligible targets
are living, touching enemies inside its attack arc. It chooses exactly one: the
target with greatest `alignment_i`. Equal alignments use the lower stable
soldier ID. If there is no eligible target, it does not strike.

This makes attack output independent of local crowd density: one ready soldier
never deals a complete strike to several enemies in the same substep.

## 12. Unified strike damage

Every legal strike uses the same damage calculation. There is no separate
impact, attrition, or pressure system, and no allied damage.

The attacker's own inward speed is:

```text
own_inward_speed_i = max(0, dot(v_i, n_ij))
damaging_speed_i = max(0, own_inward_speed_i - minimum_damage_speed)
```

The target's velocity does not contribute to this value. A stationary soldier
does not gain damage merely because an enemy runs into it.

The pairwise closing gate determines whether charge damage is available:

```text
charge_damage_i =
    damage_scale * damaging_speed_i^2
        if closing_ij > minimum_closing_speed
    0   otherwise
```

Final damage is:

```text
strike_damage_i_to_j =
    max(base_strike_damage, charge_damage_i)
    * target_vulnerability
```

Consequently, a stationary soldier still deals `base_strike_damage`, while a
fast inward charge replaces that base amount when it is larger. The target's
velocity can satisfy the closing gate but never increases the charge amount.

## 13. Attack recovery

After a soldier performs a legal strike, set its recovery timer to:

```text
ceil(attack_recovery_seconds / physics_dt)
```

The timer decreases once at the beginning of each later physics substep. A
soldier that stays in contact therefore strikes repeatedly at this cadence; it
does not need to separate and re-enter contact. Recovery affects only strike
availability, not movement, aiming, collision, or vulnerability.

## 14. Health, death, and simultaneity

For each target, sum all directed strike damage calculated from the same
pre-damage state:

```text
health_j = max(0, health_j - sum_i(damage_i_to_j))
alive_j = health_j > 0
```

Apply all health changes simultaneously. A soldier killed during a substep may
still deal damage that was valid at the start of that same damage phase. It
also receives the contact impulse and penetration correction already identified
for the first collision iteration. It is excluded from later collision
iterations and physics substeps.

There is no healing, armor, friendly fire, invulnerability window, body, corpse
collision, or random damage roll in version 0.3.

## 15. Episode termination

The battlefield is tiled by a radius-13 axial hex grid containing 547
territory cells. Control is a pure function of present force, computed as a
soft influence field after every decision step. Each living soldier
contributes `(h / H)^2 * exp(-0.5 * (d / sigma)^2)` influence to every
cell, where `d` is the distance from the soldier's center to the cell
center, `sigma = control_radius / 2`, `h` is the soldier's current health,
and `H` is the initial health. The quadratic health weight (added in
version 0.8) makes wounded soldiers project sharply less control — a
soldier at half health projects one quarter — so damage dealt moves the
score continuously instead of only at the kill threshold. Contributions
are quantized to `1 / 2^20` fixed-point units before summation so control
is exactly independent of soldier iteration order. With `I_0` and `I_1` the teams'
summed influences on a cell, the control shares are

```text
share_t = I_t / (I_0 + I_1 + kappa)        kappa = 0.125
```

so scoring requires presence: unreached ground remains neutral mass, and a
lone soldier standing on a cell holds about 89% of it. (Version 0.6 briefly
allocated the whole map by pure influence ratio; it subsidized huddled
armies whose hinterland required no presence, and was withdrawn.) Scoring
uses the continuous shares; the ternary owner used for display and local
observations is the team whose share exceeds one half, otherwise unowned.
Dead soldiers contribute nothing, and control is never remembered: shares
are recomputed from living positions every step. (The initial symmetric
ownership constant survives only as the pre-first-step reset value shown at
frame zero, and control shares are zero until the first step.)

Three non-overlapping radius-1 strongpoints are centered at axial coordinates
`(0, -8)`, `(0, 0)`, and `(0, 8)`. Each contains seven tiles. Every strongpoint
tile has territory weight `90` (raised from `30` in version 0.9); every other
tile has weight `1`, for a total territory weight of `3025`. Strongpoints
therefore carry ~62% of the score: holding the contested center decisively
outweighs out-spreading an opponent, so territory can no longer be won while
avoiding contact. (Measured at weight 30, a scripted strongpoint-seeking
charger beat a dispersing policy in only 6 of 32 games; at 90 its mean
control advantage plateaus at +0.25.)

Scoring is the time integral of held control. After each decision step's
influence update, every environment accumulates
`sum_cells(weight * (share_0 - share_1)) / total_weight` into its advantage
integral. An episode always runs for
`maximum_episode_seconds / decision_dt` decision steps, including when a team
has no living soldiers. At the end, a positive integral is a team-0 win, a
negative one a team-1 win, and a magnitude within `1e-9` is a draw. Health,
damage, and elimination affect the result only through the control their
presence or absence projects — which, because dead soldiers project nothing,
makes attrition directly territorial.

## 16. Default configuration

| Parameter | Default |
|---|---:|
| Soldiers per team | `256` |
| World width | `60.0` |
| World height | `120 / sqrt(3)` |
| Soldier radius | `0.42` |
| Initial health | `100.0` |
| Decision timestep | `0.10 s` |
| Physics substeps | `4` |
| Collision iterations per substep | `10` |
| Maximum running speed | `6.0` |
| Physics speed limit | `9.0` |
| Movement acceleration | `11.0` |
| Linear drag | `1.8 s^-1` |
| Maximum attack turn rate | `3.4 rad/s` |
| Allied restitution | `0.12` |
| Enemy restitution | `1.25` |
| Wall restitution | `0.45` |
| Collision relaxation | `0.95` |
| Attack recovery | `0.40 s` |
| Damaging arc | `120 degrees` |
| Base strike damage | `5.0` |
| Minimum charge damage speed | `1.6` |
| Minimum pair closing speed | `0.20` |
| Charge damage scale | `5.0` |
| Flank/rear damage multiplier | `2.0` |
| Maximum episode duration | `45.0 s` |
| Control radius | `8.0` |
| Territory grid | radius `13` (`547` cells) |
| Strongpoints | radius `1` at `(0,-8)`, `(0,0)`, `(0,8)` |
| Strongpoint tile weight | `30` |
| Planar slope height | scenario-defined; `0.0` by default |
| Slope gravity coefficient | `24.0` |
| River width | scenario-defined; `0.0` by default |
| Bridge width | `10.0` |

## 17. Required invariants

An implementation must maintain these invariants after every completed physics
substep:

- All living positions and velocities are finite.
- All living centers are inside the world and outside blocked terrain.
- Speed does not exceed `physics_speed_limit`, within floating-point tolerance.
- Health is finite and non-negative.
- Dead soldiers have zero velocity and cannot affect living soldiers.
- Each unordered physical pair is resolved once per collision iteration.
- Each ready soldier selects at most one strike target per damage phase.
- Pair effects and health changes are simultaneous rather than dependent on
  soldier iteration order.
- Separate batched environments cannot interact.
- Collision broad-phase overflow must be detected and reported; it must never
  silently omit contacts.

Bitwise equality between CPU and GPU is not required. Short controlled scenes
must agree within documented floating-point and iterative-solver tolerances.

## 18. Minimum conformance cases

Before optimization, a new implementation should pass at least these cases:

1. **Movement/facing independence:** moving north while aiming east changes
   `y` velocity without rotating the attack surface north.
2. **Acceleration limit:** a resting soldier does not reach full running speed
   in one decision step, but converges to it under a held control input.
3. **Turn limit:** a large aim change advances by exactly
   `maximum_turn_rate * decision_dt` when unobstructed.
4. **Coincident discs:** two soldiers at the same position separate
   deterministically to at least `0.99D` with default solver settings.
5. **No friendly damage:** allied collision produces physical response and zero
   damage.
6. **Own-speed charge:** with damage threshold `0`, damage scale `1`, and
   frontal target vulnerability, attacker inward speed `4` deals `16` damage
   regardless of target speed.
7. **Binary arc:** a target at 59 degrees receives the complete strike, while a
   target at 61 degrees receives none for a 120-degree arc.
8. **Flank damage:** the same charge against the target's side deals `32`
   damage with multiplier `2`.
9. **Single target:** a ready soldier touching several eligible enemies strikes
   only the most centered one.
10. **Repeated contact:** a stationary frontal contact deals one base strike,
    none during recovery, and another immediately when recovery reaches zero.
11. **Slope-only advantage:** mirrored controls on a positive slope yield lower
    uphill speed and higher downhill speed without a team or damage modifier.
12. **River collision:** a disc is projected out of water, while the same path
    through the bridge remains passable.
13. **Simultaneous elimination:** two lethal attacks in the same phase kill both
    soldiers without ending the episode; the control integral alone determines
    the result.
14. **Weighted control:** a soldier holding the center strongpoint cluster
    outscores an opponent holding a similar-sized disc of plain tiles.
15. **Influence control:** equidistant opposing soldiers split a cell below
    the ownership threshold; a sole nearby soldier majority-owns it; moving
    far away or dying releases it to neutral the same step, and the
    fixed-point sums make the result independent of soldier order.
16. **Score observability:** every soldier's observation carries its team's
    current signed control advantage and the banked advantage integral as
    fractions of the episode, negated in the enemy-relative view.

## 19. Explicit non-goals for version 0.3

The following are intentionally absent unless a later rules version adds them:

- rotational rigid-body physics;
- heterogeneous mass, size, health, or unit classes;
- projectiles, weapon inventory, attack animations, or explicit attack buttons;
- pair contact history, a distinct attrition/pressure system, or movement
  suppression after attacking;
- direct elevation or high-ground combat bonuses;
- stamina, morale, suppression, healing, or resurrection;
- friendly fire;
- fluid simulation for rivers;
- collision-based damage from allied contact or walls;
- variable timestep simulation;
- a requirement to use dense all-pairs collision checks.

These omissions are deliberate. The first implementation should establish that
coherent tactics can emerge from movement, facing, contact physics, damage,
and terrain before adding more mechanics.
