# Task-Driven UE Mobility + Anticipatory World-Model TX Placement

Written 2026-09-08, prompted by testing whether the Dual-Mind World Model idea
(`V2X-WM`, arXiv:2510.24546 — see `../../multi-modal-sched`'s exploration of it
this session) can replace `~/wisegrt-eval`'s existing MAPPO coverage agents
(`THREE_AGENT_COVERAGE_SCHEDULING_MARL.md`). Extends that doc and
`HETEROGENEOUS_MARL_AND_WORLD_MODELS_LITERATURE_REVIEW.md` rather than
duplicating them — read those first. Same citation convention (WebFetch-
verified vs. search-snippet-only).

## 1. What already exists (don't rebuild)

`coverage_marl_env.py` + `mappo_{env_wrapper,networks,train}.py` (built
2026-09-06/07) already solve the "minimum SNR" half of the user's ask: 2
MAPPO coverage agents, a soft hinge penalty at `-6.9dB` (this project's own
CQI-1 threshold), 3 training runs, converging to mean SNR ~26.5dB with
constraint violation shrinking to ~1.35dB. **UEs currently do a pure
rejection-sampled random walk** (`_step_ue_random_walk`, `coverage_marl_env.py`
lines 128-137) — this is the one piece genuinely still open, and the subject
of this doc.

## 2. Reframing what a "world model" should predict here

The instinct to import DMWM's System1(RSSM)+System2(logic) machinery wholesale
needs one correction first: **DMWM's RSSM exists to approximate a channel
physics model that is otherwise expensive to query live** (Sionna ray tracing
per training step). This project's coverage sub-problem does not have that
problem — `precompute_tx_radio_map_cache.py` already gives an **exact,
non-learned, 47ms-per-solve lookup table** covering the full TX/azimuth
candidate grid. Training a *learned, approximate* dynamics model to replace an
*exact, already-fast* one would be strictly worse, not a genuine adoption of
DMWM's contribution.

**What DMWM's mechanism is actually useful for here**: predicting the one
thing that genuinely isn't known in advance — **where the UEs will be**. Right
now `coverage_marl_env.py`'s TX agents are purely *reactive*: they only ever
see each UE's current position (`_get_observations`), never where it's headed.
An RSSM-style latent-state predictor of future UE trajectory (possibly
System-2-style logic-guided if UE motion follows task rules, not physics rules)
would let TX agents **pre-position** ahead of a UE's move into their room or
toward a door, rather than only reacting once it's already there — directly
mirroring this project's own already-validated finding that position value is
concentrated *anticipatorily, before* a fade/event, not during it
(`~/.claude/projects/.../s4_los_episode_position_value_status.md`, outdoor
S4 work). This reframes the DMWM adoption from "replace the channel model"
(redundant) to "add a UE-motion world model on top of the existing exact
channel cache" (genuinely additive) — closer to the S4/Prophet-S4 predictor
this project already has for a different sub-problem than to DMWM's RSSM
role in its own paper.

**But an anticipatory predictor only pays off if there's something structured
to anticipate.** A memoryless random walk has no predictable future beyond
"probably nearby" — this is exactly why UE mobility has to stop being random
before an anticipatory world model has any real signal to learn. That is the
actual dependency order: fix mobility first, then a UE-motion world model
becomes a meaningful addition rather than a predictor of noise.

## 3. Why not random movement — literature grounding

- **Yoon, Liu, Noble, "Random Waypoint Considered Harmful"** (IEEE INFOCOM
  2003). **search-snippet-only** (WebFetch of the original PDF returned
  HTTP 418; this is nonetheless one of the most heavily-cited, uncontested
  negative results in mobile-network simulation, not a fringe claim).
  Finding: the random waypoint model **never reaches a steady state** — mean
  node speed decays monotonically over the simulation (routing-protocol
  metrics measured to drop up to 40% over a 900s run purely from this
  artifact), because nodes with a slow randomly-drawn speed and a far
  randomly-drawn destination take disproportionately long to finish each leg.
  Directly relevant: `coverage_marl_env.py`'s current walk is a *simpler*
  fixed-step rejection-sampled random walk, not literal random waypoint, so it
  doesn't inherit this specific speed-decay bug — but it inherits the more
  basic problem the critique is really about: **a memoryless random process
  has no task structure, so nothing at the coverage-agent level can be
  anticipated from it** beyond short-horizon diffusion.

## 4. A concrete, citable task structure: RWARE-style box relocation

- **Papoudakis, Christianos, Schäfer, Albrecht, "Benchmarking Multi-Agent Deep
  Reinforcement Learning Algorithms in Cooperative Tasks"** (arXiv:2006.07869,
  NeurIPS 2021 Datasets & Benchmarks — introduces RWARE). **WebFetch-verified**
  (via the reference implementation's README, `github.com/semitable/
  robotic-warehouse`, since the arXiv abstract itself is thin on mechanics):
  agents move on a discrete grid, **pick up a requested shelf, deliver it to a
  workstation, then return it to an empty storage location** — a genuine
  source-to-goal-to-return task cycle, not a random walk, with partial
  observability (each agent sees only a small local window) and sparse
  reward tied to task completion, not movement itself.
- **Direct mapping onto `coverage_marl_env.py`'s 4-room grid**: replace
  `_step_ue_random_walk` with a **goal-directed walk toward a current task
  waypoint** — each UE is assigned a "pickup" location (e.g. always in room A
  or B) and a "dropoff" location (the diagonal-opposite room, forcing a
  traversal through both doors), moves toward its current waypoint using the
  exact same bounded-velocity primitive already shared by UEs and TX agents
  (`_move_toward_point`, `STEP_SIZE`), and on arrival either switches to the
  dropoff leg or gets a freshly (re-)sampled pickup/dropoff pair. This is a
  **minimal code change** — the movement primitive already exists, only the
  target-selection logic changes from "random direction, one step" to "current
  waypoint, multi-step until reached, then reassign."
- This also gives the scenario a genuine wireless-relevant framing rather
  than a borrowed game abstraction: a UE carrying a "box" between two fixed
  zones **is** the real-world task profile of an AGV/AMR in a factory, not a
  metaphor for one.

## 5. Grounding as a real wireless/industrial-IoT scenario (why this is attractive, not just a game reskin)

- Factory-automation/URLLC literature already treats AGV/AMR mobility as
  the reference mobile-node type for indoor industrial wireless research,
  with reliability requirements as strict as 10⁻⁹/<10ms and channel models
  (3GPP TR 38.901) already fit to AGV environments specifically
  (search-snippet-only, multiple 2025-2026 arXiv hits on RL-based resource
  allocation and URLLC scheduling for factory automation systems — titles not
  independently WebFetched this pass, listed for follow-up: arXiv:2208.12662,
  arXiv:2503.05313, arXiv:2501.12792). This confirms task-driven AGV-style
  mobility (not random pedestrian-style movement) is the standard framing
  the wireless-networking literature itself already uses for indoor factory
  scenarios — adopting it here is following convention, not inventing an
  unusual scenario.
- A targeted search this session for "movable AP placement + digital twin +
  mobile robots + RL" found digital-twin-for-manufacturing and digital-twin-
  for-robot-control results, but **no paper combining movable/adaptive
  wireless infrastructure placement with mobile-robot digital twins or
  task-driven mobility** — consistent with, and now a second independent
  confirmation of, `THREE_AGENT_COVERAGE_SCHEDULING_MARL.md` §10.3's earlier
  finding that placement/coverage decisions and motion/channel prediction
  stay in separate literatures. Combining (a) task-driven (RWARE-style) UE
  mobility, (b) an anticipatory UE-motion world model, and (c) learned TX
  placement under a min-SNR constraint, appears to be a genuinely open
  combination.

## 6. Proposed evaluation ladder (answers "a few SNR tests, hard for a basic or RL counterpart")

Sweep the min-SNR floor across several thresholds — reuse `-6.9dB` (CQI 1,
already used) as the easy end, then tighten (e.g. `0dB`, `+5dB`, `+10dB`) —
and compare three tiers at each threshold:

1. **Static/heuristic baseline**: one fixed TX pose per room chosen by an
   offline `RadioMapSolver` sweep for best *average*-case coverage (no
   reaction to UE position at all) — the "basic" counterpart named in the
   ask.
2. **Reactive RL (existing)**: the current MAPPO agents, which see only
   current UE position, no forecast.
3. **Anticipatory world-model RL (proposed)**: MAPPO (or a value-based
   variant) augmented with the UE-motion predictor's forecast (§2) as
   additional observation, under the RWARE-style task-driven mobility (§4)
   — anticipation is only meaningful once motion is structured, so this tier
   depends on §4 being in place first.

**Hypothesis, directly modeled on this project's own prior result**: the gap
between tiers 2 and 3 should *widen* as the min-SNR floor tightens (a stricter
constraint has less slack for a purely reactive policy to recover from a
late reaction) and should be small-to-absent if UE mobility stays random
(nothing to anticipate) — making the mobility-model fix in §4 a *precondition*
for the tier-3 result being meaningful, not an independent side improvement.
Metric per threshold: constraint-violation depth/time-below-floor per UE,
not just mean SNR (mean SNR alone doesn't show *which* UEs suffer, and the
existing runs already show mean-SNR-only optimization can under-serve the
worst-off UE).

## 7. Open questions / not done yet

- Exact UE-motion predictor architecture (RSSM-only vs. RSSM+logic; per-UE
  or joint over all 4) is undesigned — start with the simplest RSSM-only
  version and only add logic-guidance if plain latent prediction proves
  insufficiently consistent over the horizon needed (mirrors this doc's
  general preference for the cheapest thing that could work first, same
  as §6.2's discrete-before-continuous action-space call in the parent doc).
- Whether pickup/dropoff zones should be fixed points or fixed *rooms* (with
  a random point inside) is undecided — fixed rooms is more analogous to
  RWARE's workstation/storage-location structure and avoids the model
  learning to predict an exact point.
- The `-6.9dB` floor and the proposed tighter thresholds (`0/5/10dB`) are not
  yet empirically checked against what the 4-room grid's actual dynamic range
  supports (README's Run 2/3 numbers suggest ~18-26dB mean SNR is achievable,
  so `+10dB` as a floor may already be close to infeasible for far corners —
  needs a quick radio-map-cache check before committing to a sweep range).
- §10.3's three factory-automation/URLLC arXiv citations (2208.12662,
  2503.05313, 2501.12792) are search-snippet-only and need WebFetch
  verification before citing in an actual paper draft.
- No code written yet — this doc is the research/design pass the user asked
  for, not an implementation.
