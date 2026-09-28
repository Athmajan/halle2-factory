# Plan: moving-UE CIR study + TR 38.901 comparison on Halle2 (FINAL 2026-09-29, NOT started)

**HOLD:** do not start any step until the user says so. Another agent is running MacGyver-framework data collection on
this machine; nothing here may compete with it (no GPU/CPU-heavy jobs, no Jupyter kernel restarts, no shared-env
package installs) until the user confirms it has finished.

Goal: repeat the `~/inf-factory-scene` workflow on the 6G Rocket "Halle2" scene: one TX, UE(s) walking through the
hall, a per-tick CIR from Sionna RT, then DS / ASA / K-factor / LoS probability / path loss / fade statistics compared
with the TR 38.901 InF numbers. Waiting for the user's go.

## What carries over from inf-factory-scene (reuse, do not rewrite)
| Piece | Source | Change needed for Halle2 |
|---|---|---|
| Per-tick solver | `scripts/cir_common.py: solve_full` (1 rx per call, cap 1e6, 1e7 rays, depth 10, specular+diffraction, K=200 strongest valid paths + angles) | scene path, FC, TX; heavier mesh -> re-run the convergence test |
| Convergence test | `scripts/converge_nlos_test.py` (coherent sum\|a\|^2 vs radio map on ~400 NLoS points) | new points, new radio map |
| Motion | `scripts/agent_motion.py` (`PathMover`, random waypoint, ORCA), `run_cir_multiue.py` (0.5 m/s, tick = lambda/10, checkpoint + resume) | `FreeSpaceGrid` reads `clutter.json` and a 120x60 m box -> needs a mesh-derived walkable grid |
| LoS flag | `validate_inf_stats.py: los()` (Mitsuba ray cast) | none (generic) |
| Spec stats | `build_team_report_data.py` (DS, ASA, K pooled LoS/NLoS) | spec medians recomputed for this hall's V/S and freq |
| PL vs spec | `compare_pl_spec.py` | InF sub-scenario per the clutter audit |
| Lessons | STATUS.md sec. 5 | 1 rx/call only (shared path buffer); >=1400 rays/cell for 5 cm radio maps; sliding-window (10-20 lambda) fade normalisation, not whole-route mean |

## User decisions (2026-09-29)
1. TX height 8 m (high BS). Note: TX (10,40,8) gave 0 paths (inside roof structure, roof z 7.3-13 m) -> pick an (x,y)
   where 8 m is clear of the roof/trusses and metal; if none works, report back before changing the height.
2. 3.85 GHz (lambda = 7.79 cm). Bandwidth 100 MHz, NF 9 dB kept from the previous study (my assumption, not restated by user).
3. TX power 15 dBm -> SNR = path gain + 15 + 85 dB (noise -85 dBm). Power is applied downstream (PathSolver ignores it).
4. 4 UEs, no other moving objects. Motion from the box-delivery task driven by MAPPO or a simple agent, deliberately
   not straight lines all the time (hard to learn/predict).
5. Compare against both references we have: TR 38.901 spec values AND the inf-factory-scene synthetic hall results
   (Halle2 = third data set). No single sub-scenario choice needed; report the audit numbers alongside.
6. Materials: keep as exported if Sionna maps them to ITU (it does) and they make sense. The audit only flags
   something clearly wrong (e.g. the 1.3 M-face "Doors" mesh at z 0-2 m labelled glass).

## Prior model being reused (read before building)
- `~/wisegrt-eval/scripts/coverage_marl_env.py` (632 lines): 4 UEs doing continuing box relocation. Poisson box arrivals
  at a pickup wall; dropoff drawn from a discrete non-uniform goal set on the opposite side; box types small/big with
  per-trip speed and unload-time ranges; battery drain -> detour to 1 of 8 charging stations (fast/slow) then resume;
  Gauss-Markov heading wander (`_gauss_markov_step`, circular blend, clamped deviation); random door choice per trip.
  UEs also halt when SNR < threshold.
- `mappo_{env_wrapper,networks,train}.py`: CTDE MAPPO (shared actor, centralised critic, GAE, continuing task),
  but there the agents were the 2 TX coverage agents; UEs were scripted.
- Design notes: `~/ocudu/tasks/multimodal-scheduler-comparison/TASK_DRIVEN_UE_MOBILITY_AND_ANTICIPATORY_WORLD_MODEL_PLACEMENT.md`
  (RWARE-style task rationale: structured but unpredictable motion).

## PHASE SPLIT (user, 2026-09-29)
- **Phase 1 (no ray tracer):** 4 UEs deliver boxes from one side of the Halle2 floor to the other, reusing the tested
  box-delivery model (coverage_marl_env.py task layer), with a MAPPO agent trained to minimise an objective. Output:
  trained policy + 4-UE trajectories (positions, timestamps, task state). Only the 2-D walkable map from the scene is needed.
- **Phase 2 (ray tracer):** TX 8 m, 3.85 GHz, 15 dBm; CIRs along the Phase 1 trajectories; stats vs spec and inf-factory hall.

### Phase 1 objective - DECIDED: minimise mean box sojourn time
**Minimise mean box sojourn time** (arrival at pickup -> dropped off). Per-step team reward
r = -(number of boxes in the system: queued + being carried) - c_col * collisions - c_batt * (UE with empty battery).
By Little's law, maximising sum(r) = minimising mean sojourn time for the given arrival rate, so the reward is dense
(every tick) rather than a sparse delivery bonus, and UEs are pushed to cross the floor quickly and share the work.
Not chosen: (b) maximise deliveries per hour (sparse, same optimum at steady state if the queue is stable);
(c) (b) minus energy / distance travelled (shorter, straighter paths - works against the "not straight" goal);
(d) minimise worst-case (p95) sojourn time (fairness, harder to train).
What MAPPO controls: per-UE motion (heading change + speed within the task-state range) toward the current task
goal; optionally also which queued box a free UE takes (task allocation). Plan: motion only first, allocation second.
Arrival rate must be set so 4 UEs keep the queue stable but busy (utilisation ~0.7-0.8), otherwise the objective
saturates; measured with the simple-agent baseline first.
Training: CTDE MAPPO, shared actor, centralised critic (reuse mappo_networks.py + GAE from mappo_train.py), vectorised
numpy env on the 0.1 m walkable grid, CPU/GPU, no Sionna.

### Phase 1 motion model: variable speed + ORCA (user, 2026-09-29)
- **Kinematics:** each UE has its own speed v in [0, v_max(task state)] (empty / small box / big box caps) and heading.
  Speed changes only through bounded acceleration/braking: |dv/dt| <= a_max (accelerate), braking up to b_max (> a_max),
  heading rate <= w_max. Proposal (my numbers, to confirm): a_max 0.5 m/s^2, b_max 1.0 m/s^2, w_max 90 deg/s,
  v_max empty 1.0, small 0.9, big 0.5 m/s. Control step 0.1 s (trajectory later resampled per lambda/10 of distance in Phase 2).
- **MAPPO action (per UE, continuous):** (longitudinal accel in [-b_max, a_max], heading rate in [-w_max, w_max]) ->
  a *preferred* velocity for the next step. The policy therefore learns when to speed up and when to brake.
- **ORCA as the safety layer:** the preferred velocities of all 4 UEs go through RVO2 ORCA (python-rvo2 is already
  installed and wrapped in `~/inf-factory-scene/scripts/agent_motion.py: OrcaGroup`), which returns the closest
  collision-free velocity; that is what is executed. Two changes vs the old wrapper:
  1. static obstacles: the old OrcaGroup gave ORCA only the other agents (A* handled clutter). Here the policy does not
     plan with A*, so the Halle2 obstacles go into ORCA too (`addObstacle` with simplified polygons of the
     non-walkable region from the audit map) - DECIDED by user.
  2. RVO2 changes velocity instantly; after ORCA the result is clipped to the accel/brake/turn limits. Clipping can
     break ORCA's guarantee, so a residual-collision check is logged (target: 0 contacts).
- **Reward additions:** besides the chosen objective (default: minimise mean box sojourn time), a small penalty on the
  ORCA correction |v_orca - v_pref| so the policy learns to brake/steer itself instead of leaning on ORCA, plus a
  jerk penalty on |dv| to avoid bang-bang speed. Actual contacts stay heavily penalised.
- **Obs additions:** own speed, and for the 3 other UEs relative position AND relative velocity (needed to anticipate).
- **Baseline:** same kinematics + ORCA, but the preferred velocity comes from A* toward the goal at v_max (no learning),
  so the benefit of the learned accelerate/brake behaviour is measurable (sojourn time, ORCA interventions, speed profile).

## Motion design for Halle2 (new: learned UE motion)
Two layers:
- **Task layer (scripted, ported from coverage_marl_env.py):** box arrivals, pickup/dropoff/charging assignment, box type,
  wait/unload times, battery. It hands each UE its current goal. Port it, replacing the 4-room geometry (rows/doors,
  `is_valid_floor`) with the Halle2 walkable grid; zones placed from the audit map.
- **Motion layer (learned):** MAPPO over the 4 UEs, shared actor + centralised critic (reuse the networks/GAE code).
  Obs per UE: own pos/heading/speed, goal vector, local occupancy patch (e.g. 16x16 around it at 0.5 m), relative
  positions of the other 3 UEs, carrying/box type, battery. Action: heading change + speed within the task-state
  speed range (continuous, or ~8 headings x 3 speeds discrete). Reward: progress to goal + delivery bonus
  - collision (UE-UE <0.6 m, obstacle clearance <0.5 m) - small time cost. Training is geometry-only on the 2-D grid
  (no ray tracing in the loop; UEs are not in the RT scene, so their motion does not change each other's channels).
  SNR-halt rule from the old env: off by default (TX fixed, not an agent here) - confirm.
- **Why paths are not straight:** random pickup/dropoff/charging goals, box-type speeds and waits, learned policies
  that swerve around each other and clutter, and a stochastic policy at rollout. The Gauss-Markov wander can be added
  on top of the policy's action if the trajectories still look too regular.
- **Simple-agent baseline (built first, also the fallback):** A* on the grid + Gauss-Markov wander + ORCA avoidance
  (`~/inf-factory-scene/scripts/agent_motion.py` has PathMover/ORCA). Used to debug the task layer and as the
  comparison for the MAPPO trajectories (path straightness, turn-rate, predictability).
- **Rollout -> CIR sampling:** trajectories are generated first (cheap), ray tracing afterwards. Sample each UE by
  distance travelled (every lambda/10 = 7.79 mm) and store timestamps; while a UE waits/unloads/charges its channel is
  static (nothing else moves), so one solve covers the whole wait. This keeps the RT cost proportional to distance,
  not time, even with variable speeds.

## Steps
0. **Scene audit** (no RT): walkable area at UE height (slice meshes 0.1-2 m), clutter density/height/size, V/S,
   roof clearance map (for the 8 m TX), check of odd meshes/materials. Output: top-down map + audit table + spec
   values for this hall (DS from V/S, ASA, K, P_LoS k).
1. **Walkable grid** (0.1 m cells, 0.5 m clearance, largest connected region) as a generalised `FreeSpaceGrid`.
2. **Task zones** on the map: pickup side, dropoff goal set, charging stations (proposal from the audit, user to OK).
3. **TX at 8 m + radio map** at 1.5 m, 3.85 GHz, 15 dBm: pick TX (x,y) (user to OK), baseline SNR map on the 3D scene.
4. **Motion:** (a) port task layer + simple-agent baseline, check deliveries/throughput and no wall/UE collisions;
   (b) MAPPO env + training (CPU/GPU, minutes-hours); (c) roll out 4 UEs x T min, compare path statistics with
   the baseline; plot trajectories.
5. **Convergence test** on ~400 NLoS trajectory points (1 rx/call; rays/cap vs radio map within ~1 dB; path count
   stable); measure ms/point on this 5.6 M-face mesh -> time budget for step 6.
6. **CIR run**: 4 UEs, distance-sampled points, background segments with checkpoint/resume. Estimate: 30 min at
   ~0.6 m/s mean with ~70% moving ~ 750 m/UE ~ 96k points/UE; at 80-200 ms/point ~ 2-5 h per UE (to be fixed by step 5).
7. **Analysis vs the two references**: DS, ASA, K (LoS), P_LoS(d), PL vs InF formulas; narrowband/wideband/per-RB
   fade CDFs vs Rayleigh (sliding 10-20 lambda window), level-crossing rate / fade duration, SNR time series at 15 dBm.
   Side-by-side table: TR 38.901 | inf-factory-scene hall | Halle2.

## Still to confirm when starting (proposals will be shown from the audit map)
1. TX (x,y) for the 8 m BS (after the roof-clearance map; Phase 2 only).
2. Task zones on the Halle2 map: pickup side, dropoff goal set, charging stations.
3. Kinematic numbers: a_max 0.5 m/s^2, b_max 1.0 m/s^2, w_max 90 deg/s, v_max 1.0 / 0.9 / 0.5 m/s (empty/small/big), 0.1 s step.
4. Box arrival rate (set from the baseline for ~70-80% UE utilisation) and run length (30 min like before?).
5. Final dataset from MAPPO if it trains to reliable deliveries with 0 contacts; otherwise report and ask.
6. SNR-halt rule: off (default; TX is fixed and not an agent).

## Execution order when the user says go
Phase 1: audit (0) -> walkable grid + ORCA obstacle polygons (1) -> task zones (2, user OK) -> task layer port +
A*/ORCA baseline, tune arrival rate (4a) -> MAPPO env + training (4b) -> 4-UE rollout, trajectory plots, path stats
vs baseline (4c). Deliverable: policy checkpoint, trajectories (pos, vel, t, task state), report.
Phase 2: TX 8 m + SNR radio map (3) -> convergence test (5) -> CIR run (6) -> analysis vs TR 38.901 + inf-factory hall (7).

## Known facts about Halle2 (2026-09-29)
- Loads in Sionna RT 2.0.1 with no errors (one harmless HDRFilm warning). ITU materials map correctly.
- Hall ~150 x 143 m (walls x -60..94, y -23..120), roof z 7.3-13 m, floor plate 222 x 160 m. 5.6 M triangles
  (MetalStuff 3.5 M, Doors 1.3 M). Unused in the XML: `Metall.ply`, `Roof_VIsualization.ply`.
- TX at z = 8 m gave 0 paths (inside the roof structure); TX at (10, 50, 5) m works: 80% radio-map coverage,
  median SNR 23.7 dB at 24 dBm / 100 MHz / NF 9 dB.
- Viewer notebook: `view_halle2.ipynb` (symlinked into Jupyter root as `~/inf-factory-scene/6grocket_halle2`).

## Phase 3 (candidate): replay the Phase 2 channel in MAC-Gyver (checked 2026-09-29, read-only)
MAC-Gyver (`~/MAC-Gyver`, commit a8d0629) is a MAC-level emulator: no IQ/PHY. Per slot it computes per-subcarrier
SNR_k = SNR_avg * |H_k|^2 -> EESM -> Richards BLER curve -> ACK/NACK (`openair2/MAC_EMULATOR/macemu_channel.h:1-15`).
What the channel YAML accepts today:
- Path loss: none / free_space / fixed / UMi, UMa LoS+NLoS (no InF) / global sinusoidal swing, or a per-UE
  `pathloss_trace_csv` ("time_s,ue0,ue1,..."; dB; linear interpolation; loops; wall-clock time)
  (`macemu_channel_yaml.c:649`, `macemu_channel.c:427-462`). Converter already exists: `examples_macemu/sionna_to_pathloss.py`.
- Small-scale fading: `type: tdl` with BUILT-IN tap profiles only (EPA, EVA, ETU, TDLA30, TDLB100, TDLC300, TWO-TAP-100,
  FLAT; `macemu_channel.c:73-110`, max 24 taps) + Jakes Doppler; or `type: cdl` A..E scaled by `delay_spread_ns`
  (MIMO, ULA). Optional move/pause time-warp. No custom taps, no CIR file input.
- Limits: <= 106 PRB (40 MHz at 30 kHz; BLER curves stop there) -> the 100 MHz ray-tracing bandwidth cannot be used
  as is; the ray-traced paths are bandwidth-independent, so taps/SNR are recomputed for 40 MHz.
Options to fit the ray tracer:
- A (no code change): per-UE path-loss trace from Phase 2 + a built-in fading profile. Use the local-mean path gain
  (sliding 10-20 lambda) + CDL/TDL with delay spread and Doppler set to the ray-traced values (hybrid), OR the
  coherent wideband gain + FLAT with doppler 0 (keeps ray-traced time variation, loses frequency selectivity).
  Do not feed coherent (faded) gain AND a fading profile: small-scale fading would be counted twice.
- B (small C change): custom per-UE tap profile (delays/powers) in YAML, fitted to the ray-traced PDP; Jakes on top.
- C (full fidelity, moderate C change): new per-UE CIR trace: taps on a fixed delay grid (sinc taps, as
  `~/inf-factory-scene/scripts/cir_to_srk_file.py` does), amplitudes per tick; the existing per-subcarrier phase
  table stays valid because delays are fixed, only the complex tap gains change per slot. SISO/rank-1.
DECIDED (user, 2026-09-29): full ray-traced CIRs, no radio-map/statistical-fading shortcut. Phase 2 stays the dense
per-tick CIR run (lambda/10 of travel, 1 rx/call, convergence-tested). MAC-Gyver replay = Option C only (per-UE CIR
trace, fixed delay grid, ray-traced complex tap gains per tick; fading profile off). Options A/B are not used.
Phase 3 needs: (1) the C change in macemu_channel.c (new tap source replacing Jakes for traced UEs), (2) CIR->taps
converter for 40 MHz / 106 PRB (reuse cir_to_srk_file.py logic), (3) playback clock: trace time vs slot time
(slot freezes shift UE position), (4) SNR filter so UEs stay attached at 15 dBm. Scope/confirm when Phase 3 starts.
