# Halle2 factory: task-driven 4-UE mobility + ray-traced CIRs (+ MAC-Gyver replay)

Final plan agreed with the user on 2026-09-29. **Nothing has been run yet.** Read `STATUS.md` first for the hold
condition and the current state.

Goal: repeat the `~/inf-factory-scene` workflow on the 6G Rocket "Halle2" scene. 4 UEs move through the hall doing a
box-delivery job (learned with MAPPO, ORCA for collision avoidance). Proper ray-traced CIRs are computed along their
trajectories, compared against TR 38.901 and the inf-factory-scene hall, and later replayed in MAC-Gyver.

## 1. Fixed parameters (user decisions, 2026-09-29)

| Item | Value | Note |
|---|---|---|
| Scene | `scene/Halle2.xml` (Mitsuba XML exported from Blender, ITU materials) | Loads in Sionna RT 2.0.1 without errors |
| Carrier | 3.85 GHz (lambda = 7.79 cm) | same as inf-factory-scene fine-fading study |
| **Cell bandwidth** | **20 MHz, 30 kHz SCS, 51 PRB** | changed from 100 MHz on 2026-09-29; matches the earlier OCUDU/SRK cell size |
| Noise | -174 + 10log10(20e6) + 9 dB NF = **-92.0 dBm** | NF 9 dB kept from the previous study (assumption, TR 38.901 Table 7.8-7 UT NF) |
| TX | 1 BS, isotropic V-pol, **height 8 m**, **15 dBm** | (x,y) still to pick: 8 m must clear the roof structure (roof z 7.3-13 m) |
| SNR label | path gain [dB] + 15 + 92.0 | TX power is applied downstream; PathSolver ignores `power_dbm` |
| UEs | **4**, height 1.5 m, **no other moving objects** | UEs are not part of the RT geometry |
| Materials | keep as exported | Sionna maps all `mat-itu_*` to ITU materials; flag only something clearly wrong |
| Channel | **full ray-traced CIRs, per tick** | user: "no half-baked cakes" - no radio-map + statistical-fading shortcut |
| Comparison | TR 38.901 InF values AND inf-factory-scene hall results | Halle2 is the third data set |

## 2. Phase 1 - task-driven motion with MAPPO + ORCA (no ray tracer)

### 2.1 Task layer (scripted, ported)
Port the box-relocation state machine from `~/wisegrt-eval/scripts/coverage_marl_env.py` (tested earlier):
Poisson box arrivals on the pickup side; dropoff drawn from a discrete non-uniform goal set on the opposite side;
small/big box types with their own speed caps and unload-time ranges; battery drain -> detour to one of the charging
stations (fast/slow) -> resume the interrupted task. Replace the 4-room geometry (`_row_of`, doors,
`is_valid_floor`) with the Halle2 walkable grid; zones placed on the audit map. SNR-halt rule: off (TX is fixed).
In that old work MAPPO drove the 2 TX coverage agents and the UEs were scripted; here MAPPO drives the UEs.

### 2.2 Objective (decided): minimise mean box sojourn time
Per-step team reward
`r = -(boxes in system: queued + carried) - c_col*contacts - c_batt*(UEs with empty battery) - c_orca*|v_orca - v_pref| - c_jerk*|dv|`.
By Little's law, maximising sum(r) minimises mean sojourn time at a given arrival rate; dense reward every tick.
The ORCA-correction term makes the policy brake/steer by itself instead of relying on ORCA; the jerk term avoids
bang-bang speed. Arrival rate tuned with the baseline for ~70-80% UE utilisation (stable but busy queue).

### 2.3 Kinematics (decided: variable speed)
Per UE: speed v in [0, v_max(task state)], heading. Speed changes only via bounded acceleration / braking, heading via
bounded turn rate. Proposed numbers (to confirm): a_max 0.5 m/s^2, b_max 1.0 m/s^2, w_max 90 deg/s;
v_max 1.0 (empty) / 0.9 (small box) / 0.5 (big box) m/s; control step 0.1 s.

### 2.4 MAPPO
CTDE: shared actor, centralised critic, GAE, continuing task (reuse `~/wisegrt-eval/scripts/mappo_networks.py` and the
GAE/loop in `mappo_train.py`). Action per UE (continuous): longitudinal acceleration in [-b_max, a_max] + heading rate
in [-w_max, w_max] -> preferred velocity. Observation per UE: own position, heading, speed, goal vector, local
occupancy patch (~16x16 at 0.5 m), relative position AND velocity of the other 3 UEs, box type, battery.
Critic: global state of all UEs + queue. Task allocation (which UE takes which box) scripted first; learned later only
if wanted. Stochastic policy at rollout.

### 2.5 ORCA (decided: static obstacles included)
All 4 preferred velocities go through RVO2 ORCA (`python-rvo2` installed; wrapper `OrcaGroup` in
`~/inf-factory-scene/scripts/agent_motion.py`). Changes vs that wrapper:
1. static obstacles added with `addObstacle` (simplified polygons of the non-walkable region from the audit map);
2. ORCA output clipped to the accel/brake/turn limits; residual contacts are logged (target 0).

### 2.6 Baseline / fallback
Same kinematics + ORCA, preferred velocity from A* toward the goal at v_max plus Gauss-Markov heading wander
(`_gauss_markov_step` from coverage_marl_env.py). Built first to debug the task layer and tune the arrival rate;
also the comparison for MAPPO (sojourn time, ORCA interventions, speed profile, path straightness, turn rate).

### 2.7 Phase 1 deliverable
Policy checkpoint; 4-UE trajectories (t, x, y, v, heading, task state, box type, battery) for the run length (30 min
proposed); trajectory plots; statistics vs baseline.

## 3. Phase 2 - ray-traced CIRs along the trajectories

- **Sampling:** per UE, one CIR every lambda/10 = 7.79 mm of distance travelled, with timestamps. While a UE is stopped
  (waiting, unloading, charging) the channel is static because nothing else moves, so one solve covers the stop.
  RT cost scales with distance, not time.
- **Solver:** `~/inf-factory-scene/scripts/cir_common.py: solve_full` settings as the starting point (1 rx per call,
  depth 10, LoS + specular + diffraction, keep K=200 strongest valid paths + angles). **1 rx per call only**:
  `max_num_paths_per_src` is shared by all receivers in a call and silently truncates paths otherwise
  (inf-factory-scene STATUS.md sec. 5).
- **Convergence test first** (`converge_nlos_test.py` pattern): ~400 NLoS trajectory points, rays and path cap until
  coherent sum|a|^2 is within ~1 dB of the radio map and path counts stop growing; measure ms/point on this
  5.6 M-triangle mesh -> time budget.
- **Run:** background segments with checkpoint/resume (`run_cir_multiue.py` pattern). Rough size: 30 min, ~0.6 m/s
  mean, ~70% moving -> ~750 m/UE -> ~96k points/UE -> ~2-5 h per UE at 80-200 ms/point (fixed by the convergence test).
- **Analysis (20 MHz):** wideband power over the 20 MHz band (sinc-pair formula in `cir_common.powers` with BW=20e6),
  narrowband, per-RB (51 RB); DS, ASA, K-factor (LoS), P_LoS(d), path loss vs InF formulas; fade CDFs vs Rayleigh with
  a sliding 10-20 lambda local mean (not the whole-route mean); level-crossing rate and fade duration from the timed
  series; SNR time series at 15 dBm. Table: TR 38.901 | inf-factory-scene hall | Halle2.
  Note: DS/ASA/K are properties of the paths and do not depend on the cell bandwidth; only the power/fade series do.

## 4. Phase 3 - replay in MAC-Gyver (scope when Phase 2 is done)

MAC-Gyver (`~/MAC-Gyver`, commit a8d0629) is a MAC-level emulator: per slot, SNR_k = SNR_avg * |H_k|^2 per subcarrier
-> EESM -> Richards BLER curve -> ACK/NACK (`openair2/MAC_EMULATOR/macemu_channel.h:1-15`). Its YAML today accepts
path-loss models or a per-UE path-loss trace CSV, plus built-in TDL profiles (EPA/EVA/ETU/TDLA30/TDLB100/TDLC300/
TWO-TAP-100/FLAT, Jakes Doppler) or CDL-A..E. **No custom taps and no CIR input.** Max 106 PRB; a 20 MHz cell
(51 PRB) is inside the BLER-curve range.

Decided: **Option C** - per-UE CIR trace: taps on a fixed delay grid (sinc taps at the 20 MHz cell's sample rate, as
`~/inf-factory-scene/scripts/cir_to_srk_file.py` did at 23.04 Msps for the same cell size), ray-traced complex tap
gains per tick, built-in fading off. The per-subcarrier phase table stays valid because the delays are fixed.
Work items: (1) new tap source in `macemu_channel.c` for traced UEs, (2) CIR -> taps converter, (3) playback clock
(trace time vs slot time; slot freezes shift UE position), (4) gnb.yaml for a 51-PRB cell, (5) check that UEs stay
attached at 15 dBm. The path-loss-trace + statistical-fading option was rejected by the user.

## 5. Steps in order

Phase 1
0. Scene audit (no RT): walkable area at UE height (mesh slice 0.1-2 m), clutter density/height/size, V/S, roof
   clearance map for the 8 m TX, odd meshes (the 1.3 M-triangle "Doors" mesh at z 0-2 m labelled glass). Output:
   top-down map, audit table, TR 38.901 reference values for this hall.
1. Walkable grid (0.1 m, 0.5 m clearance, largest connected region) + ORCA obstacle polygons.
2. Task zones (pickup side, dropoff goal set, charging stations) -> **user OK**.
3. Task layer port + A*/ORCA baseline; tune the arrival rate.
4. MAPPO env + training; 4-UE rollout; plots and statistics vs baseline.

Phase 2
5. TX (x,y) at 8 m -> **user OK**; SNR radio map at 1.5 m (15 dBm, 20 MHz) on the 3D view.
6. Convergence test -> time budget.
7. CIR run (background, checkpointed).
8. Analysis and the three-way comparison.

Phase 3: section 4.

## 6. Still to confirm when starting
1. TX (x,y) for the 8 m BS (from the roof-clearance map).
2. Task zones on the Halle2 map.
3. Kinematic numbers in 2.3.
4. Arrival rate (from the baseline) and run length (30 min?).
5. Final dataset from MAPPO if it trains to reliable deliveries with 0 contacts; otherwise report and ask.

## 7. Reuse map
| Need | Source |
|---|---|
| Task state machine, Gauss-Markov wander | `~/wisegrt-eval/scripts/coverage_marl_env.py` |
| MAPPO networks / training loop | `~/wisegrt-eval/scripts/mappo_networks.py`, `mappo_train.py`, `mappo_env_wrapper.py` |
| Task-mobility rationale | `~/ocudu/tasks/multimodal-scheduler-comparison/TASK_DRIVEN_UE_MOBILITY_AND_ANTICIPATORY_WORLD_MODEL_PLACEMENT.md` |
| ORCA wrapper, A* mover, free-space grid | `~/inf-factory-scene/scripts/agent_motion.py` |
| CIR solve, powers, DS, K | `~/inf-factory-scene/scripts/cir_common.py` |
| Convergence test | `~/inf-factory-scene/scripts/converge_nlos_test.py` |
| Multi-UE run with checkpoint/resume | `~/inf-factory-scene/scripts/run_cir_multiue.py` |
| LoS ray cast | `~/inf-factory-scene/scripts/validate_inf_stats.py: los()` |
| Spec stats pooling (DS/ASA/K) | `~/inf-factory-scene/scripts/build_team_report_data.py` |
| PL vs spec | `~/inf-factory-scene/scripts/compare_pl_spec.py` |
| CIR -> sinc taps | `~/inf-factory-scene/scripts/cir_to_srk_file.py` |
| TR 38.901 facts used before | `~/inf-factory-scene/STATUS.md` sec. 2 |
