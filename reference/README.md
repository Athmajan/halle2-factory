# reference/ - code this project reuses (copies, 2026-09-29)

Verbatim copies so the plan can be executed on another host. **Not modified.** Adapt copies into `scripts/`;
do not edit these in place.

## box_delivery/ - task-driven UE mobility + MAPPO (from `~/wisegrt-eval/scripts`, commit 05f0c2e)
- `coverage_marl_env.py`: box-relocation state machine (Poisson arrivals, small/big boxes, pickup/dropoff,
  battery + charging, Gauss-Markov wander). In the original, MAPPO drove 2 TX coverage agents and the 4 UEs were
  scripted; here the task layer is ported and MAPPO drives the UEs (PLAN.md sec. 2). Geometry is the old 4-room grid
  and it expects a radio-map cache (`scene_fixed/...npz`) that is NOT included (not needed for the port).
- `mappo_env_wrapper.py`, `mappo_networks.py`, `mappo_train.py`: CTDE MAPPO (shared actor, centralised critic, GAE).
- `visualize_ue_task_mobility.py`: trajectory animation of the task.
- `TASK_DRIVEN_UE_MOBILITY_AND_ANTICIPATORY_WORLD_MODEL_PLACEMENT.md`: design rationale (from `~/ocudu/tasks/...`).

## inf_factory_scripts/ - ray-tracing pipeline (from `~/inf-factory-scene/scripts`, commit 236cffa)
- `cir_common.py` (scene/solve/powers/DS/K; **1 receiver per PathSolver call**), `converge_nlos_test.py`,
  `run_cir_multiue.py` (+ `agent_motion.py` ORCA/A*, `blockers.py`, `geom.py`), `validate_inf_stats.py` (LoS ray cast),
  `build_team_report_data.py`, `compare_pl_spec.py`, `analyze_finefade*.py`, `cir_to_srk_file.py` (CIR -> sinc taps).
- Hard-coded for the 120x60 m synthetic hall (scene path, FreeSpaceGrid box, TX, FC=3.85 GHz, BW=100 MHz): change
  for Halle2 (20 MHz, TX 8 m, 15 dBm).
- `INF_FACTORY_STATUS_2026-09-21.md`: that project's resume doc (TR 38.901 facts, convergence lessons).
