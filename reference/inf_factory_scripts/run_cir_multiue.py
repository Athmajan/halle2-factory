"""Multi-UE CIR run with real agent-agent collision avoidance (2026-09-23). Generalizes
run_cir_multiagent.py's single-UE architecture to N simultaneous UEs (each needs its own PathSolver
call per tick - rx=1/call is the only setting that converges, confirmed this session, so N UEs cost
N x a single UE's per-tick time, not a shared cost). Two motion modes:
  A: random-waypoint (existing model), now with ORCA collision avoidance added (the gap flagged
     all session - movers previously had none).
  B: corner-to-diagonal-corner (matches the LiDAR-beam-prediction paper's ORCA setup as closely as
     this scene allows - see agent_motion.py's make_corner_diagonal_goal docstring for exactly
     where this deviates from the paper, e.g. our scene's clutter needs A* routing theirs didn't).
--n_agv 0 for this run (per the user's explicit request) - AGVs, if present, always use Mode A
goal-picking regardless of --mode, and join the same ORCA group as the UEs.
"""
import sys, os, glob, json, time, argparse, numpy as np
sys.path.insert(0, "scripts")
from cir_common import *
from agent_motion import FreeSpaceGrid, PathMover, OrcaGroup, random_waypoint_goal, make_corner_diagonal_goal
from blockers import make_blocker, floor_position, BLOCKER_SPECS
from sionna.rt import Receiver

ap = argparse.ArgumentParser()
ap.add_argument("--n_ue", type=int, default=4)
ap.add_argument("--n_agv", type=int, default=0)
ap.add_argument("--mode", choices=["A", "B"], default="A")
ap.add_argument("--sim_seconds", type=float, default=60.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--spotcheck", type=int, default=0, help="if >0, run only this many ticks, no checkpointing")
ap.add_argument("--scene_dir", default="scenes/inf_sl_r30_s0_concrete_ceiling")
ap.add_argument("--resume_from", default=None,
                 help="path to a previous segment's final multiue_cir.npz - if given, UEs start "
                      "from their exact final positions there and the RNG stream continues from "
                      "its saved final state (a companion <path>.rng_state.json next to it), "
                      "instead of a fresh random start. For seamless multi-segment datasets.")
ap.add_argument("--out_tag", default="",
                 help="appended to the output dir name, e.g. 'seg2', to avoid collisions "
                      "between segments of the same (n_ue,mode,n_agv,seed) run")
a = ap.parse_args()

UE_SPEED = 0.5
UE_RADIUS = 0.3          # ASSUMPTION: no real UE/body size established; small enough to not force wide berths
AGV_RADIUS = 1.677       # circumscribed circle of the real 3x1.5 m footprint (sqrt(1.5^2+0.75^2))
dt = (LAM / 10) / UE_SPEED
N_TICKS = a.spotcheck if a.spotcheck > 0 else int(round(a.sim_seconds / dt))
CAP, SAMPLES, CHUNK = 1e6, 1e7, 500
tag = f"_{a.out_tag}" if a.out_tag else ""
OUT = f"results/multiue_n{a.n_ue}_mode{a.mode}_agv{a.n_agv}_seed{a.seed}{tag}"; os.makedirs(OUT, exist_ok=True)

dB = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:7.1f}s]", *x, flush=True)
log(f"n_ue={a.n_ue} n_agv={a.n_agv} mode={a.mode} dt={dt*1e3:.2f}ms N_TICKS={N_TICKS} ({N_TICKS*dt:.1f}s simulated) "
    f"cap={CAP:.0e} samples={SAMPLES:.0e} resume_from={a.resume_from}")

rng = np.random.default_rng(a.seed)
resume_ue_pos0 = None
if a.resume_from:
    prev = np.load(a.resume_from)
    resume_ue_pos0 = prev["ue_pos"][-1]  # (n_ue, 2), exact final positions of the previous segment
    rng_state_path = a.resume_from.replace(".npz", ".rng_state.json")
    if os.path.exists(rng_state_path):
        rng.bit_generator.state = json.load(open(rng_state_path))
        log(f"resuming from {a.resume_from}: UE final positions {resume_ue_pos0.tolist()}, "
            f"RNG state restored from {rng_state_path}")
    else:
        # segment predates the RNG-state-saving feature (e.g. the original 30-min run) - position
        # continuity is preserved (no teleport) but the RNG stream restarts fresh at --seed, not a
        # bit-for-bit continuation of what an uninterrupted single run would have produced.
        log(f"resuming from {a.resume_from}: UE final positions {resume_ue_pos0.tolist()}, "
            f"but NO {rng_state_path} found (segment predates RNG-state saving) - RNG stream "
            f"restarts fresh at --seed={a.seed}, position continuity only")

grid = FreeSpaceGrid(a.scene_dir)
agv_speed_fn = lambda r: r.uniform(0.5, 1.0) * BLOCKER_SPECS["agv"]["max_speed_mps"]
agv_dwell_fn = lambda r: r.exponential(7.5)
ue_speed_fn = lambda r: UE_SPEED
ue_dwell_fn = lambda r: 0.0

if a.mode == "A":
    ue_movers = [PathMover(grid, rng, resume_ue_pos0[u] if resume_ue_pos0 is not None else grid.sample_point(rng),
                            ue_speed_fn, ue_dwell_fn, random_waypoint_goal)
                 for u in range(a.n_ue)]
else:
    corners = np.array([[0.0, 0.0], [grid.L, 0.0], [0.0, grid.W], [grid.L, grid.W]])
    goal_b = make_corner_diagonal_goal(grid)
    ue_movers = []
    for i in range(a.n_ue):
        c = i % 4
        pos0 = resume_ue_pos0[i] if resume_ue_pos0 is not None else grid.nearest_free(corners[c])
        ue_movers.append(PathMover(grid, rng, pos0, ue_speed_fn, ue_dwell_fn, goal_b, init_state={"corner_idx": c}))
    log(f"mode B: UE start corners = {[i % 4 for i in range(a.n_ue)]}")

agv_movers = [PathMover(grid, rng, grid.sample_point(rng), agv_speed_fn, agv_dwell_fn, random_waypoint_goal)
              for _ in range(a.n_agv)]

all_movers = ue_movers + agv_movers
radii = [UE_RADIUS] * a.n_ue + [AGV_RADIUS] * a.n_agv
max_speeds = [UE_SPEED] * a.n_ue + [BLOCKER_SPECS["agv"]["max_speed_mps"]] * a.n_agv
orca = OrcaGroup(dt, all_movers, radii, max_speeds)

sc = make_scene()
agv_objs = [make_blocker("agv", name=f"agv{i}") for i in range(a.n_agv)]
if agv_objs: sc.edit(add=agv_objs)


def solve_one(pos_xy, seed):
    for n in list(sc.receivers): sc.remove(n)
    sc.add(Receiver("ue", position=[float(pos_xy[0]), float(pos_xy[1]), 1.5]))
    return solve_full(sc, np.array([pos_xy]), samples=SAMPLES, max_num_paths_per_src=CAP, seed=seed, K=200)


if a.spotcheck > 0:
    log("SPOT-CHECK MODE (no checkpointing)")
    for k in range(N_TICKS):
        orca.step()
        for i, obj in enumerate(agv_objs):
            obj.position = floor_position("agv", agv_movers[i].pos[0], agv_movers[i].pos[1])
        t_tick0 = time.time()
        nvalid_list, inc_list = [], []
        for u in range(a.n_ue):
            res = solve_one(ue_movers[u].pos, seed=k * 1000 + u)
            nv = (np.abs(res[0][0]) > 0).sum(); inc = (np.abs(res[0][0]) ** 2).sum()
            nvalid_list.append(int(nv)); inc_list.append(dB(inc))
        if k % 10 == 0 or k == N_TICKS - 1:
            log(f"  tick {k}: {time.time()-t_tick0:.3f}s/tick(all UEs) "
                f"UE_pos={[m.pos.round(2).tolist() for m in ue_movers]} "
                f"valid_paths={nvalid_list} inc_dB={[f'{x:.1f}' for x in inc_list]}")
    log(f"spot-check done: {N_TICKS} ticks in {time.time()-t0:.1f}s ({(time.time()-t0)/N_TICKS:.3f} s/tick, "
        f"{(time.time()-t0)/N_TICKS/max(a.n_ue,1)*1000:.1f} ms/UE-solve)")
    sys.exit(0)

# ---- full checkpointed run
ckpt_dir = f"{OUT}/_ckpt"; os.makedirs(ckpt_dir, exist_ok=True)
final_path = f"{OUT}/multiue_cir.npz"
if os.path.exists(final_path):
    log(f"{final_path} already exists, nothing to do"); sys.exit(0)

existing = glob.glob(f"{ckpt_dir}/chunk_*_*.npz")
ends = [int(os.path.basename(f).split("_")[2].split(".")[0]) for f in existing]
start = max(ends) if ends else 0
if start:
    log(f"resuming from tick {start}/{N_TICKS} ({len(existing)} chunks found) - NOTE: mover/ORCA state is not "
        f"checkpointed, only per-tick data, same caveat as run_cir_multiagent.py")

td0 = time.time()
for i in range(start, N_TICKS, CHUNK):
    j = min(i + CHUNK, N_TICKS)
    A_ = [[] for _ in range(a.n_ue)]; TAU_ = [[] for _ in range(a.n_ue)]
    PHI_ = [[] for _ in range(a.n_ue)]; TH_ = [[] for _ in range(a.n_ue)]; DR_ = [[] for _ in range(a.n_ue)]
    UEPOS, AGVPOS = [], []
    for k in range(i, j):
        orca.step()
        for idx, obj in enumerate(agv_objs):
            obj.position = floor_position("agv", agv_movers[idx].pos[0], agv_movers[idx].pos[1])
        for u in range(a.n_ue):
            res = solve_one(ue_movers[u].pos, seed=k * 1000 + u)
            A_[u].append(res[0]); TAU_[u].append(res[1]); PHI_[u].append(res[2]); TH_[u].append(res[3]); DR_[u].append(res[4])
        UEPOS.append(np.array([m.pos.copy() for m in ue_movers]))
        if a.n_agv: AGVPOS.append(np.array([m.pos.copy() for m in agv_movers]))
    save_kwargs = dict(ue_pos=np.array(UEPOS))
    if a.n_agv: save_kwargs["agv_pos"] = np.array(AGVPOS)
    for u in range(a.n_ue):
        save_kwargs[f"a_ue{u}"] = np.concatenate(A_[u]); save_kwargs[f"tau_ue{u}"] = np.concatenate(TAU_[u])
        save_kwargs[f"phi_ue{u}"] = np.concatenate(PHI_[u]); save_kwargs[f"theta_ue{u}"] = np.concatenate(TH_[u])
        save_kwargs[f"dropped_ue{u}"] = np.concatenate(DR_[u])
    tmp = f"{ckpt_dir}/.tmp_chunk_{i}_{j}.npz"
    np.savez_compressed(tmp, **save_kwargs)
    os.replace(tmp, f"{ckpt_dir}/chunk_{i}_{j}.npz")
    el = time.time() - td0; rate = (j - start) / el if el > 0 else 0; eta = (N_TICKS - j) / rate if rate > 0 else float("nan")
    log(f"{j}/{N_TICKS} ({100*j/N_TICKS:5.1f}%) elapsed {el:6.0f}s rate {rate:5.2f} tick/s ETA {eta:6.0f}s")
    json.dump(dict(done=j, total=N_TICKS, elapsed_s=el, rate_tick_s=rate, eta_s=eta), open(f"{ckpt_dir}/progress.json", "w"))

chunks = sorted(glob.glob(f"{ckpt_dir}/chunk_*_*.npz"), key=lambda f: int(os.path.basename(f).split("_")[1]))
keys = ["ue_pos"] + (["agv_pos"] if a.n_agv else [])
for u in range(a.n_ue): keys += [f"a_ue{u}", f"tau_ue{u}", f"phi_ue{u}", f"theta_ue{u}", f"dropped_ue{u}"]
parts = {k: [] for k in keys}
for f in chunks:
    d = np.load(f)
    for k in keys: parts[k].append(d[k])
out = {k: np.concatenate(v) for k, v in parts.items()}
assert len(out["ue_pos"]) == N_TICKS
np.savez_compressed(final_path, dt=dt, n_ue=a.n_ue, n_agv=a.n_agv, mode=a.mode, seed=a.seed,
                     fc=FC, bw=BW, noise_dbm=NOISE_DBM, **out)
rng_state_path = final_path.replace(".npz", ".rng_state.json")
json.dump(rng.bit_generator.state, open(rng_state_path, "w"))
log(f"saved {final_path} ({os.path.getsize(final_path)/1e6:.0f} MB), {N_TICKS} ticks, "
    f"and {rng_state_path} for seamless continuation")
