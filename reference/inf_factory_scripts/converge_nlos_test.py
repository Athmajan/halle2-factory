"""Convergence test (2026-09-22): find (receivers_per_call, max_num_paths_per_src, samples_per_src) so that
coherent NLoS power matches the 3.85 GHz radio map within ~1 dB and path counts stop growing, on ~400 NLoS points.
Root cause under test: PathSolver's max_num_paths_per_src buffer is SHARED across all receivers added in one
call (default 1e6; the invalid full run used 2000 rx/call -> ~500 paths/rx budget). Do not use these results for
anything beyond sizing the rerun; report to the user before launching a full run.
"""
import sys, time, os, subprocess, threading, numpy as np
sys.path.insert(0, "scripts")
from cir_common import *
from sionna.rt import Receiver, PathSolver, RadioMapSolver

OUT = "results/converge_nlos_test"; os.makedirs(OUT, exist_ok=True)
t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:6.1f}s]", *x, flush=True)


class MemWatch:
    """Polls nvidia-smi memory.used (MiB) in a background thread to catch the peak during a call."""
    def __init__(self, period=0.3):
        self.period = period; self.peak = 0; self._stop = False; self._t = None
    def _run(self):
        while not self._stop:
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                v = int(out.decode().splitlines()[0].strip()); self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(self.period)
    def __enter__(self):
        self._stop = False; self.peak = 0; self._t = threading.Thread(target=self._run, daemon=True); self._t.start(); return self
    def __exit__(self, *a):
        self._stop = True; self._t.join(timeout=2)


def solve_capped(scene, pts, samples, cap, seed=0):
    """Like cir_common.solve() but with an explicit max_num_paths_per_src (cap), and reports p.valid counts."""
    for n in list(scene.receivers): scene.remove(n)
    for i, (x, y) in enumerate(pts): scene.add(Receiver(f"r{i}", position=[float(x), float(y), 1.5]))
    p = PathSolver()(scene, max_depth=10, max_num_paths_per_src=int(cap), samples_per_src=int(samples),
                      los=True, specular_reflection=True, diffraction=True, seed=seed)
    a, tau = p.cir(out_type="numpy"); a = np.squeeze(a); tau = np.squeeze(tau)
    v = np.array(p.valid).reshape(len(pts), -1)
    return a.astype(np.complex64), tau.astype(np.float32), v


def dB(x): return 10 * np.log10(np.maximum(x, 1e-30))


# ---- 1. ground-truth 3.85 GHz radio map (same recipe as analyze_finefade.py; cached for reuse)
rm_path = f"{OUT}/radio_map_3.85GHz.npz"
sc = make_scene()
if os.path.exists(rm_path):
    RM = np.load(rm_path)["pg"]; log("radio map loaded from cache", RM.shape)
else:
    log("computing 3.85 GHz radio map (0.05 m cells, 4e9 rays)...")
    rm = RadioMapSolver()(sc, cell_size=(.05, .05), samples_per_tx=int(4e9), max_depth=10,
                           center=[60, 30, 1.5], orientation=[0, 0, 0], size=[120, 60], seed=5, diffraction=True)
    RM = np.array(rm.path_gain)[0]
    np.savez_compressed(rm_path, pg=RM)
    log("radio map done", RM.shape)


def rm_lookup(pos):
    ix = np.clip((pos[:, 0] / .05).astype(int), 0, RM.shape[1] - 1)
    iy = np.clip((pos[:, 1] / .05).astype(int), 0, RM.shape[0] - 1)
    return RM[iy, ix]


# ---- 2. pick ~400 NLoS points, spread across tracks for spatial diversity (geometric LoS flag, independent of the truncation bug)
z = np.load("results/finefade_3.85GHz_concrete/tracks_cir.npz")
nlos_idx = np.where(~z["los"])[0]
pick = nlos_idx[np.linspace(0, len(nlos_idx) - 1, 400).astype(int)]
pts = z["pos"][pick]
rm_vals = rm_lookup(pts)
log(f"selected {len(pts)} NLoS points, radio-map path gain range {dB(rm_vals).min():.1f}..{dB(rm_vals).max():.1f} dB "
    f"(mean {dB(rm_vals.mean()):.1f} dB, mean-of-dB {dB(rm_vals).mean():.1f} dB)")
np.savez_compressed(f"{OUT}/test_points.npz", pos=pts, idx=pick, rm=rm_vals)

FREE0 = int(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"]).decode().split()[0])
log(f"GPU free memory at start: {FREE0} MiB")

# ---- 3. sweep combos: (label, rx_per_call, cap, samples)
combos = [
    ("rx1_cap1e6_s1e7",    1,   1e6,  1e7),   # reference: ~no dilution, moderate rays
    ("rx1_cap1e6_s3e7",    1,   1e6,  3e7),   # does more rays alone help (fixed cap)?
    ("rx20_cap2e6_s1e7",   20,  2e6,  1e7),   # 1e5 paths/rx budget
    ("rx20_cap6e6_s3e7",   20,  6e6,  3e7),   # 3e5 paths/rx budget
    ("rx100_cap1e7_s1e7",  100, 1e7,  1e7),   # 1e5 paths/rx budget, bigger batch
    ("rx100_cap3e7_s3e7",  100, 3e7,  3e7),   # 3e5 paths/rx budget
    ("rx400_cap4e7_s1e7",  400, 4e7,  1e7),   # 1e5 paths/rx, single call over all 400
    ("rx400_cap1.2e8_s3e7", 400, 1.2e8, 3e7), # 3e5 paths/rx, single call - watch memory closely
]
# rx1 combos are expensive per-point (per-call scene overhead); use a 40-point subset for those only
sub40 = np.linspace(0, len(pts) - 1, 40).astype(int)

rows = []
for label, rxpc, cap, samples in combos:
    use_pts, use_rm = (pts[sub40], rm_vals[sub40]) if rxpc == 1 else (pts, rm_vals)
    N = len(use_pts)
    nb_all, inc_all, npaths_all, nvalid_all = [], [], [], []
    tC0 = time.time()
    with MemWatch() as mw:
        try:
            for i in range(0, N, rxpc):
                batch = use_pts[i:i + rxpc]
                a, tau, v = solve_capped(sc, batch, samples=samples, cap=cap, seed=1000 + i)
                if a.ndim == 1: a = a[None, :]; v = v[None, :]
                nb_all.append(np.abs((a * v).sum(-1)) ** 2)          # narrowband coherent power (valid taps only)
                inc_all.append((np.abs(a) ** 2 * v).sum(-1))          # incoherent sum
                npaths_all.append(v.sum(-1))
        except Exception as e:
            log(f"{label}: FAILED - {type(e).__name__}: {e}")
            rows.append(dict(label=label, rx_per_call=rxpc, cap=cap, samples=samples, failed=str(e)))
            continue
    dt = time.time() - tC0
    nb = np.concatenate(nb_all); inc = np.concatenate(inc_all); npv = np.concatenate(npaths_all)
    zero_frac = (npv == 0).mean()
    deficit_nb = dB(nb.mean()) - dB(use_rm.mean())
    deficit_inc = dB(inc.mean()) - dB(use_rm.mean())
    per_point_time = dt / N
    log(f"{label}: N={N} rx/call={rxpc} cap={cap:.0e} samples={samples:.0e} | {dt:.1f}s ({1000*per_point_time:.1f} ms/pt) | "
        f"mem peak {mw.peak} MiB (free was {FREE0}) | paths/rx median {np.median(npv):.0f} (p5 {np.percentile(npv,5):.0f}, p95 {np.percentile(npv,95):.0f}) | "
        f"zero-path frac {100*zero_frac:.2f}% | deficit vs RM: incoherent {deficit_inc:+.2f} dB, narrowband(coherent) {deficit_nb:+.2f} dB")
    rows.append(dict(label=label, rx_per_call=rxpc, cap=cap, samples=samples, N=N, time_s=dt, ms_per_pt=1000*per_point_time,
                      mem_peak_mib=mw.peak, gpu_free_mib=FREE0, paths_median=float(np.median(npv)), paths_p5=float(np.percentile(npv,5)),
                      paths_p95=float(np.percentile(npv,95)), zero_frac=zero_frac, deficit_inc_db=deficit_inc, deficit_nb_db=deficit_nb))

import json
with open(f"{OUT}/summary.json", "w") as f: json.dump(rows, f, indent=2, default=str)
log(f"ALL COMBOS DONE, total {time.time()-t0:.0f}s. Summary -> {OUT}/summary.json")
