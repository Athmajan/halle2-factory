"""Ray-traced path loss vs TR 38.901 Table 7.4.1-1 InF-SL (fc=3.5 GHz), split by geometric LoS/NLoS to the TX.
 LoS : PL = 31.84 + 21.5 log10(d3D) + 19 log10(fc)                       (sigma_SF 4.3 dB)
 NLoS: PL = max(PL_LoS, 33 + 25.5 log10(d3D) + 20 log10(fc))             (sigma_SF 5.7 dB)"""
import sys, numpy as np, sionna.rt as rt
from validate_inf_stats import los
scene_dir = sys.argv[1] if len(sys.argv) > 1 else "scenes/inf_sl_r30_s0"; out = "results/power_sweep_" + scene_dir.rstrip("/").split("/")[-1]
d = np.load(f"{out}/path_gain_dB.npz"); pg = d["pg_db"]; x, y, tx = d["x"], d["y"], d["tx"]
X, Y = np.meshgrid(x, y); ok = ~np.isnan(pg)
p = np.c_[X[ok], Y[ok], np.full(ok.sum(), 1.5)]; pl = -pg[ok]
d3 = np.linalg.norm(p - tx, axis=1); scene = rt.load_scene(f"{scene_dir}/hall.xml")
L = los(scene, np.tile(tx, (len(p), 1)), p); fc = 3.5
pl_los = 31.84 + 21.5 * np.log10(d3) + 19 * np.log10(fc); pl_nlos = np.maximum(pl_los, 33 + 25.5 * np.log10(d3) + 20 * np.log10(fc))
fspl = 20 * np.log10(4 * np.pi * d3 * 3.5e9 / 3e8)
bad = pl < fspl - 6                     # more than +6 dB above free space (2-ray max) => not physical
print(f"cells {ok.sum()}; geometric LoS {L.mean()*100:.1f}%; non-physical (gain > FSPL+6 dB): {bad.sum()} at", [(round(float(a),1), round(float(b),1), round(float(c),1)) for a, b, c in zip(p[bad,0], p[bad,1], (fspl-pl)[bad])][:6])
print("\n d3D bin   | LoS: N  sim-PL  spec-PL  diff | NLoS: N  sim-PL  spec-PL  diff   (sim - spec, dB; + = sim loses more)")
for lo, hi in [(1, 10), (10, 20), (20, 30), (30, 45), (45, 60), (60, 90)]:
    r = [f" {lo:3d}-{hi:<3d} m |"]
    for name, m, sp in (("LoS", L, pl_los), ("NLoS", ~L, pl_nlos)):
        s = m & ~bad & (d3 >= lo) & (d3 < hi)
        r.append(f" {s.sum():5d} {pl[s].mean():7.1f} {sp[s].mean():7.1f} {pl[s].mean()-sp[s].mean():+6.1f} |" if s.sum() > 5 else "     -       -       -      - |")
    print("".join(r))
nl = ~L & ~bad; print(f"\nNLoS extra loss over free space: sim {np.mean((pl-fspl)[nl]):.1f} dB (median {np.median((pl-fspl)[nl]):.1f}), spec {np.mean((pl_nlos-fspl)[nl]):.1f} dB;  LoS: sim {np.mean((pl-fspl)[L&~bad]):.1f} vs spec {np.mean((pl_los-fspl)[L&~bad]):.1f}")
