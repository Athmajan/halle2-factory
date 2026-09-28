"""Check a clutter layout against TR 38.901 InF-SL statistics (geometric ray-cast on the real meshes, z=1.5 m both ends).
 Test 1: P(LoS) vs 2D distance vs Table 7.4.2-1:  exp(-d/k), k = -d_clutter/ln(1-r)  (InF-SL/DL)
 Test 2: LoS-state autocorrelation along a UE path; Table 7.6.3.1-2 gives LoS-state correlation distance d_clutter/2 (=5 m).
         NOTE: the spec's 5 m is for the underlying correlated random variable, not the binary state, so this is approximate."""
import sys, json, numpy as np, sionna.rt as rt, mitsuba as mi
D_CL = 10.0

def los(scene, p, q):
    p, q = np.asarray(p, np.float32), np.asarray(q, np.float32)
    v = q - p; dist = np.linalg.norm(v, axis=1)
    ray = mi.Ray3f(o=mi.Point3f(p[:, 0], p[:, 1], p[:, 2]), d=mi.Vector3f(*(v / dist[:, None]).T))
    ray.maxt = mi.Float((dist - 1e-3).astype(np.float32))
    return ~np.array(scene.mi_scene.ray_intersect(ray).is_valid())

def inside_clutter(pts, objs):
    ok = np.zeros(len(pts), bool)
    for o in objs:
        if o["type"] == "box": ok |= (abs(pts[:, 0] - o["cx"]) < o["dx"] / 2 + .1) & (abs(pts[:, 1] - o["cy"]) < o["dy"] / 2 + .1)
        else: ok |= np.hypot(pts[:, 0] - o["cx"], pts[:, 1] - o["cy"]) < o["r"] + .1
    return ok

def run(scene_dir, n=60000, seed=0):
    scene = rt.load_scene(f"{scene_dir}/hall.xml"); cj = json.load(open(f"{scene_dir}/clutter.json")); cl = cj["objects"]
    K = -D_CL / np.log(1 - cj.get("r_target", 0.20))
    rng = np.random.default_rng(seed); L, W = 120, 60
    bs = np.c_[rng.uniform(1, L - 1, n), rng.uniform(1, W - 1, n), np.full(n, 1.5)]
    ue = np.c_[rng.uniform(1, L - 1, n), rng.uniform(1, W - 1, n), np.full(n, 1.5)]
    keep = ~inside_clutter(bs, cl) & ~inside_clutter(ue, cl); bs, ue = bs[keep], ue[keep]
    d = np.linalg.norm(bs[:, :2] - ue[:, :2], axis=1); s = los(scene, bs, ue)
    edges = np.array([5, 15, 25, 35, 50, 65, 80, 100]); rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & (d < hi); dm = d[m].mean()
        rows.append((lo, hi, m.sum(), s[m].mean(), np.exp(-dm / K)))
    # Test 2: 400 straight UE chords (0.5 m steps) vs an independent random BS; points inside clutter are masked out
    # (a UE cannot be inside a machine); autocorrelation uses only pairs of valid points at each lag.
    lags = np.arange(0, 21) * 0.5; num = np.zeros(len(lags)); den = np.zeros(len(lags)); allst = []; cnt = 0
    for _ in range(400):
        b = np.array([rng.uniform(1, L - 1), rng.uniform(1, W - 1)]);
        if inside_clutter(b[None], cl).any(): continue
        a0 = np.array([rng.uniform(1, L - 1), rng.uniform(1, W - 1)]); th = rng.uniform(0, 2 * np.pi)
        t = np.arange(0, 60, 0.5); pts = a0 + np.c_[np.cos(th) * t, np.sin(th) * t]
        inb = (pts[:, 0] > 1) & (pts[:, 0] < L - 1) & (pts[:, 1] > 1) & (pts[:, 1] < W - 1)
        pts = pts[inb]
        if len(pts) < 40: continue
        v = ~inside_clutter(pts, cl); pts3 = np.c_[pts, np.full(len(pts), 1.5)]
        st = los(scene, np.tile([*b, 1.5], (len(pts), 1)), pts3).astype(float)
        allst.append((st, v)); cnt += 1
    mu = np.mean(np.concatenate([st[v] for st, v in allst])); var = mu * (1 - mu)
    for st, v in allst:
        x = st - mu
        for i in range(len(lags)):
            m = v[:len(x) - i] & v[i:]
            num[i] += np.sum(x[:len(x) - i][m] * x[i:][m]); den[i] += m.sum()
    ac = num / den / var; lag1e = next((lags[i] for i in range(len(ac)) if ac[i] < np.exp(-1)), None)
    dm = np.array([(lo + hi) / 2 for lo, hi, *_ in rows]); pm = np.array([r_[3] for r_ in rows])
    k_fit = np.sum(dm ** 2) / -np.sum(dm * np.log(pm))          # LS fit of ln P = -d/k through origin
    return rows, lags, ac, lag1e, cnt, s.mean(), k_fit, K

if __name__ == "__main__":
    print(f"spec LoS-state correlation distance = d_clutter/2 = {D_CL/2} m")
    for sd in sys.argv[1:]:
        rows, lags, ac, lag1e, cnt, mean, k_fit, K = run(sd)
        print(f"\n== {sd}  overall LoS frac {mean:.3f}   fitted k = {k_fit:.1f} m (spec {K:.1f} m)")
        print(" d bin (m)     N   P(LoS) sim   spec exp(-d/k)")
        for lo, hi, nn, ps, pt in rows: print(f" {lo:3d}-{hi:<3d}  {nn:7d}   {ps:6.3f}       {pt:6.3f}")
        print(f" LoS-state autocorr (paths={cnt}): lag 0/2.5/5/10 m =", [round(float(ac[int(l / .5)]), 2) for l in (0, 2.5, 5, 10)], f" 1/e lag ~ {lag1e} m")
