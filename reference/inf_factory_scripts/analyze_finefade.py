"""Analysis of the coherent 3.85 GHz CIR data: wideband (100 MHz), narrowband (centre tone), per-RB (273 RB x 12 subcarriers x 30 kHz = 360 kHz),
rms delay spread, K-factor, arrival-azimuth spread, and consistency with the 3.85 GHz radio map."""
import sys, time, os, numpy as np
sys.path.insert(0, "scripts")
from cir_common import *
from sionna.rt import RadioMapSolver
out = "results/finefade_3.85GHz_concrete"; t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:5.0f}s]", *x, flush=True)
NRB, RBW = 273, 360e3; NU = (np.arange(NRB) - (NRB - 1) / 2) * RBW                      # RB centre offsets from the carrier, Hz
snr = lambda pw: 10 * np.log10(np.maximum(pw, 1e-40)) + PTX - NOISE_DBM               # linear path gain -> SNR dB

def metrics(a, tau, chunk=200):
    N = len(a); inc = np.zeros(N); nb = np.zeros(N); wb = np.zeros(N); rb = np.zeros((N, NRB), np.float32)
    for i in range(0, N, chunk):
        A = a[i:i + chunk]; T = tau[i:i + chunk]; kc = max(int((np.abs(A) > 0).sum(1).max()), 1)
        order = np.argsort(-np.abs(A), 1)[:, :kc]; A = np.take_along_axis(A, order, 1); T = np.take_along_axis(T, order, 1)   # strongest kc paths (all valid ones)
        inc[i:i + chunk] = (np.abs(A) ** 2).sum(1); nb[i:i + chunk] = np.abs(A.sum(1)) ** 2
        dt = T[:, :, None] - T[:, None, :]; wb[i:i + chunk] = np.real((A[:, :, None] * np.conj(A[:, None, :]) * np.sinc(BW * dt)).sum((1, 2)))
        E = np.exp(-2j * np.pi * NU[None, :, None].astype(np.float32) * T[:, None, :].astype(np.float32)).astype(np.complex64)
        rb[i:i + chunk] = np.abs(np.einsum("nrk,nk->nr", E, A)) ** 2
    return inc, nb, wb, rb

def az_spread_deg(a, phi):
    p = np.abs(a) ** 2; P = np.maximum(p.sum(1), 1e-40); mu = np.angle((p * np.exp(1j * phi)).sum(1))[:, None]
    d = np.mod(phi - mu + np.pi, 2 * np.pi) - np.pi; return np.degrees(np.sqrt((p * d ** 2).sum(1) / P))

# ---- 3.85 GHz radio map for the consistency check
sc = make_scene(); rm = RadioMapSolver()(sc, cell_size=(.05, .05), samples_per_tx=int(4e9), max_depth=10, center=[60, 30, 1.5], orientation=[0, 0, 0], size=[120, 60], seed=5, diffraction=True)
RM = np.array(rm.path_gain)[0]; log("radio map", RM.shape)
res = {}
for name in ("tracks", "routes"):
    z = np.load(f"{out}/{name}_cir.npz"); a, tau, phi = z["a"], z["tau"], z["phi_r"]; pos, los, pid = z["pos"], z["los"], z["pid"]
    nvalid = (np.abs(a) > 0).sum(1); ok = nvalid > 0
    log(f"[{name}] {len(a)} points, {int((~ok).sum())} with NO valid paths ({100*(~ok).mean():.3f}%), LoS {100*los.mean():.1f}%, paths/pt median {np.median(nvalid[ok]):.0f} max {nvalid.max()}")
    inc, nb, wb, rb = metrics(a, tau); ds = rms_ds(a, tau); K = kfactor_db(a); asa = az_spread_deg(a, phi)
    ix = np.clip((pos[:, 0] / .05).astype(int), 0, RM.shape[1] - 1); iy = np.clip((pos[:, 1] / .05).astype(int), 0, RM.shape[0] - 1); rmv = RM[iy, ix]
    res[name] = dict(pos=pos, pid=pid, los=los, ok=ok, inc=inc, nb=nb, wb=wb, ds=ds, K=K, asa=asa, rm=rmv, s=z["s"], t=z["t"])
    np.savez_compressed(f"{out}/{name}_metrics.npz", snr_rb_db=snr(rb).astype(np.float16), **{k: v for k, v in res[name].items()})
    log(f"[{name}] metrics saved")
np.save(f"{out}/_done.npy", np.array([1]))
log("ANALYSIS STAGE 1 DONE")
