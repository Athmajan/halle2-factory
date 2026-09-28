import numpy as np, sionna.rt as rt
from sionna.rt import Transmitter, Receiver, PathSolver
FC, BW, PTX, TX = 3.85e9, 100e6, 24.0, [60.0, 30.0, 1.5]  # PTX reverted to 24 dBm (2026-09-22): a
# 2026-09-22 attempt to set this to 15.0 to get a "lower power" CIR dataset had NO effect on any
# PathSolver-based output (confirmed: 15 vs 24 dBm runs of the same seed produced identical `a`,
# delta 0.003 dB = float noise) - Transmitter.power_dbm is only ever consumed by RadioMapSolver
# (sionna/rt/radio_map_solvers/{radio_map,radio_map_solver}.py), never by PathSolver/paths.cir().
# A different TX power scenario for CIR-derived quantities (SNR, CQI, ...) must be applied
# explicitly downstream as pathgain_dB + PTX - NOISE_DBM, not via this constant.
LAM = 299792458.0 / FC
NOISE_DBM = -174 + 10 * np.log10(BW) + 9

def make_scene(xml="scenes/inf_sl_r30_s0_concrete_ceiling/hall.xml"):
    s = rt.load_scene(xml); s.frequency, s.bandwidth = FC, BW
    s.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V"); s.rx_array = s.tx_array
    s.add(Transmitter("tx", position=TX, power_dbm=PTX)); return s

def solve(scene, pts, samples=1e7, seed=0, diffraction=True):
    """pts: (N,2) xy at 1.5 m. Returns complex a (N,P), tau (N,P) seconds (delays normalised per rx; magnitudes/differences unaffected)."""
    for n in list(scene.receivers): scene.remove(n)
    for i, (x, y) in enumerate(pts): scene.add(Receiver(f"r{i}", position=[float(x), float(y), 1.5]))
    p = PathSolver()(scene, max_depth=10, samples_per_src=int(samples), los=True, specular_reflection=True, diffraction=diffraction, seed=seed)
    a, tau = p.cir(out_type="numpy"); a = np.squeeze(a); tau = np.squeeze(tau)
    return a.astype(np.complex64), tau.astype(np.float32)

def powers(a, tau, chunk=1500):
    """wideband (band-averaged over BW), narrowband (centre tone), incoherent sum; linear path gain. Chunked (N x P x P)."""
    inc = np.zeros(len(a)); nb = np.zeros(len(a)); wb = np.zeros(len(a))
    for i in range(0, len(a), chunk):
        A = np.where(np.abs(a[i:i + chunk]) > 0, a[i:i + chunk], 0); T = tau[i:i + chunk]
        inc[i:i + chunk] = (np.abs(A) ** 2).sum(-1); nb[i:i + chunk] = np.abs(A.sum(-1)) ** 2
        dt = T[:, :, None] - T[:, None, :]
        wb[i:i + chunk] = np.real((A[:, :, None] * np.conj(A[:, None, :]) * np.sinc(BW * dt)).sum((1, 2)))
    return inc, nb, wb

def rms_ds(a, tau):
    p = np.abs(a) ** 2; m = np.abs(a) > 0; p = np.where(m, p, 0); t = np.where(m, tau, 0)
    P = p.sum(-1); mu = (p * t).sum(-1) / np.maximum(P, 1e-40); return np.sqrt(np.maximum((p * t ** 2).sum(-1) / np.maximum(P, 1e-40) - mu ** 2, 0))

def kfactor_db(a):
    p = np.abs(a) ** 2; pm = p.max(-1); return 10 * np.log10(np.maximum(pm, 1e-40) / np.maximum(p.sum(-1) - pm, 1e-40))


def solve_full(scene, pts, samples=1e7, seed=0, K=200, max_num_paths_per_src=1e6):
    """Like solve() but keeps only VALID paths, the K strongest per receiver, plus arrival angles. Returns a,tau,phi_r,theta_r (N,K) and dropped power fraction (N,).
    max_num_paths_per_src is a buffer SHARED across all receivers passed in one call (2026-09-22 convergence test:
    only converges to ~1 dB of the radio map at 1 receiver/call; batching >1 receivers dilutes it and does not converge
    even with a much larger cap - see results/converge_nlos_test/summary.json). Call with one point in `pts` at a time."""
    for n in list(scene.receivers): scene.remove(n)
    for i, (x, y) in enumerate(pts): scene.add(Receiver(f"r{i}", position=[float(x), float(y), 1.5]))
    p = PathSolver()(scene, max_depth=10, samples_per_src=int(samples), max_num_paths_per_src=int(max_num_paths_per_src),
                      los=True, specular_reflection=True, diffraction=True, seed=seed)
    a, tau = p.cir(out_type="numpy"); N = len(pts); a = np.array(a).reshape(N, -1); tau = np.array(tau).reshape(N, -1)
    v = np.array(p.valid).reshape(N, -1); phi = np.array(p.phi_r).reshape(N, -1); th = np.array(p.theta_r).reshape(N, -1)
    pw = np.where(v, np.abs(a) ** 2, 0.0); order = np.argsort(-pw, axis=1)[:, :K]; g = lambda x: np.take_along_axis(x, order, 1)
    kept = g(pw); tot = pw.sum(1); dropped = 1 - kept.sum(1) / np.maximum(tot, 1e-40); m = kept > 0
    out = [np.where(m, g(x), 0) for x in (a, tau, phi, th)]
    pad = K - out[0].shape[1]
    if pad > 0: out = [np.pad(x, ((0, 0), (0, pad))) for x in out]          # fixed width K so batches can be stacked
    return out[0].astype(np.complex64), out[1].astype(np.float32), out[2].astype(np.float32), out[3].astype(np.float32), dropped
