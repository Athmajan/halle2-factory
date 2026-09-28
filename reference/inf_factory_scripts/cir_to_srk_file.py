"""Convert the ray-traced multi-UE dataset into the Sionna Research Kit channel emulator's file format
(config.json + cirs.bin), following the SionnaRT GUI's own export recipe (2026-09-26).
See notes/2026-09-26_e2e_ocudu_cir_plan.md.

Recipe followed (NVlabs/sionna-rt-gui @ srk-channel-emulator, srk_batch_export.py + sionna_utils.py):
  1. per true ray-traced solve: baseband path coefficients a_i and delays tau_i (Paths.cir(normalize_delays=True))
     -> OUR saved dataset already holds exactly this per tick (cir_common.solve_full: p.cir(), K=200 strongest
     paths, arrival angles theta_r/phi_r).
  2. Doppler extrapolation between true solves (Paths.cir(num_time_steps=...)): a_i(t) = a_i exp(j2pi f_i t),
     f_i = v_rx . r_hat(theta_R, phi_R) / lambda  (Sionna Paths.doppler, static scene, only the RX moves).
     The GUI solves once per slot (interpolation_factor=14); we have one solve per tick (15.57 ms = lambda/10 of
     travel at 3.85 GHz, 0.5 m/s), so each extrapolation spans one tick.
     Deviation (documented): the common phase of the first-arriving path (tau=0 after delay normalisation) is
     carried forward across ticks instead of being reset at every solve, so the emulated channel has no
     artificial common-phase step at tick boundaries (which fall mid-slot here).
  3. Paths.taps(): h_l(t) = sum_i a_i(t) sinc(l - W tau_i), l = l_min..l_max, W = sampling rate of the cell.
  4. keep the num_taps largest |h_l| per symbol + their uint16 indices (prepare_and_normalize_cir_batch).
  5. norm = ||h_selected||, taps /= norm; config sigma_scaling = 10^(-snr_offset_db/20),
     sigma_max = 10^(max_noise_std_db/10) (config.py SrkDemoConfig.max_noise_std) - the kernel then uses
     noise_std = min(sigma_scaling / norm, sigma_max) per symbol (chn_emu_cuda.cu).
  6. cirs.bin: per symbol, float32 norm, float32 taps[2*num_taps] (re, im interleaved), uint16 idx[num_taps];
     config.json "channel_emulation" section as in the tutorial, plus provenance.
CIR time base: cir_sampling_frequency_hz = 14 symbols / 0.5 ms slot = 28 kHz (the GUI's own knob), so
trajectory time == slot time on the 30 kHz SCS cell.

One output per (segment, UE) or window of it; all 16 (4 UEs x 4 segments) are converted separately and
played as separate runs (user, 2026-09-26: use all data, not one continuous 8 h stretch). Streams to disk.

Usage (2-min pilot, segment 1, UE0, Phase 1 chunk 0):
  jupyter_env/bin/python3 scripts/cir_to_srk_file.py --src results/multiue_n4_modeA_agv0_seed0/multiue_cir.npz \
      --ue 0 --tick0 0 --tick1 7705 --out results/srk_cir/pilot_seg1_ue0_chunk0
"""
import argparse, json, os, time
import numpy as np

C = 299792458.0


def rhat(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="one multiue_cir.npz segment")
    ap.add_argument("--ue", type=int, required=True)
    ap.add_argument("--tick0", type=int, default=0)
    ap.add_argument("--tick1", type=int, default=None, help="end tick, exclusive (default: whole segment)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample_rate_hz", type=float, default=23.04e6)   # OCUDU cell srate (configs/ocudu_du1.yml)
    ap.add_argument("--cir_sampling_frequency_hz", type=float, default=28e3)  # 14 symbols / 0.5 ms
    ap.add_argument("--num_taps", type=int, default=10)                # GUI PathsConfig.num_taps default
    ap.add_argument("--l_min", type=int, default=0)                    # GUI default
    ap.add_argument("--l_max", type=int, default=100)                  # GUI default
    ap.add_argument("--snr_offset_db", type=float, default=60.0)       # GUI default (calibration knob, see notes)
    ap.add_argument("--max_noise_std_db", type=float, default=18.0)    # GUI SrkDemoConfig default
    ap.add_argument("--speed_mps", type=float, default=None, help="default: the dataset's own (0.5 m/s)")
    ap.add_argument("--block_ticks", type=int, default=512)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    W = a.sample_rate_hz
    l = np.arange(a.l_min, a.l_max + 1)
    L = a.num_taps
    entry = np.dtype([("norm", "<f4"), ("taps", "<f4", (2 * L,)), ("idx", "<u2", (L,))])
    assert entry.itemsize == 4 + 8 * L + 2 * L and entry.itemsize % 4 == 0
    fs_cir = a.cir_sampling_frequency_hz
    t0 = time.time()

    d = np.load(a.src)
    dt_data = float(d["dt"]); fc = float(d["fc"]); lam = C / fc
    speed0 = (lam / 10) / dt_data
    speed = a.speed_mps or speed0
    dt = (lam / 10) / speed          # speed emulation (Phase 4 logic): fixed spatial step, only its duration changes
    A = d[f"a_ue{a.ue}"]; TAU = d[f"tau_ue{a.ue}"]; PHI = d[f"phi_ue{a.ue}"]; TH = d[f"theta_ue{a.ue}"]
    pos = d["ue_pos"][:, a.ue]
    T = len(A)
    k0, k1 = a.tick0, (a.tick1 if a.tick1 is not None else T)
    vel = np.zeros((T, 3))
    vel[:-1, :2] = (pos[1:] - pos[:-1]) / dt
    vel[-1] = vel[-2]

    # symbol n (from the window start) sits at time n/fs_cir, inside tick k0 + floor(n*... / dt)
    n_total = int(np.floor((k1 - k0) * dt * fs_cir))
    n_total -= n_total % 14                                   # num_cirs must be a multiple of 14 (cir_file.c)
    sym_tick = k0 + np.floor(np.arange(n_total) / fs_cir / dt).astype(np.int64)
    sym_off = np.arange(n_total) / fs_cir - (sym_tick - k0) * dt   # time since the tick's solve, [0, dt)

    common_phase = 0.0   # carried first-arrival Doppler phase (deviation note in the docstring)
    fbin = open(os.path.join(a.out, "cirs.bin"), "wb")
    first_path_fD = np.zeros(k1 - k0)
    for b0 in range(k0, k1, a.block_ticks):
        b1 = min(k1, b0 + a.block_ticks)
        out_blocks = []
        for k in range(b0, b1):
            lo = np.searchsorted(sym_tick, k); hi = np.searchsorted(sym_tick, k, side="right")
            ak = A[k].astype(np.complex128); tk = TAU[k].astype(np.float64)
            valid = np.abs(ak) > 0
            ak, tk = ak[valid], tk[valid]
            fD = (rhat(TH[k][valid].astype(np.float64), PHI[k][valid].astype(np.float64)) @ vel[k]) / lam
            f0 = fD[np.argmin(tk)] if len(tk) else 0.0
            first_path_fD[k - k0] = f0
            if hi > lo:
                ts = sym_off[lo:hi]
                # Sionna Doppler phase exp(j2pi fD t), minus the first path's own term which is instead carried
                # continuously in common_phase (so at t=dt the next solve continues without a phase step)
                ph = np.exp(1j * (2 * np.pi * np.outer(ts, fD - f0) + common_phase + 2 * np.pi * f0 * ts[:, None]))
                G = np.sinc(l[None, :] - W * tk[:, None])                              # Paths.taps sinc
                h = (ph * ak[None, :]) @ G                                             # [S, l_max-l_min+1]
                order = np.argsort(-np.abs(h), axis=1)[:, :L]
                hs = np.take_along_axis(h, order, axis=1)
                norm = np.sqrt((np.abs(hs) ** 2).sum(1))
                hs = hs / np.where(norm > 0, norm, 1.0)[:, None]
                e = np.empty(hi - lo, dtype=entry)
                e["norm"] = norm
                e["taps"][:, 0::2] = hs.real; e["taps"][:, 1::2] = hs.imag
                e["idx"] = (order + a.l_min).astype(np.uint16)
                out_blocks.append(e)
            common_phase = float(np.angle(np.exp(1j * (common_phase + 2 * np.pi * f0 * dt))))
        if out_blocks:
            np.concatenate(out_blocks).tofile(fbin)
        print(f"[{time.time()-t0:7.1f}s] ue{a.ue} ticks {b1}/{k1}", flush=True)
    fbin.close()

    cfg = {
        "channel_emulation": {
            "center_frequency": fc,
            "num_taps": L,
            "num_cirs": n_total,
            "sigma_scaling": 10 ** (-a.snr_offset_db / 20),
            "sigma_max": 10 ** (a.max_noise_std_db / 10),
        },
        "batch": {
            "src": a.src, "ue": a.ue, "tick0": k0, "tick1": k1, "dt_s": dt, "speed_mps": speed,
            "sampling_frequency_hz": fs_cir, "sample_rate_hz": W, "l_min": a.l_min, "l_max": a.l_max,
            "snr_offset_db": a.snr_offset_db, "max_noise_std_db": a.max_noise_std_db,
            "num_symbols_per_slot": 14, "n_cirs": n_total, "duration_s": n_total / fs_cir,
            "symbol_to_tick": "tick = tick0 + floor((n / sampling_frequency_hz) / dt_s)",
            "built_by": "scripts/cir_to_srk_file.py",
        },
    }
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=4)
    np.save(os.path.join(a.out, "first_path_doppler_hz.npy"), first_path_fD)
    print(f"done: {n_total} CIRs ({n_total/fs_cir:.1f}s, {n_total*entry.itemsize/1e6:.1f} MB) -> {a.out}")


if __name__ == "__main__":
    main()
