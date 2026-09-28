# STATUS / RESUME DOCUMENT - inf-factory-scene (written 2026-09-21, end of session 1)

Read this first. `README.md` in the same folder is the chronological technical log; this file is the state + what to do next.

## 0. Working rules the user set (MUST follow)
1. **Check with the user BEFORE diverging from the agreed plan or from the 3GPP calibration table.** Do not run "extra" experiments (e.g. the diffuse-scattering / edge-diffraction detour on 2026-09-21 was NOT authorised and annoyed the user). If something unexpected appears (e.g. "fading is shallow"), report it and ask; the user said shallow fading is irrelevant - just run the plan.
2. Clarify before executing when the user asks for it; verify claims from the spec text / source code, not memory. State when a number is my assumption, not a spec value.
3. Report progress plainly; do not block the user with long foreground waits (use background jobs). Tone: concise, honest, own mistakes.
4. Everything for this project lives in `/home/avivekan23/inf-factory-scene` (new, separate from the older `~/wisegrt-eval` four-room work, which is untouched). Python env: `/home/avivekan23/multi-modal-sched/jupyter_env/bin/python` (sionna 2.0.1, mitsuba 3.8, GPU/CUDA works). Run scripts from the project root; `PYTHONPATH=scripts` where needed.

## 1. Goal
Evaluate deep fades in a 3GPP TR 38.901 **InF-SL** (indoor factory, sparse clutter, low BS) hall with realistic Tx power (max 24 dBm), using Sionna RT, as groundwork for an anticipatory (fade-predicting) MAC-scheduler paper. Original concern: how to see deep fades while keeping realistic transmit power. Radio maps (large-scale only) then a coherent (complex path) UE-motion study for small-scale fading.

## 2. Spec facts (TR 38.901 V19.4.0; ETSI PDF https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/19.04.00_60/tr_138901v190400p.pdf ; text extracted at scratchpad, re-extract with `pdftotext -layout`)
- InF definition Table 7.2-4 (p.27); LoS prob. Table 7.4.2-1 (`exp(-d/k)`, `k=-d_clutter/ln(1-r)`); PL formulas Table 7.4.1-1 (LoS `31.84+21.5log10(d3D)+19log10(fc)`, NLoS SL `max(LoS, 33+25.5log10(d3D)+20log10(fc))`); calibration Table 7.8-7 (120x60 m or 300x150 m hall, height 10 m, 18 BS on a lattice D=20 m at D/2 from walls, BS 1.5 m (SL/DL) or 8 m (SH/DH), UT 1.5 m, UT NF 9 dB, 3.5 & 28 GHz, 100 MHz, clutter density 20% (low) / 60%, h_c 2 m / 6 m, d_clutter 10 m / 2 m, isotropic V-pol antennas); LSPs Table 7.5-6 Part-3 + 7.5-11; correlation distance Table 7.6.3.1-2 (LoS state d_clutter/2); blockers Table 7.6.4.2-5 (human <=3 km/h, AGV <=30 km/h, robot <=3 m/s).
- **TX power is NOT given for InF.** 24 dBm is the Indoor-office (InH) figure (Tables 7.8-1/7.8-2). Also TX at (60,30) is not on the spec lattice.
- `d_clutter` ("typical clutter size") and clutter placement are NOT defined by the spec; R1-1909704 (calibration report) would define them but 3GPP FTP returns 403 from here. **Ask the user's supervisor / obtain R1-1909704.**
- Spec range 0.5-100 GHz; Sionna ITU materials (concrete, metal) support **1-100 GHz only** (ValueError at load outside).
- Spec InF medians for our empty hall (V/S = 4.0 m): DS 52.7 ns (LoS) / 55.2 ns (NLoS); ASD 36-37 deg; ASA(3.85 GHz) 45.3 / 52.5 deg; K 7 dB (sd 8) LoS.

## 3. Scene (built, validated)
- `scripts/build_inf_hall.py` (+`geom.py`, `place_clutter.py`): 120x60x10 m hall; walls `itu_concrete`; ceiling `itu_metal` (default) or `--ceiling concrete`; metal-coated window band (itu_metal, z 3-5 m, long walls; my layout); floor concrete (my assumption); clutter = random non-overlapping metal cuboids/cylinders, h=2 m, >=1 m gaps (my assumption), exactly 30% floor coverage (user decision; calibration table says 20% -> documented divergence).
- Scenes: `scenes/inf_sl_r30_s{0..4}` (30%, metal ceiling), `scenes/inf_sl_r30_s0_concrete_ceiling` (**the fixed choice for the fine-fading study**), `scenes/inf_sl_clutter_s{0..4}` (20%), `scenes/inf_sl_hall` (empty).
- Statistical check vs spec (`validate_inf_stats.py`): LoS probability decays exponentially but with fitted k ~0.57 x spec at BOTH r=0.2 and r=0.3 (26 vs 44.8 m; 15.9 vs 28.0 m) -> layout blocks ~1.7x too much; unresolved (needs the d_clutter definition). Grid-of-cells hypothesis tested (`--layout grid`): no better.
- Path loss vs spec (`compare_pl_spec.py`): metal ceiling gives NLoS 8-15 dB better than spec NLoS PL; concrete ceiling within ~1-5 dB. **Ceiling default was left as a pending user decision; user later "fixed concrete ceiling" for the fine-fading study.**
- Notebook `notebooks/view_hall.ipynb` (Sionna preview + radio map). JupyterLab was launched on 127.0.0.1:8888 (SSH tunnel `ssh -N -L 8888:localhost:8888 <user>@<this-host>`); restart it if needed (token is per launch).

## 4. Radio-map results (large-scale only; power-summed, NO small-scale fading - verified in Sionna source: radio map g_i = cell average of sum|a_i|^2, `planar_radio_map.add_paths` adds per-ray |E|^2)
- TX (60,30,1.5), isotropic, 100 MHz, noise = -174+10log10(BW)+9 = -85 dBm, UE plane 1.5 m, mask: clutter cells, <1 m from TX, solver artifacts (>20 dB above free space).
- Power sweep 24->0 dBm (independent RT per power, 0.05 m cells, 4e9 rays): `results/hist_rt_*`, `results/median_snr_vs_tx_power.png`. Median shifts 1 dB/dB; concrete ceiling 6.64 dB below metal. % of cells < 0 dB (concrete): 19.6% @0 dBm, 9.7% @2, 2.0% @4, 0.3% @6, 0 above.
- **Convergence lesson**: at 5 cm cells need >= ~1400 rays/cell (4e9 rays); 2e8 rays biased the median low and inflated tails.
- Frequency sweep at 24 dBm (1,1.5,3,3.5,6,12,25,45,90 GHz, diffraction on): `results/freq_rt_*`, `median_snr_vs_frequency.png`. Concrete: % < 0 dB = 0 up to 12 GHz, 0.04% @25, 8.0% @45, 36.4% @90; metal 0.65% @90. Diffraction on/off differ <=0.03 dB in medians (`freq_diffraction_overlay_*.png`). Unexplained: 1->1.5 GHz median drop 6.8 dB (concrete) vs 3.5 dB free space.
- Cell-size facts: no minimum in Sionna; 2 cm ok; 0.86 cm starts starving rays.

## 5. Fine-fading (coherent) study - IN PROGRESS
User decisions: goals = BOTH spatial fade statistics AND time series; **3.85 GHz** (user-intended; NOT the calibration 3.5 GHz; lambda = 7.79 cm), 100 MHz wideband first then per-RB (273 RB x 12 subcarriers x 30 kHz = 360 kHz), **concrete-ceiling scene**, TX (60,30,1.5) 24 dBm, **UE speed 0.5 m/s** (=1.8 km/h, inside spec "up to 3 km/h"), UE clearance 0.5 m (my assumption).
- Motion plan DONE: `scripts/plan_ue_paths.py` -> `trajectories/ue_plan.npz|png`: 188 straight 2 m local-area tracks (376 m) + 6 A* AMR routes (1378 m = 45.9 min @0.5 m/s), all >= 0.5 m clear of clutter/walls; free area 61.4% of hall, fully connected.
- Coherent trace: `scripts/cir_common.py` (`make_scene`, `solve_full`, `powers`...), `scripts/run_cir_all.py` (tracks 48,316 pts + routes 176,968 pts at lambda/10 = 7.79 mm; specular + wedge diffraction, 10 bounces, 1e7 rays, batch 2000 rx, keeps K=200 strongest valid paths). Analysis: `scripts/analyze_finefade.py` (wideband via sinc pair formula, narrowband, per-RB at RB-centre tone, DS, K-factor, az. spread, radio-map check) and `analyze_finefade2.py` (statistics).
- Validation that PASSED: carrier phase is included in Sionna path coefficients (|sum a|^2 varies sub-wavelength, sum|a|^2 smooth); smooth point-to-point (0% adjacent jumps >3 dB on 3 test tracks); LoS points: coherent power = radio map within +0.3..+0.8 dB.

### !!! THE FULL RUN (2026-09-21) IS INVALID FOR NLoS - DO NOT USE `results/finefade_3.85GHz_concrete/*` FOR NLoS (see INVALID_NLOS.txt there)
Cause (my parameter error): `PathSolver(max_num_paths_per_src)` (default 1e6) is a buffer **shared by all receivers in one call**. Validation used 257 rx/call (~3900 paths/rx, looked fine); the full run used 2000 rx/call (~500 paths/rx) -> paths silently truncated. Symptoms: median only 6-7 valid paths per point, 2.4% (tracks) / 2.0% (routes) points with NO valid path (all NLoS), NLoS coherent power median -17 dB below the radio map (tail < -70 dB), path count DEcreasing as rays increase.
Test on 400 NLoS points (1e7 rays), deficit vs radio map: cap 1e6 -> -16.0 dB (17 paths); 1e7 -> -15.6 dB (116); **1e8 -> -8.3 dB (1110 paths)**; cap 1e8 with 1e8 rays -> out-of-memory RuntimeError. So converged NLoS channels have >1000 paths (spec: 25 clusters x 20 rays = 500); **my earlier claim "one or two paths dominate so fading is shallow" was a truncation artifact - retract it.** Still 8 dB short at cap 1e8 -> not converged.
Second analysis mistake: route "fade depth" was normalised to the whole-route mean (mixes path loss). Needs a sliding window (~10-20 lambda). Track numbers (2 m local mean) are OK in method.
Reliable (LoS only, may still shift): tracks LoS n=9166: narrowband std 1.08 dB, wideband 0.44 dB, per-RB 1.09 dB, K median 15 dB; DS ~5 ns (spec 52.7); az. spread ~0 deg (strong paths = LoS + floor/ceiling bounces share one azimuth - real, not a bug).

## 6. NEXT STEPS (in order; confirm with user before deviating)
1. **10-minute convergence test** (user's option 1): on ~400 NLoS points find (receivers per call, `max_num_paths_per_src`, rays) such that coherent sum|a|^2 matches the radio map within ~1 dB and path counts stop growing; watch GPU memory (cap 1e8 x 1e8 rays crashed). Consider per-call budget >= ~1e5-1e6 paths per rx, few rx per call, and the time cost (rough guess 1 h for tracks; routes may need lambda/4 spacing - ask user).
2. Re-run tracks (then routes) with the fixed budget; fix route fade normalisation (sliding window); verify zero "no valid path" points and radio-map agreement in NLoS.
3. Then: wideband first, per-RB second; fade-depth CDFs vs Rayleigh (P(<-3/-6/-10 dB) = 39.4/22.2/9.5%), level-crossing rate / fade duration at 0.5 m/s (route time series, `t = s/0.5`), DS / ASA / K vs spec medians above, and the radio-map consistency check. Figures planned: example tracks (narrowband/wideband/per-RB vs distance in lambda), fade-depth CDFs LoS vs NLoS with Rayleigh reference, route SNR-vs-time, spec comparison histograms.
4. Only after asking: diffuse scattering / other realism additions (spec gives no scattering coefficient).

## 7. Open questions / unresolved (do not silently decide)
- Ceiling default for the "main" scene (concrete fixed for fine-fading only). Clutter size/placement interpretation and the 0.57 k-fit gap (need R1-1909704). Coverage 30% vs calibration 20% (user chose 30%, following Dinh/Maman/Calvanese-Strinati 2022).
- 24 dBm source for InF; 18-BS lattice not used (single TX); some lattice points fall inside clutter (spec silent).
- 1->1.5 GHz slope anomaly; path-count oddities seen with large ray counts (now explained by the buffer cap).
- 12 of 200 strata had no room for a 2 m track (fine).

## 8. File map
`scripts/`: scene (`build_inf_hall.py geom.py place_clutter.py render_layout.py build_notebook.py`), spec checks (`validate_hall.py validate_inf_stats.py compare_pl_spec.py`), radio-map sweeps (`power_sweep.py power_hist_rt.py freq_hist_rt.py plot_median_vs_power.py plot_median_vs_freq.py plot_freq_diffraction_overlay.py`), fine fading (`plan_ue_paths.py cir_common.py validate_cir.py run_cir_all.py analyze_finefade*.py`; `run_tracks_cir.py` is superseded). `results/`: all plots/CSVs; `results/*/snr_cells.npz` allow `--replot`. `trajectories/`: motion plan.
