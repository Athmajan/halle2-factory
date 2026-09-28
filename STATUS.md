# STATUS - halle2-factory (2026-09-29)

**State: planned, not started.** The plan is `PLAN.md`.

**HOLD:** do not start until the user says so. Another agent is running MAC-Gyver data collection on this host, and
nothing here may compete with it: no GPU/CPU-heavy jobs, no Jupyter kernel restarts, no package installs in the
shared env. Its own rule: the gNB uses CPUs 8-11 and the UEs 12-23; anything unavoidable goes on `taskset -c 0-5`
(`~/6grocket/T2.2_armB_status.md` sec. 3).

## Environment
- Python: `/home/avivekan23/multi-modal-sched/jupyter_env/bin/python` (sionna 2.0.1, mitsuba 3.8, CUDA, python-rvo2).
- JupyterLab on 127.0.0.1:8888 (root `~/inf-factory-scene`). This project is reachable there through the symlink
  `~/inf-factory-scene/6grocket_halle2 -> ~/halle2-factory`; viewer: `6grocket_halle2/notebooks/view_halle2.ipynb`.
- Work rules carried over from inf-factory-scene: check with the user before diverging from the plan or the 3GPP
  table, no unrequested experiments, and label my own assumptions as such.

## Done so far
- Scene source: `~/6grocket/Halle2.xml.zip` (Halle2.xml + Halle2.blend 1.1 GB + meshes.7z). Extracted to `scene/`
  (meshes.7z unpacked with py7zr installed only in a session scratch dir, not in the shared env).
- Import check: `sionna.rt.load_scene` works, 0 errors. The only message is the usual HDRFilm "Monochrome mode" warning.
  With default `merge_shapes=True`, Doors + Walls-itu_glass merge into `no-name-1`; the notebook uses `merge_shapes=False`.
- Geometry: hall ~150 x 143 m (walls x -60..94, y -23..120), roof z 7.3-13 m, floor plate 222 x 160 m, 5.6 M triangles
  (MetalStuff 3.5 M, Doors 1.3 M, Roof 0.59 M). Meshes in meshes.7z but not in the XML: `Metall.ply`, `Roof_VIsualization.ply`.
- Materials at 3.5 GHz: plasterboard eps 2.73, glass 6.31, concrete 5.24, ceiling_board 1.48, metal sigma 1e7 S/m.
- RT sanity: TX (10,40,8) -> 0 paths (inside the roof structure). TX (10,50,5): PathSolver OK; radio map covers 80%
  of cells; median SNR 23.7 dB at 24 dBm / 100 MHz / NF 9 (older settings, before 15 dBm / 20 MHz were chosen).
- Notebook `notebooks/view_halle2.ipynb`: load + inspect, 3D preview (clipped at 7 m), RSS radio map, SNR
  (`rm_metric="sinr"`, noise via `scene.temperature = 290*10^(NF/10)` set before solving) on the 3D scene.
  Defaults now 3.85 GHz, 20 MHz, 15 dBm. Do not save it executed: outputs are ~420 MB.
- Plan history: `notes/2026-09-29_plan_draft_superseded.md` (the working draft, incl. the rejected MAC-Gyver options A/B).

## Compute host (2026-09-29)
The original host is busy (MAC-Gyver data collection), so the CIR work is to run on another host. This repo is
pushed to GitHub (public: github.com/Athmajan/halle2-factory) with plan, reference code and environment.
**The scene is deliberately NOT in the repo** (user decision): copy `scene/Halle2.xml` + `scene/meshes.7z` by hand. Paths like `~/inf-factory-scene`, `~/wisegrt-eval`, `~/MAC-Gyver` in PLAN.md refer to
the original host; their needed files are under `reference/`. The Jupyter/hold notes above apply to the original host.

## Next when the user says go
PLAN.md sec. 5, step 0 (scene audit).
