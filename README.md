# halle2-factory

Sionna RT study on the 6G Rocket "Halle2" factory hall. 4 UEs do a box-delivery job (MAPPO + ORCA). Ray-traced CIRs
are computed along their trajectories at 3.85 GHz for a 20 MHz cell, compared with TR 38.901 and the
`~/inf-factory-scene` hall, and later replayed in MAC-Gyver.

- `STATUS.md`: current state, hold condition, environment. Read first.
- `PLAN.md`: the agreed plan (phases, decisions, steps, reuse map).
- `scene/`: **not in the repo.** Copy `Halle2.xml` + `meshes.7z` from the original host (`~/halle2-factory/scene/`,
  source `~/6grocket/Halle2.xml.zip`) into `scene/` on each host.
- `notebooks/view_halle2.ipynb`: 3D viewer + RSS/SNR radio map.
- `scripts/`, `results/`, `trajectories/`: empty until work starts.
- `notes/`: dated notes and superseded drafts.

## Setup on a new host
```bash
git clone https://version.oulu.fi/avivekan23/halle2-factory.git && cd halle2-factory
pip install -r environment.txt            # see notes in the file (pyrvo2 from GitHub)
# copy Halle2.xml + meshes.7z into scene/ first, e.g.
# scp <user>@<original-host>:halle2-factory/scene/{Halle2.xml,meshes.7z} scene/
cd scene && 7z x meshes.7z                 # or: python -c "import py7zr; py7zr.SevenZipFile('meshes.7z').extractall('.')"
python -c "import sionna.rt as rt; s=rt.load_scene('Halle2.xml'); print(len(s.objects), 'objects')"   # expect 5
```
- `reference/`: verbatim copies of the box-delivery/MAPPO code and the inf-factory ray-tracing scripts (see its README).
- `environment.txt`: package versions of the original host.
