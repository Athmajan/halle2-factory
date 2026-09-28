"""Gather all numbers/series needed for the team presentation artifact (2026-09-22):
(1) spec-alignment stats pooled across all 5 multi-agent runs (DS, K-factor, ASA vs TR 38.901
    InF-SL medians), (2) a downsampled example slow-fading trajectory (n_agv=6, busiest run) for
    the interactive time-series chart. Writes results/team_report/report_data.json.
"""
import sys, json, numpy as np
sys.path.insert(0, "scripts")
from cir_common import *
from validate_inf_stats import los

dB = lambda x: 10 * np.log10(np.maximum(x, 1e-30))
OUT = "results/team_report"
import os; os.makedirs(OUT, exist_ok=True)

RUNS = [("n4_seed1", 4), ("n4_seed2", 4), ("n2_seed1", 2), ("n3_seed1", 3), ("n6_seed1", 6)]

spec = dict(ds_los=52.7, ds_nlos=55.2, asa_los=45.3, asa_nlos=52.5, k_los_mu=7.0, k_los_sd=8.0)


def az_spread_deg(a, phi):
    p = np.abs(a) ** 2; P = np.maximum(p.sum(1), 1e-40); mu = np.angle((p * np.exp(1j * phi)).sum(1))[:, None]
    d = np.mod(phi - mu + np.pi, 2 * np.pi) - np.pi
    return np.degrees(np.sqrt((p * d ** 2).sum(1) / P))


sc = make_scene()
all_ds, all_k, all_asa, all_los, all_nvalid = [], [], [], [], []
per_run = {}
for tag, n_agv in RUNS:
    d = np.load(f"results/multiagent_{tag}/multiagent_cir.npz")
    a, tau, phi, ue = d["a"], d["tau"], d["phi_r"], d["ue_pos"]
    nvalid = (np.abs(a) > 0).sum(-1)
    ds = rms_ds(a, tau) * 1e9
    k = kfactor_db(a)
    asa = az_spread_deg(a, phi)
    N = len(a)
    L = np.concatenate([los(sc, np.tile(TX, (len(ue[j:j+20000]), 1)), np.c_[ue[j:j+20000], np.full(len(ue[j:j+20000]), 1.5)])
                         for j in range(0, N, 20000)])
    ok = nvalid > 0
    per_run[tag] = dict(n_agv=n_agv, N=N, los_frac=float(L.mean()), no_path_frac=float((~ok).mean()))
    all_ds.append(ds[ok]); all_k.append(k[ok & L]); all_asa.append(asa[ok]); all_los.append(L[ok]); all_nvalid.append(nvalid[ok])
    print(f"{tag}: N={N} LoS%={100*L.mean():.2f} no-path%={100*(~ok).mean():.3f}")

ds_all = np.concatenate(all_ds); k_all = np.concatenate(all_k); asa_all = np.concatenate(all_asa); los_all = np.concatenate(all_los)

def q(x, p): return float(np.percentile(x, p)) if len(x) else None

result = dict(
    spec=spec,
    per_run=per_run,
    n_ticks_total=int(sum(v["N"] for v in per_run.values())),
    ds_los=dict(median=q(ds_all[los_all], 50), p10=q(ds_all[los_all], 10), p90=q(ds_all[los_all], 90), n=int(los_all.sum())),
    ds_nlos=dict(median=q(ds_all[~los_all], 50), p10=q(ds_all[~los_all], 10), p90=q(ds_all[~los_all], 90), n=int((~los_all).sum())),
    asa_los=dict(median=q(asa_all[los_all], 50), n=int(los_all.sum())),
    asa_nlos=dict(median=q(asa_all[~los_all], 50), n=int((~los_all).sum())),
    k_los=dict(median=q(k_all, 50), mean=float(k_all.mean()) if len(k_all) else None, sd=float(k_all.std()) if len(k_all) else None, n=int(len(k_all))),
)
with open(f"{OUT}/spec_stats.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
