import sys, numpy as np
sys.path.insert(0, "scripts")
out = "results/finefade_3.85GHz_concrete"; PTX, NOISE = 24.0, -85.0
db = lambda x: 10 * np.log10(np.maximum(x, 1e-40)); snr = lambda p: db(p) + PTX - NOISE
def local_norm_db(x, pid, ok):                       # power relative to its own track mean (linear mean), dB
    r = np.full(len(x), np.nan)
    for t in np.unique(pid):
        m = (pid == t) & ok
        if m.sum() > 20: r[m] = db(x[m] / x[m].mean())
    return r
def stats(f, lab):
    f = f[~np.isnan(f)]; return f"{lab:>16s}: n {len(f):8d} std {f.std():5.2f} dB | p1 {np.percentile(f,1):6.2f} p5 {np.percentile(f,5):6.2f} min {f.min():6.2f} | P(<-3 dB) {100*np.mean(f<-3):6.3f}% P(<-6) {100*np.mean(f<-6):6.4f}% P(<-10) {100*np.mean(f<-10):7.5f}%"
fc = 3.85; L = np.log10(1 + fc); spec = dict(ds_los=10 ** (np.log10(26 * 4 + 14) - 9.35) * 1e9, ds_nlos=10 ** (np.log10(30 * 4 + 32) - 9.44) * 1e9, asa_los=10 ** (-0.18 * L + 1.78), asa_nlos=10 ** 1.72)
print("Rayleigh reference (power rel. to mean): std 5.57 dB | P(<-3 dB) 49.9%... wait compute:", "P(<-3)=%.1f%% P(<-6)=%.1f%% P(<-10)=%.1f%%" % tuple(100 * (1 - np.exp(-10 ** (np.array([-3, -6, -10]) / 10)))))
for name in ("tracks", "routes"):
    z = np.load(f"{out}/{name}_metrics.npz"); ok = z["ok"]; los = z["los"]; pid = z["pid"]
    print(f"\n================ {name.upper()}: {len(ok)} points, {int((~ok).sum())} holes ({100*(~ok).mean():.2f}%) ; LoS among holes: {100*los[~ok].mean():.1f}% (overall LoS {100*los.mean():.1f}%)")
    s = snr(z["inc"]); print(f"incoherent SNR (24 dBm): median {np.median(s[ok]):.1f} dB, p5 {np.percentile(s[ok],5):.1f}, min {s[ok].min():.1f}, max {s[ok].max():.1f}; wideband median {np.median(snr(z['wb'])[ok]):.1f}; narrowband median {np.median(snr(z['nb'])[ok]):.1f}")
    for tag, mk in (("LoS", ok & los), ("NLoS", ok & ~los), ("all", ok)):
        print(f"--- {tag}")
        for lab, x in (("narrowband", z["nb"]), ("wideband 100MHz", z["wb"])):
            print("  " + stats(local_norm_db(x, pid, mk), lab))
        rb = 10 ** (z["snr_rb_db"].astype(np.float32) / 10)                                     # per-RB linear (relative)
        fr = np.concatenate([local_norm_db(rb[:, r], pid, mk)[mk] for r in range(0, 273, 13)]); print("  " + stats(fr, "per-RB (21 RBs)"))
    for tag, mk in (("LoS", ok & los), ("NLoS", ok & ~los)):
        print(f"spread {tag}: DS median {np.median(z['ds'][mk]*1e9):.1f} ns (p10 {np.percentile(z['ds'][mk]*1e9,10):.1f}, p90 {np.percentile(z['ds'][mk]*1e9,90):.1f}) spec {spec['ds_'+tag.lower()]:.1f} | ASA median {np.median(z['asa'][mk]):.1f} deg spec {spec['asa_'+tag.lower()]:.1f} | K median {np.median(z['K'][mk]):.1f} dB (mean {z['K'][mk].mean():.1f}, sd {z['K'][mk].std():.1f})" + ("  spec K mu 7 sd 8" if tag == "LoS" else "  (spec: N/A for NLoS)"))
    d = db(z["inc"]) - db(z["rm"]); m = ok & (z["rm"] > 0)
    print(f"radio-map check: 10log10(sum|a|^2 at point) - 10log10(radio-map cell) : median {np.median(d[m]):+.2f} dB, p5 {np.percentile(d[m],5):+.2f}, p95 {np.percentile(d[m],95):+.2f}, mean-abs {np.abs(d[m]).mean():.2f}  (LoS median {np.median(d[m&los]):+.2f}, NLoS {np.median(d[m&~los]):+.2f})")
    dm = [db(z['nb'][(pid == t) & ok].mean()) - db(z['rm'][(pid == t) & ok].mean()) for t in np.unique(pid) if ((pid == t) & ok).sum() > 20]
    print(f"                 per-track mean coherent (narrowband) power vs radio map: median {np.median(dm):+.2f} dB, p5 {np.percentile(dm,5):+.2f}, p95 {np.percentile(dm,95):+.2f}")
