"""Aggregate results_*.json into markdown tables (results_tables.md)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
COLS = [("knn_euc", "kNN-euc"), ("knn_cos", "kNN-cos"), ("linprobe", "LinProbe"), ("trust", "Trust@10"),
        ("contin", "Cont@10"), ("k_skew", "k-skew@10"), ("eff_rank", "EffRank"), ("fit_time", "fit s")]
out = []
for ds in ("pets", "digits", "synthetic"):
    R = json.loads((HERE / f"results_{ds}.json").read_text())
    out.append(f"\n### {ds}  train {R['shape_train']} test {R['shape_test']}  info={R['info']}\n")
    out.append("| method | dim | " + " | ".join(c[1] for c in COLS) + " |")
    out.append("|---" * (len(COLS) + 2) + "|")
    order = []
    for r in R["results"]:
        if r["method"] not in order: order.append(r["method"])
    ekinfo = []
    for m in order:
        rs = [r for r in R["results"] if r["method"] == m]
        errs = [r for r in rs if "error" in r]
        rs = [r for r in rs if "error" not in r]
        if errs: out.append(f"| {m} | ERROR: {errs[0]['error']} |")
        if not rs: continue
        cells = []
        dims = sorted({r["dim"] for r in rs})
        for k, _ in COLS:
            v = np.array([r[k] for r in rs], float)
            f = "{:.1f}" if k in ("eff_rank", "fit_time") else ("{:.2f}" if k == "k_skew" else "{:.3f}")
            cells.append(f.format(v.mean()) + (" ± " + f.format(v.std(ddof=1)) if len(v) > 1 else ""))
        out.append(f"| {m}{' (n=%d)' % len(rs) if len(rs) > 1 else ''} | {'/'.join(map(str, dims))} | " + " | ".join(cells) + " |")
        if "target_dim" in rs[0]:
            for r in rs:
                h = r["history"].get("k_skewness", [])
                ekinfo.append(f"- {m} seed {r['seed']}: target_dim={r['target_dim']}, stopped_epoch={r['stopped_epoch']}/{r['epochs']}, "
                              f"ID estimates={ {k: round(v, 2) for k, v in r['analysis_id_estimates'].items()} }, "
                              f"monitored k_skewness history (every 5 ep)={[round(x, 3) for x in h]}, "
                              f"final loss={r['history']['loss'][-1]:.3f}, warnings={r['warnings']}")
    out.append("\nEmbedKit run details:\n" + "\n".join(ekinfo))
    base_warn = {w for r in R["results"] if "target_dim" not in r for w in r.get("warnings", [])}
    out.append(f"\nBaseline warnings: {sorted(base_warn)}")
S = json.loads((HERE / "results_step4.json").read_text())
out.append("\n### Step 4 trajectories (synthetic)\n")
for run in S["runs"]:
    out.append(f"\nseed {run['seed']} target_dim={run['target_dim']} rho(kskew,acc)={run['spearman_kskew_vs_acc']} "
               f"rho(unif,acc)={run['spearman_uniformity_vs_acc']} argmin_kskew_ep={run['argmin_kskew_epoch']} "
               f"argmax_acc_ep={run['argmax_acc_epoch']} sim_ES_stop={run['simulated_es_stop_epoch']} acc@stop={run['acc_at_simulated_stop']:.4f}\n")
    out.append("| epoch | loss | k-skew (held-out) | uniformity (held-out) | test kNN acc |\n|---|---|---|---|---|")
    for t in run["trajectory"]:
        out.append(f"| {t['epoch']} | {t['loss']:.3f} | {t['k_skew_heldout']:.3f} | {t['uniformity_heldout']:.3f} | {t['test_knn_acc']:.4f} |")
(HERE / "results_tables.md").write_text("\n".join(out))
print("\n".join(out))
