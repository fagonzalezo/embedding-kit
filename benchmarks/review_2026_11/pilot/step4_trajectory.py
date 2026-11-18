import sys; sys.modules["faiss"] = None
import json, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.neighbors import KNeighborsClassifier

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from datasets import make_synthetic  # noqa

import embedkit.utils.neighbors as nbrs  # noqa
from embedkit import EmbedKitAnalyzer  # noqa
from embedkit.analysis.hubness import HubnessAnalyzer  # noqa
from embedkit.analysis.geometry import UniformityScore  # noqa
from embedkit.improvement.augmentation import KNNPairs  # noqa
from embedkit.improvement.losses import NTXentLoss, AlignUniformLoss, CombinedLoss  # noqa
from embedkit.improvement.model import EmbeddingRefiner  # noqa
from embedkit.improvement.trainer import Trainer  # noqa

nbrs._MAX_CACHE = 0  # disable address-keyed kNN cache (stale-result bug)

Xtr_all, ytr_all, Xte, yte, info = make_synthetic(sep=1.0, nuis_scale=2.0)
out = {"info": info, "runs": []}

for seed in (0, 1, 2):
    nbrs.clear_cache()
    # same analysis EmbedKit.fit runs (on the full train split) -> auto target_dim
    rep = EmbedKitAnalyzer(random_state=seed).fit(Xtr_all)
    td = int(rep.suggested_target_dim)
    # same held-out carve-out as Trainer(val_split=0.1, random_state=seed)
    n = len(Xtr_all)
    n_val = max(1, int(n * 0.1))
    val_idx = np.random.default_rng(seed).choice(n, n_val, replace=False)
    mask = np.ones(n, bool); mask[val_idx] = False
    Xfit, yfit, Xho = Xtr_all[mask], ytr_all[mask], Xtr_all[val_idx]

    traj = []

    class TrajTrainer(Trainer):
        def _evaluate(self, X_np):
            nbrs.clear_cache()
            Zho = self.transform(Xho)
            Zfit = self.transform(Xfit)
            Zte = self.transform(Xte)
            ep = len(self.history["loss"])
            row = dict(epoch=ep, loss=self.history["loss"][-1],
                       k_skew_heldout=float(HubnessAnalyzer().fit(Zho).k_skewness),
                       uniformity_heldout=float(UniformityScore().fit(Zho).uniformity),
                       test_knn_acc=float(KNeighborsClassifier(5).fit(Zfit, yfit).score(Zte, yte)))
            traj.append(row)
            return {"k_skewness": row["k_skew_heldout"], "uniformity": row["uniformity_heldout"]}

    model = EmbeddingRefiner(input_dim=Xfit.shape[1], target_dim=td, hidden_dim=256, n_layers=2)
    loss = CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])
    tr = TrajTrainer(model=model, augmentation=KNNPairs(k=5), loss=loss, epochs=100, batch_size=256,
                     optimizer="adam", lr=3e-4, weight_decay=1e-4, scheduler="cosine", eval_every=5,
                     eval_metrics=["k_skewness"], early_stopping_patience=None, monitor="k_skewness",
                     val_split=0.0, random_state=seed)
    t0 = time.perf_counter(); tr.fit(Xfit); dt = time.perf_counter() - t0
    acc = np.array([r["test_knn_acc"] for r in traj])
    ks = np.array([r["k_skew_heldout"] for r in traj])
    un = np.array([r["uniformity_heldout"] for r in traj])
    rho_k = spearmanr(ks, acc); rho_u = spearmanr(un, acc)
    # what would early stopping (patience 10 evals, min k_skew, no weight restore) have picked?
    best, pc, stop = np.inf, 0, None
    for i, v in enumerate(ks):
        if v < best: best, pc = v, 0
        else: pc += 1
        if pc >= 10: stop = i; break
    run = dict(seed=seed, target_dim=td, fit_time=dt, trajectory=traj,
               spearman_kskew_vs_acc=[float(rho_k.statistic), float(rho_k.pvalue)],
               spearman_uniformity_vs_acc=[float(rho_u.statistic), float(rho_u.pvalue)],
               argmin_kskew_epoch=int(traj[int(np.argmin(ks))]["epoch"]),
               argmax_acc_epoch=int(traj[int(np.argmax(acc))]["epoch"]),
               simulated_es_stop_epoch=(int(traj[stop]["epoch"]) if stop is not None else None),
               acc_at_simulated_stop=(float(acc[stop]) if stop is not None else float(acc[-1])))
    out["runs"].append(run)
    print(json.dumps({k: v for k, v in run.items() if k != "trajectory"}), flush=True)
    for r in traj:
        print("  ", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, flush=True)
    (HERE / "results_step4.json").write_text(json.dumps(out, indent=1))
