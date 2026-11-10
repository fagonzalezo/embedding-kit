import sys; sys.modules["faiss"] = None  # FAISS segfaults in this env -> sklearn fallback
import json, time, warnings, traceback
from pathlib import Path

import numpy as np
from scipy.stats import skew
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import trustworthiness
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
from datasets import load_pets, load_digits_split, make_synthetic  # noqa: E402

import embedkit.utils.neighbors as nbrs  # noqa: E402
from embedkit import EmbedKit  # noqa: E402

# The kNN cache is keyed on the array memory address and returns stale results when
# an address is reused. Disable it entirely (equivalent to clear_cache() before every call,
# including the calls made inside Trainer._evaluate and KNNPairs.precompute).
nbrs._MAX_CACHE = 0
nbrs.clear_cache()

DS = sys.argv[1]
SEEDS = [0, 1, 2]
OUT = HERE / f"results_{DS}.json"


def get_data():
    if DS == "pets":
        return load_pets() + ({"note": "ResNet50 ImageNet features, Oxford-IIIT Pets"},)
    if DS == "digits":
        return load_digits_split() + ({"note": "raw 8x8 pixels, NOT a learned embedding"},)
    if DS == "synthetic":
        return make_synthetic(sep=1.0, nuis_scale=2.0)
    raise ValueError(DS)


Xtr, ytr, Xte, yte, info = get_data()
n, d = Xtr.shape
print(DS, Xtr.shape, Xte.shape, info, flush=True)
rng = np.random.default_rng(0)
tw_idx = np.sort(rng.choice(len(Xte), min(2000, len(Xte)), replace=False))


def metrics(Ztr, Zte):
    Ztr = np.asarray(Ztr, dtype=np.float64); Zte = np.asarray(Zte, dtype=np.float64)
    nbrs.clear_cache()
    m = {}
    m["knn_euc"] = KNeighborsClassifier(5).fit(Ztr, ytr).score(Zte, yte)
    m["knn_cos"] = KNeighborsClassifier(5, metric="cosine", algorithm="brute").fit(Ztr, ytr).score(Zte, yte)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(Ztr, ytr)
    m["linprobe"] = lr.score(Zte, yte)
    Xs = Xte[tw_idx].astype(np.float64); Zs = Zte[tw_idx]
    m["trust"] = float(trustworthiness(Xs, Zs, n_neighbors=10))
    m["contin"] = float(trustworthiness(Zs, Xs, n_neighbors=10))
    _, ind = NearestNeighbors(n_neighbors=11).fit(Zte).kneighbors(Zte)
    Nk = np.bincount(ind[:, 1:].ravel(), minlength=len(Zte))
    m["k_skew"] = float(skew(Nk))
    ev = np.clip(np.linalg.eigvalsh(np.cov(Zte, rowvar=False)), 0, None)
    p = ev / ev.sum(); p = p[p > 0]
    m["eff_rank"] = float(np.exp(-(p * np.log(p)).sum()))
    return m


results = []


def record(name, fn, seed=None, extra_fn=None):
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            Ztr, Zte, extra = fn()
        fit_time = time.perf_counter() - t0
        m = metrics(Ztr, Zte)
        m.update(method=name, seed=seed, fit_time=fit_time, dim=int(np.asarray(Zte).shape[1]),
                 warnings=sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in wl}), **(extra or {}))
        if not np.all(np.isfinite(Zte)):
            m["nonfinite"] = True
    except Exception as e:
        m = dict(method=name, seed=seed, error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
    results.append(m)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
           if k not in ("history", "tb", "warnings")}, flush=True)
    OUT.write_text(json.dumps(dict(dataset=DS, info=info, shape_train=[n, d], shape_test=list(Xte.shape),
                                   results=results), indent=1, default=float))


# ---------------- training-free baselines ----------------
record("raw", lambda: (Xtr, Xte, None))


def centre_l2():
    mu = Xtr.mean(0)
    f = lambda X: (X - mu) / (np.linalg.norm(X - mu, axis=1, keepdims=True) + 1e-12)
    return f(Xtr), f(Xte), None
record("centre+L2", centre_l2)


def zscore():
    s = StandardScaler().fit(Xtr)
    return s.transform(Xtr), s.transform(Xte), None
record("standardise", zscore)

ks = sorted({min(k, d) for k in (16, 32, 64, 128)})
# rank of the centred train data (digits has constant pixels -> whitening would divide by 0)
rank = int(np.linalg.matrix_rank(Xtr - Xtr.mean(0)))
for k in ks:
    def pca(k=k):
        p = PCA(k, random_state=0).fit(Xtr)
        return p.transform(Xtr), p.transform(Xte), None
    record(f"PCA-{k}", pca)
for k in ks:
    kk = min(k, rank)
    def pcaw(kk=kk):
        p = PCA(kk, whiten=True, random_state=0).fit(Xtr)
        return p.transform(Xtr), p.transform(Xte), None
    record(f"PCA-whiten-{k}" + (f" (capped to rank {kk})" if kk != k else ""), pcaw)

Ds = []
for D in (1, 3, max(1, d // 100)):
    if D not in Ds:
        Ds.append(D)
for D in Ds:
    def abtt(D=D):
        mu = Xtr.mean(0)
        U = PCA(D, random_state=0).fit(Xtr).components_
        f = lambda X: (X - mu) - ((X - mu) @ U.T) @ U
        return f(Xtr), f(Xte), None
    record(f"ABTT-D{D}", abtt)


# ---------------- EmbedKit ----------------
def ek_run(**kw):
    def fn():
        nbrs.clear_cache()
        ek = EmbedKit(**kw)
        if kw.get("mode") == "supervised":
            ek.fit(Xtr, ytr)
        else:
            ek.fit(Xtr)
        Ztr = ek.transform(Xtr); Zte = ek.transform(Xte)
        tr = ek._trainer
        extra = dict(target_dim=int(ek._config["target_dim"]), stopped_epoch=int(tr.stopped_epoch_),
                     epochs=int(tr.epochs),
                     history={k: [float(v) for v in vs] for k, vs in tr.history.items()},
                     analysis_consensus_id=float(ek.analysis_report.intrinsic_dim.consensus),
                     analysis_id_estimates={k: float(v) for k, v in ek.analysis_report.intrinsic_dim.estimates.items()})
        return Ztr, Zte, extra
    return fn


for s in SEEDS:
    record("EmbedKit-default", ek_run(mode="self_supervised", random_state=s), seed=s)
for s in SEEDS:
    record("EmbedKit-td64-noES-80ep", ek_run(mode="self_supervised", target_dim=64, early_stopping_patience=None,
                                             val_split=0.0, epochs=80, random_state=s), seed=s)
for s in SEEDS:
    record("EmbedKit-supervised", ek_run(mode="supervised", random_state=s), seed=s)
print("DONE", DS, flush=True)
