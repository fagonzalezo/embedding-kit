"""Scalar metrics for judging an embedding or a transform of one.

Every function takes NumPy arrays (``torch.Tensor`` is accepted for the 2-D embedding
arguments), has no hidden state and returns a Python ``float``. Metrics that compare a transformed
space ``Z`` with an original space ``X`` take the two arrays row-aligned: row ``i`` of ``Z`` is the
image of row ``i`` of ``X``.

Held-out metrics (:func:`knn_accuracy`, :func:`linear_probe`, :func:`recall_at_k`) take the
training and test splits separately, so that the caller cannot leak test points into the
neighbour search or the fit.
"""

from __future__ import annotations

import numpy as np

from embedkit.utils.validation import _to_numpy, encode_labels

__all__ = [
    "knn_accuracy",
    "knn_regression_r2",
    "linear_probe",
    "recall_at_k",
    "trustworthiness",
    "continuity",
    "knn_overlap",
    "linear_cka",
    "effective_rank",
    "k_skewness",
    "nc1",
    "silhouette",
]

_PROBE_CS = np.logspace(-3, 2, 6)
_PROBE_ALPHAS = np.logspace(-3, 5, 9)
_PROBE_MAX_ITER = 500


# ----------------------------------------------------------------------------- helpers
def _labels_to_numpy(y) -> np.ndarray:
    try:
        import torch

        if isinstance(y, torch.Tensor):
            return y.detach().cpu().numpy()
    except ImportError:  # pragma: no cover
        pass
    return np.asarray(y)


def _joint_labels(y_a, y_b, name_a="y_train", name_b="y_test"):
    """Encode two label sets with one code book. Returns ``(codes_a, codes_b, kind)``."""
    a, b = _labels_to_numpy(y_a), _labels_to_numpy(y_b)
    if a.ndim != b.ndim or (a.ndim == 2 and a.shape[1] != b.shape[1]):
        raise ValueError(f"{name_a} and {name_b} have incompatible shapes {a.shape}, {b.shape}")
    both = np.concatenate([a, b], axis=0)
    codes, kind = encode_labels(both)
    codes = codes.numpy()
    return codes[: len(a)], codes[len(a):], kind


def _check_rows(Z, y, name):
    if len(y) != Z.shape[0]:
        raise ValueError(f"{name}: got {len(y)} labels for {Z.shape[0]} rows")


def _to_float64(Z) -> np.ndarray:
    """2-D float64 array from an ndarray or a tensor, without the float32 round trip."""
    try:
        import torch

        if isinstance(Z, torch.Tensor):
            Z = Z.detach().cpu().double().numpy()
    except ImportError:  # pragma: no cover
        pass
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {Z.shape}")
    return Z


def _check_aligned(X, Z):
    if X.shape[0] != Z.shape[0]:
        raise ValueError(
            f"X and Z must have the same number of rows, got {X.shape[0]} and {Z.shape[0]}"
        )


def _subsample_rows(n: int, max_samples: int | None, random_state: int) -> np.ndarray | None:
    """Sorted random row indices (fixed seed), or ``None`` when no subsampling is needed."""
    if max_samples is None or n <= max_samples:
        return None
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(n, size=int(max_samples), replace=False))


def _check_same_dim(A, B, names=("train", "test")):
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"{names[0]} and {names[1]} have different dimensions: {A.shape[1]} and {B.shape[1]}"
        )


def _reduced_neighbors(reduction, metric, Z_ref, Z_query, k):
    """Indices of the ``k`` neighbours under a hubness reduction built on ``metric``.

    ``Z_query=None`` searches the reference rows among themselves.
    """
    r = reduction.clone(metric=metric).fit(Z_ref)
    k = int(min(k, len(Z_ref) - (1 if Z_query is None else 0)))
    return r.kneighbors(Z_query, n_neighbors=k, return_distance=False)


# ----------------------------------------------------------------------------- held-out
def knn_accuracy(
    Z_train,
    y_train,
    Z_test,
    y_test,
    k: int = 5,
    metric: str = "cosine",
    return_per_point: bool = False,
    hubness_reduction=None,
):
    """Test accuracy of a ``k``-nearest-neighbour majority vote.

    The neighbours of every test point are searched among the training points only. Ties in the
    vote go to the smallest class code (the sorted order of the label values). A test label that
    never occurs in the training set can never be predicted and counts as an error.

    Labels may be integers, strings or booleans. Continuous labels raise ``ValueError``.

    With ``return_per_point=True`` the result is ``(accuracy, correct)`` where ``correct`` is a
    float array of shape ``(n_test,)`` holding 1.0 for a correct prediction and 0.0 otherwise
    (its mean is the accuracy), e.g. for bootstrapping paired differences over test points.

    ``hubness_reduction`` is an object from :mod:`embedkit.hubness_reduction`, e.g.
    ``MutualProximity()``. The neighbours are then those under its secondary distance, built
    on ``metric`` (a copy with that metric is fitted on the training points).
    """
    from sklearn.neighbors import KNeighborsClassifier

    Ztr, Zte = _to_numpy(Z_train), _to_numpy(Z_test)
    _check_same_dim(Ztr, Zte)
    ytr, yte, kind = _joint_labels(y_train, y_test)
    if kind != "class":
        raise ValueError("knn_accuracy needs class labels, got continuous labels")
    _check_rows(Ztr, ytr, "train")
    _check_rows(Zte, yte, "test")
    if hubness_reduction is not None:
        idx = _reduced_neighbors(hubness_reduction, metric, Ztr, Zte, k)
        votes = np.zeros((len(Zte), int(ytr.max()) + 1))
        np.add.at(votes, (np.arange(len(Zte))[:, None], ytr[idx]), 1.0)
        pred = votes.argmax(axis=1)  # ties: the smallest class code, as in scikit-learn
        correct = (pred == yte).astype(np.float64)
    else:
        clf = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute")
        clf.fit(Ztr, ytr)
        correct = (clf.predict(Zte) == yte).astype(np.float64)
    value = float(correct.mean())
    return (value, correct) if return_per_point else value


def linear_probe(
    Z_train,
    y_train,
    Z_test,
    y_test,
    task: str = "auto",
    cv: int = 5,
    random_state: int = 0,
    return_per_point: bool = False,
    C: float | None = None,
    max_iter: int | None = None,
):
    """Test score of a linear model fitted on the training split.

    Features are standardised with the training mean and standard deviation. Columns whose
    training standard deviation is below ``1e-6`` times the largest one are only centred, so
    that rounding residue in a (numerically) constant column is not scaled up into a feature.
    The regularisation strength is chosen on the training split only:

    - ``task="classification"``: multinomial logistic regression (L-BFGS, ``max_iter=500``) with
      the inverse regularisation ``C`` chosen from ``logspace(-3, 2, 6)`` by ``cv``-fold
      stratified cross-validation on accuracy (the smaller ``C`` wins ties; ``cv`` is lowered to
      the size of the smallest class when that class has fewer than ``cv`` members; with a class
      of one member a fixed ``C = 1`` is used). The loop over ``C`` is explicit rather than
      ``LogisticRegressionCV``, whose API is changing across scikit-learn releases and which
      warns on every call. Returns the **test accuracy**.
    - ``task="regression"``: ridge regression with the penalty chosen from
      ``logspace(-3, 5, 9)`` by efficient leave-one-out cross-validation (``RidgeCV``; ``cv`` is
      not used). For 2-D targets one penalty is shared by all outputs. Returns the **test R^2**,
      averaged over outputs for 2-D targets.
    - ``task="auto"``: classification when :func:`embedkit.utils.validation.encode_labels` reports
      class labels, regression for continuous labels.

    ``C`` fixes the inverse regularisation of the classification probe instead of choosing it by
    cross-validation (``C=1.0, max_iter=3000`` reproduces the review pilot's probe). ``max_iter``
    overrides the L-BFGS iteration limit (default 500).

    With ``return_per_point=True`` the result is ``(value, per_point)``: for classification
    ``per_point`` is the float array of shape ``(n_test,)`` of 0/1 correctness (its mean is the
    accuracy); for regression it is ``1 - e_i^2 / v`` with ``e_i`` the residual of test point
    ``i`` and ``v`` the (population) variance of the test targets, whose mean is exactly R^2
    (see :func:`_r2_terms`). For 2-D targets ``per_point`` is ``None``.

    Convergence warnings from L-BFGS are not silenced. The largest ``C`` on the grid is close to
    unregularised and may not converge within ``max_iter`` on separable data; that candidate then
    usually loses the cross-validation, and a caller who wants a quiet run can filter
    ``ConvergenceWarning`` itself.
    """
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"task must be 'auto', 'classification' or 'regression', got {task!r}")
    Ztr, Zte = _to_numpy(Z_train), _to_numpy(Z_test)
    _check_same_dim(Ztr, Zte)
    Ztr, Zte = _standardize(Ztr, Zte)

    ytr, yte, kind = _joint_labels(y_train, y_test)
    _check_rows(Ztr, ytr, "train")
    _check_rows(Zte, yte, "test")
    if task == "auto":
        task = "classification" if kind == "class" else "regression"

    if task == "classification":
        if kind != "class":
            raise ValueError("task='classification' needs class labels, got continuous labels")
        correct = _probe_classification(
            Ztr, ytr, Zte, yte, cv, random_state, C=C, max_iter=max_iter or _PROBE_MAX_ITER
        )
        value = float(correct.mean())
        return (value, correct) if return_per_point else value
    value, per_point = _probe_regression(Ztr, y_train, Zte, y_test)
    return (value, per_point) if return_per_point else value


# A column whose training standard deviation is below this fraction of the largest column's is
# treated as constant: centred but not scaled. Scaling it to unit variance would turn rounding
# residue into a feature. Example: all-but-the-top on scikit-learn digits leaves ~1e-17 in an
# all-zero pixel column, proportional to the removed principal component; unit scaling put that
# component back and raised the probe accuracy by 3 to 6 points.
_CONSTANT_RTOL = 1e-6


def _standardize(Ztr: np.ndarray, Zte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score with training statistics; numerically constant columns are only centred."""
    Ztr = Ztr.astype(np.float64)
    Zte = Zte.astype(np.float64)
    mean = Ztr.mean(axis=0)
    std = Ztr.std(axis=0)
    constant = std <= _CONSTANT_RTOL * max(float(std.max(initial=0.0)), np.finfo(np.float64).tiny)
    scale = np.where(constant, 1.0, std)
    return (Ztr - mean) / scale, (Zte - mean) / scale


def _probe_classification(
    Ztr, ytr, Zte, yte, cv, random_state, C=None, max_iter=None
) -> np.ndarray:
    """0/1 correctness of the tuned (or fixed-C) logistic regression on each test point."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    min_count = int(np.bincount(np.unique(ytr, return_inverse=True)[1]).min())
    n_splits = min(cv, min_count)
    max_iter = max_iter or _PROBE_MAX_ITER
    best_C = 1.0 if C is None else float(C)
    if C is None and n_splits >= 2:
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
        scores = [
            cross_val_score(
                LogisticRegression(C=c, solver="lbfgs", max_iter=max_iter),
                Ztr,
                ytr,
                cv=folds,
                scoring="accuracy",
            ).mean()
            for c in _PROBE_CS
        ]
        best_C = float(_PROBE_CS[int(np.argmax(scores))])  # first maximum: the smaller C
    model = LogisticRegression(C=best_C, solver="lbfgs", max_iter=max_iter)
    model.fit(Ztr, ytr)
    return (model.predict(Zte) == yte).astype(np.float64)


def _probe_regression(Ztr, y_train, Zte, y_test) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    try:
        ytr = np.asarray(_labels_to_numpy(y_train), dtype=np.float64)
        yte = np.asarray(_labels_to_numpy(y_test), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("task='regression' needs numeric targets") from exc
    if ytr.ndim == 2 and ytr.shape[1] == 1:
        ytr, yte = ytr[:, 0], yte.reshape(len(yte), -1)[:, 0]
    model = RidgeCV(alphas=_PROBE_ALPHAS).fit(Ztr, ytr)
    pred = model.predict(Zte)
    value = float(r2_score(yte, pred, multioutput="uniform_average"))
    return value, (_r2_terms(yte, pred) if yte.ndim == 1 else None)


def _r2_terms(y, pred) -> np.ndarray:
    """Per-point terms ``1 - (y_i - pred_i)^2 / var(y)``, whose mean is R^2.

    With a fixed test set the denominator is a constant, so a paired bootstrap over these terms
    resamples R^2 differences as differences of squared errors.
    """
    y = np.asarray(y, dtype=np.float64)
    v = float(np.mean((y - y.mean()) ** 2))
    if v == 0.0:
        raise ValueError("R^2 is undefined for constant test targets")
    return 1.0 - (y - np.asarray(pred, dtype=np.float64)) ** 2 / v


def knn_regression_r2(
    Z_train, y_train, Z_test, y_test, k: int = 10, metric: str = "cosine",
    return_per_point: bool = False,
):
    """Test R^2 of ``k``-nearest-neighbour regression (mean target of the neighbours).

    Neighbours of each test point are searched among the training points. Targets must be
    continuous and 1-D (integers are accepted and treated as numbers). With
    ``return_per_point=True`` the result is ``(r2, terms)``, with the terms of :func:`_r2_terms`.
    """
    from sklearn.neighbors import KNeighborsRegressor

    Ztr, Zte = _to_numpy(Z_train), _to_numpy(Z_test)
    _check_same_dim(Ztr, Zte)
    ytr = np.asarray(_labels_to_numpy(y_train), dtype=np.float64)
    yte = np.asarray(_labels_to_numpy(y_test), dtype=np.float64)
    if ytr.ndim != 1 or yte.ndim != 1:
        raise ValueError("knn_regression_r2 needs 1-D targets")
    _check_rows(Ztr, ytr, "train")
    _check_rows(Zte, yte, "test")
    reg = KNeighborsRegressor(n_neighbors=min(k, len(Ztr)), metric=metric, algorithm="brute")
    terms = _r2_terms(yte, reg.fit(Ztr, ytr).predict(Zte))
    value = float(terms.mean())
    return (value, terms) if return_per_point else value


def recall_at_k(
    Z_query,
    Z_gallery,
    relevance,
    k: int = 10,
    metric: str = "cosine",
    exclude_self: bool = False,
    return_per_point: bool = False,
    hubness_reduction=None,
):
    """Recall@k as used in metric learning: the hit rate of the top ``k`` gallery items.

    For each query the score is 1 if at least one relevant gallery item is among its ``k``
    nearest gallery items, else 0; the result is the mean over queries. Queries without any
    relevant gallery item are left out of the mean; if no query has one, ``ValueError`` is raised.

    ``relevance`` is either

    - a **tuple** ``(y_query, y_gallery)``: a gallery item is relevant to a query when their
      labels are equal, or
    - a boolean array of shape ``(n_query, n_gallery)``.

    ``exclude_self=True`` is for the case where queries and gallery are the same set: the item
    with the same index is neither retrieved nor counted as relevant.

    ``metric`` is ``"cosine"`` or ``"euclidean"``. ``k`` larger than the gallery means the whole
    gallery. Ties at the ``k``-th distance are broken arbitrarily.

    With ``return_per_point=True`` the result is ``(recall, hits)``, where ``hits`` is the float
    array of 0/1 hits of the queries that are counted (those with at least one relevant item, in
    query order); its mean is the recall.

    ``hubness_reduction``: rank the gallery under that object's secondary distance (see
    :func:`knn_accuracy`).
    """
    Q, G = _to_numpy(Z_query), _to_numpy(Z_gallery)
    _check_same_dim(Q, G, ("query", "gallery"))
    nq, ng = Q.shape[0], G.shape[0]
    if metric not in ("cosine", "euclidean"):
        raise ValueError(f"metric must be 'cosine' or 'euclidean', got {metric!r}")
    if exclude_self and nq != ng:
        raise ValueError("exclude_self=True needs as many queries as gallery items")

    if isinstance(relevance, tuple):
        if len(relevance) != 2:
            raise ValueError("relevance as a tuple must be (y_query, y_gallery)")
        yq, yg = _joint_codes_any(relevance[0], relevance[1])
        _check_rows(Q, yq, "query")
        _check_rows(G, yg, "gallery")
        rel_full = None
    else:
        rel_full = np.asarray(relevance)
        if rel_full.dtype != bool:
            raise ValueError("relevance must be a boolean array or a (y_query, y_gallery) tuple")
        if rel_full.shape != (nq, ng):
            raise ValueError(f"relevance has shape {rel_full.shape}, expected {(nq, ng)}")

    n_candidates = ng - 1 if exclude_self else ng
    if n_candidates < 1:
        raise ValueError("The gallery has no item to retrieve")
    k = int(min(k, n_candidates))
    if k < 1:
        raise ValueError("k must be at least 1")

    if hubness_reduction is not None:
        top = _reduced_neighbors(hubness_reduction, metric, G, None if exclude_self else Q, k)
        if rel_full is None:
            hit = (yg[top] == yq[:, None]).any(axis=1)
            if exclude_self:
                has_rel = np.bincount(yg)[yq] > 1
            else:
                has_rel = np.isin(yq, yg)
        else:
            rel = rel_full.copy()
            if exclude_self:
                np.fill_diagonal(rel, False)
            hit = np.take_along_axis(rel, top, axis=1).any(axis=1)
            has_rel = rel.any(axis=1)
        hit_all = hit[has_rel].astype(np.float64)
        if hit_all.size == 0:
            raise ValueError("No query has a relevant gallery item")
        value = float(hit_all.mean())
        return (value, hit_all) if return_per_point else value

    if metric == "cosine":
        norm = np.linalg.norm
        Qn = Q / np.maximum(norm(Q, axis=1, keepdims=True), 1e-12)
        Gn = G / np.maximum(norm(G, axis=1, keepdims=True), 1e-12)
        G_sq = None
    else:
        Qn, Gn = Q, G
        G_sq = np.einsum("ij,ij->i", G, G, dtype=np.float64)

    hit_parts = []
    chunk = max(1, min(nq, 2_000_000 // max(ng, 1)))
    for start in range(0, nq, chunk):
        stop = min(nq, start + chunk)
        q = Qn[start:stop]
        if metric == "cosine":
            dist = 1.0 - q @ Gn.T
        else:
            q_sq = np.einsum("ij,ij->i", q, q, dtype=np.float64)
            dist = q_sq[:, None] - 2.0 * (q @ Gn.T) + G_sq[None, :]
        rows = np.arange(stop - start)
        if rel_full is None:
            rel = yq[start:stop, None] == yg[None, :]
        else:
            rel = rel_full[start:stop].copy()
        if exclude_self:
            dist[rows, start + rows] = np.inf
            rel[rows, start + rows] = False
        top = np.argpartition(dist, k - 1, axis=1)[:, :k]
        has_rel = rel.any(axis=1)
        hit = np.take_along_axis(rel, top, axis=1).any(axis=1)
        hit_parts.append(hit[has_rel])
    hit_all = np.concatenate(hit_parts).astype(np.float64)
    if hit_all.size == 0:
        raise ValueError("No query has a relevant gallery item")
    value = float(hit_all.mean())
    return (value, hit_all) if return_per_point else value


def _joint_codes_any(y_a, y_b):
    """Integer codes for two label arrays of any dtype, sharing one code book."""
    a, b = _labels_to_numpy(y_a), _labels_to_numpy(y_b)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Labels for recall_at_k must be 1-D")
    _, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
    inv = inv.reshape(-1)
    return inv[: len(a)], inv[len(a):]


# ----------------------------------------------------------- neighbourhood preservation
def trustworthiness(
    X,
    Z,
    k: int = 10,
    max_samples: int = 5000,
    random_state: int = 0,
    metric: str = "euclidean",
) -> float:
    """Trustworthiness of ``Z`` as a representation of ``X`` (Venna and Kaski 2001).

    Wraps :func:`sklearn.manifold.trustworthiness`: 1 means that no point enters the ``k``
    nearest neighbours in ``Z`` without already being near in ``X``. ``metric`` is used in both
    spaces. When there are more than ``max_samples`` rows, the same randomly chosen rows (fixed
    by ``random_state``) of ``X`` and ``Z`` are used. ``k`` must be below ``n / 2``
    (scikit-learn's requirement, applied to the subsample).
    """
    from sklearn.manifold import trustworthiness as _sk_trustworthiness

    X, Z = _to_numpy(X), _to_numpy(Z)
    _check_aligned(X, Z)
    idx = _subsample_rows(X.shape[0], max_samples, random_state)
    if idx is not None:
        X, Z = X[idx], Z[idx]
    return float(_sk_trustworthiness(X, Z, n_neighbors=k, metric=metric))


def continuity(
    X,
    Z,
    k: int = 10,
    max_samples: int = 5000,
    random_state: int = 0,
    metric: str = "euclidean",
) -> float:
    """Continuity of ``Z`` with respect to ``X``: trustworthiness with the roles swapped.

    1 means that no neighbour of a point in ``X`` is pushed out of its ``k`` nearest neighbours
    in ``Z``. Same arguments and subsampling as :func:`trustworthiness`.
    """
    return trustworthiness(Z, X, k=k, max_samples=max_samples, random_state=random_state,
                           metric=metric)


def knn_overlap(
    X,
    Z,
    k: int = 10,
    metric_x: str = "euclidean",
    metric_z: str = "cosine",
    max_samples: int | None = None,
    random_state: int = 0,
) -> float:
    """Mean fraction of shared ``k``-nearest neighbours between two spaces.

    ``mean_i |N_k^X(i) & N_k^Z(i)| / k`` with exact neighbours from
    :func:`embedkit.utils.neighbors.knn`, each space with its own metric (the defaults follow the
    usual choice of Euclidean for raw features and cosine for embeddings). Independent spaces of
    ``n`` points give about ``k / (n - 1)``. With ``max_samples`` set and ``n`` larger, the same
    random rows (fixed by ``random_state``) of both spaces are used and the neighbours are those
    within the subsample.
    """
    from embedkit.utils.neighbors import knn

    X, Z = _to_numpy(X), _to_numpy(Z)
    _check_aligned(X, Z)
    idx = _subsample_rows(X.shape[0], max_samples, random_state)
    if idx is not None:
        X, Z = X[idx], Z[idx]
    _, nx = knn(X, k, metric=metric_x)
    _, nz = knn(Z, k, metric=metric_z)
    shared = (nx[:, :, None] == nz[:, None, :]).any(axis=2).sum(axis=1)
    return float(np.mean(shared / k))


# ----------------------------------------------------------------------- representation
def linear_cka(X, Z) -> float:
    """Centred linear CKA between two representations of the same ``n`` points.

    Kornblith et al. (2019): with column-centred ``Xc`` and ``Zc``,
    ``||Xc^T Zc||_F^2 / (||Xc^T Xc||_F * ||Zc^T Zc||_F)``. It is 1 for ``Z`` equal to ``X`` up to
    rotation and isotropic scaling, and near 0 for independent data. Computed in float64, in
    the ``n x n`` Gram form (``<K, L>_F / (||K||_F ||L||_F)``) when that is cheaper than the
    feature-space form; the two are equal. Returns ``nan`` when either input is constant.
    """
    X, Z = _to_float64(X), _to_float64(Z)
    _check_aligned(X, Z)
    n, p = X.shape
    q = Z.shape[1]
    Xc = X - X.mean(axis=0)
    Zc = Z - Z.mean(axis=0)
    if n * n * (p + q + 3) < n * (p * q + p * p + q * q):
        K, L = Xc @ Xc.T, Zc @ Zc.T
        num = float(np.sum(K * L))
        den = float(np.linalg.norm(K) * np.linalg.norm(L))
    else:
        num = float(np.linalg.norm(Xc.T @ Zc) ** 2)
        den = float(np.linalg.norm(Xc.T @ Xc) * np.linalg.norm(Zc.T @ Zc))
    if den <= 0.0:
        return float("nan")
    return num / den


def effective_rank(Z, center: bool = False) -> float:
    """Effective rank of the rows of ``Z`` (RankMe, Garrido et al. 2023).

    ``exp(-sum_k p_k log p_k)`` with ``p_k = s_k / sum_j s_j``, where ``s`` are **all** the
    singular values of ``Z`` (of ``Z`` minus its column mean when ``center=True``), not the
    eigenvalues of a covariance matrix and not a truncated spectrum. Singular values that are zero
    up to rounding (``<= eps * max(n, d) * ||Z||_F``) are ignored. The value lies in
    ``[1, min(n, d)]``: 1 for rank-1 data and ``r`` for ``r`` equal non-zero singular values.
    Returns ``nan`` when there is no non-zero singular value (all-zero ``Z``, or constant rows
    with ``center=True``).
    """
    Z = _to_float64(Z)
    tol = np.finfo(np.float64).eps * max(Z.shape) * np.linalg.norm(Z)
    if center:
        Z = Z - Z.mean(axis=0)
    s = np.linalg.svd(Z, compute_uv=False)
    s = s[s > tol]
    if s.size == 0:
        return float("nan")
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def k_skewness(Z, k: int = 10, metric: str = "cosine", hubness_reduction=None) -> float:
    """Skewness of the ``k``-occurrence distribution of ``Z`` (hubness; larger is hubbier).

    Same computation as ``HubnessAnalyzer(k=k, metric=metric).fit(Z).k_skewness`` (they share
    :func:`embedkit.analysis.hubness.k_occurrence_skewness`), with exact neighbours on all rows;
    ``HubnessAnalyzer`` subsamples above 20,000 rows by default, this function does not.

    ``hubness_reduction``: the neighbours under that object's secondary distance (see
    :func:`knn_accuracy`), i.e. the hubness that remains after the reduction.
    """
    from embedkit.analysis.hubness import k_occurrence, k_occurrence_skewness
    from embedkit.utils.neighbors import knn

    Z = _to_numpy(Z)
    if hubness_reduction is not None:
        indices = _reduced_neighbors(hubness_reduction, metric, Z, None, k)
    else:
        _, indices = knn(Z, k, metric=metric)
    return k_occurrence_skewness(k_occurrence(indices, Z.shape[0]))


# ------------------------------------------------------------------------ class structure
def nc1(Z, y) -> float:
    """Neural-collapse NC1 variability (Papyan, Han and Donoho 2020): ``tr(Sw Sb^+) / C``.

    ``Sw`` is the within-class covariance, the average over **all** samples of the outer product of
    a sample's deviation from its class mean. ``Sb`` is the between-class covariance of the class
    means around the global mean, with every class weighted equally and the global mean taken as
    the mean of the class means (as in Papyan et al., who use balanced classes; with unbalanced
    classes this differs from the sample-weighted mean). ``Sb^+`` is its Moore-Penrose
    pseudo-inverse (``Sb`` has rank at most ``C - 1``) and ``C`` the number of classes. 0 for
    perfectly collapsed classes; larger values mean more within-class spread relative to the spread
    of the class means. Invariant to the scale of ``Z``.

    Computed without forming ``d x d`` matrices: with ``Sb = M^T M / C`` and the thin SVD
    ``M = U S V^T`` of the centred class means, ``NC1 = sum_k ||R v_k||^2 / (n s_k^2)`` over the
    non-negligible ``s_k``, where ``R`` stacks the within-class deviations.

    Class labels only; continuous labels and fewer than two classes raise ``ValueError``.
    """
    Z = _to_float64(Z)
    codes, kind = encode_labels(y, n_samples=Z.shape[0])
    if kind != "class":
        raise ValueError("nc1 needs class labels, got continuous labels")
    _, inv = np.unique(codes.numpy(), return_inverse=True)
    inv = inv.reshape(-1)
    n_classes = int(inv.max()) + 1
    if n_classes < 2:
        raise ValueError("nc1 needs at least two classes")
    n = Z.shape[0]
    counts = np.bincount(inv, minlength=n_classes).astype(np.float64)
    means = np.zeros((n_classes, Z.shape[1]))
    np.add.at(means, inv, Z)
    means /= counts[:, None]
    R = Z - means[inv]
    centred = means - means.mean(axis=0)
    _, s, Vt = np.linalg.svd(centred, full_matrices=False)
    keep = s > s.max() * max(centred.shape) * np.finfo(np.float64).eps
    if not keep.any():
        return float("nan")
    proj = R @ Vt[keep].T
    return float(np.sum(np.sum(proj**2, axis=0) / (n * s[keep] ** 2)))


def silhouette(
    Z,
    y,
    metric: str = "cosine",
    max_samples: int = 5000,
    random_state: int = 0,
) -> float:
    """Mean silhouette coefficient of the class labels ``y`` in ``Z``, in ``[-1, 1]``.

    Wraps :func:`sklearn.metrics.silhouette_score`, using ``sample_size=max_samples`` (a random
    subsample fixed by ``random_state``) when ``Z`` has more rows than that.
    """
    from sklearn.metrics import silhouette_score

    Z = _to_numpy(Z)
    codes, kind = encode_labels(y, n_samples=Z.shape[0])
    if kind != "class":
        raise ValueError("silhouette needs class labels, got continuous labels")
    sample_size = max_samples if Z.shape[0] > max_samples else None
    return float(
        silhouette_score(
            Z, codes.numpy(), metric=metric, sample_size=sample_size, random_state=random_state
        )
    )
