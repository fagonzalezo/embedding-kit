"""Training-free embedding transforms, to use as baselines for learned refinement.

Every class has the same small interface:

- ``fit(X, y=None) -> self``; ``y`` is accepted for pipeline compatibility and ignored;
- ``transform(X) -> np.ndarray`` (float32), using only statistics computed in ``fit``;
- ``fit_transform(X, y=None)``;
- ``get_params() -> dict`` with the constructor arguments, so a benchmark can record the
  configuration;
- after ``fit``: ``n_features_in_``, ``n_features_out_`` and the learned statistics, all with a
  trailing underscore. No training rows are stored.

Inputs may be NumPy arrays or torch tensors (they are converted to float32 on entry, as
everywhere in EmbedKit). Computation is done in float64 and the result is returned as float32.
``transform`` raises ``sklearn.exceptions.NotFittedError`` before ``fit`` and ``ValueError`` when
the number of columns differs from the one seen in ``fit``.

Covariances use the unbiased ``n - 1`` normalisation, except :class:`Standardize` (population
standard deviation, as in scikit-learn's ``StandardScaler``) and :class:`ConceptorNegation`
(``1 / n``, as in the paper).
"""
from __future__ import annotations

import inspect
import warnings

import numpy as np
from sklearn.exceptions import NotFittedError

from embedkit.utils.validation import _to_numpy

__all__ = [
    "Identity",
    "CenterNormalize",
    "Standardize",
    "PCAReduce",
    "ZCAWhiten",
    "AllButTheTop",
    "ConceptorNegation",
    "RandomProjection",
]


# ---------------------------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------------------------


class _Baseline:
    """Common plumbing: input validation, fitted checks, ``get_params``, ``fit_transform``."""

    def _fit_input(self, X, min_samples: int = 1) -> np.ndarray:
        X = _to_numpy(X).astype(np.float64)
        if X.shape[0] < min_samples:
            raise ValueError(
                f"{type(self).__name__} needs at least {min_samples} samples, got {X.shape[0]}")
        if X.shape[1] < 1:
            raise ValueError("X has no columns")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; all entries must be finite")
        self.n_features_in_ = X.shape[1]
        return X

    def _transform_input(self, X) -> np.ndarray:
        if not hasattr(self, "n_features_in_"):
            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet; call fit() first.")
        X = _to_numpy(X).astype(np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} was fitted with "
                f"{self.n_features_in_} features")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values; all entries must be finite")
        return X

    def get_params(self) -> dict:
        """Constructor arguments, so that ``type(obj)(**obj.get_params())`` rebuilds it."""
        params = inspect.signature(type(self).__init__).parameters.values()
        skip = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        names = [p.name for p in params if p.name != "self" and p.kind not in skip]
        return {name: getattr(self, name) for name in names}

    def fit_transform(self, X, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


def _out(Z: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(Z, dtype=np.float32)


def _check_int(name: str, value, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _rank_tolerance(S: np.ndarray, shape: tuple[int, int]) -> float:
    """Singular values at or below this are numerically zero.

    This is the rule of ``np.linalg.matrix_rank`` with the float32 machine epsilon: inputs are
    converted to float32 on entry, so directions whose singular value is at the rounding-noise
    level of the data are not real variance and must not be whitened.
    """
    return (S.max() if S.size else 0.0) * max(shape) * np.finfo(np.float32).eps


def _svd_sign_fix(Vt: np.ndarray) -> np.ndarray:
    """Make the largest-magnitude loading of every component positive (deterministic signs)."""
    idx = np.abs(Vt).argmax(axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), idx])
    signs[signs == 0] = 1.0
    return Vt * signs[:, None]


def _pca_fit(Xc: np.ndarray):
    """Principal axes of already centred data.

    Returns ``(components, explained_variance, rank)`` with all ``min(n, d)`` components in
    descending order; ``components`` has deterministic signs; only the first ``rank`` rows are
    meaningful (the rest span the null space).
    """
    n = Xc.shape[0]
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = int((S > _rank_tolerance(S, Xc.shape)).sum())
    return _svd_sign_fix(Vt), S ** 2 / max(n - 1, 1), rank


def _capped(name: str, requested: int, rank: int, what: str) -> int:
    if requested > rank:
        warnings.warn(
            f"{name}={requested} exceeds the rank of the centred training data ({rank}); "
            f"using {what}={rank}.", UserWarning, stacklevel=3)
        return rank
    return requested


def _remove_directions(C: np.ndarray, U: np.ndarray) -> np.ndarray:
    return C - (C @ U.T) @ U


# ---------------------------------------------------------------------------------------------
# transforms
# ---------------------------------------------------------------------------------------------


class Identity(_Baseline):
    """Return the input unchanged (as float32). The "raw embedding" row of a baseline table."""

    def fit(self, X, y=None):
        self._fit_input(X)
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        return _out(self._transform_input(X))


class CenterNormalize(_Baseline):
    """Subtract the training mean, then scale every row to unit L2 norm.

    Rows whose norm is below ``1e-12`` after centring (for example a vector equal to the training
    mean) are left as zero vectors instead of being divided by zero.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
    """

    def fit(self, X, y=None):
        X = self._fit_input(X)
        self.mean_ = X.mean(axis=0)
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        C = self._transform_input(X) - self.mean_
        norms = np.linalg.norm(C, axis=1, keepdims=True)
        return _out(C / np.where(norms > 1e-12, norms, 1.0))


class Standardize(_Baseline):
    """Per-column z-score with the training mean and (population) standard deviation.

    Columns with zero variance in the training data are centred but not scaled (``scale_`` is 1
    for them), so they map to 0 on the training data and never produce NaN or infinity.

    Attributes
    ----------
    mean_, scale_ : ndarray (n_features,)
    """

    def fit(self, X, y=None):
        X = self._fit_input(X)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0)
        constant = scale <= 1e-12 * np.abs(self.mean_)
        scale[constant | (scale == 0)] = 1.0
        self.scale_ = scale
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        return _out((self._transform_input(X) - self.mean_) / self.scale_)


class PCAReduce(_Baseline):
    """Project onto the top principal components of the training data, optionally whitened.

    ``n_components`` is capped at the rank of the centred training data (a ``UserWarning`` is
    issued) so that whitening never divides by a zero eigenvalue; the value used is
    ``n_components_``. With ``whiten=True`` each output column is divided by the square root of
    its eigenvalue, so the training output has identity covariance.

    The signs of the components are fixed by making the largest-magnitude loading of each
    component positive, so results do not depend on the SVD implementation or the row order.
    The decomposition is an exact SVD, so ``random_state`` has no effect; it is kept so that the
    signature matches the other randomised methods and the benchmark configuration.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
    components_ : ndarray (n_components_, n_features)
    explained_variance_ : ndarray (n_components_,)   eigenvalues of the covariance (``n - 1``)
    n_components_ : int
    rank_ : int   numerical rank of the centred training data
    """

    def __init__(self, n_components: int, whiten: bool = False, random_state: int | None = 0):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, y=None):
        k = _check_int("n_components", self.n_components, 1)
        X = self._fit_input(X, min_samples=2)
        self.mean_ = X.mean(axis=0)
        comps, ev, rank = _pca_fit(X - self.mean_)
        if rank == 0:
            raise ValueError("The training data has no variance (all rows are identical)")
        self.rank_ = rank
        self.n_components_ = _capped("n_components", k, rank, "n_components")
        self.components_ = comps[: self.n_components_]
        self.explained_variance_ = ev[: self.n_components_]
        self.n_features_out_ = self.n_components_
        return self

    def transform(self, X) -> np.ndarray:
        Z = (self._transform_input(X) - self.mean_) @ self.components_.T
        if self.whiten:
            Z = Z / np.sqrt(self.explained_variance_)
        return _out(Z)


class ZCAWhiten(_Baseline):
    """ZCA (Mahalanobis) whitening: ``x -> (x - mean) W`` with ``W = U diag(1/sqrt(λ + eps)) U^T``.

    ``U, λ`` are the eigenvectors and eigenvalues of the training covariance (``n - 1``). With
    small ``eps`` and well-conditioned data the training output has identity covariance to
    about ``eps / λ_min``.

    Rank-deficient data: eigenvalues are clipped at 0 before adding ``eps``, so ``W`` is always
    finite. Directions with no variance in the training data are multiplied by ``1/sqrt(eps)``
    (about 316 for the default); the training data has no component there, so its output is
    unaffected, but new points with a component in those directions are strongly amplified.
    Increase ``eps`` or use :class:`PCAReduce` (which caps at the rank) if that matters.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
    whitening_ : ndarray (n_features, n_features)   the symmetric matrix ``W``
    eigenvalues_ : ndarray (n_features,)   training covariance eigenvalues, ascending
    """

    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def fit(self, X, y=None):
        if not self.eps >= 0:
            raise ValueError(f"eps must be >= 0, got {self.eps!r}")
        X = self._fit_input(X, min_samples=2)
        self.mean_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False).reshape(X.shape[1], X.shape[1])
        lam, U = np.linalg.eigh(cov)
        lam = np.maximum(lam, 0.0)
        denom = np.sqrt(lam + self.eps)
        if (denom == 0).any():
            raise ValueError("eps=0 with a rank-deficient covariance; use eps > 0")
        self.eigenvalues_ = lam
        self.whitening_ = (U / denom) @ U.T
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        return _out((self._transform_input(X) - self.mean_) @ self.whitening_)


class AllButTheTop(_Baseline):
    """All-but-the-top post-processing of Mu and Viswanath (2018).

    Subtract the training mean and remove the projections on the top ``n_remove`` principal
    directions of the training data (the "common" directions that dominate the variance):
    ``x -> c - sum_i (u_i . c) u_i`` with ``c = x - mean``. The output keeps the dimension.

    With ``n_components`` set this is the reduce-then-clean variant of Raunak, Gupta and Metze
    (2019), Algorithm 2: (1) post-process with ``n_remove``, (2) PCA to ``n_components``,
    (3) post-process the reduced vectors again with the same ``n_remove``. Each stage fits its own
    mean and directions on the training data.

    ``n_remove`` (and ``n_components``) larger than the rank of the data the stage sees are capped
    at that rank with a ``UserWarning``. ``n_remove = 0`` only centres.

    References
    ----------
    Mu, Bhat, Viswanath. All-but-the-Top: Simple and Effective Postprocessing for Word
    Representations. ICLR 2018 (arXiv:1702.01417).

    Raunak, Gupta, Metze. Effective Dimensionality Reduction for Word Embeddings. RepL4NLP
    (ACL workshop) 2019. Algorithm 2 is "PPA, PCA, PPA" with ``D = 7`` and ``N = d/2``.

    Attributes
    ----------
    mean_ : ndarray (n_features,)
    components_ : ndarray (n_remove_, n_features)   removed directions of the first stage
    n_remove_ : int
    pca_mean_, pca_components_ : only with ``n_components`` (the PCA stage; ``pca_mean_`` is ~0
        because the data is already centred)
    post_mean_, post_components_ : only with ``n_components`` (the second post-processing stage)
    n_components_ : int, only with ``n_components``
    """

    def __init__(self, n_remove: int, n_components: int | None = None):
        self.n_remove = n_remove
        self.n_components = n_components

    @staticmethod
    def _fit_ppa(X: np.ndarray, n_remove: int):
        mean = X.mean(axis=0)
        comps, _, rank = _pca_fit(X - mean)
        n = _capped("n_remove", n_remove, rank, "n_remove")
        return mean, comps[:n]

    def fit(self, X, y=None):
        r = _check_int("n_remove", self.n_remove, 0)
        k = None if self.n_components is None else _check_int("n_components", self.n_components, 1)
        X = self._fit_input(X, min_samples=2)

        self.mean_, self.components_ = self._fit_ppa(X, r)
        self.n_remove_ = self.components_.shape[0]
        self.n_features_out_ = self.n_features_in_
        if k is None:
            return self

        S1 = _remove_directions(X - self.mean_, self.components_)
        self.pca_mean_ = S1.mean(axis=0)
        comps, _, rank = _pca_fit(S1 - self.pca_mean_)
        if rank == 0:
            raise ValueError("No variance is left after removing the top directions")
        self.n_components_ = _capped("n_components", k, rank, "n_components")
        self.pca_components_ = comps[: self.n_components_]
        S2 = (S1 - self.pca_mean_) @ self.pca_components_.T
        self.post_mean_, self.post_components_ = self._fit_ppa(S2, r)
        self.n_features_out_ = self.n_components_
        return self

    def transform(self, X) -> np.ndarray:
        S1 = _remove_directions(self._transform_input(X) - self.mean_, self.components_)
        if self.n_components is None:
            return _out(S1)
        S2 = (S1 - self.pca_mean_) @ self.pca_components_.T
        return _out(_remove_directions(S2 - self.post_mean_, self.post_components_))


class ConceptorNegation(_Baseline):
    """Conceptor negation post-processing of Liu, Ungar and Sedoc (2019).

    From the training data compute the correlation matrix ``R = X^T X / n``, the conceptor
    ``C = R (R + alpha^-2 I)^-1`` and map ``x -> (I - C) x``. This soft-suppresses the
    high-variance directions: along an eigen-direction of ``R`` with eigenvalue ``λ`` the output is
    scaled by ``alpha^-2 / (λ + alpha^-2)``.

    Following Algorithm 2 of the paper, the vectors are **not centred**: ``R`` is the second
    moment of the raw vectors and ``transform`` is applied to the raw vectors. (The paper remarks
    that for zero-mean vectors ``R`` is the covariance, and that this makes the method a soft
    version of all-but-the-top; the mean direction itself is strongly suppressed because it
    dominates ``R``.) ``alpha = 2`` is the value the paper fixes for Word2Vec and GloVe vectors;
    it is not scale-free, so it may need tuning for embeddings with a very different norm.

    Reference
    ---------
    Liu, Ungar, Sedoc. Unsupervised Post-Processing of Word Vectors via Conceptor Negation.
    AAAI 2019 (arXiv:1811.11001).

    Attributes
    ----------
    conceptor_ : ndarray (n_features, n_features)   ``C``
    negation_ : ndarray (n_features, n_features)    ``I - C``
    """

    def __init__(self, alpha: float = 2.0):
        self.alpha = alpha

    def fit(self, X, y=None):
        if not self.alpha > 0:
            raise ValueError(f"alpha must be > 0, got {self.alpha!r}")
        X = self._fit_input(X)
        R = X.T @ X / X.shape[0]
        lam, U = np.linalg.eigh(R)
        lam = np.maximum(lam, 0.0)
        ridge = float(self.alpha) ** -2
        self.conceptor_ = (U * (lam / (lam + ridge))) @ U.T
        self.negation_ = (U * (ridge / (lam + ridge))) @ U.T
        self.n_features_out_ = self.n_features_in_
        return self

    def transform(self, X) -> np.ndarray:
        return _out(self._transform_input(X) @ self.negation_)  # negation_ is symmetric


class RandomProjection(_Baseline):
    """Gaussian random projection with Johnson-Lindenstrauss scaling.

    ``x -> x P^T`` with ``P`` of shape ``(n_components, n_features)`` and i.i.d. ``N(0,
    1/n_components)`` entries, so squared distances are preserved in expectation. No centring.
    The matrix depends only on ``random_state`` and the number of columns, not on the values in
    ``X``; ``n_components`` may exceed the input dimension. ``random_state=None`` draws a fresh
    matrix at each ``fit``.

    Attributes
    ----------
    components_ : ndarray (n_components, n_features)
    """

    def __init__(self, n_components: int, random_state: int | None = 0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = _check_int("n_components", self.n_components, 1)
        self._fit_input(X)
        rng = np.random.default_rng(self.random_state)
        self.components_ = rng.standard_normal((k, self.n_features_in_)) / np.sqrt(k)
        self.n_features_out_ = k
        return self

    def transform(self, X) -> np.ndarray:
        return _out(self._transform_input(X) @ self.components_.T)
