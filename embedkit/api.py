"""EmbedKit: high-level API orchestrating analyze → configure → train → transform."""

from __future__ import annotations

import json
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from embedkit.analysis.report import EmbedKitAnalyzer, EmbedKitReport
from embedkit.baselines import CenterNormalize, Identity
from embedkit.improvement.augmentation import EmbeddingMixup, GaussianNoise, KNNPairs
from embedkit.improvement.losses import (
    AlignUniformLoss,
    CombinedLoss,
    MatryoshkaLoss,
    NTXentLoss,
    RankNContrastLoss,
    SupConLoss,
)
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
from embedkit.utils.io import load_bundle, save_bundle
from embedkit.utils.validation import _to_numpy, trusted_input, validate_embeddings


class EmbedKit:
    """Analyse, configure, train and transform in one estimator.

    **Strategy (WP 7.2).** ``strategy="select"`` (the default) picks the output by validation,
    and ``strategy="fixed"`` builds the configuration given by ``method`` and ``target_dim``.

    ``"select"`` without labels returns the input centred on the training mean and scaled to unit
    L2 norm (:class:`~embedkit.baselines.CenterNormalize`). No label-free score can tell whether a
    transform helps (``knn_overlap``, which ranked refiner runs in E2, gives the unchanged input
    the top score), and in E4 no training-free fix beat the raw embedding on most of 35 natural
    sets; centring and L2 normalisation was the most consistent small gain on Euclidean kNN (16
    sets up, 1 down) and changes nothing for a cosine search beyond removing the mean
    (``benchmarks/results/e4/REPORT.md``). PCA to the report's ``suggested_target_dim``, the
    0.5.0 default, lost accuracy on 32 of the 35 sets.

    ``"select"`` with labels holds out ``val_fraction`` of the rows (stratified for classes),
    fits every candidate on the rest and scores it on the held-out rows: ``k``-NN accuracy for
    classes, ``k``-NN regression R^2 for continuous targets (k = 10, ``selection_metric``,
    cosine by default; ``knn_overlap`` is not used, see above). The candidates are ``Identity``
    (the input as given), ``CenterNormalize``, PCA at 64, 128 and 256 (those below the input
    dimension; the ``target_dim`` given instead, if any) followed by L2 normalisation, and, in
    supervised and regression mode, the refiner of that mode at ``target_dim`` (``"auto"``:
    ``min(128, d // 2)``, as in E8). A candidate replaces ``Identity`` only when it beats it on
    the held-out rows with a paired bootstrap interval (95%, 2,000 resamples of the held-out
    rows) above 0; among those the best score wins. Otherwise the result is ``Identity``, so the
    default never returns something worse than the input by its own criterion. The winner is
    refitted on all the rows. ``selected_`` names it and ``selection_report_`` holds the table.
    Most gains in E4 were below one point, so the interval keeps noise from picking a transform.

    **Large inputs.** With more than ``selection_max_rows`` rows (50,000 by default), the
    candidates are fitted and scored on a random subsample of that many rows (stratified by
    class), split as above; only the winner is fitted on all the rows. In E9, selection on all
    the rows took 1.8 to 9.7 hours at a million rows on 16 CPUs, up to 11 times the input in
    memory, mostly for fitting candidates that lost. The refit costs nothing more for
    ``Identity`` and little for ``CenterNormalize`` and PCA; a refiner that wins is trained on
    all the rows. ``selection_report_.attrs`` records ``n_rows`` and ``n_selection_rows``.
    ``None`` selects on all the rows (the 0.6.0 behaviour).
    Under ``"select"`` no analysis report is computed (``analysis_report`` is ``None``): no
    decision reads it. Run :class:`~embedkit.analysis.report.EmbedKitAnalyzer` for one.

    ``mode="auto"`` (the default) is ``"self_supervised"`` without labels and ``"supervised"``
    with them; continuous targets need ``mode="regression"``. With ``mode="self_supervised"``
    and labels, ``"select"`` uses the labels to choose among the training-free candidates only:
    in E1 the self-supervised refiner never beat PCA and cost about 100 times more.

    **Fixed configurations.** ``strategy="fixed"``, or a ``method`` other than ``"auto"`` (which
    implies it; ``strategy_`` records what ran), builds ``method``: ``"refiner"`` trains an
    :class:`~embedkit.improvement.model.EmbeddingRefiner` with a contrastive loss. ``"pca"``
    trains nothing: it is the ``linear`` refiner at its initialisation, the top ``target_dim``
    principal directions followed by L2 normalisation, with the same ``transform``, ``dim=``,
    ``save`` and ``load``. ``"auto"`` is ``"pca"`` for ``mode="self_supervised"`` and
    ``"refiner"`` otherwise; ``target_dim="auto"`` is the report's ``suggested_target_dim``. In
    E1, E6 and E7 (``docs/BENCHMARK_E1_E2.md``, ``docs/BENCHMARK_E6_E7.md``) the self-supervised
    refiner never beat the better of PCA and PCA-whitening, lost to them when fitted on fewer
    than a few thousand rows, and cost about 100 times more. The training parameters
    (``augmentation``, ``loss``, ``architecture``, ``nested_dims``, ``epochs`` and the
    optimiser settings) apply to the refiner only; setting ``augmentation``, ``loss``,
    ``nested_dims`` or a non-linear ``architecture`` when no refiner is fitted or compared
    raises, so that a training configuration is never silently ignored. PCA prefixes are
    nested by construction, so ``transform(X, dim=m)`` is PCA at ``m`` (renormalised) without
    ``nested_dims``.

    ``nested_dims`` (WP 5.3) trains every prefix of the output as an embedding of its own, by
    wrapping the loss in :class:`~embedkit.improvement.losses.MatryoshkaLoss`; read a prefix
    with ``transform(X, dim=m)``. ``"auto"`` nests the powers of two from 32 below the output
    dimension, as in E6, where prefixes of one model at 256 matched linear models trained at
    32, 64 and 128 (``benchmarks/results/e6/REPORT.md``). A sequence gives the dimensions
    directly. ``None`` (the default) trains the full output only.

    Modes of the refiner (WP 6.1). ``"supervised"`` (class labels): ``SupConLoss`` alone on the
    ``"residual"`` architecture with ``KNNPairs(5)`` views, ``val_knn_accuracy`` as the monitor.
    ``"regression"`` (continuous targets, cast to float): ``RankNContrastLoss`` on the ``"mlp"``
    architecture with ``GaussianNoise(0.05)`` views, ``val_ridge_r2`` (a ridge probe) as the
    monitor; a fixed refiner warns with 1000 rows or fewer, where in E8 no refiner reliably beat
    the raw embeddings. ``"self_supervised"``: the ``"linear"`` architecture with ``KNNPairs(5)``
    views and NT-Xent plus 0.5 alignment-uniformity, monitored by ``k_skewness``. These are the
    ``"auto"`` values of ``architecture`` and ``augmentation``; ``docs/BENCHMARK_SUPERVISED.md``
    has the evidence (E8).
    ``monitor="auto"`` and ``early_stopping_patience="auto"`` resolve per mode: with labels, 10%
    of the rows are held out, training stops after 10 evaluations without improvement of the
    monitor and the best weights are restored; without labels there is no early stopping (E2
    found no label-free signal for it). In supervised mode ``history`` also records NC1 on the
    training subsample and the validation rows (``train_nc1``, ``val_nc1``).
    """

    def __init__(
        self,
        mode: Literal["auto", "self_supervised", "supervised", "regression"] = "auto",
        strategy: Literal["select", "fixed"] = "select",
        method: Literal["auto", "pca", "refiner"] = "auto",
        augmentation: Any = "auto",
        loss: Any = "auto",
        target_dim: int | Literal["auto"] = "auto",
        hidden_dim: int = 256,
        n_layers: int = 2,
        architecture: Literal["auto", "mlp", "residual", "linear"] = "auto",
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: str = "adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler: str | None = "cosine",
        eval_every: int = 5,
        early_stopping_patience: int | Literal["auto"] | None = "auto",
        monitor: str = "auto",
        val_split: float | None = None,
        device: str | torch.device | None = None,
        id_methods: list[str] | None = None,
        random_state: int | None = 42,
        nested_dims: Sequence[int] | Literal["auto"] | None = None,
        val_fraction: float = 0.2,
        selection_metric: Literal["cosine", "euclidean"] = "cosine",
        selection_max_rows: int | None = 50_000,
    ):
        self.mode = mode
        self.strategy = strategy
        self.method = method
        self.augmentation_cfg = augmentation
        self.loss_cfg = loss
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.architecture = architecture
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.eval_every = eval_every
        self.early_stopping_patience = early_stopping_patience
        self.monitor = monitor
        self.val_split = val_split
        self.device = device
        self.id_methods = id_methods
        self.random_state = random_state
        self.nested_dims = nested_dims
        self.val_fraction = val_fraction
        self.selection_metric = selection_metric
        self.selection_max_rows = selection_max_rows

        self.analysis_report: EmbedKitReport | None = None
        self._trainer: Trainer | None = None
        self._baseline = None  # a fitted Identity or CenterNormalize when one is the output
        self._config: dict = {}

    def fit(self, X, y=None) -> EmbedKit:
        if self.mode not in ("auto", "self_supervised", "supervised", "regression"):
            raise ValueError(f"mode must be 'auto', 'self_supervised', 'supervised' or "
                             f"'regression', got {self.mode!r}")
        if self.strategy not in ("select", "fixed"):
            raise ValueError(f"strategy must be 'select' or 'fixed', got {self.strategy!r}")
        # The one validation of the call (NaN or infinite values raise; constant columns and
        # duplicate rows warn); the analyzer, the candidates and the trainer below stay silent.
        X_np = validate_embeddings(X, min_samples=5)
        with trusted_input():
            return self._fit(X_np, y)

    def _fit(self, X_np: np.ndarray, y) -> EmbedKit:
        self.mode_ = self._resolve_mode(y)
        if self.mode_ != "self_supervised" and y is None:
            raise ValueError(f"mode={self.mode!r} needs labels: call fit(X, y)")
        if y is not None:
            y = _labels_array(y, self.mode_)
            if len(y) != len(X_np):
                raise ValueError(f"y has {len(y)} entries for {len(X_np)} rows")
        self._trainer, self._baseline = None, None
        self.selected_, self.selection_report_ = None, None
        # An explicit method is a fixed configuration: there is nothing to select.
        self.strategy_ = "fixed" if self.method != "auto" else self.strategy
        if self.strategy_ == "fixed":
            return self._fit_fixed(X_np, y)
        self.analysis_report = None
        if y is None:
            self._check_no_training_config("strategy='select' without labels fits no refiner")
            return self._fit_baseline(X_np, "center_l2")
        return self._fit_select(X_np, y)

    # ------------------------------------------------------------------------------------
    # strategy="select"
    # ------------------------------------------------------------------------------------
    def _fit_select(self, X_np: np.ndarray, y: np.ndarray) -> EmbedKit:
        from embedkit.evaluation.metrics import knn_accuracy, knn_regression_r2

        n, d = X_np.shape
        classes = self.mode_ != "regression"
        refiner = self.mode_ in ("supervised", "regression")
        if not refiner:
            self._check_no_training_config(
                "mode='self_supervised' with strategy='select' compares training-free "
                "candidates only")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in (0, 1), got {self.val_fraction!r}")
        cap = self.selection_max_rows
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, (int, np.integer))
                                or cap < 2):
            raise ValueError(f"selection_max_rows must be an integer of at least 2 or None, "
                             f"got {cap!r}")
        if cap is not None and n > cap:
            sub = _selection_subsample(y, int(cap), classes, self.random_state)
            tr, va = _selection_split(y[sub], self.val_fraction, classes, self.random_state)
            tr, va = sub[tr], sub[va]
        else:
            tr, va = _selection_split(y, self.val_fraction, classes, self.random_state)
        if len(va) < _MIN_VAL_ROWS or len(tr) < _MIN_VAL_ROWS:
            warnings.warn(
                f"Too few rows to select a transform ({len(tr)} to fit, {len(va)} to score, "
                f"at least {_MIN_VAL_ROWS} each needed); returning the input unchanged "
                "(Identity). Pass strategy='fixed' to build a configuration anyway.",
                UserWarning, stacklevel=3)
            return self._fit_baseline(X_np, "identity")
        Xtr, Xva, ytr, yva = X_np[tr], X_np[va], y[tr], y[va]

        candidates: dict[str, tuple[int, np.ndarray, np.ndarray]] = {}
        candidates["identity"] = (d, Xtr, Xva)
        cn = CenterNormalize().fit(Xtr)
        candidates["center_l2"] = (d, cn.transform(Xtr), cn.transform(Xva))
        mean, comps, rank = _pca(Xtr)
        for k in self._pca_dims(d, rank):
            P = comps[:k].T
            candidates[f"pca-{k}"] = (k, _l2((Xtr - mean) @ P), _l2((Xva - mean) @ P))
        if refiner:
            t = self._refiner_dim(d)
            cand = self._fixed_refiner(t)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Regression refiner fitted on")
                cand.fit(Xtr, ytr)
            candidates[f"refiner-{t}"] = (t, _np(cand.transform(Xtr)), _np(cand.transform(Xva)))

        def score(Ztr, Zva):
            if classes:
                return knn_accuracy(Ztr, ytr, Zva, yva, k=_SELECTION_K,
                                    metric=self.selection_metric, return_per_point=True)
            return knn_regression_r2(Ztr, ytr, Zva, yva, k=_SELECTION_K,
                                     metric=self.selection_metric, return_per_point=True)

        scores = {name: score(Ztr, Zva) for name, (_, Ztr, Zva) in candidates.items()}
        base = scores["identity"][1]
        rng = np.random.default_rng(self.random_state)
        boot_idx = rng.integers(0, len(va), size=(_N_BOOT, len(va)))
        rows = []
        for name, (dim, _, _) in candidates.items():
            value, per_point = scores[name]
            diff = per_point - base
            if name == "identity":
                lo = hi = 0.0
            else:
                boot = diff[boot_idx].mean(axis=1)
                lo, hi = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
            rows.append(dict(candidate=name, dim=dim, score=float(value),
                             diff_vs_identity=float(diff.mean()), diff_lo=lo, diff_hi=hi,
                             beats_identity=bool(name != "identity" and lo > 0)))
        table = pd.DataFrame(rows)
        winners = table[table["beats_identity"]]
        winner = ("identity" if winners.empty
                  else str(winners.sort_values("score", ascending=False, kind="stable")
                           ["candidate"].iloc[0]))
        table["selected"] = table["candidate"] == winner
        table.attrs.update(criterion=("knn_accuracy" if classes else "knn_regression_r2"),
                           k=_SELECTION_K, metric=self.selection_metric, n_fit=len(tr),
                           n_val=len(va), n_boot=_N_BOOT, n_rows=n,
                           n_selection_rows=len(tr) + len(va))

        # Refit the winner on all the rows.
        if winner in ("identity", "center_l2"):
            self._fit_baseline(X_np, winner)
        elif winner.startswith("pca-"):
            self._fit_pca_model(X_np, int(winner.split("-")[1]))
            self.method_ = "pca"
        else:
            final = self._fixed_refiner(int(winner.split("-")[1]))
            with warnings.catch_warnings():  # the refiner has just beaten Identity here
                warnings.filterwarnings("ignore", message="Regression refiner fitted on")
                final.fit(X_np, y)
            self._adopt(final)
        self.selected_, self.selection_report_ = winner, table
        self._config.update(strategy="select", mode=self.mode_, selected=winner,
                            selection_report=json.loads(table.to_json(orient="records")),
                            selection_attrs=dict(table.attrs))
        return self

    def _pca_dims(self, d: int, rank: int) -> list[int]:
        dims = _PCA_DIMS if self.target_dim == "auto" else (int(self.target_dim),)
        return [k for k in dims if k < d and k <= rank]

    def _refiner_dim(self, d: int) -> int:
        return min(128, max(1, d // 2)) if self.target_dim == "auto" else int(self.target_dim)

    def _fixed_refiner(self, target_dim: int) -> EmbedKit:
        """The refiner of this mode as a fixed configuration, with this object's settings."""
        params = {name: getattr(self, attr) for name, attr in _PARAMS.items()}
        params.update(mode=self.mode_, strategy="fixed", method="refiner",
                      target_dim=target_dim)
        return EmbedKit(**params)

    def _adopt(self, other: EmbedKit) -> None:
        for attr in ("_trainer", "_config", "method_", "architecture_", "nested_dims_",
                     "monitor_", "early_stopping_patience_", "analysis_report"):
            if hasattr(other, attr):
                setattr(self, attr, getattr(other, attr))
        self._baseline = None

    def _fit_baseline(self, X_np: np.ndarray, name: str) -> EmbedKit:
        self._baseline = _fitted_baseline(name, X_np)
        self._trainer = None
        self.method_, self.selected_ = name, name
        self.nested_dims_ = None
        self._config = {"mode": self.mode_, "strategy": self.strategy_, "method": name,
                        "input_dim": X_np.shape[1], "selected": name}
        return self

    def _check_no_training_config(self, why: str) -> None:
        given = [name for name, value, default in (
            ("augmentation", self.augmentation_cfg, "auto"), ("loss", self.loss_cfg, "auto"),
            ("nested_dims", self.nested_dims, None),
        ) if not (value is default or value == default)]
        if self.architecture not in ("auto", "linear"):
            given.append("architecture")
        if given:
            raise ValueError(f"{', '.join(given)} configure training, but {why}; pass "
                             "strategy='fixed', method='refiner' to train")

    def _resolve_mode(self, y) -> str:
        if self.mode != "auto":
            return self.mode
        return "self_supervised" if y is None else "supervised"

    # ------------------------------------------------------------------------------------
    # strategy="fixed"
    # ------------------------------------------------------------------------------------
    def _fit_fixed(self, X_np: np.ndarray, y) -> EmbedKit:
        n, d = X_np.shape
        if self.mode_ == "regression" and n <= _SMALL_REGRESSION_ROWS \
                and self._resolve_method() == "refiner":
            warnings.warn(
                f"Regression refiner fitted on {n} rows: with {_SMALL_REGRESSION_ROWS} "
                "training rows or fewer, no refiner reliably beat the raw embeddings in E8 "
                "(docs/BENCHMARK_SUPERVISED.md); compare with the raw embeddings or PCA.",
                UserWarning, stacklevel=4)

        # Step 1: analyze
        # Only suggested_target_dim is read here, so the null calibration, the ID bootstrap and
        # the ID scale curve are skipped.
        analyzer = EmbedKitAnalyzer(
            id_methods=self.id_methods,
            random_state=self.random_state,
            calibrate=False,
            id_bootstrap=0,
            id_scale_curve=False,
        )
        self.analysis_report = analyzer.fit(X_np, y)
        report = self.analysis_report

        # Step 2: auto-configure
        target_dim = self._resolve_target_dim(d, report)
        method = self._resolve_method()
        self.method_ = method
        if method == "pca":
            return self._fit_pca(X_np, target_dim)
        augmentation = self._resolve_augmentation(report)
        loss_fn = self._resolve_loss(report)
        nested = self._resolve_nested_dims(target_dim)
        architecture = self._resolve_architecture()
        self.architecture_ = architecture
        if nested is not None:
            if isinstance(loss_fn, MatryoshkaLoss):
                raise ValueError("Pass nested_dims or a MatryoshkaLoss, not both")
            loss_fn = MatryoshkaLoss(loss_fn, dims=nested)
        self.nested_dims_ = loss_fn.dims if isinstance(loss_fn, MatryoshkaLoss) else None

        monitor = self._resolve_monitor()
        patience = self._resolve_patience()
        eval_metrics = [monitor]
        if self.mode_ == "supervised":
            eval_metrics += ["train_nc1"] + (["val_nc1"] if self._resolve_val_split() > 0 else [])
        self.monitor_, self.early_stopping_patience_ = monitor, patience

        self._config = {
            "mode": self.mode_,
            "strategy": "fixed",
            "method": "refiner",
            "target_dim": target_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "architecture": architecture,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "input_dim": d,
            "nested_dims": list(self.nested_dims_) if self.nested_dims_ else None,
        }

        # Step 3: build model + trainer
        # Draw the initial weights from random_state without touching the caller's RNG state.
        with torch.random.fork_rng(devices=[], enabled=self.random_state is not None):
            if self.random_state is not None:
                torch.default_generator.manual_seed(self.random_state)
            model = EmbeddingRefiner(
                input_dim=d,
                target_dim=target_dim,
                hidden_dim=self.hidden_dim,
                n_layers=self.n_layers,
                architecture=architecture,
            )
        self._trainer = Trainer(
            model=model,
            augmentation=augmentation,
            loss=loss_fn,
            epochs=self.epochs,
            batch_size=self.batch_size,
            optimizer=self.optimizer,
            lr=self.lr,
            weight_decay=self.weight_decay,
            scheduler=self.scheduler,
            eval_every=self.eval_every,
            eval_metrics=list(dict.fromkeys(eval_metrics)),
            early_stopping_patience=patience,
            monitor=monitor,
            val_split=self._resolve_val_split(),
            device=self.device,
            random_state=self.random_state,
        )
        self._trainer.fit(X_np, y)
        return self

    def _resolve_method(self) -> str:
        if self.method not in ("auto", "pca", "refiner"):
            raise ValueError(f"method must be 'auto', 'pca' or 'refiner', got {self.method!r}")
        if self.method != "auto":
            return self.method
        return "pca" if self.mode_ == "self_supervised" else "refiner"

    def _fit_pca(self, X_np: np.ndarray, target_dim: int) -> EmbedKit:
        """``method="pca"``: the linear refiner at its PCA initialisation, not trained."""
        self._check_no_training_config(
            "method='pca' does not train (method='auto' is 'pca' in self-supervised mode)")
        return self._fit_pca_model(X_np, target_dim)

    def _fit_pca_model(self, X_np: np.ndarray, target_dim: int) -> EmbedKit:
        model = EmbeddingRefiner(input_dim=X_np.shape[1], target_dim=target_dim,
                                 hidden_dim=self.hidden_dim, n_layers=self.n_layers,
                                 architecture="linear")
        model.initialize_from_data(X_np)
        self.nested_dims_ = None
        self.architecture_ = "linear"
        self._config = {
            "mode": self.mode_,
            "strategy": self.strategy_,
            "method": "pca",
            "target_dim": target_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "architecture": "linear",
            "input_dim": X_np.shape[1],
        }
        self._trainer = _inference_trainer(model, self.device)
        self._baseline = None
        return self

    def _resolve_architecture(self) -> str:
        # E8 (docs/BENCHMARK_SUPERVISED.md): with class labels the residual adapter matched the
        # MLP on full training sets and, unlike it, did not collapse with 1-5 labels per class;
        # for regression the MLP won on every dataset.
        if self.architecture != "auto":
            return self.architecture
        return {"supervised": "residual", "regression": "mlp"}.get(self.mode_, "linear")

    def _resolve_monitor(self) -> str:
        if self.monitor != "auto":
            return self.monitor
        if self._resolve_val_split() <= 0:  # nothing held out: a label-free statistic, logged
            return "k_skewness"
        return {"supervised": "val_knn_accuracy", "regression": "val_ridge_r2"}.get(
            self.mode_, "k_skewness")

    def _resolve_patience(self) -> int | None:
        # Without labels early stopping is off: in E2 (docs/BENCHMARK_E1_E2.md) no label-free
        # monitor tracked test accuracy within a run. With labels the monitor is a held-out score.
        if self.early_stopping_patience != "auto":
            return self.early_stopping_patience
        return None if self.mode_ == "self_supervised" else 10

    def _resolve_val_split(self) -> float:
        # Hold rows out only when a monitor decides something.
        if self.val_split is not None:
            return self.val_split
        return 0.1 if self._resolve_patience() is not None else 0.0

    def transform(self, X, dim: int | None = None) -> np.ndarray | torch.Tensor:
        """Refine ``X``. With ``dim``, keep the first ``dim`` output coordinates and renormalise.

        Truncation is meant for models trained with
        :class:`~embedkit.improvement.losses.MatryoshkaLoss`, whose prefixes are trained to be
        embeddings on their own (WP 5.3); for other models a prefix is just a projection.
        """
        if self._baseline is not None:
            Z = self._baseline.transform(X)
            if isinstance(X, torch.Tensor):
                Z = torch.from_numpy(Z)
            normalized = isinstance(self._baseline, CenterNormalize)
        elif self._trainer is not None:
            Z = self._trainer.transform(X)
            normalized = getattr(self._trainer.model, "normalize_output", True)
        else:
            raise RuntimeError("Call fit() before transform()")
        if dim is None:
            return Z
        d = Z.shape[1]
        if not 1 <= dim <= d:
            raise ValueError(f"dim must be between 1 and the output dimension {d}, got {dim}")
        Z = Z[:, :dim]
        if normalized:
            if isinstance(Z, torch.Tensor):
                Z = torch.nn.functional.normalize(Z, dim=1)
            else:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z

    def fit_transform(self, X, y=None, dim: int | None = None) -> np.ndarray | torch.Tensor:
        self.fit(X, y)
        return self.transform(X, dim=dim)

    def save(self, directory: str | Path) -> None:
        if self._baseline is not None:
            arrays = ({"mean": self._baseline.mean_} if isinstance(self._baseline, CenterNormalize)
                      else {})
            save_bundle(directory, None, self._config, self.analysis_report, arrays=arrays)
            return
        if self._trainer is None:
            raise RuntimeError("Nothing fitted yet")
        save_bundle(directory, self._trainer.model, self._config, self.analysis_report)

    @classmethod
    def load(cls, directory: str | Path, device: str | torch.device | None = None) -> EmbedKit:
        """Load a bundle written by :meth:`save`, for ``transform``.

        ``device`` is where ``transform`` runs; ``None`` picks CUDA, then Apple MPS, then CPU.
        Pass the device the model was fitted on to reproduce its outputs to float precision:
        different backends round differently (about 1e-4 between CPU and MPS).
        """
        bundle = load_bundle(directory)
        cfg = bundle["config"]
        if cfg.get("method") in ("identity", "center_l2"):
            obj = cls(mode=cfg.get("mode", "self_supervised"),
                      strategy=cfg.get("strategy", "select"))
            base = Identity() if cfg["method"] == "identity" else CenterNormalize()
            base.n_features_in_ = base.n_features_out_ = cfg["input_dim"]
            if cfg["method"] == "center_l2":
                base.mean_ = np.asarray(bundle["arrays"]["mean"], dtype=np.float64)
            obj._baseline, obj.device = base, device
            obj.method_ = cfg["method"]
            _restore_selection(obj, cfg)
            return obj
        obj = cls(
            mode=cfg.get("mode", "self_supervised"),
            strategy="fixed",
            method=cfg.get("method", "refiner"),
            target_dim=cfg.get("target_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 2),
            architecture=cfg.get("architecture", "mlp"),
        )
        model = EmbeddingRefiner(
            input_dim=cfg["input_dim"],
            target_dim=cfg["target_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
            architecture=cfg.get("architecture", "mlp"),
        )
        model.load_state_dict(bundle["state_dict"])
        obj._trainer = _inference_trainer(model, device)
        obj.device = device
        obj.method_ = cfg.get("method", "refiner")
        _restore_selection(obj, cfg)
        return obj

    def _resolve_target_dim(self, d: int, report: EmbedKitReport) -> int:
        if self.target_dim != "auto":
            return int(self.target_dim)
        return report.suggested_target_dim

    def _resolve_nested_dims(self, target_dim: int) -> tuple[int, ...] | None:
        if self.nested_dims is None:
            return None
        if isinstance(self.nested_dims, str):
            if self.nested_dims != "auto":
                raise ValueError(f"nested_dims must be 'auto', a sequence or None, "
                                 f"got {self.nested_dims!r}")
            dims, m = [], 32
            while m < target_dim:
                dims.append(m)
                m *= 2
            return tuple(dims) or (target_dim,)
        dims = tuple(int(m) for m in self.nested_dims)
        if any(m > target_dim for m in dims):
            raise ValueError(f"nested_dims {dims} exceed the output dimension {target_dim}")
        return dims

    def _resolve_augmentation(self, report: EmbedKitReport):
        if self.augmentation_cfg != "auto":
            return self.augmentation_cfg
        if self.mode_ == "regression":
            # E8: noise views beat input-space neighbour pairs with the MLP on 3 of 4 datasets.
            return GaussianNoise(std=0.05)
        return KNNPairs(k=5)

    def _resolve_loss(self, report: EmbedKitReport):
        if self.loss_cfg != "auto":
            return self.loss_cfg

        if self.mode_ == "supervised":
            # No uniformity term: it spreads each class over the sphere against SupCon's pull
            # (review 3.4, WP 6.1). The 0.2.0 default was SupCon + 0.5 AlignUniform.
            return SupConLoss(temperature=0.07)
        if self.mode_ == "regression":
            return RankNContrastLoss()
        return CombinedLoss([(NTXentLoss(temperature=0.07), 1.0), (AlignUniformLoss(), 0.5)])


_SMALL_REGRESSION_ROWS = 1000
_SELECTION_K = 10        # neighbours of the selection score
_PCA_DIMS = (64, 128, 256)  # PCA candidates of strategy="select" (those below d)
_N_BOOT = 2000           # paired bootstrap resamples of the held-out rows
_MIN_VAL_ROWS = 20       # fewer rows to fit or to score: no selection, Identity
# Constructor arguments passed on to the refiner candidate (name -> attribute).
_PARAMS = {name: name for name in (
    "hidden_dim", "n_layers", "architecture", "epochs", "batch_size", "optimizer", "lr",
    "weight_decay", "scheduler", "eval_every", "early_stopping_patience", "monitor",
    "val_split", "device", "id_methods", "random_state", "nested_dims", "val_fraction",
    "selection_metric", "selection_max_rows")} | {
    "augmentation": "augmentation_cfg", "loss": "loss_cfg"}


def _restore_selection(obj: EmbedKit, cfg: dict) -> None:
    """Set the fitted attributes that a bundle records (``selection_report_`` since WP 7.2)."""
    obj._config = cfg
    obj.mode_ = cfg.get("mode", "self_supervised")
    obj.strategy_ = cfg.get("strategy", "fixed")
    obj.selected_ = cfg.get("selected")
    obj.selection_report_ = None
    if cfg.get("selection_report"):
        obj.selection_report_ = pd.DataFrame(cfg["selection_report"])
        obj.selection_report_.attrs.update(cfg.get("selection_attrs", {}))


def _labels_array(y, mode: str) -> np.ndarray:
    y = _to_numpy(y) if isinstance(y, torch.Tensor) else np.asarray(y)
    if mode == "regression":
        return np.asarray(y, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError(f"class labels must be 1-D, got shape {y.shape}")
    if y.dtype.kind == "f" and not np.all(np.isfinite(y) & (y == np.round(y))):
        raise ValueError("y holds continuous values; pass mode='regression' for continuous "
                         "targets (mode='auto' treats labels as classes)")
    return y


def _selection_split(y, fraction: float, classes: bool, random_state):
    """Indices of the rows to fit and to score, stratified by class when every class has 2."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    stratify = None
    if classes:
        _, counts = np.unique(y, return_counts=True)
        n_val = int(np.ceil(fraction * len(y)))
        if counts.min() >= 2 and len(counts) <= min(n_val, len(y) - n_val):
            stratify = y
    return train_test_split(idx, test_size=fraction, random_state=random_state,
                            stratify=stratify)


def _selection_subsample(y, size: int, classes: bool, random_state) -> np.ndarray:
    """Sorted indices of ``size`` rows to select on, stratified by class when every class has 2."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    stratify = None
    if classes:
        _, counts = np.unique(y, return_counts=True)
        if counts.min() >= 2 and len(counts) <= min(size, len(y) - size):
            stratify = y
    sub, _ = train_test_split(idx, train_size=size, random_state=random_state,
                              stratify=stratify)
    return np.sort(sub)


def _fitted_baseline(name: str, X: np.ndarray):
    """``Identity`` or ``CenterNormalize`` fitted on validated ``X``.

    The same result as ``.fit(X)``, without the float64 copy of ``X`` that the baselines make
    (two times a float32 input; at a million rows of 4,096 columns, 33 GB).
    """
    base = Identity() if name == "identity" else CenterNormalize()
    base.n_features_in_ = base.n_features_out_ = X.shape[1]
    if name == "center_l2":
        base.mean_ = X.mean(axis=0, dtype=np.float64)
    return base


def _pca(X: np.ndarray):
    from embedkit.baselines import _pca_fit

    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    comps, _, rank = _pca_fit(X - mean)
    return mean, comps, rank


def _l2(Z: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return (Z / np.where(norms > 1e-12, norms, 1.0)).astype(np.float32)


def _np(Z) -> np.ndarray:
    return Z.numpy() if isinstance(Z, torch.Tensor) else np.asarray(Z)


def _inference_trainer(model: EmbeddingRefiner, device=None) -> Trainer:
    """A trainer that is only used for ``transform`` (loaded bundles and ``method="pca"``)."""
    return Trainer(model=model, augmentation=EmbeddingMixup(), loss=NTXentLoss(), epochs=0,
                   device=device)
