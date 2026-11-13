"""EmbedKit: high-level API orchestrating analyze → configure → train → transform."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from embedkit.analysis.report import EmbedKitAnalyzer, EmbedKitReport
from embedkit.improvement.augmentation import EmbeddingMixup, KNNPairs
from embedkit.improvement.losses import (
    AlignUniformLoss,
    CombinedLoss,
    MatryoshkaLoss,
    NTXentLoss,
    SupConLoss,
)
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
from embedkit.utils.io import load_bundle, save_bundle
from embedkit.utils.validation import _to_numpy


class EmbedKit:
    """Analyse, configure, train and transform in one estimator.

    ``nested_dims`` (WP 5.3) trains every prefix of the output as an embedding of its own, by
    wrapping the loss in :class:`~embedkit.improvement.losses.MatryoshkaLoss`; read a prefix
    with ``transform(X, dim=m)``. ``"auto"`` nests the powers of two from 32 below the output
    dimension, as in E6, where prefixes of one model at 256 matched linear models trained at
    32, 64 and 128 (``benchmarks/results/e6/REPORT.md``). A sequence gives the dimensions
    directly. ``None`` (the default) trains the full output only.
    """

    def __init__(
        self,
        mode: Literal["self_supervised", "supervised"] = "self_supervised",
        augmentation: Any = "auto",
        loss: Any = "auto",
        target_dim: int | Literal["auto"] = "auto",
        hidden_dim: int = 256,
        n_layers: int = 2,
        architecture: Literal["mlp", "residual", "linear"] = "linear",
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: str = "adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler: str | None = "cosine",
        eval_every: int = 5,
        early_stopping_patience: int | None = None,
        monitor: str = "k_skewness",
        val_split: float | None = None,
        device: str | torch.device | None = None,
        id_methods: list[str] | None = None,
        random_state: int | None = 42,
        nested_dims: Sequence[int] | Literal["auto"] | None = None,
    ):
        self.mode = mode
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

        self.analysis_report: EmbedKitReport | None = None
        self._trainer: Trainer | None = None
        self._config: dict = {}

    def fit(self, X, y=None) -> EmbedKit:
        X_np = _to_numpy(X)
        n, d = X_np.shape

        # Step 1: analyze
        analyzer = EmbedKitAnalyzer(
            id_methods=self.id_methods,
            random_state=self.random_state,
        )
        self.analysis_report = analyzer.fit(X_np, y)
        report = self.analysis_report

        # Step 2: auto-configure
        target_dim = self._resolve_target_dim(d, report)
        augmentation = self._resolve_augmentation(report)
        loss_fn = self._resolve_loss(report)
        nested = self._resolve_nested_dims(target_dim)
        if nested is not None:
            if isinstance(loss_fn, MatryoshkaLoss):
                raise ValueError("Pass nested_dims or a MatryoshkaLoss, not both")
            loss_fn = MatryoshkaLoss(loss_fn, dims=nested)
        self.nested_dims_ = loss_fn.dims if isinstance(loss_fn, MatryoshkaLoss) else None

        self._config = {
            "mode": self.mode,
            "target_dim": target_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "architecture": self.architecture,
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
                architecture=self.architecture,
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
            eval_metrics=[self.monitor],
            early_stopping_patience=self.early_stopping_patience,
            monitor=self.monitor,
            val_split=self._resolve_val_split(),
            device=self.device,
            random_state=self.random_state,
        )
        self._trainer.fit(X_np, y)
        return self

    def _resolve_val_split(self) -> float:
        # Early stopping is off by default: in E2 (docs/BENCHMARK_E1_E2.md) no label-free monitor
        # tracked test accuracy within a run. Hold rows out only when a monitor decides something.
        if self.val_split is not None:
            return self.val_split
        return 0.1 if self.early_stopping_patience is not None else 0.0

    def transform(self, X, dim: int | None = None) -> np.ndarray | torch.Tensor:
        """Refine ``X``. With ``dim``, keep the first ``dim`` output coordinates and renormalise.

        Truncation is meant for models trained with
        :class:`~embedkit.improvement.losses.MatryoshkaLoss`, whose prefixes are trained to be
        embeddings on their own (WP 5.3); for other models a prefix is just a projection.
        """
        if self._trainer is None:
            raise RuntimeError("Call fit() before transform()")
        Z = self._trainer.transform(X)
        if dim is None:
            return Z
        d = Z.shape[1]
        if not 1 <= dim <= d:
            raise ValueError(f"dim must be between 1 and the output dimension {d}, got {dim}")
        Z = Z[:, :dim]
        if getattr(self._trainer.model, "normalize_output", True):
            if isinstance(Z, torch.Tensor):
                Z = torch.nn.functional.normalize(Z, dim=1)
            else:
                Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        return Z

    def fit_transform(self, X, y=None, dim: int | None = None) -> np.ndarray | torch.Tensor:
        self.fit(X, y)
        return self.transform(X, dim=dim)

    def save(self, directory: str | Path) -> None:
        if self._trainer is None:
            raise RuntimeError("Nothing fitted yet")
        save_bundle(directory, self._trainer.model, self._config, self.analysis_report)

    @classmethod
    def load(cls, directory: str | Path) -> EmbedKit:
        bundle = load_bundle(directory)
        cfg = bundle["config"]
        obj = cls(
            mode=cfg.get("mode", "self_supervised"),
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
        # minimal trainer wrapper just for transform
        from embedkit.improvement.losses import NTXentLoss
        trainer = Trainer(
            model=model,
            augmentation=EmbeddingMixup(),
            loss=NTXentLoss(),
            epochs=0,
        )
        obj._trainer = trainer
        obj._config = cfg
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
        return KNNPairs(k=5)

    def _resolve_loss(self, report: EmbedKitReport):
        if self.loss_cfg != "auto":
            return self.loss_cfg

        if self.mode == "self_supervised":
            base = NTXentLoss(temperature=0.07)
        else:
            base = SupConLoss(temperature=0.07)

        return CombinedLoss([(base, 1.0), (AlignUniformLoss(), 0.5)])
