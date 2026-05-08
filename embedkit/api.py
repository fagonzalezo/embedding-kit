"""EmbedKit: high-level API orchestrating analyze → configure → train → transform."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from embedkit.analysis.report import EmbedKitAnalyzer, EmbedKitReport
from embedkit.improvement.augmentation import EmbeddingMixup, KNNPairs
from embedkit.improvement.losses import (
    AlignUniformLoss,
    CombinedLoss,
    NTXentLoss,
    SupConLoss,
)
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.improvement.trainer import Trainer
from embedkit.utils.io import load_bundle, save_bundle
from embedkit.utils.validation import _to_numpy


class EmbedKit:
    def __init__(
        self,
        mode: Literal["self_supervised", "supervised"] = "self_supervised",
        augmentation: Any = "auto",
        loss: Any = "auto",
        target_dim: int | Literal["auto"] = "auto",
        hidden_dim: int = 256,
        n_layers: int = 2,
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: str = "adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler: str | None = "cosine",
        eval_every: int = 5,
        early_stopping_patience: int | None = 10,
        monitor: str = "k_skewness",
        val_split: float = 0.1,
        device: str | torch.device | None = None,
        id_methods: list[str] | None = None,
        random_state: int | None = 42,
    ):
        self.mode = mode
        self.augmentation_cfg = augmentation
        self.loss_cfg = loss
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
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

        self.analysis_report: EmbedKitReport | None = None
        self._trainer: Trainer | None = None
        self._config: dict = {}

    def fit(self, X, y=None) -> "EmbedKit":
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

        self._config = {
            "mode": self.mode,
            "target_dim": target_dim,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "input_dim": d,
        }

        # Step 3: build model + trainer
        model = EmbeddingRefiner(
            input_dim=d,
            target_dim=target_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
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
            val_split=self.val_split,
            device=self.device,
            random_state=self.random_state,
        )
        self._trainer.fit(X_np, y)
        return self

    def transform(self, X) -> np.ndarray | torch.Tensor:
        if self._trainer is None:
            raise RuntimeError("Call fit() before transform()")
        return self._trainer.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray | torch.Tensor:
        self.fit(X, y)
        return self.transform(X)

    def save(self, directory: str | Path) -> None:
        if self._trainer is None:
            raise RuntimeError("Nothing fitted yet")
        save_bundle(directory, self._trainer.model, self._config, self.analysis_report)

    @classmethod
    def load(cls, directory: str | Path) -> "EmbedKit":
        bundle = load_bundle(directory)
        cfg = bundle["config"]
        obj = cls(
            mode=cfg.get("mode", "self_supervised"),
            target_dim=cfg.get("target_dim", 64),
            hidden_dim=cfg.get("hidden_dim", 256),
            n_layers=cfg.get("n_layers", 2),
        )
        model = EmbeddingRefiner(
            input_dim=cfg["input_dim"],
            target_dim=cfg["target_dim"],
            hidden_dim=cfg["hidden_dim"],
            n_layers=cfg["n_layers"],
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
