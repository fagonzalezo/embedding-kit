"""Trainer: training loop, evaluation, early stopping."""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from embedkit.improvement.augmentation.base import BaseAugmentation
from embedkit.improvement.losses.base import BaseLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.utils.validation import _to_numpy, _to_tensor


class Trainer:
    def __init__(
        self,
        model: EmbeddingRefiner,
        augmentation: BaseAugmentation,
        loss: BaseLoss,
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: Literal["adam", "sgd", "lars"] = "adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler: Literal["cosine", "step", "plateau", None] = "cosine",
        warmup_epochs: int = 10,
        eval_every: int = 10,
        eval_metrics: list[str] | None = None,
        eval_subsample: int = 5_000,
        early_stopping_patience: int | None = None,
        monitor: str = "uniformity",
        val_split: float = 0.0,
        device: str | torch.device | None = None,
        random_state: int | None = 42,
    ):
        self.model = model
        self.augmentation = augmentation
        self.loss = loss
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_name = optimizer
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.warmup_epochs = warmup_epochs
        self.eval_every = eval_every
        self.eval_metrics = eval_metrics or ["uniformity"]
        self.eval_subsample = eval_subsample
        self.early_stopping_patience = early_stopping_patience
        self.monitor = monitor
        self.val_split = val_split
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.random_state = random_state
        self.history: dict[str, list] = {"loss": []}

    def fit(self, X, y=None) -> "Trainer":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_np = _to_numpy(X)
        n = X_np.shape[0]
        has_labels = y is not None

        # Carve a held-out validation split for early stopping when requested.
        # Precompute and the DataLoader use only the training portion so the
        # val set never contributes gradients or neighbor structure.
        X_val_np: np.ndarray | None = None
        if self.val_split > 0.0:
            rng_split = np.random.default_rng(self.random_state)
            n_val = max(1, int(n * self.val_split))
            val_idx = rng_split.choice(n, n_val, replace=False)
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            X_val_np = X_np[val_idx]
            X_np = X_np[train_mask]
            if has_labels:
                y = np.asarray(y)
                y = y[train_mask]
            n = X_np.shape[0]

        # Precompute neighbor structures for augmentations that support it
        # (e.g. KNNPairs), so global dataset-level positives can be used.
        if hasattr(self.augmentation, "precompute"):
            self.augmentation.precompute(X_np)

        # Keep data on CPU; pin_memory lets CUDA transfers overlap with compute.
        # Global row indices (position 1) let index-aware augmentations look up
        # dataset-level neighbors even under a shuffled DataLoader.
        X_cpu = torch.from_numpy(X_np)
        idx_cpu = torch.arange(n, dtype=torch.long)
        tensors = [X_cpu, idx_cpu]
        if has_labels:
            y_arr = np.asarray(y)
            tensors.append(torch.from_numpy(y_arr))

        dataset = TensorDataset(*tensors)
        pin = self.device.type == "cuda"
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            drop_last=True, pin_memory=pin, num_workers=0,
        )

        # Determine the array used for early-stopping evaluation.
        # Prefer the held-out val split; fall back to a subsample of training data.
        if X_val_np is not None:
            X_eval = X_val_np
        elif n > self.eval_subsample and self.random_state is not None:
            rng = np.random.default_rng(self.random_state)
            eval_idx = rng.choice(n, self.eval_subsample, replace=False)
            X_eval = X_np[eval_idx]
        else:
            X_eval = X_np

        self.model.to(self.device)
        self.loss.to(self.device)
        opt = self._build_optimizer()
        sched = self._build_scheduler(opt, len(loader))

        best_score = float("inf")
        patience_count = 0
        self.stopped_epoch_: int = self.epochs

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            ep_losses = []
            for batch in loader:
                x_batch = batch[0].to(self.device, non_blocking=pin)
                idx_batch = batch[1]  # global row indices; kept on CPU for array indexing
                lbl_batch = batch[2].to(self.device, non_blocking=pin) if has_labels else None
                x_i, x_j = self.augmentation(x_batch, idx_batch)
                z_i = self.model(x_i)
                z_j = self.model(x_j)
                l = self.loss(z_i, z_j, lbl_batch)
                opt.zero_grad()
                l.backward()
                opt.step()
                if sched is not None and self.scheduler_name != "plateau":
                    sched.step()
                ep_losses.append(l.item())

            mean_loss = float(np.mean(ep_losses))
            self.history["loss"].append(mean_loss)

            if epoch % self.eval_every == 0:
                metrics = self._evaluate(X_eval)
                for k, v in metrics.items():
                    self.history.setdefault(k, []).append(v)

                if self.scheduler_name == "plateau" and sched is not None and self.monitor in metrics:
                    sched.step(metrics[self.monitor])

                if self.early_stopping_patience is not None and self.monitor in metrics:
                    score = metrics[self.monitor]
                    if score < best_score:
                        best_score = score
                        patience_count = 0
                    else:
                        patience_count += 1
                    if patience_count >= self.early_stopping_patience:
                        self.stopped_epoch_ = epoch
                        break

        return self

    def transform(self, X, batch_size: int | None = None) -> np.ndarray | torch.Tensor:
        """Transform X in mini-batches to avoid GPU OOM on large datasets."""
        was_tensor = isinstance(X, torch.Tensor)
        X_np = _to_numpy(X)
        bs = batch_size or max(self.batch_size * 4, 1024)
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, X_np.shape[0], bs):
                x_chunk = _to_tensor(X_np[start: start + bs], device=self.device)
                chunks.append(self.model(x_chunk).cpu())
        Z = torch.cat(chunks, dim=0)
        return Z if was_tensor else Z.numpy()

    def _evaluate(self, X_np: np.ndarray) -> dict[str, float]:
        metrics: dict[str, float] = {}
        try:
            from embedkit.analysis.geometry import UniformityScore, IsotropyAnalyzer
            from embedkit.analysis.hubness import HubnessAnalyzer
            Z_np = self.transform(X_np)
            if isinstance(Z_np, torch.Tensor):
                Z_np = Z_np.numpy()
            if "uniformity" in self.eval_metrics:
                metrics["uniformity"] = UniformityScore().fit(Z_np).uniformity
            if "isotropy" in self.eval_metrics:
                metrics["isotropy"] = IsotropyAnalyzer().fit(Z_np).isotropy_score
            if "k_skewness" in self.eval_metrics:
                metrics["k_skewness"] = HubnessAnalyzer().fit(Z_np).k_skewness
        except Exception as e:
            warnings.warn(f"Eval step failed: {e}")
        return metrics

    def _build_optimizer(self):
        params = list(self.model.parameters()) + list(self.loss.parameters())
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        if self.optimizer_name == "lars":
            try:
                from torch.optim import SGD
                return SGD(params, lr=self.lr * 10, momentum=0.9, weight_decay=self.weight_decay)
            except Exception:
                return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _build_scheduler(self, opt, steps_per_epoch: int):
        if self.scheduler_name is None:
            return None
        total_steps = self.epochs * steps_per_epoch
        if self.scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        if self.scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, self.epochs // 3))
        if self.scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5)
        return None
