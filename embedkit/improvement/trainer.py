"""Trainer: training loop, evaluation, early stopping."""

from __future__ import annotations

import contextlib
import copy
import warnings
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from embedkit.improvement.augmentation.base import BaseAugmentation
from embedkit.improvement.losses.base import BaseLoss
from embedkit.improvement.model import EmbeddingRefiner
from embedkit.utils.validation import _to_numpy, _to_tensor, encode_labels

# ---------------------------------------------------------------------------------------------
# Monitors
# ---------------------------------------------------------------------------------------------
# A monitor is a statistic of the refined embedding Z of the evaluation split. Each entry is
# (function, direction): direction "min" means lower is better, "max" means higher is better.
# Early stopping, best-weight tracking and ReduceLROnPlateau read the direction from here.
#
# Functions are called as fn(Z, **context) and should accept **_ for context they do not use.
# The context holds: k, metric, random_state (from eval_k, eval_metric, random_state); X (the
# input rows of the evaluation split); y (their labels, or None); Z_ref and y_ref (refined
# training rows and labels, the neighbour pool for val_knn_accuracy, or None); trainer.

MonitorFn = Callable[..., float]


def _monitor_uniformity(Z: np.ndarray, *, random_state=None, **_) -> float:
    from embedkit.analysis.geometry import UniformityScore

    return float(UniformityScore(random_state=random_state).fit(Z).uniformity)


def _monitor_isotropy(Z: np.ndarray, **_) -> float:
    from embedkit.analysis.geometry import IsotropyAnalyzer

    return float(IsotropyAnalyzer().fit(Z).isotropy_score)


def _monitor_k_skewness(Z: np.ndarray, *, k: int, metric: str, random_state=None, **_) -> float:
    from embedkit.analysis.hubness import HubnessAnalyzer

    k_eff = min(k, Z.shape[0] - 1)
    return float(
        HubnessAnalyzer(k=k_eff, metric=metric, random_state=random_state).fit(Z).k_skewness
    )


def _monitor_knn_overlap(Z, *, X, k: int, metric: str, random_state=None, **_) -> float:
    from embedkit.evaluation.metrics import knn_overlap

    return float(knn_overlap(X, Z, k=min(k, Z.shape[0] - 1), metric_x="euclidean",
                             metric_z=metric, max_samples=2000, random_state=random_state or 0))


def _monitor_effective_rank(Z, **_) -> float:
    from embedkit.evaluation.metrics import effective_rank

    return float(effective_rank(Z))


def _monitor_val_loss(Z, *, X, y, trainer, **_) -> float:
    return trainer._held_out_loss(X, y)


def _monitor_val_knn_accuracy(Z, *, y, Z_ref, y_ref, metric: str, **_) -> float:
    from embedkit.evaluation.metrics import knn_accuracy

    if y is None or Z_ref is None:
        raise ValueError("val_knn_accuracy needs labels and val_split > 0")
    return float(knn_accuracy(Z_ref, y_ref, Z, y, k=min(5, len(Z_ref)), metric=metric))


MONITORS: dict[str, tuple[MonitorFn, Literal["min", "max"]]] = {
    # Wang & Isola uniformity, log E[exp(-t ||z_i - z_j||^2)]: lower is more uniform.
    "uniformity": (_monitor_uniformity, "min"),
    # Normalised effective rank of the covariance spectrum: higher is more isotropic.
    "isotropy": (_monitor_isotropy, "max"),
    # Skewness of the k-occurrence distribution: lower means less hubness.
    "k_skewness": (_monitor_k_skewness, "min"),
    # Mean fraction of each point's eval_k input-space (Euclidean) neighbours kept in the output
    # (eval_metric), on at most 2,000 evaluation rows: higher preserves more local structure.
    "knn_overlap": (_monitor_knn_overlap, "max"),
    # RankMe effective rank of the refined evaluation rows: higher is less dimensional collapse.
    "effective_rank": (_monitor_effective_rank, "max"),
    # The training loss on the validation split, with the augmentation (fixed draws): lower is
    # better. Needs val_split > 0.
    "val_loss": (_monitor_val_loss, "min"),
    # 5-NN accuracy of the validation rows, neighbours from (a subsample of) the training rows.
    # Needs labels and val_split > 0.
    "val_knn_accuracy": (_monitor_val_knn_accuracy, "max"),
}

_NEEDS_VAL_SPLIT = {"val_loss", "val_knn_accuracy"}
_NEEDS_LABELS = {"val_knn_accuracy"}


def _check_monitor_name(name: str) -> None:
    if name not in MONITORS:
        raise ValueError(f"Unknown monitor {name!r}. Available: {sorted(MONITORS)}")


def _is_improvement(score: float, best: float | None, direction: str) -> bool:
    if not np.isfinite(score):
        return False
    if best is None:
        return True
    return score < best if direction == "min" else score > best


# ---------------------------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------------------------


def _default_device() -> torch.device:
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Train an :class:`EmbeddingRefiner` with a contrastive loss.

    Parameters (selected)
    ---------------------
    scheduler:
        ``"cosine"`` (cosine annealing, stepped per batch), ``"step"`` (learning rate times
        0.1 every ``epochs // 3`` epochs, stepped per epoch), ``"plateau"``
        (``ReduceLROnPlateau`` on the monitor, stepped at each evaluation) or ``None``.
    warmup_epochs:
        Linear learning-rate warm-up over this many epochs before the scheduler takes over.
        0 (the default) disables it.
    eval_every:
        Evaluate the monitors every this many epochs.
    eval_metrics:
        Names from :data:`MONITORS` to log in ``history`` at each evaluation. The ``monitor``
        is always added. ``val_loss`` and ``val_knn_accuracy`` need ``val_split > 0``, and
        ``val_knn_accuracy`` needs labels; with ``val_split > 0`` the labels of the validation
        rows are kept for it.
    eval_k, eval_metric:
        Neighbourhood size and metric used by neighbour-based monitors (``k_skewness``,
        ``knn_overlap``; ``val_knn_accuracy`` uses 5 neighbours and ``eval_metric``). The
        default metric is cosine because the refiner's output is L2-normalised.
    early_stopping_patience:
        Stop after this many consecutive **evaluations** (not epochs) without improvement of
        the monitor. With ``eval_every=5`` and a patience of 10, that is 50 epochs. ``None``
        disables early stopping.
    monitor:
        Name in :data:`MONITORS`. Its direction (minimise or maximise) comes from the registry.
    restore_best:
        When early stopping is enabled, load the weights of the best evaluation at the end of
        :meth:`fit`, whether training stopped early or ran out of epochs.
    device:
        ``None`` selects CUDA, then Apple MPS, then CPU. If the loss cannot run on MPS, the
        trainer falls back to CPU with a warning.

    Attributes set by :meth:`fit`
    -----------------------------
    best_score_, best_epoch_:
        Best monitor value and the epoch at which it was reached (``None`` if no evaluation ran).
    stopped_epoch_:
        Last epoch trained.
    history:
        ``"loss"`` and ``"lr"`` (learning rate at the end of the epoch) per epoch;
        ``"eval_epoch"`` and one list per monitor, per evaluation.
    """

    def __init__(
        self,
        model: EmbeddingRefiner,
        augmentation: BaseAugmentation,
        loss: BaseLoss,
        epochs: int = 100,
        batch_size: int = 256,
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        scheduler: Literal["cosine", "step", "plateau", None] = "cosine",
        warmup_epochs: int = 0,
        eval_every: int = 10,
        eval_metrics: list[str] | None = None,
        eval_subsample: int = 5_000,
        eval_k: int = 10,
        eval_metric: str = "cosine",
        early_stopping_patience: int | None = None,
        monitor: str = "uniformity",
        restore_best: bool = True,
        val_split: float = 0.0,
        device: str | torch.device | None = None,
        random_state: int | None = 42,
    ):
        if optimizer not in ("adam", "sgd"):
            raise ValueError(
                f"Unknown optimizer {optimizer!r}. Choose 'adam' or 'sgd'. "
                "('lars' was removed in 0.1.3: it was SGD with a 10x learning rate, not LARS.)"
            )
        if scheduler not in ("cosine", "step", "plateau", None):
            raise ValueError(f"Unknown scheduler {scheduler!r}")
        if warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        _check_monitor_name(monitor)
        metrics = list(eval_metrics) if eval_metrics is not None else ["uniformity"]
        for name in metrics:
            _check_monitor_name(name)
        if monitor not in metrics:
            metrics.append(monitor)

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
        self.eval_metrics = metrics
        self.eval_subsample = eval_subsample
        self.eval_k = eval_k
        self.eval_metric = eval_metric
        self.early_stopping_patience = early_stopping_patience
        self.monitor = monitor
        self.restore_best = restore_best
        self.val_split = val_split
        self.device = torch.device(device) if device is not None else _default_device()
        self.random_state = random_state
        self.history: dict[str, list] = {"loss": []}
        self.best_score_: float | None = None
        self.best_epoch_: int | None = None
        self.stopped_epoch_: int = 0

    @property
    def monitor_mode(self) -> Literal["min", "max"]:
        return MONITORS[self.monitor][1]

    # -----------------------------------------------------------------------------------------
    def fit(self, X, y=None) -> Trainer:
        """Train the model on ``X`` (and labels ``y`` for supervised losses).

        With an integer ``random_state`` the run is reproducible on CPU, and the caller's global
        NumPy and torch random states are left untouched: data order, augmentation draws and
        dropout all come from generators derived from ``random_state``. (The model's initial
        weights are drawn when the model is built; build it under a fixed seed too.)
        """
        with _rng_scope(self.random_state is not None):
            return self._fit(X, y)

    def _fit(self, X, y=None) -> Trainer:
        loader_gen: torch.Generator | None = None
        if self.random_state is not None:
            ss_loader, ss_aug_torch, ss_aug_np, ss_global = np.random.SeedSequence(
                self.random_state
            ).spawn(4)
            loader_gen = torch.Generator().manual_seed(_seed_int(ss_loader))
            aug_torch_gen = torch.Generator().manual_seed(_seed_int(ss_aug_torch))
            aug_np_rng = np.random.default_rng(ss_aug_np)
            # Global torch state (dropout inside the model); restored on exit by _rng_scope.
            torch.manual_seed(_seed_int(ss_global))
        else:
            aug_torch_gen, aug_np_rng = None, None
        if hasattr(self.augmentation, "set_rng"):
            self.augmentation.set_rng(aug_np_rng, aug_torch_gen)

        X_np = _to_numpy(X)
        n = X_np.shape[0]
        has_labels = y is not None
        y_t: torch.Tensor | None = None
        if has_labels:
            y_t, _ = encode_labels(y, n_samples=n)

        # Carve a held-out validation split for early stopping when requested.
        # Precompute and the DataLoader use only the training portion so the
        # val set never contributes gradients or neighbor structure.
        missing_val = [m for m in self.eval_metrics if m in _NEEDS_VAL_SPLIT]
        if missing_val and self.val_split <= 0.0:
            raise ValueError(f"Monitors {missing_val} need val_split > 0")
        missing_y = [m for m in self.eval_metrics if m in _NEEDS_LABELS]
        if missing_y and not has_labels:
            raise ValueError(f"Monitors {missing_y} need labels: call fit(X, y)")

        X_val_np: np.ndarray | None = None
        y_val: np.ndarray | None = None
        if self.val_split > 0.0:
            rng_split = np.random.default_rng(self.random_state)
            n_val = max(1, int(n * self.val_split))
            val_idx = rng_split.choice(n, n_val, replace=False)
            train_mask = np.ones(n, dtype=bool)
            train_mask[val_idx] = False
            X_val_np = X_np[val_idx]
            X_np = X_np[train_mask]
            if y_t is not None:
                y_val = y_t[torch.from_numpy(val_idx)].numpy()
                y_t = y_t[torch.from_numpy(train_mask)]
            n = X_np.shape[0]

        if n < 2:
            raise ValueError(
                f"Trainer.fit needs at least 2 training samples, got {n}"
                + (f" after val_split={self.val_split}" if self.val_split > 0.0 else "")
                + "."
            )

        # Data-dependent initialisation (e.g. the residual refiner starts at PCA), on the
        # training rows only.
        if hasattr(self.model, "initialize_from_data"):
            self.model.initialize_from_data(X_np)

        self._select_device(X_np, y_t)

        # Precompute neighbor structures for augmentations that support it
        # (e.g. KNNPairs), so global dataset-level positives can be used.
        if hasattr(self.augmentation, "precompute"):
            self.augmentation.precompute(X_np)
        if hasattr(self.loss, "precompute"):
            self.loss.precompute(X_np)  # e.g. input-space neighbours for NTXentLoss(neighbor_mask)
        pass_indices = getattr(self.loss, "uses_indices", False)

        # Keep data on CPU; pin_memory lets CUDA transfers overlap with compute.
        # Global row indices (position 1) let index-aware augmentations look up
        # dataset-level neighbors even under a shuffled DataLoader.
        tensors = [torch.from_numpy(X_np), torch.arange(n, dtype=torch.long)]
        if y_t is not None:
            tensors.append(y_t)

        # Contrastive losses need at least two samples per batch. With drop_last=True,
        # a batch_size larger than n would yield zero batches per epoch: no training
        # and a NaN loss (mean of an empty list). Cap the batch at n instead.
        batch_size = self.batch_size
        if batch_size > n:
            warnings.warn(
                f"batch_size={self.batch_size} exceeds the number of training samples ({n}); "
                f"using batch_size={n} (one batch per epoch).",
                UserWarning,
                stacklevel=2,
            )
            batch_size = n

        pin = self.device.type == "cuda"
        loader = DataLoader(
            TensorDataset(*tensors), batch_size=batch_size, shuffle=True,
            drop_last=True, pin_memory=pin, num_workers=0, generator=loader_gen,
        )

        # Array used for evaluation: the held-out split, else a subsample of the training data.
        # The training subsample is also the neighbour pool for val_knn_accuracy.
        rng = np.random.default_rng(self.random_state)
        sub_idx = (
            np.sort(rng.choice(n, self.eval_subsample, replace=False))
            if n > self.eval_subsample else np.arange(n)
        )
        if X_val_np is not None:
            X_eval, y_eval = X_val_np, y_val
        else:
            X_eval, y_eval = X_np[sub_idx], None
        self._eval_ctx = dict(
            y=y_eval,
            X_ref=X_np[sub_idx] if y_val is not None else None,
            y_ref=y_t[torch.from_numpy(sub_idx)].numpy() if y_val is not None else None,
        )
        self._val_batch_size = batch_size

        self.model.to(self.device)
        self.loss.to(self.device)
        opt = self._build_optimizer()
        steps_per_epoch = len(loader)
        batch_sched, epoch_sched, plateau_sched = self._build_schedulers(opt, steps_per_epoch)

        self.history = {"loss": [], "lr": [], "eval_epoch": []}
        self.best_score_ = None
        self.best_epoch_ = None
        best_state: dict[str, Any] | None = None
        patience_count = 0
        self.stopped_epoch_ = 0
        track_best = self.early_stopping_patience is not None

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
                if pass_indices:
                    loss_value = self.loss(z_i, z_j, lbl_batch, indices=idx_batch)
                else:
                    loss_value = self.loss(z_i, z_j, lbl_batch)
                opt.zero_grad()
                loss_value.backward()
                opt.step()
                if batch_sched is not None:
                    batch_sched.step()
                ep_losses.append(loss_value.item())
            if epoch_sched is not None:
                epoch_sched.step()
            self.stopped_epoch_ = epoch

            self.history["loss"].append(float(np.mean(ep_losses)))
            self.history["lr"].append(float(opt.param_groups[0]["lr"]))

            if epoch % self.eval_every != 0:
                continue
            metrics = self._evaluate(X_eval, **self._eval_ctx)
            self.history["eval_epoch"].append(epoch)
            for name, value in metrics.items():
                self.history.setdefault(name, []).append(value)
            score = metrics[self.monitor]

            if plateau_sched is not None and epoch > self.warmup_epochs:
                plateau_sched.step(score)

            if _is_improvement(score, self.best_score_, self.monitor_mode):
                self.best_score_ = score
                self.best_epoch_ = epoch
                patience_count = 0
                if track_best and self.restore_best:
                    best_state = _copy_state(self.model)
            else:
                patience_count += 1

            if track_best and patience_count >= self.early_stopping_patience:
                break

        if track_best and self.restore_best and best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def transform(self, X, batch_size: int | None = None) -> np.ndarray | torch.Tensor:
        """Transform X in mini-batches to avoid GPU OOM on large datasets."""
        was_tensor = isinstance(X, torch.Tensor)
        X_np = _to_numpy(X)
        bs = batch_size or max(self.batch_size * 4, 1024)
        self.model.to(self.device)
        self.model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, X_np.shape[0], bs):
                x_chunk = _to_tensor(X_np[start: start + bs], device=self.device)
                chunks.append(self.model(x_chunk).cpu())
        Z = torch.cat(chunks, dim=0)
        return Z if was_tensor else Z.numpy()

    def evaluate(self, X, y=None, X_ref=None, y_ref=None) -> dict[str, float]:
        """Compute every monitor in ``eval_metrics`` on the refined embedding of ``X``.

        ``y`` are the labels of ``X``; ``X_ref`` and ``y_ref`` the labelled neighbour pool
        (``val_knn_accuracy`` needs all three).
        """
        y_np = encode_labels(y, n_samples=len(X))[0].numpy() if y is not None else None
        y_ref_np = (
            encode_labels(y_ref, n_samples=len(X_ref))[0].numpy() if y_ref is not None else None
        )
        return self._evaluate(
            _to_numpy(X), y=y_np, X_ref=None if X_ref is None else _to_numpy(X_ref), y_ref=y_ref_np
        )

    def _evaluate(self, X_np: np.ndarray, y=None, X_ref=None, y_ref=None) -> dict[str, float]:
        Z_np = self.transform(X_np)
        needs_ref = any(name in _NEEDS_LABELS for name in self.eval_metrics)
        Z_ref = self.transform(X_ref) if X_ref is not None and needs_ref else None
        context = dict(
            X=X_np, y=y, Z_ref=Z_ref, y_ref=y_ref, trainer=self,
            k=self.eval_k, metric=self.eval_metric, random_state=self.random_state,
        )
        return {name: float(MONITORS[name][0](Z_np, **context)) for name in self.eval_metrics}

    def _held_out_loss(self, X_np: np.ndarray, y=None) -> float:
        """The training loss on held-out rows, with the augmentation, without gradients.

        The augmentation's random draws come from a generator reseeded at every call, so the
        value changes only when the model does. Index-based augmentations (``KNNPairs``) use
        their in-batch path, since the rows are not in the training set.
        """
        aug = self.augmentation
        saved = (getattr(aug, "_np_rng", None), getattr(aug, "_torch_gen", None))
        seed = 0 if self.random_state is None else int(self.random_state)
        if hasattr(aug, "set_rng"):
            aug.set_rng(np.random.default_rng(seed), torch.Generator().manual_seed(seed))
        bs = max(2, min(getattr(self, "_val_batch_size", self.batch_size), len(X_np)))
        total, count = 0.0, 0
        self.model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(X_np), bs):
                    xb = torch.from_numpy(np.ascontiguousarray(X_np[start: start + bs]))
                    if len(xb) < 2:
                        continue
                    xb = xb.to(self.device)
                    yb = None
                    if y is not None:
                        yb = torch.as_tensor(np.asarray(y[start: start + bs])).to(self.device)
                    x_i, x_j = aug(xb, None)
                    value = self.loss(self.model(x_i), self.model(x_j), yb)
                    total += float(value) * len(xb)
                    count += len(xb)
        finally:
            if hasattr(aug, "set_rng"):
                aug.set_rng(*saved)
        return total / max(count, 1)

    # -----------------------------------------------------------------------------------------
    def _select_device(self, X_np: np.ndarray, y_t: torch.Tensor | None) -> None:
        """Fall back from MPS to CPU when the loss or augmentation does not run there."""
        if self.device.type != "mps":
            return
        m = min(8, X_np.shape[0])
        try:
            x = torch.from_numpy(X_np[:m]).to(self.device)
            lbl = y_t[:m].to(self.device) if y_t is not None else None
            probe = copy.deepcopy(self.model).to(self.device)
            probe.eval()  # no dropout: the probe must not consume random numbers
            loss = copy.deepcopy(self.loss).to(self.device)
            z_i, z_j = probe(x), probe(x.roll(1, dims=0))
            loss(z_i, z_j, lbl).backward()
        except (NotImplementedError, RuntimeError, TypeError) as exc:
            warnings.warn(
                f"{type(self.loss).__name__} does not run on MPS ({exc}); training on CPU.",
                UserWarning,
                stacklevel=3,
            )
            self.device = torch.device("cpu")

    def _build_optimizer(self):
        params = list(self.model.parameters()) + list(self.loss.parameters())
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(
                params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

    def _build_schedulers(self, opt, steps_per_epoch: int):
        """Return (per-batch scheduler, per-epoch scheduler, plateau scheduler).

        At most one of the first two is set. Warm-up is linear and is stepped at the same
        granularity as the scheduler it precedes (per batch for cosine and None, per epoch for
        step and plateau).
        """
        from torch.optim import lr_scheduler as S

        W = self.warmup_epochs
        name = self.scheduler_name

        def warmup(total_iters: int):
            return S.LinearLR(opt, start_factor=1.0 / (total_iters + 1), total_iters=total_iters)

        if name == "cosine":
            total = self.epochs * steps_per_epoch
            w = min(W * steps_per_epoch, max(total - 1, 0))
            cosine = S.CosineAnnealingLR(opt, T_max=max(1, total - w))
            if w == 0:
                return cosine, None, None
            return S.SequentialLR(opt, [warmup(w), cosine], milestones=[w]), None, None
        if name == "step":
            step = S.StepLR(opt, step_size=max(1, self.epochs // 3), gamma=0.1)
            if W == 0:
                return None, step, None
            return None, S.SequentialLR(opt, [warmup(W), step], milestones=[W]), None
        if name == "plateau":
            plateau = S.ReduceLROnPlateau(opt, mode=self.monitor_mode, patience=5)
            return None, (warmup(W) if W > 0 else None), plateau
        # No scheduler: warm-up only, if requested.
        if W > 0:
            return warmup(W * steps_per_epoch), None, None
        return None, None, None


def _seed_int(ss: np.random.SeedSequence) -> int:
    return int(ss.generate_state(1, dtype=np.uint64)[0] % (2**63 - 1))


@contextlib.contextmanager
def _rng_scope(enabled: bool):
    """Save the global torch RNG states (CPU, CUDA, MPS) and restore them on exit."""
    if not enabled:
        yield
        return
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))  # CPU
        if torch.cuda.is_available():
            stack.enter_context(
                torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
            )
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            stack.enter_context(torch.random.fork_rng(devices=[0], device_type="mps"))
        yield


def _copy_state(model: torch.nn.Module) -> dict[str, Any]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
