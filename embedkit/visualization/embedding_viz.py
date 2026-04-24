"""EmbeddingVisualizer: 2-D projections with UMAP."""

from __future__ import annotations

from typing import Literal

import numpy as np


class EmbeddingVisualizer:
    def __init__(
        self,
        method: Literal["umap", "pca"] = "umap",
        n_components: int = 2,
        random_state: int | None = 42,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

    def _reduce(self, X: np.ndarray) -> np.ndarray:
        if self.method == "pca" or X.shape[1] <= self.n_components:
            from sklearn.decomposition import PCA
            return PCA(n_components=self.n_components, random_state=self.random_state).fit_transform(X)
        import umap
        return umap.UMAP(
            n_components=self.n_components,
            random_state=self.random_state,
        ).fit_transform(X)

    def plot_comparison(self, X_before: np.ndarray, X_after: np.ndarray, labels=None):
        import matplotlib.pyplot as plt
        from embedkit.utils.validation import _to_numpy
        Xb = _to_numpy(X_before)
        Xa = _to_numpy(X_after)

        emb_b = self._reduce(Xb)
        emb_a = self._reduce(Xa)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax, emb, title in zip(axes, [emb_b, emb_a], ["Before refinement", "After refinement"]):
            sc = ax.scatter(
                emb[:, 0], emb[:, 1],
                c=labels if labels is not None else "steelblue",
                cmap="tab10" if labels is not None else None,
                s=5, alpha=0.6,
            )
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        if labels is not None:
            fig.colorbar(sc, ax=axes[1], label="Label")
        fig.tight_layout()
        return fig

    def plot_training_trajectory(self, history: dict):
        import matplotlib.pyplot as plt
        keys = [k for k in history if k != "loss"]
        n_plots = 1 + len(keys)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
        if n_plots == 1:
            axes = [axes]

        axes[0].plot(history["loss"])
        axes[0].set_title("Training Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")

        for ax, key in zip(axes[1:], keys):
            ax.plot(history[key])
            ax.set_title(key)
            ax.set_xlabel(f"Eval step (every {len(history['loss']) // max(1, len(history[key]))} epochs)")

        fig.tight_layout()
        return fig

    def plot_knn_graph(self, X: np.ndarray, k: int = 5, highlight_hubs: np.ndarray | None = None):
        import matplotlib.pyplot as plt
        from embedkit.utils.neighbors import knn
        from embedkit.utils.validation import _to_numpy
        X_np = _to_numpy(X)
        emb = self._reduce(X_np)
        _, indices = knn(X_np, k)

        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ["steelblue"] * X_np.shape[0]
        if highlight_hubs is not None:
            for h in highlight_hubs:
                colors[h] = "red"

        for i, nbrs in enumerate(indices):
            for j in nbrs:
                ax.plot(
                    [emb[i, 0], emb[j, 0]],
                    [emb[i, 1], emb[j, 1]],
                    "k-", alpha=0.05, linewidth=0.5,
                )
        ax.scatter(emb[:, 0], emb[:, 1], c=colors, s=8, zorder=2)
        ax.set_title(f"kNN Graph (k={k})")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        return fig
