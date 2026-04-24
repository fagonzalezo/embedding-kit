"""AnalysisPlotter: visualize EmbedKitReport results."""

from __future__ import annotations

import numpy as np


class AnalysisPlotter:
    def __init__(self, report):
        self.report = report

    def plot_eigenvalue_spectrum(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ev = self.report.geometry.isotropy.eigenvalue_spectrum
        ax.plot(np.arange(1, len(ev) + 1), ev, marker="o", markersize=3)
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Eigenvalue Spectrum")
        ax.set_yscale("log")
        fig.tight_layout()
        return fig

    def plot_distance_histogram(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        dc = self.report.geometry.distance_concentration
        centers = (dc.bin_edges[:-1] + dc.bin_edges[1:]) / 2
        ax.bar(centers, dc.distance_histogram, width=np.diff(dc.bin_edges), alpha=0.7)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Pairwise Distance Distribution  "
            f"(contrast={dc.relative_contrast:.3f})"
        )
        fig.tight_layout()
        return fig

    def plot_k_occurrence(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        N_k = self.report.hubness.k_occurrence
        ax.hist(N_k, bins=30, alpha=0.7)
        ax.set_xlabel("k-occurrence (N_k)")
        ax.set_ylabel("Count")
        ax.set_title(f"k-Occurrence Distribution  (skewness={self.report.hubness.k_skewness:.2f})")
        fig.tight_layout()
        return fig

    def plot_kernel_spectrum(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        kev = self.report.kernel.eigenvalues
        ax.plot(np.arange(1, len(kev) + 1), kev, marker="o", markersize=3)
        ax.set_xlabel("Component")
        ax.set_ylabel("Eigenvalue")
        ax.set_title(
            f"Kernel Eigenvalue Spectrum  "
            f"(eff_rank={self.report.kernel.effective_rank:.1f})"
        )
        ax.set_yscale("log")
        fig.tight_layout()
        return fig

    def plot_id_local_map(self, X):
        import matplotlib.pyplot as plt
        from embedkit.analysis.intrinsic_dim import IntrinsicDimensionEstimator
        from embedkit.utils.validation import _to_numpy
        X_np = _to_numpy(X)
        est = IntrinsicDimensionEstimator(methods=["TwoNN"])
        result = est.fit(X_np)
        local = result.local_estimates.get("TwoNN")
        fig, ax = plt.subplots(figsize=(6, 4))
        if local is not None and len(local) > 0:
            ax.hist(local, bins=30, alpha=0.7)
            ax.set_xlabel("Local ID estimate")
            ax.set_ylabel("Count")
            ax.set_title("Per-point Intrinsic Dimension (TwoNN)")
        else:
            ax.text(0.5, 0.5, "No local ID estimates available", ha="center", va="center")
        fig.tight_layout()
        return fig

    def plot_hubness_map(self, X):
        import matplotlib.pyplot as plt
        from embedkit.utils.validation import _to_numpy
        X_np = _to_numpy(X)
        N_k = self.report.hubness.k_occurrence
        hubs = set(self.report.hubness.hubs.tolist())
        antihubs = set(self.report.hubness.antihubs.tolist())
        colors = []
        for i in range(X_np.shape[0]):
            if i in hubs:
                colors.append("red")
            elif i in antihubs:
                colors.append("blue")
            else:
                colors.append("gray")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(range(len(N_k)), N_k, c=colors, s=5, alpha=0.5)
        ax.set_xlabel("Point index")
        ax.set_ylabel("k-occurrence")
        ax.set_title("Hubness Map (red=hub, blue=antihub, gray=normal)")
        fig.tight_layout()
        return fig

    def plot_full_report(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        def _bar(ax, vals, labels, title):
            ax.bar(labels, vals, alpha=0.7)
            ax.set_title(title)

        # 1: eigenvalue spectrum
        ev = self.report.geometry.isotropy.eigenvalue_spectrum
        axes[0, 0].plot(ev, marker="o", markersize=2)
        axes[0, 0].set_title("Eigenvalue Spectrum")
        axes[0, 0].set_yscale("log")

        # 2: distance histogram
        dc = self.report.geometry.distance_concentration
        centers = (dc.bin_edges[:-1] + dc.bin_edges[1:]) / 2
        axes[0, 1].bar(centers, dc.distance_histogram, width=np.diff(dc.bin_edges), alpha=0.7)
        axes[0, 1].set_title("Distance Distribution")

        # 3: k-occurrence
        axes[0, 2].hist(self.report.hubness.k_occurrence, bins=20, alpha=0.7)
        axes[0, 2].set_title(f"k-Occurrence (skew={self.report.hubness.k_skewness:.1f})")

        # 4: kernel spectrum
        kev = self.report.kernel.eigenvalues
        axes[1, 0].plot(kev, marker="o", markersize=2)
        axes[1, 0].set_title("Kernel Spectrum")
        axes[1, 0].set_yscale("log")

        # 5: summary metrics
        names = ["PR", "Isotropy", "NC", "Hub ratio", "Antihub ratio"]
        vals = [
            self.report.geometry.isotropy.participation_ratio,
            self.report.geometry.isotropy.isotropy_score,
            self.report.geometry.neighbor_consistency.mean_consistency,
            self.report.hubness.hub_ratio,
            self.report.hubness.antihub_ratio,
        ]
        axes[1, 1].bar(names, vals, alpha=0.7)
        axes[1, 1].set_title("Key Metrics")
        axes[1, 1].tick_params(axis="x", rotation=30)

        # 6: severity text
        axes[1, 2].axis("off")
        axes[1, 2].text(
            0.1, 0.9,
            f"Severity: {self.report.severity.upper()}\n"
            f"ID: {self.report.intrinsic_dim.consensus:.1f}\n"
            f"Uniformity: {self.report.geometry.uniformity.uniformity:.3f}\n\n"
            + "\n".join(f"• {r[:60]}..." if len(r) > 60 else f"• {r}"
                        for r in self.report.recommendations[:4]),
            transform=axes[1, 2].transAxes,
            va="top",
            fontsize=8,
        )

        fig.suptitle("EmbedKit Full Report", fontsize=14)
        fig.tight_layout()
        return fig
