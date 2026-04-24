"""Smoke tests for visualization modules."""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import numpy as np
import pytest

from embedkit.utils.neighbors import clear_cache


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture(scope="module")
def report(small_X):
    from embedkit.analysis.report import EmbedKitAnalyzer
    return EmbedKitAnalyzer(id_methods=["TwoNN"]).fit(small_X)


class TestAnalysisPlotter:
    def test_eigenvalue_spectrum(self, report):
        from embedkit.visualization.plots import AnalysisPlotter
        import matplotlib.pyplot as plt
        plotter = AnalysisPlotter(report)
        fig = plotter.plot_eigenvalue_spectrum()
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_distance_histogram(self, report):
        from embedkit.visualization.plots import AnalysisPlotter
        import matplotlib.pyplot as plt
        plotter = AnalysisPlotter(report)
        fig = plotter.plot_distance_histogram()
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_k_occurrence(self, report):
        from embedkit.visualization.plots import AnalysisPlotter
        import matplotlib.pyplot as plt
        plotter = AnalysisPlotter(report)
        fig = plotter.plot_k_occurrence()
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_kernel_spectrum(self, report):
        from embedkit.visualization.plots import AnalysisPlotter
        import matplotlib.pyplot as plt
        plotter = AnalysisPlotter(report)
        fig = plotter.plot_kernel_spectrum()
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_full_report(self, report):
        from embedkit.visualization.plots import AnalysisPlotter
        import matplotlib.pyplot as plt
        plotter = AnalysisPlotter(report)
        fig = plotter.plot_full_report()
        assert hasattr(fig, "savefig")
        plt.close(fig)


class TestEmbeddingVisualizer:
    def test_plot_comparison(self, small_X):
        from embedkit.visualization.embedding_viz import EmbeddingVisualizer
        import matplotlib.pyplot as plt
        viz = EmbeddingVisualizer(method="pca", random_state=0)
        X_b = small_X
        X_a = small_X + 0.1
        fig = viz.plot_comparison(X_b, X_a)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_training_trajectory(self):
        from embedkit.visualization.embedding_viz import EmbeddingVisualizer
        import matplotlib.pyplot as plt
        history = {"loss": [1.0, 0.8, 0.6], "uniformity": [-0.5, -0.6]}
        viz = EmbeddingVisualizer(method="pca")
        fig = viz.plot_training_trajectory(history)
        assert hasattr(fig, "savefig")
        plt.close(fig)

    def test_plot_knn_graph(self, small_X):
        from embedkit.visualization.embedding_viz import EmbeddingVisualizer
        import matplotlib.pyplot as plt
        viz = EmbeddingVisualizer(method="pca", random_state=0)
        fig = viz.plot_knn_graph(small_X, k=3)
        assert hasattr(fig, "savefig")
        plt.close(fig)
