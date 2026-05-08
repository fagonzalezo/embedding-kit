"""Integration tests for EmbedKit high-level API."""

import numpy as np
import pytest

from embedkit.utils.neighbors import clear_cache


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


class TestEmbedKit:
    def test_self_supervised_fit_transform(self, small_X):
        from embedkit import EmbedKit
        X_refined = EmbedKit(mode="self_supervised", epochs=3, eval_every=100).fit_transform(small_X)
        assert X_refined.shape[0] == small_X.shape[0]
        assert X_refined.shape[1] > 0

    def test_supervised_fit_transform(self, small_Xy):
        from embedkit import EmbedKit
        X, y = small_Xy
        X_refined = EmbedKit(mode="supervised", epochs=3, eval_every=100).fit_transform(X, y)
        assert X_refined.shape[0] == X.shape[0]

    def test_analysis_report_stored(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit(epochs=2, eval_every=100)
        ek.fit(small_X)
        assert ek.analysis_report is not None
        assert ek.analysis_report.severity in ("low", "medium", "high")

    def test_explicit_target_dim(self, small_X):
        from embedkit import EmbedKit
        X_refined = EmbedKit(target_dim=4, epochs=2, eval_every=100).fit_transform(small_X)
        assert X_refined.shape[1] == 4

    def test_save_load(self, small_X, tmp_path):
        from embedkit import EmbedKit
        ek = EmbedKit(target_dim=8, epochs=2, eval_every=100)
        ek.fit(small_X)
        ek.save(tmp_path / "bundle")
        ek2 = EmbedKit.load(tmp_path / "bundle")
        Z = ek2.transform(small_X)
        assert Z.shape == (small_X.shape[0], 8)

    def test_transform_before_fit_raises(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit()
        with pytest.raises(RuntimeError):
            ek.transform(small_X)

