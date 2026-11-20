"""Integration tests for EmbedKit high-level API."""

import pytest


class TestEmbedKit:
    def test_self_supervised_fit_transform(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit(mode="self_supervised", method="refiner", epochs=3, eval_every=100)
        X_refined = ek.fit_transform(small_X)
        assert X_refined.shape[0] == small_X.shape[0]
        assert X_refined.shape[1] > 0

    def test_supervised_fit_transform(self, small_Xy):
        from embedkit import EmbedKit
        X, y = small_Xy
        X_refined = EmbedKit(mode="supervised", epochs=3, eval_every=100).fit_transform(X, y)
        assert X_refined.shape[0] == X.shape[0]

    def test_analysis_report_stored(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit(method="pca")
        ek.fit(small_X)
        assert ek.analysis_report is not None
        # EmbedKit reads only suggested_target_dim, so it skips the null calibration.
        assert ek.analysis_report.calibration is None
        # method="auto" reads no report, so it computes none.
        assert EmbedKit().fit(small_X).analysis_report is None

    def test_no_analysis_report_with_explicit_target_dim(self, small_X):
        # The report is read only for target_dim="auto", so an explicit target_dim skips it.
        from embedkit import EmbedKit
        assert EmbedKit(method="pca", target_dim=8).fit(small_X).analysis_report is None
        ek = EmbedKit(method="refiner", target_dim=8, epochs=1, eval_every=100).fit(small_X)
        assert ek.analysis_report is None
        assert ek.transform(small_X).shape == (small_X.shape[0], 8)

    def test_explicit_target_dim(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit(method="refiner", target_dim=4, epochs=2, eval_every=100)
        X_refined = ek.fit_transform(small_X)
        assert X_refined.shape[1] == 4

    def test_save_load(self, small_X, tmp_path):
        from embedkit import EmbedKit
        ek = EmbedKit(method="refiner", target_dim=8, epochs=2, eval_every=100)
        ek.fit(small_X)
        ek.save(tmp_path / "bundle")
        ek2 = EmbedKit.load(tmp_path / "bundle")
        Z = ek2.transform(small_X)
        assert Z.shape == (small_X.shape[0], 8)

    def test_no_early_stopping_by_default(self, small_X):
        # E2: no label-free monitor tracks test accuracy, so the default is a fixed epoch budget
        # trained on every row.
        from embedkit import EmbedKit
        ek = EmbedKit(method="refiner", target_dim=4, epochs=4, eval_every=1,
                      device="cpu").fit(small_X)
        assert ek._trainer.early_stopping_patience is None
        assert ek._trainer.val_split == 0.0
        assert ek._trainer.stopped_epoch_ == 4

    def test_early_stopping_holds_out_validation(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit(method="refiner", target_dim=4, epochs=2, eval_every=1,
                      early_stopping_patience=5,
                      device="cpu").fit(small_X)
        assert ek._trainer.val_split == 0.1
        assert EmbedKit(early_stopping_patience=5, val_split=0.2)._resolve_val_split() == 0.2

    def test_transform_before_fit_raises(self, small_X):
        from embedkit import EmbedKit
        ek = EmbedKit()
        with pytest.raises(RuntimeError):
            ek.transform(small_X)

