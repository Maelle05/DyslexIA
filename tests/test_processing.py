import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch

from dyslexia.processing.process_data_justine import get_files_path, process_dataset, split


def _write_metrics_csv(folder, sid, mean_fix=250.0, n_fix=20, mean_sacc=15.0):
    df = pd.DataFrame({
        "sid": [sid],
        "mean_fix_dur_trial": [mean_fix],
        "n_fix_trial": [n_fix],
        "mean_sacc_ampl_trial": [mean_sacc],
    })
    path = os.path.join(folder, f"Subject_{sid:03d}_T4_Meaningful_Text_metrics.csv")
    df.to_csv(path, index=False)
    return path


def _write_fixations_csv(folder, sid, n=30):
    rng = np.random.default_rng(sid)
    df = pd.DataFrame({
        "sid": [sid] * n,
        "duration_ms": rng.integers(100, 500, n).astype(float),
        "fix_x": rng.integers(0, 1920, n).astype(float),
        "fix_y": rng.integers(0, 1080, n).astype(float),
        "aoi_line": rng.integers(1, 8, n),
    })
    path = os.path.join(folder, f"Subject_{sid:03d}_T4_Meaningful_Text_fixations.csv")
    df.to_csv(path, index=False)
    return path


def _make_labels(sids):
    return pd.DataFrame({
        "subject_id": sids,
        "class_id": [i % 2 for i in sids],
    })


def _mock_read_csv_factory(labels_df):
    """Return a side-effect for pd.read_csv that injects a labels DataFrame."""
    real_read = pd.read_csv

    def _side_effect(path, **kwargs):
        if "label" in str(path):
            return labels_df
        return real_read(path, **kwargs)

    return _side_effect


class TestGetFilesPath:
    def test_finds_metrics_and_fixations(self, tmp_path):
        _write_metrics_csv(str(tmp_path), 1)
        _write_fixations_csv(str(tmp_path), 1)
        metrics, fixations = get_files_path(str(tmp_path))
        assert len(metrics) == 1
        assert len(fixations) == 1

    def test_ignores_non_matching_files(self, tmp_path):
        (tmp_path / "random.csv").write_text("x,y\n1,2\n")
        metrics, fixations = get_files_path(str(tmp_path))
        assert metrics == []
        assert fixations == []

    def test_empty_directory(self, tmp_path):
        metrics, fixations = get_files_path(str(tmp_path))
        assert metrics == [] and fixations == []

    def test_multiple_subjects(self, tmp_path):
        for sid in [1, 2, 5]:
            _write_metrics_csv(str(tmp_path), sid)
            _write_fixations_csv(str(tmp_path), sid)
        metrics, fixations = get_files_path(str(tmp_path))
        assert len(metrics) == 3
        assert len(fixations) == 3

    def test_paths_are_absolute(self, tmp_path):
        _write_metrics_csv(str(tmp_path), 1)
        metrics, _ = get_files_path(str(tmp_path))
        assert os.path.isabs(metrics[0])


class TestProcessDataset:
    def _build(self, tmp_path, sids):
        for sid in sids:
            _write_metrics_csv(str(tmp_path), sid)
            _write_fixations_csv(str(tmp_path), sid)
        return get_files_path(str(tmp_path))

    def test_returns_dataframe(self, tmp_path):
        metrics, fixations = self._build(tmp_path, [1, 2])
        labels = _make_labels([1, 2])
        with patch("dyslexia.processing.process_data_justine.pd.read_csv",
                   side_effect=_mock_read_csv_factory(labels)):
            ds = process_dataset(metrics, fixations)
        assert isinstance(ds, pd.DataFrame)

    def test_expected_columns_present(self, tmp_path):
        metrics, fixations = self._build(tmp_path, [1, 2])
        labels = _make_labels([1, 2])
        with patch("dyslexia.processing.process_data_justine.pd.read_csv",
                   side_effect=_mock_read_csv_factory(labels)):
            ds = process_dataset(metrics, fixations)
        expected = {
            "sid", "mean_fix_dur_trial", "n_fix_trial", "mean_sacc_ampl_trial",
            "std_fixation_duration", "mean_fixation_x", "mean_fixation_y",
            "dispersion_x", "dispersion_y", "fixation_per_line_ratio", "class_id",
        }
        assert expected.issubset(set(ds.columns))

    def test_row_count_matches_subjects(self, tmp_path):
        sids = [1, 2, 3]
        metrics, fixations = self._build(tmp_path, sids)
        labels = _make_labels(sids)
        with patch("dyslexia.processing.process_data_justine.pd.read_csv",
                   side_effect=_mock_read_csv_factory(labels)):
            ds = process_dataset(metrics, fixations)
        assert len(ds) == 3

    def test_n_fix_trial_is_integer(self, tmp_path):
        metrics, fixations = self._build(tmp_path, [1, 2])
        labels = _make_labels([1, 2])
        with patch("dyslexia.processing.process_data_justine.pd.read_csv",
                   side_effect=_mock_read_csv_factory(labels)):
            ds = process_dataset(metrics, fixations)
        assert ds["n_fix_trial"].dtype == int

    def test_n_lines_fixated_column_dropped(self, tmp_path):
        metrics, fixations = self._build(tmp_path, [1, 2])
        labels = _make_labels([1, 2])
        with patch("dyslexia.processing.process_data_justine.pd.read_csv",
                   side_effect=_mock_read_csv_factory(labels)):
            ds = process_dataset(metrics, fixations)
        assert "n_lines_fixated" not in ds.columns


class TestSplit:
    def _make_dataset(self, n=20):
        rng = np.random.default_rng(42)
        return pd.DataFrame({
            "sid": range(n),
            "mean_fix_dur_trial": rng.uniform(100, 400, n),
            "n_fix_trial": rng.integers(5, 30, n),
            "mean_sacc_ampl_trial": rng.uniform(10, 50, n),
            "std_fixation_duration": rng.uniform(10, 80, n),
            "mean_fixation_x": rng.uniform(0, 1920, n),
            "mean_fixation_y": rng.uniform(0, 1080, n),
            "dispersion_x": rng.uniform(0, 200, n),
            "dispersion_y": rng.uniform(0, 200, n),
            "fixation_per_line_ratio": rng.uniform(1, 5, n),
            "class_id": [i % 2 for i in range(n)],
        })

    def test_returns_four_arrays(self):
        result = split(self._make_dataset())
        assert len(result) == 4

    def test_total_sample_count_preserved(self):
        ds = self._make_dataset(n=20)
        X_train, X_test, y_train, y_test = split(ds)
        assert X_train.shape[0] + X_test.shape[0] == 20

    def test_test_size_is_20_percent(self):
        ds = self._make_dataset(n=100)
        _, X_test, _, _ = split(ds)
        assert X_test.shape[0] == 20

    def test_label_arrays_match_feature_arrays(self):
        X_train, X_test, y_train, y_test = split(self._make_dataset())
        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)

    def test_sid_and_class_id_excluded_from_features(self):
        ds = self._make_dataset(n=20)
        n_feature_cols = len(ds.columns) - 2  # drop sid and class_id
        X_train, _, _, _ = split(ds)
        assert X_train.shape[1] == n_feature_cols
