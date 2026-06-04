import pandas as pd
import numpy as np
import os
import re
from sklearn.model_selection import train_test_split

def get_files_path(folder_path):
    """Scan a directory for per-subject eye-tracking CSVs.

    Args:
        folder_path: Path to the directory containing Subject CSV files.

    Returns:
        A tuple (files_metrics, files_fixations) where each element is a list
        of absolute file paths matching the naming convention
        ``Subject_<N>_T4_Meaningful_Text_metrics.csv`` or
        ``Subject_<N>_T4_Meaningful_Text_fixations.csv``.
    """
    files_metrics = []
    files_fixations = []

    for filename in os.listdir(folder_path):
        if re.match(r"Subject_\d+_T4_Meaningful_Text_metrics\.csv", filename):
            files_metrics.append(os.path.join(folder_path, filename))
        elif re.match(r"Subject_\d+_T4_Meaningful_Text_fixations\.csv", filename):
            files_fixations.append(os.path.join(folder_path, filename))

    return files_metrics, files_fixations

def process_dataset(files_metrics, files_fixations):
    """Build a labelled feature DataFrame from raw metrics and fixation CSVs.

    Joins fixation-level statistics (duration std, spatial dispersion, line
    coverage) with trial-level metrics (mean fixation duration, saccade
    amplitude, fixation count) and then merges the class labels from
    ``../../data/dyslexia_class_label.csv``.

    Args:
        files_metrics: List of paths to ``*_metrics.csv`` files.
        files_fixations: List of paths to ``*_fixations.csv`` files.

    Returns:
        A ``pd.DataFrame`` with one row per subject, containing feature columns
        and a ``class_id`` column (0 = non-dyslexic, 1 = dyslexic).
    """
    fix_dict = {}

    for f in files_fixations:
        df_f = pd.read_csv(f)
        sid = int(df_f["sid"].iloc[0])

        fix_dict[sid] = {
            "std_fixation_duration": df_f["duration_ms"].std(),
            "mean_fixation_x": df_f["fix_x"].mean(),
            "mean_fixation_y": df_f["fix_y"].mean(),
            "dispersion_x" : df_f["fix_x"].std(),
            "dispersion_y" : df_f["fix_y"].std(),
            "n_lines_fixated": df_f["aoi_line"].nunique()
        }


    rows = []

    for f in files_metrics:
        df_m = pd.read_csv(f)
        sid = int(df_m["sid"].iloc[0])

        row = {"sid": sid}
        row.update(df_m[["mean_fix_dur_trial", "n_fix_trial", "mean_sacc_ampl_trial"]].iloc[0].to_dict())
        row.update(fix_dict.get(sid, {}))

        row["fixation_per_line_ratio"] = row.get("n_fix_trial") / row.get("n_lines_fixated")

        rows.append(row)

    dataset = pd.DataFrame(rows)

    dataset["n_fix_trial"] = dataset["n_fix_trial"].astype(int)
    dataset = dataset.drop(columns=["n_lines_fixated"])

    labels = pd.read_csv("../../data/dyslexia_class_label.csv")

    dataset = dataset.merge(labels[["subject_id", "class_id"]], left_on="sid", right_on="subject_id").drop(columns=["subject_id"])

    return dataset

def split(dataset):
    """Stratified train/test split of a feature dataset.

    Args:
        dataset: DataFrame returned by :func:`process_dataset`. Must contain
            ``sid`` and ``class_id`` columns.

    Returns:
        A tuple ``(X_train, X_test, y_train, y_test)`` with an 80/20 split
        stratified by ``class_id``.
    """
    X = dataset.drop(columns=["sid", "class_id"]).values
    y = dataset["class_id"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y)

    return X_train, X_test, y_train, y_test
