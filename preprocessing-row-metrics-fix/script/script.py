"""
Eye Tracking Data Processor
----------------------------
Transforme raw.csv (time, fix_x, fix_y, gaze_x_left, gaze_y_left, gaze_x_right, gaze_y_right)
en deux fichiers :
  - fixations.csv  : duration_ms, fix_x, fix_y
  - metrics.csv    : mean_fix_dur_trial, mean_sacc_ampl_trial, n_fix_trial
"""

import pandas as pd
import numpy as np
import sys
import os

# ──────────────────────────────────────────────
# PARAMÈTRES
# ──────────────────────────────────────────────
INPUT_FILE   = "raw.csv"          # fichier d'entrée
MIN_FIX_DUR  = 80                 # durée minimale d'une fixation (ms)
DISP_THRESH  = 25                 # seuil de dispersion (pixels) pour I-DT


def load_data(path: str) -> pd.DataFrame:
    """Charge le CSV et vérifie les colonnes attendues."""
    required = {"time", "fix_x", "fix_y", "gaze_x_left", "gaze_y_left",
                "gaze_x_right", "gaze_y_right"}
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def detect_fixations_idt(df: pd.DataFrame,
                          disp_thresh: float = DISP_THRESH,
                          min_dur: float = MIN_FIX_DUR) -> pd.DataFrame:
    """
    Détection des fixations par I-DT (Identification by Dispersion Threshold).

    Une fenêtre glissante est étendue tant que la dispersion des points
    (max_x - min_x + max_y - min_y) reste sous disp_thresh.
    Si la durée de la fenêtre >= min_dur, on enregistre une fixation.

    Retourne un DataFrame avec : duration_ms, fix_x, fix_y
    """
    times  = df["time"].values.astype(float)
    x_vals = df["fix_x"].values.astype(float)
    y_vals = df["fix_y"].values.astype(float)

    fixations = []
    i = 0
    n = len(df)

    while i < n:
        j = i
        # Étendre la fenêtre
        while j < n:
            window_x = x_vals[i:j+1]
            window_y = y_vals[i:j+1]
            disp = (window_x.max() - window_x.min()) + \
                   (window_y.max() - window_y.min())
            if disp > disp_thresh:
                break
            j += 1

        window_dur = times[j-1] - times[i]

        if window_dur >= min_dur and j > i:
            cx = x_vals[i:j].mean()
            cy = y_vals[i:j].mean()
            fixations.append({
                "duration_ms": round(window_dur, 2),
                "fix_x":       round(cx, 2),
                "fix_y":       round(cy, 2),
            })
            i = j          # avancer après la fixation
        else:
            i += 1         # point isolé → saccade / bruit

    return pd.DataFrame(fixations)


def euclidean(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def compute_metrics(fixations: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les métriques globales (1 ligne = 1 trial).
      - mean_fix_dur_trial  : durée moyenne des fixations (ms)
      - mean_sacc_ampl_trial: amplitude moyenne des saccades (pixels)
      - n_fix_trial         : nombre de fixations
    """
    if fixations.empty:
        return pd.DataFrame([{
            "mean_fix_dur_trial":   None,
            "mean_sacc_ampl_trial": None,
            "n_fix_trial":          0,
        }])

    mean_fix_dur = fixations["duration_ms"].mean()
    n_fix        = len(fixations)

    # Amplitude des saccades = distance entre fixations consécutives
    if n_fix > 1:
        dx = fixations["fix_x"].diff().dropna()
        dy = fixations["fix_y"].diff().dropna()
        amplitudes = np.sqrt(dx**2 + dy**2)
        mean_sacc_ampl = amplitudes.mean()
    else:
        mean_sacc_ampl = None

    return pd.DataFrame([{
        "mean_fix_dur_trial":   round(mean_fix_dur, 4),
        "mean_sacc_ampl_trial": round(mean_sacc_ampl, 4) if mean_sacc_ampl is not None else None,
        "n_fix_trial":          n_fix,
    }])


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE

    if not os.path.exists(input_path):
        print(f"[ERREUR] Fichier introuvable : {input_path}")
        sys.exit(1)

    print(f"[1/4] Chargement de {input_path}...")
    df = load_data(input_path)
    print(f"      {len(df)} échantillons chargés.")

    print("[2/4] Détection des fixations (I-DT)...")
    fixations = detect_fixations_idt(df)
    print(f"      {len(fixations)} fixations détectées.")

    print("[3/4] Calcul des métriques...")
    metrics = compute_metrics(fixations)

    out_dir = os.path.dirname(input_path) or "."
    fix_path     = os.path.join(out_dir, "fixations.csv")
    metrics_path = os.path.join(out_dir, "metrics.csv")

    fixations.to_csv(fix_path,     index=False)
    metrics.to_csv(metrics_path,   index=False)

    print(f"[4/4] Fichiers générés :")
    print(f"      → {fix_path}")
    print(f"      → {metrics_path}")
    print()
    print("── Métriques ──────────────────────────────")
    print(metrics.to_string(index=False))
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
