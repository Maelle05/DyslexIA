import pandas as pd
import I2MC
import numpy as np

def raw_to_metrics(raw_df):
    df_row = raw_df.copy()
    df_row['time_ms'] = (df_row['time'] - df_row['time'][0]) / 1000
    freq = 1000 / df_row["time_ms"].diff().median()
    # print(f"FREQ : {freq:.1f} Hz")
    FREQ_HZ      = freq

    min_fix_ms = max(40, 2 * (1000 / FREQ_HZ))
    # print(f"MIN_FIX_MS : {min_fix_ms}ms")
    MIN_FIX_MS   = min_fix_ms

    SCREEN_X     = 1920
    SCREEN_Y     = 1080
    MISSING_VAL  = -32768

    # Merges two fixations separated by a micro-stutter if they are less than N pixels apart
    MAX_MERGE_DIST = 30
    MAX_MERGE_TIME = 30

    gaze = pd.DataFrame({
        'time': df_row['time_ms'],
        'L_X' : df_row['gaze_x_left'],
        'L_Y' : df_row['gaze_y_left'],
        'R_X' : df_row['gaze_x_right'],
        'R_Y' : df_row['gaze_y_right'],
    })
    options = {
        'xres'        : SCREEN_X,
        'yres'        : SCREEN_Y,
        'freq'        : FREQ_HZ,
        'missingx'    : MISSING_VAL,
        'missingy'    : MISSING_VAL,
        'minFixDur'   : MIN_FIX_MS,
        'maxMergeDist': MAX_MERGE_DIST,
        'maxMergeTime': MAX_MERGE_TIME,
    }
    result = I2MC.I2MC(gaze, options, logging=False)
    fix, _, _ = result
    df_fix = pd.DataFrame(fix)
    df_fix = df_fix[['dur', 'xpos', 'ypos']]
    df_fix = df_fix.rename(columns={"dur": "duration_ms", "xpos": "fix_x", "ypos": "fix_y"})

    mean_fix_dur = df_fix["duration_ms"].mean()

    # ── Nombre de fixations ──────────────────────────────────────────────
    n_fix = len(df_fix)

    # ── Amplitude moyenne des saccades ───────────────────────────────────
    # Une saccade = vecteur entre la fin d'une fixation et le début de la suivante
    # On approxime par la distance euclidienne entre centroïdes consécutifs
    if n_fix > 1:
        dx = df_fix["fix_x"].diff().dropna()
        dy = df_fix["fix_y"].diff().dropna()
        amplitudes = np.sqrt(dx**2 + dy**2)
        mean_sacc_ampl = amplitudes.mean()
    else:
        mean_sacc_ampl = None   # pas de saccade avec une seule fixation

    return pd.DataFrame([{
        "mean_fix_dur_trial":   round(mean_fix_dur,   6),
        "n_fix_trial":          n_fix,
        "mean_sacc_ampl_trial": round(mean_sacc_ampl, 6) if mean_sacc_ampl is not None else None,
    }])
