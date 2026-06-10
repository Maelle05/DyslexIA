"""
Pipeline de normalisation et extraction de features pour la prédiction de dyslexie.
Sources : Dataset1 (Tobii 250Hz, pixels), Dataset2 (50Hz, unités propriétaires),
          Application locale (angles en degrés, ~25-40Hz).

Usage:
    from dyslexia_pipeline import load_dataset1, load_dataset2, load_app, extract_features

    df = load_dataset1("Subject_1003_T1_Syllables_raw.csv")
    features = extract_features(df, subject_id=1003, label=1, source='ds1')
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# 1. CHARGEMENT ET NETTOYAGE PAR SOURCE
# ══════════════════════════════════════════════════════════════

def load_dataset1(filepath):
    """
    Dataset1 — Tobii, 250 Hz, coordonnées en pixels (résolution ~1557×1066).
    Colonnes attendues : time (µs), gaze_x/y_left, gaze_x/y_right,
                         pupil_left, pupil_right, stimfile, subject_id.
    """
    df = pd.read_csv(filepath)

    df['t_s'] = (df['time'] - df['time'].iloc[0]) / 1e6   # µs → secondes relatives

    df['x'] = (df['gaze_x_left']  + df['gaze_x_right'])  / 2
    df['y'] = (df['gaze_y_left']  + df['gaze_y_right'])  / 2
    df['x_left'] = df['gaze_x_left']
    df['y_left'] = df['gaze_y_left']
    df['x_right'] = df['gaze_x_right']
    df['y_right'] = df['gaze_y_right']
    # df['pupil'] = (df['pupil_left'] + df['pupil_right']) / 2

    df['x_left']  = (df['x_left']  - df['x_left'].min())  / (df['x_left'].max()  - df['x_left'].min())
    df['y_left']  = (df['y_left']  - df['y_left'].min())  / (df['y_left'].max()  - df['y_left'].min())
    df['x_right'] = (df['x_right'] - df['x_right'].min()) / (df['x_right'].max() - df['x_right'].min())
    df['y_right'] = (df['y_right'] - df['y_right'].min()) / (df['y_right'].max() - df['y_right'].min())

    return df[['t_s', 'x', 'y', 'x_left', 'y_left', 'x_right', 'y_right']].reset_index(drop=True)

def load_dataset2(filepath):
    """
    Dataset2 — EyeLink/SMI, 50 Hz, coordonnées en unités propriétaires
    (environ 60× les degrés d'angle visuel).
    Colonnes attendues : T (ms), LX, LY, RX, RY.
    Blinks : |LX| < 1 ET |LY| < 1.
    """
    df = pd.read_csv(filepath)

    df['t_s'] = df['T'] / 1000   # ms → secondes

    df['x'] = (df['LX'] + df['RX']) / 2
    df['y'] = (df['LY'] + df['RY']) / 2
    df['x_left'] = df['LX']
    df['y_left'] = df['LY']
    df['x_right'] = df['RX']
    df['y_right'] = df['RY']
    # df['pupil'] = np.nan   # non disponible dans ce dataset

    df['x_left']  = (df['x_left']  - df['x_left'].min())  / (df['x_left'].max()  - df['x_left'].min())
    df['y_left']  = (df['y_left']  - df['y_left'].min())  / (df['y_left'].max()  - df['y_left'].min())
    df['x_right'] = (df['x_right'] - df['x_right'].min()) / (df['x_right'].max() - df['x_right'].min())
    df['y_right'] = (df['y_right'] - df['y_right'].min()) / (df['y_right'].max() - df['y_right'].min())

    return df[['t_s', 'x', 'y', 'x_left', 'y_left', 'x_right', 'y_right']].reset_index(drop=True)

def load_app(filepath):
    """
    Application locale — ~25-40 Hz.
    Colonnes attendues : time (Unix s), angle1_l/r (°H), angle2_l/r (°V),
                         gaze_x/y_left/right (pixels), fix_x, fix_y.
    Résolution écran : 1366×768.
    Blinks/pertes : valeurs aux limites de l'écran (≥1350 ou ≥750 px).
    """
    df = pd.read_csv(filepath)

    df['t_s'] = df['time'] - df['time'].iloc[0]   # secondes relatives

    if 'angle1_l' in df.columns:
        # Format enrichi : utiliser les angles (dynamiquement calculés par le tracker)
        valid = (df['gaze_x_left']  < 1350) & (df['gaze_y_left']  < 750) & \
                (df['gaze_x_right'] < 1350) & (df['gaze_y_right'] < 750) & \
                (df['gaze_x_left']  > 20)   & (df['gaze_y_left']  > 20)
        df = df[valid].copy()
        df['x'] = (df['angle1_l'] + df['angle1_r']) / 2   # degrés horizontaux
        df['y'] = (df['angle2_l'] + df['angle2_r']) / 2   # degrés verticaux
        df['x_left'] = df['angle1_l']
        df['y_left'] = df['angle2_l']
        df['x_right'] = df['angle1_r']
        df['y_right'] = df['angle2_r']
    else:
        # Format basique : pixels uniquement (ancien format)
        valid = (df['gaze_x_left']  < 1350) & (df['gaze_y_left']  < 750) & \
                (df['gaze_x_right'] < 1350) & (df['gaze_y_right'] < 750)
        df = df[valid].copy()
        df['x'] = (df['gaze_x_left']  + df['gaze_x_right']) / 2 / 1366
        df['y'] = (df['gaze_y_left'] + df['gaze_y_right'])  / 2 / 768
        df['x_left'] = df['gaze_x_left']
        df['y_left'] = df['gaze_y_left']
        df['x_right'] = df['gaze_x_right']
        df['y_right'] = df['gaze_y_right']

    # df['pupil'] = np.nan

    return df[['t_s', 'x', 'y', 'x_left', 'y_left', 'x_right', 'y_right']].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 2. EXTRACTION DE FEATURES ROBUSTES (sans unité, comparables)
# ══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'saccade_rate',        # proportion de points en saccade
    'fixation_prop',       # proportion de points en fixation
    'regression_rate',     # part des saccades vers la gauche (régressions)
    'reg_fwd_ratio',       # amplitude régression / amplitude progression
    'vel_cv',              # coefficient de variation de la vélocité
    'x_spread_ratio',      # étalement horizontal relatif (std / range)
    'y_spread_ratio',      # étalement vertical relatif
    'y_drift',             # dérive verticale 1ère→2ème moitié de session
    'saccade_regularity',  # irrégularité de l'espacement des saccades
]


def extract_features(df, subject_id=None, label=None, source=None):
    """
    Calcule les 9 features comportementales à partir d'un DataFrame normalisé
    (colonnes : t_s, x, y).

    Toutes les features sont des ratios ou coefficients sans unité,
    directement comparables entre DS1, DS2 et l'application.

    Paramètres
    ----------
    df         : DataFrame retourné par load_dataset1/2/app
    subject_id : identifiant du sujet (optionnel)
    label      : 1 = dyslexique, 0 = contrôle, None = inconnu
    source     : 'ds1', 'ds2' ou 'app'

    Retourne
    --------
    dict avec les features + métadonnées (subject_id, dyslexia, source,
    duration_s, n_samples) ou None si données insuffisantes.
    """
    x = df['x'].values
    y = df['y'].values
    t = df['t_s'].values

    if len(x) < 10:
        return None

    # ── Vélocité instantanée ──────────────────────────────────
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)
    dt = np.where(dt < 1e-6, 1e-6, dt)
    raw_vel = np.sqrt(dx**2 + dy**2) / dt

    # Clipper les artefacts (>99.5e percentile = transitions inter-textes, clignements résiduels)
    vel_ceiling = np.percentile(raw_vel, 99.5)
    velocity = np.clip(raw_vel, 0, vel_ceiling)

    # ── Saccades vs fixations ─────────────────────────────────
    # Seuil adaptatif : 75e percentile de la session (robuste aux différences d'unité)
    v_thresh    = np.percentile(velocity, 75)
    is_saccade  = velocity > v_thresh
    is_fixation = ~is_saccade

    # ── Directions ───────────────────────────────────────────
    regression_mask = (dx < 0) & is_saccade   # vers la gauche
    forward_mask    = (dx > 0) & is_saccade   # vers la droite

    n_saccades   = int(np.sum(is_saccade))
    n_regression = int(np.sum(regression_mask))
    n_forward    = int(np.sum(forward_mask))

    fwd_amp = np.mean(np.abs(dx[forward_mask]))    if n_forward    > 0 else 1e-9
    reg_amp = np.mean(np.abs(dx[regression_mask])) if n_regression > 0 else 0.0

    vel_mean   = float(np.mean(velocity))
    vel_std    = float(np.std(velocity))

    # ── Irrégularité des saccades ─────────────────────────────
    saccade_idx = np.where(is_saccade)[0]
    if len(saccade_idx) > 2:
        ipi = np.diff(saccade_idx)   # inter-pulse intervals
        saccade_reg = float(np.std(ipi) / max(np.mean(ipi), 1))
    else:
        saccade_reg = 0.0

    # ── Dérive verticale ──────────────────────────────────────
    mid = len(y) // 2
    y_range = float(np.ptp(y))
    y_drift = float((np.mean(y[mid:]) - np.mean(y[:mid])) / (y_range + 1e-9))

    # ── Assemblage ────────────────────────────────────────────
    feat = {
        'saccade_rate':       float(np.mean(is_saccade)),
        'fixation_prop':      float(np.mean(is_fixation)),
        'regression_rate':    float(n_regression / max(n_saccades, 1)),
        'reg_fwd_ratio':      float(reg_amp / fwd_amp),
        'vel_cv':             float(vel_std / vel_mean if vel_mean > 0 else 0),
        'x_spread_ratio':     float(np.std(x) / (np.ptp(x) + 1e-9)),
        'y_spread_ratio':     float(np.std(y) / (y_range + 1e-9)),
        'y_drift':            y_drift,
        'saccade_regularity': saccade_reg,
        # Métadonnées
        'duration_s':  float(t[-1] - t[0]),
        'n_samples':   len(x),
    }

    if subject_id is not None: feat['subject_id'] = subject_id
    if label      is not None: feat['dyslexia']   = label
    if source     is not None: feat['source']      = source

    return feat


# ══════════════════════════════════════════════════════════════
# 3. DÉMONSTRATION / TEST RAPIDE
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("TEST DU PIPELINE SUR LES FICHIERS D'EXEMPLE")
    print("=" * 60)

    rows = []

    df1 = load_dataset1('/mnt/user-data/uploads/Subject_1003_T1_Syllables_raw.csv')
    f1  = extract_features(df1, subject_id=1003, label=None, source='ds1')
    rows.append(f1)
    print(f"DS1  | {len(df1):5d} échantillons valides sur {18247} bruts")

    df2 = load_dataset2('/mnt/user-data/uploads/111GM3.csv')
    f2  = extract_features(df2, subject_id='111GM3', label=None, source='ds2')
    rows.append(f2)
    print(f"DS2  | {len(df2):5d} échantillons valides sur {1499} bruts")

    app = load_app('/mnt/user-data/uploads/gaze_log_1_.csv')
    f3  = extract_features(app, subject_id='app_001', label=None, source='app')
    rows.append(f3)
    print(f"App  | {len(app):5d} échantillons valides sur {419} bruts")

    print(f"\n{'Feature':<24} {'DS1':>10} {'DS2':>10} {'App':>10}  {'Comparable?'}")
    print("─" * 70)
    thresholds = {   # seuils empiriques d'écart acceptable entre sources
        'saccade_rate': 0.05, 'fixation_prop': 0.05,
        'regression_rate': 0.15, 'reg_fwd_ratio': 0.5,
        'vel_cv': 0.5, 'x_spread_ratio': 0.1, 'y_spread_ratio': 0.1,
        'y_drift': 0.5, 'saccade_regularity': 0.8,
    }
    for k in FEATURE_NAMES:
        v1, v2, v3 = f1[k], f2[k], f3[k]
        spread = max(v1, v2, v3) - min(v1, v2, v3)
        ok = "✓" if spread < thresholds.get(k, 1.0) else "⚠"
        print(f"{k:<24} {v1:>10.3f} {v2:>10.3f} {v3:>10.3f}  {ok}")

    print("\nProchaine étape : appliquer ce pipeline aux 255 sujets étiquetés")
    print("puis entraîner un RandomForest ou XGBoost sur les features extraites.")
