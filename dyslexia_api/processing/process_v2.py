"""Signal processing helpers for gaze data.

Provides routines for fixed-length interpolation, feature extraction from
binocular gaze traces, and conversion of gaze logs into model-ready feature
vectors.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.fft import fft

pd.set_option('display.max_columns', None)

TARGET_LEN = 2000
GAZE_COLS = ['x_left', 'y_left', 'x_right', 'y_right']


def to_fixed_length(signal, target_len):
    """Interpolate a signal to a fixed number of samples.

    Args:
        signal: 1D numeric sequence representing a gaze trace.
        target_len: Desired output length.

    Returns:
        A numpy array of length ``target_len`` with linearly interpolated values.
    """
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, target_len)
    return interp1d(x_old, signal, kind='linear')(x_new)


def extract_features(df, gaze_cols=GAZE_COLS, target_len=TARGET_LEN):
    """Compute a feature vector from raw gaze coordinates.

    Args:
        df: DataFrame containing gaze log columns.
        gaze_cols: List of gaze coordinate column names to process.
        target_len: Number of samples used for fixed-length interpolation.

    Returns:
        A numpy array containing binocular divergence, vertical velocity
        percentile, horizontal saccade rate, and high-frequency energy.
    """
    signals = {}
    for col in gaze_cols:
        signals[col] = to_fixed_length(df[col].values.astype(float), target_len)

    # Divergence binoculaire moyenne — écart horizontal gauche/droite
    mean_binocular_divergence = np.mean(np.abs(signals['x_left'] - signals['x_right']))

    # Percentile 90 de la vitesse verticale — y_left
    vel_yl = np.abs(np.diff(signals['y_left']))
    p90_vertical_velocity_left = np.percentile(vel_yl, 90)

    # Taux de saccades horizontales — x_left
    vel_xl = np.abs(np.diff(signals['x_left']))
    saccade_rate_x_left = np.sum(vel_xl > np.mean(vel_xl) + 2*np.std(vel_xl)) / len(vel_xl)

    # Énergie normalisée de la bande la plus haute fréquence (bande 10/10) — x_left
    spectrum = np.abs(fft(signals['x_left']))[:len(signals['x_left'])//2]
    bands = np.array_split(spectrum, 10)
    energies = np.array([np.sum(b**2) for b in bands])
    high_freq_band_energy_x_left = (energies / (np.sum(energies) + 1e-8))[9]

    return np.array([mean_binocular_divergence, p90_vertical_velocity_left, saccade_rate_x_left, high_freq_band_energy_x_left])


def process_data(X):
    """Normalise gaze data and extract inference features.

    Args:
        X: DataFrame containing raw gaze log columns including ``time``,
           ``x_left``, ``y_left``, ``x_right``, and ``y_right``.

    Returns:
        A 1D numpy array of extracted features ready for model input.
    """
    X['time'] = X['time'] - X['time'][0]
    X['x_left'] = (X['x_left'] - X['x_left'].min()) / (X['x_left'].max() - X['x_left'].min())
    X['y_left'] = (X['y_left'] - X['y_left'].min()) / (X['y_left'].max() - X['y_left'].min())
    X['x_right'] = (X['x_right'] - X['x_right'].min()) / (X['x_right'].max() - X['x_right'].min())
    X['y_right'] = (X['y_right'] - X['y_right'].min()) / (X['y_right'].max() - X['y_right'].min())
    X_processed = extract_features(X)

    return X_processed
