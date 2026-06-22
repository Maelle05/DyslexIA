"""Signal processing helpers for gaze data.

Provides routines for fixed-length interpolation, feature extraction from
binocular gaze traces, and conversion of gaze logs into model-ready feature
vectors.
"""

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)

TARGET_LEN = 2000
GAZE_COLS = ['x_left', 'y_left', 'x_right', 'y_right']


def to_uniform(signal, timestamps):
    """Resample a signal to a uniform timeline between 0 and 1.

    Many gaze logs include irregular timestamps. This helper normalises the
    input timestamps to the [0, 1] range and returns an interpolated signal
    sampled at evenly spaced points matching the original signal length.

    Args:
        signal: 1D array-like of signal values.
        timestamps: 1D array-like of monotonically increasing timestamps.

    Returns:
        A numpy array with the same length as ``signal`` containing the
        resampled values.
    """
    t_norm = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
    x_new = np.linspace(0, 1, len(signal))
    return np.interp(x_new, t_norm, signal)


def extract_features(df):
    """Extract a compact feature vector from binocular gaze traces.

    The function computes a small set of summary statistics used as input
    features for the dyslexia classifier: mean binocular divergence, the
    ratio of 90th to 50th percentile velocities on X and Y, and the rate of
    direction changes in the cyclopean X signal.

    Args:
        df: pandas DataFrame containing ``time`` and the gaze columns defined
            in :data:`GAZE_COLS`.

    Returns:
        A 1D numpy array with four float features in the order described
        above.
    """
    timestamps = df['time'].values.astype(float)
    signals = {}
    for col in GAZE_COLS:
        signals[col] = to_uniform(df[col].values.astype(float), timestamps)

    # Divergence binoculaire moyenne
    mean_cross_divergence_x = np.mean(np.abs(signals['x_left'] - signals['x_right']))

    # Cyclope
    x = (signals['x_left'] + signals['x_right']) / 2
    y = (signals['y_left'] + signals['y_right']) / 2

    # Vélocités
    vel_x = np.abs(np.diff(x))
    x_vel_p90p50 = np.percentile(vel_x, 90) / (np.percentile(vel_x, 50) + 1e-8)

    vel_y = np.abs(np.diff(y))
    y_vel_p90p50 = np.percentile(vel_y, 90) / (np.percentile(vel_y, 50) + 1e-8)

    # Taux de changements de direction
    dx = np.diff(x)
    x_direction_changes = np.sum(np.diff(np.sign(dx)) != 0) / len(dx)

    return np.array([mean_cross_divergence_x,
                     x_vel_p90p50, y_vel_p90p50,
                     x_direction_changes])


def process_data(X):
    """Normalise gaze data and extract inference features.

    Args:
        X: DataFrame containing raw gaze log columns including ``time``,
           ``x_left``, ``y_left``, ``x_right``, and ``y_right``.

    Returns:
        A 1D numpy array of extracted features ready for model input.
    """
    X['time'] = X['time'] - X['time'][0]

    # Normalize each gaze axis safely: if max == min, produce a zero array
    for col in ['x_left', 'y_left', 'x_right', 'y_right']:
        col_min = X[col].min()
        col_max = X[col].max()
        denom = col_max - col_min
        if denom == 0:
            X[col] = 0.0
        else:
            X[col] = (X[col] - col_min) / denom

    X_processed = extract_features(X)

    return X_processed
