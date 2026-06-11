import numpy as np
import pandas as pd

from dyslexia_api.processing.process_v2 import to_uniform, process_data


def test_to_uniform_interpolates_signal():
    sample = np.array([0.0, 1.0, 2.0, 4.0])
    timestamps = np.array([0.0, 1.0, 2.0, 3.0])
    interpolated = to_uniform(sample, timestamps)

    assert interpolated.shape == sample.shape
    assert np.isclose(interpolated[0], sample[0])
    assert np.isclose(interpolated[-1], sample[-1])


def test_process_data_returns_feature_vector():
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "x_left": [0, 1, 2, 3, 4],
            "y_left": [0, 2, 4, 6, 8],
            "x_right": [1, 2, 3, 4, 5],
            "y_right": [2, 3, 4, 5, 6],
        }
    )

    features = process_data(df)

    assert features.shape == (4,)
    assert np.all(np.isfinite(features))
    assert np.all(features >= 0)


def test_process_data_handles_constant_columns():
    # Columns where min == max should not cause division by zero
    df = pd.DataFrame(
        {
            "time": [0, 1, 2, 3, 4],
            "x_left": [1, 1, 1, 1, 1],
            "y_left": [2, 2, 2, 2, 2],
            "x_right": [3, 3, 3, 3, 3],
            "y_right": [4, 4, 4, 4, 4],
        }
    )

    features = process_data(df)

    assert features.shape == (4,)
    assert np.all(np.isfinite(features))
