import numpy as np
import pandas as pd

from dyslexia_api.processing.process_v2 import to_fixed_length, process_data


def test_to_fixed_length_interpolates_signal():
    sample = np.array([0.0, 1.0, 2.0, 4.0])
    interpolated = to_fixed_length(sample, target_len=8)

    assert interpolated.shape == (8,)
    assert np.isclose(interpolated[0], 0.0)
    assert np.isclose(interpolated[-1], 4.0)


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
