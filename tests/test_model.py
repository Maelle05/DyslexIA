import numpy as np

from dyslexia_api.model.xgboost import XGBoostModel


def test_xgboost_model_can_train_save_load(tmp_path):
    X_train = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0, 0.0],
    ])
    y_train = np.array([0, 1, 0, 1])

    model = XGBoostModel(X_train, y_train, n_estimators=1, max_depth=1)
    predictions = model.predict(X_train)

    assert predictions.shape == (4,)

    model_file = tmp_path / "xgboost_model.json"
    model.save_model(str(model_file))

    loaded_model = XGBoostModel.from_file(str(model_file))
    proba = loaded_model.predict_proba(X_train)

    assert proba.shape == (4, 2)

    train_acc, test_acc = loaded_model.evaluate(X_train, y_train, X_train, y_train)
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= test_acc <= 1.0
