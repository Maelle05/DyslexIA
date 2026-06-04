import pytest
import numpy as np

from dyslexia.model.xgboost import XGBoostModel


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((40, 8))
    y = rng.integers(0, 2, 40)
    return X, y


@pytest.fixture
def trained_model(training_data):
    X, y = training_data
    return XGBoostModel(X, y), X, y


class TestXGBoostModelInit:
    def test_instantiation(self, training_data):
        X, y = training_data
        model = XGBoostModel(X, y)
        assert model.model is not None

    def test_custom_hyperparameters(self, training_data):
        X, y = training_data
        model = XGBoostModel(X, y, learning_rate=0.1, max_depth=3, n_estimators=20)
        assert model.model is not None


class TestPredict:
    def test_output_shape(self, trained_model):
        model, X, _ = trained_model
        preds = model.predict(X)
        assert preds.shape == (len(X),)

    def test_predictions_are_binary(self, trained_model):
        model, X, _ = trained_model
        preds = model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_single_sample(self, trained_model):
        model, X, _ = trained_model
        pred = model.predict(X[:1])
        assert pred.shape == (1,)


class TestPredictProba:
    def test_output_shape(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_probabilities_sum_to_one(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_probabilities_in_range(self, trained_model):
        model, X, _ = trained_model
        proba = model.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()


class TestEvaluate:
    def test_returns_two_floats(self, trained_model):
        model, X, y = trained_model
        result = model.evaluate(X, y, X, y)
        assert len(result) == 2

    def test_accuracies_in_range(self, trained_model):
        model, X, y = trained_model
        train_acc, test_acc = model.evaluate(X, y, X, y)
        assert 0.0 <= train_acc <= 1.0
        assert 0.0 <= test_acc <= 1.0

    def test_train_accuracy_above_chance(self, training_data):
        X, y = training_data
        model = XGBoostModel(X, y)
        train_acc, _ = model.evaluate(X, y, X, y)
        assert train_acc > 0.5


class TestSaveLoad:
    def test_save_and_from_file_produce_same_predictions(self, trained_model, tmp_path):
        model, X, _ = trained_model
        path = str(tmp_path / "model.json")
        model.save_model(path)
        loaded = XGBoostModel.from_file(path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_load_model_instance_method(self, trained_model, tmp_path):
        model, X, _ = trained_model
        path = str(tmp_path / "model.json")
        model.save_model(path)
        new_model = XGBoostModel.__new__(XGBoostModel)
        new_model.load_model(path)
        np.testing.assert_array_equal(model.predict(X), new_model.predict(X))

    def test_from_file_nonexistent_raises(self, tmp_path):
        with pytest.raises(Exception):
            XGBoostModel.from_file(str(tmp_path / "missing.json"))
