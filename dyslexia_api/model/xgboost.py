"""Build, save, load, and evaluate XGBoost models for dyslexia detection."""

import xgboost as xgb
from sklearn.metrics import accuracy_score


class XGBoostModel:
    """XGBoost classifier wrapper for dyslexia detection.

    Supports training, inference, evaluation, and serialisation to the XGBoost
    JSON format. Two construction paths are provided: :meth:`__init__` trains a
    new model from data, while :meth:`from_file` restores a saved one.
    """
    @classmethod
    def from_file(cls, file_path):
        """Load a previously saved model from a JSON file.

        Args:
            file_path: Path to the XGBoost JSON model file.

        Returns:
            A new :class:`XGBoostModel` instance with the loaded weights.
        """
        instance = cls.__new__(cls)   # allocates without calling __init__
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(file_path)
        return instance

    def __init__(
            self,
            X_train,
            y_train,
            learning_rate=0.05,
            max_depth=2,
            n_estimators=50,
            reg_lambda=1.0,
            reg_alpha=0.1,
            min_child_weight=3,
            imbalance_ratio=1):
        """Train a new XGBoost classifier.

        Args:
            X_train: Feature matrix of shape ``(n_samples, n_features)``.
            y_train: Binary class labels of shape ``(n_samples,)``.
            learning_rate: Boosting learning rate (eta).
            max_depth: Maximum tree depth.
            n_estimators: Number of boosting rounds.
            reg_lambda: L2 regularisation term.
            reg_alpha: L1 regularisation term.
            min_child_weight: Minimum sum of instance weight in a child node.
            imbalance_ratio: `scale_pos_weight` — ratio of negative to positive
                samples, used to handle class imbalance.
        """
        self.model = xgb.XGBClassifier(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            min_child_weight=min_child_weight,
            scale_pos_weight=imbalance_ratio
        )

        self.model.fit(X_train, y_train)

    def predict(self, X):
        """Return predicted class labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of predicted labels of shape ``(n_samples,)``.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 is the probability
            of the positive class.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X_train, y_train, X_test, y_test):
        """Compute train and test accuracy.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_test: Test feature matrix.
            y_test: Test labels.

        Returns:
            A tuple `(train_accuracy, test_accuracy)` as floats in `[0, 1]`.
        """
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        return train_acc, test_acc

    def save_model(self, file_path):
        """Persist the model to a JSON file.

        Args:
            file_path: Destination path for the XGBoost JSON file.
        """
        self.model.save_model(file_path)

    def load_model(self, file_path):
        """Replace this instance's model with one loaded from a JSON file.

        Args:
            file_path: Path to the XGBoost JSON model file.
        """
        self.model = xgb.XGBClassifier()
        self.model.load_model(file_path)
