import xgboost as xgb
from sklearn.metrics import accuracy_score

class XGBoostModel:
    @classmethod
    def from_file(cls, file_path):
        instance = cls.__new__(cls)   # allocates without calling __init__
        instance.model = xgb.XGBClassifier()
        instance.model.load_model(file_path)
        return instance

    def __init__(self, X_train, y_train, learning_rate=0.05, max_depth=2, n_estimators=50, reg_lambda=1.0, reg_alpha=0.1, min_child_weight=3, imbalance_ratio=1):
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
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_train, y_train, X_test, y_test):
        y_train_pred = self.predict(X_train)
        y_test_pred = self.predict(X_test)

        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)

        return train_acc, test_acc

    def save_model(self, file_path):
        self.model.save_model(file_path)

    def load_model(self, file_path):
      self.model = xgb.XGBClassifier()
      self.model.load_model(file_path)
