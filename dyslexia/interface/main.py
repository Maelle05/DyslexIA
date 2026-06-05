from dyslexia.processing.process_data_justine import (get_files_path,
                                                      process_dataset,
                                                      split)
from dyslexia.model.xgboost import XGBoostModel
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def main():
    """Run the full training and inference pipeline.

    Steps:
        1. Discover raw CSV files under the hard-coded data directory.
        2. Build a feature dataset with :func:
        `~dyslexia.processing.process_data_justine.process_dataset`.
        3. Split into train/test sets and apply standard scaling.
        4. Load an existing model from ``xgboost_dyslexia_model_v1.json`` if
           available; otherwise train a fresh model
           and print train/test accuracy.
        5. Print the predicted class and probability
        for the first training sample.
    """
    files_metrics, files_fixations = get_files_path(
            "/home/yoannl/code/drealKn/DyslexIA/data/data"
        )
    dataset = process_dataset(files_metrics, files_fixations)
    X_train, X_test, y_train, y_test = split(dataset)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        model = XGBoostModel.from_file("xgboost_dyslexia_model_v1.json")
    except xgb.core.XGBoostError:
        model = XGBoostModel(X_train, y_train)
        train_acc, test_acc = model.evaluate(X_train, y_train, X_test, y_test)
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

    model = XGBoostModel.from_file("xgboost_dyslexia_model_v1.json")
    # Example prediction (replace with actual data)
    prediction = model.predict(X_train[0].reshape(1, -1))
    prediction_proba = model.predict_proba(X_train[0].reshape(1, -1))
    print(f"Predicted class: {prediction[0]}")
    print(f"Predicted proba: {prediction_proba[0][prediction[0]]}")
    print(f"True class: {y_train[0]}")


if __name__ == "__main__":
    main()
