import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from data_preprocessing import load_data, get_X_y


def build_pipeline(model, X):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


if __name__ == "__main__":
    df = load_data()
    X, y = get_X_y(df)

    # Stratified split, fixed random_state; test set stays untouched until evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print("Train:", X_train.shape, "Test:", X_test.shape)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
    }

    os.makedirs("models", exist_ok=True)

    for name, model in models.items():
        pipe = build_pipeline(model, X_train)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, f"models/{name}.joblib")

        print(f"\n=== {name} ===")
        print(classification_report(
            y_test, pipe.predict(X_test),
            target_names=["Not At Risk", "At Risk"]
        ))

    # Save the test split so the evaluation step uses the same data
    joblib.dump((X_test, y_test), "models/test_data.joblib")
    print("Models saved in the models/ folder.")
