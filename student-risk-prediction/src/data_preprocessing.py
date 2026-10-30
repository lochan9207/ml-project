import pandas as pd

# Project-defined threshold: at risk if final grade G3 < 10 (out of 20)
RISK_THRESHOLD = 10

# Sensitive attributes we leave out (responsible use / privacy)
DROP_SENSITIVE = ["sex", "romantic"]


def load_data(path="data/student-mat-cleaned.csv"):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    return df


def create_target(df, threshold=RISK_THRESHOLD):
    df = df.copy()
    df["at_risk"] = (df["G3"] < threshold).astype(int)
    return df


def get_X_y(df, use_period_grades=True):
    """
    Returns input features X and target y.
    G3 is used ONLY to create the label, never as an input (no leakage).
    If use_period_grades=True, G1 and G2 are used (prediction after period 2).
    """
    df = create_target(df)
    y = df["at_risk"]
    X = df.drop(columns=["G3", "at_risk"] + DROP_SENSITIVE)
    if not use_period_grades:
        X = X.drop(columns=["G1", "G2"])
    return X, y


if __name__ == "__main__":
    df = load_data()
    X, y = get_X_y(df)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    print("X shape:", X.shape)
    print("G3 in features?", "G3" in X.columns)
    print("Categorical columns:", categorical)
    print("Numeric columns:", numeric)
    print(y.value_counts(normalize=True).round(2))