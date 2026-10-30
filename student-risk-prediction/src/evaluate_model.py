import os
import joblib
import matplotlib
matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score
)

from data_preprocessing import load_data, create_target, RISK_THRESHOLD

os.makedirs("figures", exist_ok=True)
labels = ["Not At Risk", "At Risk"]

# ---------- 1. Model evaluation on the held-out test set ----------
X_test, y_test = joblib.load("models/test_data.joblib")
names = ["logistic_regression", "random_forest"]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
print(f"{'Model':22}{'Precision':>10}{'Recall':>10}{'F1':>10}   (At Risk class)")
for ax, name in zip(axes, names):
    pipe = joblib.load(f"models/{name}.joblib")
    pred = pipe.predict(X_test)

    p = precision_score(y_test, pred)
    r = recall_score(y_test, pred)
    f = f1_score(y_test, pred)
    print(f"{name:22}{p:>10.2f}{r:>10.2f}{f:>10.2f}")

    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()

# ---------- 2. EDA charts ----------
df = create_target(load_data())
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

sns.histplot(df["G3"], bins=20, ax=axes[0])
axes[0].axvline(RISK_THRESHOLD, color="red", linestyle="--", label="Risk threshold")
axes[0].set_title("Final grade (G3) distribution")
axes[0].legend()

sns.boxplot(x="at_risk", y="absences", data=df, ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(labels)
axes[1].set_title("Absences by risk group")
axes[1].set_xlabel("")

sns.countplot(x="at_risk", data=df, ax=axes[2])
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(labels)
axes[2].set_title("Class balance")
axes[2].set_xlabel("")

plt.tight_layout()
plt.savefig("figures/eda_charts.png", dpi=150)
plt.close()

print("\nSaved: figures/confusion_matrices.png and figures/eda_charts.png")