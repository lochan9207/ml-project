
import os
from pathlib import Path
 
import joblib
import matplotlib
matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.model_selection import StratifiedKFold, cross_validate
 
from data_preprocessing import load_data, create_target, get_X_y, RISK_THRESHOLD
 
# Work from the project root no matter where the script is launched from
ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
 
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
labels = ["Not At Risk", "At Risk"]
TEAL, AMBER = "#1b7a63", "#e0a030"
 
# ---------- 1. Model evaluation on the held-out test set ----------
X_test, y_test = joblib.load("models/test_data.joblib")
names = ["logistic_regression", "random_forest"]
 
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
print(f"Test set size: {len(y_test)} students")
print(f"{'Model':22}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Brier':>10}   (At Risk class)")
 
rows = []
for ax, name in zip(axes, names):
    pipe = joblib.load(f"models/{name}.joblib")
    pred = pipe.predict(X_test)
 
    p = precision_score(y_test, pred)
    r = recall_score(y_test, pred)
    f = f1_score(y_test, pred)
    b = brier_score_loss(y_test, pipe.predict_proba(X_test)[:, 1])
    print(f"{name:22}{p:>10.2f}{r:>10.2f}{f:>10.2f}{b:>10.3f}")
    rows.append({"model": name, "precision": round(p, 2),
                 "recall": round(r, 2), "f1": round(f, 2),
                 "brier": round(b, 3)})
 
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", ax=ax,
                xticklabels=labels, yticklabels=labels)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
 
plt.tight_layout()
plt.savefig("figures/confusion_matrices.png", dpi=150)
plt.close()
 
pd.DataFrame(rows).to_csv("results/test_metrics.csv", index=False)
 
# ---------- 2. 5-fold cross-validation (more honest than one small test set) ----------
raw = load_data()
X_all, y_all = get_X_y(raw)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"precision": "precision", "recall": "recall", "f1": "f1"}
 
print("\n5-fold cross-validation (mean +/- std, At Risk class)")
cv_rows = []
for name in names:
    pipe = joblib.load(f"models/{name}.joblib")
    res = cross_validate(clone(pipe), X_all, y_all, cv=cv, scoring=scoring)
    row = {"model": name}
    for metric in ["precision", "recall", "f1"]:
        scores = res[f"test_{metric}"]
        row[f"{metric}_mean"] = round(scores.mean(), 3)
        row[f"{metric}_std"] = round(scores.std(), 3)
    cv_rows.append(row)
    print(f"{name:22} "
          f"P {row['precision_mean']:.2f}+/-{row['precision_std']:.2f}  "
          f"R {row['recall_mean']:.2f}+/-{row['recall_std']:.2f}  "
          f"F1 {row['f1_mean']:.2f}+/-{row['f1_std']:.2f}")
pd.DataFrame(cv_rows).to_csv("results/cv_metrics.csv", index=False)
 
# ---------- 3. Random Forest feature importance ----------
rf = joblib.load("models/random_forest.joblib")
feat_names = [n.replace("num__", "").replace("cat__", "")
              for n in rf.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=feat_names)
top = imp.sort_values(ascending=False).head(10).sort_values()
 
fig, ax = plt.subplots(figsize=(6, 4))
top.plot(kind="barh", ax=ax, color=TEAL)
ax.set_xlabel("Importance")
ax.set_title("Top features (Random Forest, overall)")
plt.tight_layout()
plt.savefig("figures/feature_importance.png", dpi=150)
plt.close()
 
# ---------- 4. Calibration curve (are the probabilities trustworthy?) ----------
fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
for name, colour in zip(names, [AMBER, TEAL]):
    pipe = joblib.load(f"models/{name}.joblib")
    probs = pipe.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=5, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", color=colour, label=name)
ax.set_xlabel("Predicted probability of At Risk")
ax.set_ylabel("Actual fraction At Risk")
ax.set_title("Calibration (small test set, read loosely)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/calibration_curve.png", dpi=150)
plt.close()
 
# ---------- 5. EDA charts ----------
df = create_target(raw)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
 
sns.histplot(df["G3"], bins=20, color=TEAL, ax=axes[0])
axes[0].axvline(RISK_THRESHOLD, color=AMBER, linestyle="--", linewidth=2,
                label="Risk threshold")
axes[0].set_title("Final grade (G3) distribution")
axes[0].legend()
 
sns.boxplot(x="at_risk", y="absences", hue="at_risk",
            palette={0: TEAL, 1: AMBER}, legend=False, data=df, ax=axes[1])
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(labels)
axes[1].set_title("Absences by risk group")
axes[1].set_xlabel("")
 
sns.countplot(x="at_risk", hue="at_risk", palette={0: TEAL, 1: AMBER},
              legend=False, data=df, ax=axes[2])
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(labels)
axes[2].set_title("Class balance")
axes[2].set_xlabel("")
 
plt.tight_layout()
plt.savefig("figures/eda_charts.png", dpi=150)
plt.close()
 
print("\nSaved figures: confusion_matrices.png, calibration_curve.png, "
      "feature_importance.png, eda_charts.png")
print("Saved results: results/test_metrics.csv, results/cv_metrics.csv")