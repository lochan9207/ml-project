import sys
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

sys.path.append("src")
from data_preprocessing import load_data, create_target, get_X_y, RISK_THRESHOLD

st.set_page_config(page_title="Student Risk Prediction", page_icon="🎓", layout="wide")
LABELS = ["Not At Risk", "At Risk"]
TEAL, AMBER = "#1b7a63", "#e0a030"
PALETTE = {0: TEAL, 1: AMBER}
sns.set_theme(style="whitegrid")

# ---------------- DESIGN ONLY (no content changes) ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-image:
            radial-gradient(circle at 1px 1px, rgba(27,122,99,0.12) 1px, transparent 0),
            linear-gradient(160deg, #f6f4ee 0%, #e6f3ee 55%, #fdf3e1 100%);
        background-size: 28px 28px, 100% 100%;
        background-attachment: fixed;
    }
    [data-testid="stHeader"] {background: transparent;}
    .block-container {padding-top: 2rem; max-width: 1200px;}

    .hero {background: linear-gradient(135deg, #0f1b2d 0%, #1b4d5c 60%, #1b7a63 100%);
           padding: 30px 40px; border-radius: 22px; margin-bottom: 22px;
           box-shadow: 0 8px 24px rgba(15,27,45,0.25);}
    .hero h1 {font-size: 2.1rem; margin: 0 0 8px 0; color: #f6f4ee; padding: 0;}
    .hero p {margin: 0; color: #c9d6df; font-size: 1.05rem;}

    h3 {border-left: 6px solid #e0a030; padding-left: 12px !important;}

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff; border-radius: 16px;
        box-shadow: 0 4px 14px rgba(15,27,45,0.08);
    }

    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background: #ffffff; border-radius: 12px 12px 0 0;
        padding: 10px 18px; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {background: #1b7a63;}
    .stTabs [aria-selected="true"] p {color: #ffffff;}
    .stTabs [data-baseweb="tab-highlight"] {background: #e0a030;}

    .stButton > button {
        border-radius: 12px; padding: 0.6rem 1.8rem; font-weight: 700;
        box-shadow: 0 4px 12px rgba(27,122,99,0.30);
    }
    div[data-testid="stAlert"] {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    return {
        "Random Forest": joblib.load("models/random_forest.joblib"),
        "Logistic Regression": joblib.load("models/logistic_regression.joblib"),
    }


@st.cache_data
def load_all():
    raw = load_data()
    df = create_target(raw)
    X, y = get_X_y(raw)
    X_test, y_test = joblib.load("models/test_data.joblib")
    return df, X, X_test, y_test


models = load_models()
df, X, X_test, y_test = load_all()

# Default value for every feature (median for numbers, most common for text).
# The form below overrides the ones the user can enter.
defaults = {}
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        defaults[col] = X[col].median()
    else:
        defaults[col] = X[col].mode()[0]

st.markdown(
    """
    <div class="hero">
      <h1>🎓 Student Academic Performance Risk Prediction</h1>
      <p>📘 A decision-support tool to guide supportive review. It is not a judgment about any student.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Predict", "📊 Data charts", "📈 Model evaluation", "ℹ️ About and limitations"]
)

# ---------------- TAB 1: PREDICT ----------------
with tab1:
    st.subheader("📝 Enter student details")
    model_name = st.selectbox("Model", list(models.keys()))

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.markdown("**📚 Academic record**")
            g1 = st.slider("G1: period 1 grade (0-20)", 0, 20, 10)
            g2 = st.slider("G2: period 2 grade (0-20)", 0, 20, 10)
            failures = st.selectbox("Past class failures", [0, 1, 2, 3])
            absences = st.number_input("Absences", min_value=0, max_value=75, value=4)
    with c2:
        with st.container(border=True):
            st.markdown("**🧑‍🎓 Student profile**")
            studytime = st.selectbox(
                "Weekly study time",
                [1, 2, 3, 4],
                format_func=lambda v: {1: "Under 2 hours", 2: "2 to 5 hours",
                                       3: "5 to 10 hours", 4: "Over 10 hours"}[v],
                index=1,
            )
            age = st.slider("Age", 15, 22, 17)
            higher = st.selectbox("Wants higher education", ["yes", "no"])
            internet = st.selectbox("Internet at home", ["yes", "no"])
    with c3:
        with st.container(border=True):
            st.markdown("**🤝 Support and lifestyle**")
            schoolsup = st.selectbox("Extra school support", ["no", "yes"])
            famsup = st.selectbox("Family educational support", ["yes", "no"])
            paid = st.selectbox("Paid extra classes", ["no", "yes"])
            goout = st.slider("Going out with friends (1-5)", 1, 5, 3)
            health = st.slider("Health status (1-5)", 1, 5, 3)

    if st.button("🔍 Predict risk", type="primary"):
        # ---- Input validation ----
        problems = []
        notes = []
        if g1 == 0 and g2 == 0:
            problems.append("Both G1 and G2 are 0. Please check the grades.")
        if abs(g1 - g2) > 8:
            notes.append("G1 and G2 differ by more than 8 points, which is unusual. Please double-check them.")
        if absences > 40:
            notes.append("Absences above 40 are rare in the training data, so the prediction may be less reliable.")
        if problems:
            for p in problems:
                st.error(p)
            st.stop()
        for n in notes:
            st.info(n)

        user_inputs = {
            "G1": g1, "G2": g2, "failures": failures, "absences": absences,
            "studytime": studytime, "age": age, "higher": higher,
            "internet": internet, "schoolsup": schoolsup, "famsup": famsup,
            "paid": paid, "goout": goout, "health": health,
        }
        row = pd.DataFrame([{**defaults, **user_inputs}])[X.columns]

        pipe = models[model_name]
        prob = pipe.predict_proba(row)[0][1]

        if prob >= 0.5:
            st.error(f"At Risk (estimated probability {prob:.0%})")
        else:
            st.success(f"Not At Risk (estimated risk probability {prob:.0%})")

        st.warning(
            "Use this result to guide supportive review, "
            "not to punish or label students."
        )

        st.subheader("🔎 Factors the Random Forest relies on most (overall)")
        rf = models["Random Forest"]
        names = rf.named_steps["prep"].get_feature_names_out()
        imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
        top = imp.sort_values(ascending=False).head(8).sort_values()
        top.index = [n.replace("num__", "").replace("cat__", "") for n in top.index]
        fig, ax = plt.subplots(figsize=(6, 3))
        top.plot(kind="barh", ax=ax, color=TEAL)
        ax.set_xlabel("Importance")
        plt.tight_layout()
        st.pyplot(fig)
        st.caption(
            "This shows what the model uses across all students, "
            "not a full explanation of this one prediction."
        )

# ---------------- TAB 2: DATA CHARTS ----------------
with tab2:
    st.subheader("📊 Exploring the data")
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            fig, ax = plt.subplots()
            sns.histplot(df["G3"], bins=20, color=TEAL, ax=ax)
            ax.axvline(RISK_THRESHOLD, color=AMBER, linestyle="--", linewidth=2, label="Risk threshold")
            ax.set_title("Final grade (G3) distribution")
            ax.legend()
            st.pyplot(fig)

    with c2:
        with st.container(border=True):
            fig, ax = plt.subplots()
            sns.boxplot(x="at_risk", y="absences", hue="at_risk", palette=PALETTE,
                        legend=False, data=df, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(LABELS)
            ax.set_xlabel("")
            ax.set_title("Absences by risk group")
            st.pyplot(fig)

    with c3:
        with st.container(border=True):
            fig, ax = plt.subplots()
            sns.countplot(x="at_risk", hue="at_risk", palette=PALETTE,
                          legend=False, data=df, ax=ax)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(LABELS)
            ax.set_xlabel("")
            ax.set_title("Class balance")
            st.pyplot(fig)

# ---------------- TAB 3: MODEL EVALUATION ----------------
with tab3:
    st.subheader("📈 Performance on the held-out test set (At Risk class)")
    rows = []
    for name, pipe in models.items():
        pred = pipe.predict(X_test)
        rows.append({
            "Model": name,
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
        })
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True)
    st.caption(f"The test set has only {len(y_test)} students, so small differences may not be reliable.")

    cols = st.columns(2)
    for col, (name, pipe) in zip(cols, models.items()):
        with col:
            with st.container(border=True):
                cm = confusion_matrix(y_test, pipe.predict(X_test))
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(cm, annot=True, fmt="d", cmap="BuGn", ax=ax,
                            xticklabels=LABELS, yticklabels=LABELS)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title(name)
                st.pyplot(fig)

# ---------------- TAB 4: ABOUT ----------------
with tab4:
    with st.container(border=True):
        st.subheader("🏷️ How the risk label is defined")
        st.write(
            f"A student is labelled **At Risk** if the final grade G3 is below "
            f"**{RISK_THRESHOLD}** (out of 20). This is a project assumption, "
            "not a universal standard."
        )
    with st.container(border=True):
        st.subheader("🔒 Avoiding data leakage")
        st.write(
            "G3 is used only to create the label and is never an input. "
            "The model uses G1 and G2, so it predicts after the second-period exam."
        )
    with st.container(border=True):
        st.subheader("⚖️ Limitations and responsible use")
        st.markdown(
            "- The dataset may not represent students at your institution.\n"
            "- The risk threshold is a project assumption.\n"
            "- Predictions can be wrong and may reflect bias in historical data.\n"
            "- Sex and relationship status are excluded to protect privacy.\n"
            "- Use predictions to prompt supportive help, not as a final judgment "
            "about ability or future success."
        )