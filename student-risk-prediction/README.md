# Student Academic Performance Risk Prediction Using Machine Learning

A small end-to-end ML prototype that estimates whether a student may be at risk
of low academic performance. The output is a decision-support indicator to guide
supportive review, not a judgment about any student.

## Dataset
UCI Student Performance dataset (`student-mat.csv`, 395 students).

## Risk label
A student is **At Risk** if the final grade `G3` is below 10 (out of 20).
This is a project-defined threshold, not a universal standard.

## Avoiding data leakage
`G3` is used only to create the label and is never an input. The model uses
`G1` and `G2`, so it predicts after the second-period exam. `sex` and `romantic`
are excluded to protect privacy.

## Models
- Logistic Regression (baseline)
- Random Forest (comparison)

Both use a scikit-learn Pipeline with a stratified 80/20 split and `random_state=42`.

## Results (At Risk class, test set of 79 students)
| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression | 0.81 | 0.85 | 0.83 |
| Random Forest | 0.85 | 0.88 | 0.87 |

The test set is small, so differences between models may not be reliable.

## How to run
```
pip install -r requirements.txt
python src/clean_data.py
python src/train_model.py
python src/evaluate_model.py
python -m streamlit run app.py

## Project structure
```
student-risk-prediction/
    data/           dataset CSV
    src/            data_preprocessing.py, train_model.py, evaluate_model.py
    models/         saved models
    figures/        saved charts
    screenshots/    dashboard screenshots
    app.py          Streamlit dashboard
    requirements.txt
    README.md
```

## Limitations and responsible use
- The dataset may not represent students at your institution.
- The risk threshold is a project assumption.
- Predictions can be wrong and may reflect bias in historical data.
- Use predictions to prompt supportive help, not as a final judgment about
  ability or future success.