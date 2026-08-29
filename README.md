# Political Affiliation Classification — Multiclass Logistic Regression

Predicting self-reported political affiliation (Republican, Democrat, or
Independent) from public opinion survey responses, using regularized
multiclass logistic regression.

## Overview

This project frames political affiliation prediction as a three-class
classification problem. It covers preprocessing, a regularized logistic
regression model tuned via grid search, and generating predictions on a
held-out test set.

**Note:** this is a modeling exercise using a real public-opinion survey as
the data source. The goal is prediction accuracy, not commentary on the
survey's content — the analysis here is not an endorsement of any views
expressed in the underlying data.

## Dataset

The data comes from a monthly public-opinion poll conducted by Survey
Sampling International, a professional research firm, on behalf of a
2017–2018 polling series that asked a nationally representative sample of
the American public a mix of lighthearted and substantive questions. This
project uses one wave of that survey (a subset of questions, cleaned for
the exercise) to predict each respondent's self-reported political
affiliation from their other responses.

*Data files (`CAH-201803-train.csv`, `CAH-201803-test.csv`) are not included
in this repo, as they were distributed privately as part of a course
assignment (a cleaned, question-trimmed subset of the original survey wave).
The original poll and its public analysis are available at
[thepulseofthenation.com](https://thepulseofthenation.com/#poll-12) for
reference, though the raw survey microdata used for this exercise is not
independently published there.*

## Approach

1. **Target encoding** — map `political_affiliation` to numeric classes
   (Republican, Democrat, Independent).
2. **Preprocessing pipeline** — a `ColumnTransformer` that standardizes
   numeric predictors and one-hot encodes categorical predictors, wrapped in
   a single reproducible `Pipeline`.
3. **Model** — multiclass logistic regression with an elastic-net penalty
   (blending L1/L2 regularization), fit with the `saga` solver.
4. **Hyperparameter tuning** — `GridSearchCV` over the inverse
   regularization strength (`C`) and the L1/L2 mix (`l1_ratio`), using
   10-fold cross-validation scored on accuracy.
5. **Prediction** — class predictions are mapped back to their original
   labels and exported for submission.

## Results

| Metric | Value |
|---|---|
| Best `C` | _fill in from `grid_log.best_params_`_ |
| Best `l1_ratio` | _fill in from `grid_log.best_params_`_ |
| Cross-validated accuracy | _fill in from `grid_log.best_score_`_ |

## Tech Stack

Python · pandas · NumPy · scikit-learn (`Pipeline`, `ColumnTransformer`,
`LogisticRegression`, `GridSearchCV`)

## How to Run

*Note: the course-provided data files are required to run this notebook
as-is and are not included in this repo (see [Dataset](#dataset) above).
If you have your own copy of the assignment files, follow the steps below;
otherwise this notebook is best read as a walkthrough of the approach.*

1. Place `CAH-201803-train.csv` and `CAH-201803-test.csv` in the project
   root.
2. Render the notebook with [Quarto](https://quarto.org):
   ```
   quarto render political_affiliation_classification.qmd
   ```
   or run the embedded Python cells directly in a Jupyter/VS Code notebook.
3. Predictions are written to `RealFinalclass_fileupdated.csv`
   (`id_num`, `political_affiliation_predicted`).

## File Structure

```
political-affiliation-classification/
├── political_affiliation_classification.qmd   # full modeling pipeline
├── requirements.txt                            # Python dependencies
└── README.md
```
