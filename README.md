# House Price Prediction — Regularized Regression

Predicting residential sale prices in Ames, Iowa using an Elastic Net regression
pipeline, tuned via grid search and cross-validation.

## Overview

This project builds a regularized regression model to predict home sale prices
from a structured housing dataset. It covers the full applied workflow: feature
engineering, a reproducible preprocessing pipeline, hyperparameter tuning, and
generating predictions for unseen data.

## Dataset

The data covers residential property sales in Ames, Iowa, with each row
representing one sale. Features describe the physical and locational
characteristics of the property — lot size and street access, neighborhood,
dwelling type and style, overall quality and condition ratings, year built,
roof style, and related structural details. The target variable, `SalePrice`,
is the property's sale price in dollars. Each property has a unique
identifier (`PID`). The dataset is a simplified subset of the well-known Ames
housing data, prepared for a modeling exercise.

*Data files (`train_new.csv`, `test_new.csv`) are not included in this repo,
since they were distributed privately as part of a course assignment.* The
underlying data is a trimmed version of the well-known
[Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
on Kaggle — that public version has more columns than the course subset used
here, but is a close substitute for reproducing the pipeline end-to-end.

## Approach

1. **Cleaning & feature engineering** — drop rows with missing values; add an
   engineered `GrLivArea_sq` term (squared above-grade living area) to capture
   a nonlinear relationship between living space and price.
2. **Preprocessing pipeline** — a `ColumnTransformer` that standardizes
   numeric features and one-hot encodes categorical features, wrapped in a
   single reproducible `Pipeline`.
3. **Model** — Elastic Net regression, which blends L1 (Lasso) and L2 (Ridge)
   regularization to handle correlated predictors while still performing
   feature selection.
4. **Target transform** — `SalePrice` is log-transformed before fitting to
   stabilize variance across the price range; predictions are exponentiated
   back to dollar terms before export.
5. **Hyperparameter tuning** — `GridSearchCV` over the regularization
   strength (`alpha`) and the L1/L2 mix (`l1_ratio`), using 10-fold
   cross-validation scored on negative MSE.
6. **Evaluation** — cross-validated RMSE (on the log scale) is used to
   compare candidate hyperparameters and select the final model.

## Results

| Metric | Value |
|---|---|
| Best `alpha` | _fill in from `gscv_fitted.best_params_`_ |
| Best `l1_ratio` | _fill in from `gscv_fitted.best_params_`_ |
| Cross-validated RMSE (log scale) | _fill in from printed `rmse_elastic`_ |

## Tech Stack

Python · pandas · NumPy · scikit-learn (`Pipeline`, `ColumnTransformer`,
`ElasticNet`, `GridSearchCV`)

## How to Run

1. Obtain the Ames housing data — either the original course files or the
   public [Ames Housing dataset](https://www.kaggle.com/datasets/prevek18/ames-housing-dataset)
   on Kaggle, split into `train_new.csv` and `test_new.csv` (column names
   should match those referenced in the notebook; trim to the course's
   column subset if using the public version, or adjust the drop list in
   the notebook to match your columns).
2. Place both files in the project root.
3. Render the notebook with [Quarto](https://quarto.org):
   ```
   quarto render house_price_regression.qmd
   ```
   or run the embedded Python cells directly in a Jupyter/VS Code notebook.
4. Predictions are written to `finalpred_file1.csv` (`PID`, `SalePrice`).

## File Structure

```
house-price-prediction/
├── house_price_regression.qmd   # full modeling pipeline
├── requirements.txt             # Python dependencies
└── README.md
```
