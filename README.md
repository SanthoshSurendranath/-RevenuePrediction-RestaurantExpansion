# Revenue Prediction — Restaurant Expansion

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-orange?logo=scikit-learn) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen) ![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This project develops a machine learning pipeline to predict restaurant revenue in support of an expansion strategy. Using historical restaurant data across Turkish cities, the models are trained to estimate revenue for over 100,000 candidate locations, enabling data-driven site selection and investment decisions.

The project is structured in two parts:

- **Part A** — Data preprocessing, missing value treatment, and feature engineering
- **Part B** — Model training, evaluation, ensemble stacking, and prediction generation

---

## Business Objective

Identify high-revenue potential locations for new restaurant openings by predicting revenue from restaurant-level features such as city, opening date, restaurant type, and operational attributes (P1–P37).

---

## Dataset

| Dataset | Records | Features | Description |
|---|---|---|---|
| `train.csv` | 137 | 43 | Labelled historical restaurant data including revenue |
| `test.csv` | 100,000 | 42 | Unlabelled candidate locations for revenue prediction |

> **Note:** Data is sourced from Google Drive and loaded via the Google Colab environment.

---

## Project Structure

```
RevenuePrediction/
│
├── RevenuePrediction-RestaurantExpansion (Part A).ipynb   # Preprocessing pipeline
├── RevenuePrediction-RestaurantExpansion (Part B).ipynb   # Modelling & predictions
└── README.md
```

---

## Methodology

### Part A — Preprocessing

1. **Data Loading** — Train and test datasets loaded from Google Drive
2. **Combined Processing** — Train and test data concatenated to ensure consistent transformations
3. **Missing Value Treatment**
   - Columns with fewer than 60% non-null values dropped
   - Numeric columns imputed with column mean
   - Categorical columns imputed with mode or placeholder (`"N"`)
4. **Feature Engineering** — Date parsing to extract year, month, and day from `Open Date`
5. **One-Hot Encoding** — Applied to categorical variables: `City`, `City Group`, `Type`, `Open Date`
6. **Export** — Preprocessed train and test sets saved as `Preprocess_Train_Assignment4.csv` and `Preprocess_Test_Assignment4.csv`

### Part B — Modelling

| Model | Training RMSE |
|---|---|
| Decision Tree Regressor | — |
| Random Forest Regressor | ~1,020,363,853,565 |
| Gradient Boosting Regressor | ~211,059,290,151 |
| Stacked Ensemble (GB + RF + DT → GB meta-model) | ~399,110,582,712 |
| Stacked Ensemble with GridSearchCV (tuned) | Best params via CV |

**Stacking Strategy:**
- Base learners: Gradient Boosting, Random Forest, Decision Tree
- Meta-learner: Gradient Boosting Regressor (via `vecstack`) / Linear Regression (via `StackingRegressor`)
- Cross-validation: 4-fold, shuffled, with `random_state=42`

---

## Technologies & Libraries

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, evaluation, hyperparameter tuning |
| `vecstack` | Model stacking utility |
| `Google Colab` | Cloud-based notebook environment |
| `Google Drive` | Data storage and I/O |

---

## Setup & Usage

### Prerequisites

```bash
pip install vecstack scikit-learn pandas numpy
```

### Running the Notebooks

1. Mount Google Drive in Colab:
   ```python
   from google.colab import drive
   drive.mount('/gdrive')
   ```

2. Place the following files in your Google Drive root:
   - `train.csv`
   - `test.csv`

3. Run **Part A** notebook to generate preprocessed datasets.

4. Run **Part B** notebook to train models and generate prediction CSVs.

### Output Files

| File | Description |
|---|---|
| `Preprocess_Train_Assignment4.csv` | Cleaned training data |
| `Preprocess_Test_Assignment4.csv` | Cleaned test data |
| `DT_Test17.csv` | Decision Tree predictions |
| `RF_Test18.csv` / `RF_Test19.csv` | Random Forest predictions |
| `ABC_Test17.csv` / `GB_Test17.csv` | Gradient Boosting predictions |
| `Stacking_Test17.csv` | Stacked ensemble predictions |
| `Stacked_Model_Predictions.csv` | Tuned stacked model predictions |

---

## Key Findings

- **Gradient Boosting** outperformed individual Decision Tree and Random Forest models in training RMSE
- **Stacking** provided a competitive ensemble combining the strengths of all three base learners
- The small training set (137 records) presents a challenge for generalisation; model variance was monitored via 4-fold cross-validation

---

## Author

Santhosh Surendranath \
Data Scientist \

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
