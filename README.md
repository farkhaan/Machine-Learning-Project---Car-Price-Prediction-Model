## Car Price Prediction — End-to-End ML Pipeline
Predicting used car sale prices using ensemble models, custom feature engineering, and full scikit-learn pipelines. Built as part of the *Hands-on Machine Learning and Data Science* course at KU Eichstätt-Ingolstadt.

## Project Overview:
This project tackles a **supervised regression problem**: given a set of features about a car listing (brand, model, engine specs, location, seller type, etc.), predict the **sale price**.
The full ML workflow is covered — from raw data exploration all the way to hyperparameter-tuned, submission-ready sklearn `Pipeline` objects.
**Best model: HistGradientBoosting Regressor — R² = 0.69 on held-out test set**

---

## Repository Structure:
```
├── group_P_version_35_project.ipynb   # Main notebook (EDA + modeling + evaluation)
├── topic21_v35_train.csv              # Training dataset
└── README.md
```

---

## Dataset

| Property | Detail |
|---|---|
| Source | Course-provided car sales dataset (unique per group) |
| Instances | ~8,000 |
| Numerical features | 5 (some anonymized) |
| Categorical features | ~14 (brand, model, fuel type, transmission, color, city, etc.) |
| Target variable | `price` (continuous, USD) |
| Task type | Supervised Regression |

Notable data quality challenges:
- Engine capacity and horsepower stored as **string ranges** (e.g. `"600 - 699 HP"`)
- Missing values encoded as `"Unknown"` or `"Other"` in several columns
- Highly **right-skewed** price distribution requiring log transformation
- Rare car brands needing grouping to avoid sparse encoding

---

## Feature Engineering & Preprocessing

Three **custom scikit-learn transformers** were built from scratch:

### `RangeToNumeric`
Parses string-formatted engine/horsepower ranges into usable floats.
```
"1000 - 1500 cc"  →  1250.0
"1500+ cc"        →  1500.0
"Unknown"         →  NaN
```

### `WarrantyTransformer`
Converts categorical warranty labels to binary.
```
"Yes"             →  1
"No" / "Does Not Apply"  →  0
```

### `RareBrandGrouper`
Groups infrequent car brands (below a count threshold) into a single `"Other"` category to reduce noise in one-hot encoding.

**Additional steps:**
- Log transformation on skewed numerical features (`1_log`, `3_log`)
- Median imputation for numerical NaNs
- Most-frequent imputation for categorical NaNs
- `OneHotEncoder` for categorical features
- `StandardScaler` for numerical features
- All steps embedded in a single sklearn `Pipeline` — **no data leakage possible**

---

## Models Trained & Compared

| Model | Type | Test R² | Notes |
|---|---|---|---|
| Ridge Regression | Linear | ~0.45 | Stable, underfits non-linear patterns |
| K-Nearest Neighbors | Instance-based | ~0.63 | Overfits heavily (Train R² = 1.00) |
| Decision Tree | Tree-based | ~0.52 | Underfits without tuning |
| Random Forest | Ensemble (Bagging) | — | Tuned via GridSearchCV |
| Extra Trees | Ensemble (Randomized) | — | Tuned via GridSearchCV |
| **HistGradientBoosting** | **Ensemble (Boosting)** | **~0.69** | **Best model — submitted to leaderboard** |

All ensemble models were tuned using **GridSearchCV with 5-fold cross-validation**.

Best hyperparameters for HistGradientBoosting:
```
learning_rate:    0.01
max_depth:        None
min_samples_leaf: 3
max_leaf_nodes:   100
```

---

##Evaluation Metrics

Models were evaluated using:
- **MSE** — Mean Squared Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination *(primary metric)*

HistGradientBoosting (best model) results:
```
Train R²:  0.945   |   Train RMSE: ~43,925
Test  R²:  0.687   |   Test  RMSE: ~110,022
```

The gap between train and test R² indicates moderate overfitting — a known challenge with boosting on this dataset's price range.

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas 2.2 | Data manipulation |
| NumPy 2.0 | Numerical operations |
| scikit-learn 1.6 | Modeling, pipelines, preprocessing |
| seaborn / matplotlib | EDA visualizations |
| Google Colab | Development environment |

---

## Key Learnings

- Building **custom sklearn-compatible transformers** (`BaseEstimator`, `TransformerMixin`) for domain-specific preprocessing
- Structuring ML workflows as proper **Pipelines** to eliminate data leakage
- The trade-off between model complexity and generalization — simpler models (Ridge) were more stable but ensemble methods captured non-linear price patterns much better
- The importance of handling messy, real-world data (string ranges, placeholder values, skewed distributions) before any modeling

---

## Authors

Group P — KU Eichstätt-Ingolstadt — Farkhan Mustafazade, Elena Erlygina, and Iuliia Dudulina
Course: *Hands-on Machine Learning and Data Science* (Prof. Felix Voigtlaender)
