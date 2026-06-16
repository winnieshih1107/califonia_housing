# L7-Multiple Linear Regression Workflow Log

**Date:** June 15, 2026
**Objective:** Build a robust, self-documenting Machine Learning pipeline to predict Boston neighborhood housing values based on ecological, structural, and demographic profiles, adhering strictly to the CRISP-DM methodology.

---

## 1. Data Acquisition & Sourcing
*   **Action:** Sourced the classic `Boston Housing` dataset from an open-source GitHub repository after verifying it contained exactly 506 records.
*   **Result:** Saved the dataset locally as `boston_housing.csv` and `data.csv`. 
*   **Dataset Features:** `crim`, `zn`, `indus`, `chas` (Charles River dummy), `nox`, `rm`, `age`, `dis`, `rad`, `tax`, `ptratio`, `b`, `lstat`, and `medv` (Target Variable).

## 2. CRISP-DM Step 1: Business Understanding
*   **Objective Defined:** Predict the median home value (`medv`) of a neighborhood based on its local profiles. This predictive capability allows home buyers to negotiate fair pricing, aids real estate agencies in valuations, and assists urban planning departments in assessing neighborhood development strategies.

## 3. CRISP-DM Step 2: Data Understanding (EDA)
*   **Action:** Integrated a massive Exploratory Data Analysis (EDA) suite directly into the core Python script (`solve_boston_housing_crispdm.py`) using `pandas` and print diagnostics.
*   **Checks Performed:** 
    *   Dataset shape (506, 14), Data Types, and Info.
    *   Missing Values (None found).
    *   Duplicate Records (None found).
    *   Descriptive statistics (min, max, mean, std) for all features.
    *   Charles River dummy frequency (471 bounds, 35 bounding river).
    *   Target correlation ranking.
*   **Key EDA Findings:** `rm` (Average rooms) possesses a high positive correlation (0.70) with home value. `lstat` (% lower status) possesses a high negative correlation (-0.74). Proximity to the Charles River (`chas`) adds a positive premium (avg value is $28.4k vs $22.1k).

## 4. CRISP-DM Step 3: Data Preparation
*   **Action:** Constructed a robust preprocessing pipeline using Scikit-learn's `ColumnTransformer` and `StandardScaler` to scale all 13 numerical columns.
*   **Feature Scaling:** Standardized all inputs to mean=0 and std=1, ensuring coefficients in the regression model are directly comparable and regularized methods (like Lasso) perform stably.
*   **Data Splitting:** Applied an 80/20 Train-Test split (`train_test_split`) using `random_state=42` for reproducibility.

## 5. CRISP-DM Step 3.11: Advanced Feature Selection Analysis
*   **Action:** Implemented a feature selection suite to compare 10 different algorithms (Pearson, Spearman, VarianceThreshold, F-test, Mutual Info, RFE, SFS, SBS, Lasso, and Random Forest) at a target feature size.
*   **Key Findings:** For a target subset size of $k=3$, the consensus selections are `rm` (rooms), `lstat` (% lower status), and `ptratio` (pupil-teacher ratio), demonstrating a robust consensus across wrappers, filters, and embedded estimators.

## 6. CRISP-DM Step 4 & 5: Modeling & Evaluation
*   **Action:** Evaluated and compared 4 linear regression configurations:
    *   *Model 1*: rooms (`rm`) only.
    *   *Model 2*: rooms + lower status (`rm` + `lstat`).
    *   *Model 3*: rooms + lower status + pupil-teacher ratio (`rm` + `lstat` + `ptratio`).
    *   *Model 4*: All 13 Features.
*   **Results:**
    *   Adding features incrementally increased generalization performance: Cross-validated $R^2$ rose from **0.473** (Model 1) to **0.715** (Model 4).
    *   Selected Model 4 as the final deployed configuration due to stable CV standard deviations (0.037) and robust performance.

## 7. CRISP-DM Step 6: Deployment
*   **Action:** Refit the optimal configuration on the entire 506-row dataset and saved it using `joblib`.
*   **Result:** Exported as `boston_housing_model.pkl`. Decoupled and fully prepared for use in web applications and dashboard environments.
