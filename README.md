# L7-Multiple Linear Regression Workflow

This repository implements a modular Scikit-learn regression solution to predict Boston neighborhood housing values based on ecological, structural, and demographic indicators, adhering strictly to the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

---

## 📂 Project Structure & Deliverables

- **[solve_boston_housing_crispdm.py](file:///d:/Huan%20Chen/L6%2050%20Startup/solve_boston_housing_crispdm.py)**: The main CRISP-DM pipeline script containing modular functions representing Business Understanding, Data Understanding (EDA), Data Preparation (Pipeline with `ColumnTransformer` + `StandardScaler`), Modeling, Evaluation (5-Fold Cross Validation), and Deployment.
- **[boston_housing.csv](file:///d:/Huan%20Chen/L6%2050%20Startup/boston_housing.csv)**: The primary Boston Housing dataset containing 506 rows and 14 columns.
- **[boston_housing_model.pkl](file:///d:/Huan%20Chen/L6%2050%20Startup/boston_housing_model.pkl)**: Serialized final pipeline model (refitted on all 506 samples using the optimal features).
- **[feature_selection_performance_allinone.png](file:///d:/Huan%20Chen/L6%2050%20Startup/feature_selection_performance_allinone.png)**: Comparative visual comparing 9 top feature selection algorithms (Pearson, Spearman, F-test, Mutual Info, RFE, SFS, SBS, Lasso, and Random Forest).
- **[hw7.md](file:///d:/Huan%20Chen/L6%2050%20Startup/hw7.md)**: A concise homework summary report of today's work and results.

---

## 🚀 Execution & Running Instructions

Make sure the required libraries are installed:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit altair
```

### 1. Run the Main Machine Learning Pipeline
This command executes the full 6-step CRISP-DM pipeline, evaluates 4 feature-set models, prints selection justifications, and runs the deployment simulation:
```bash
python solve_boston_housing_crispdm.py
```

### 2. Run the Feature Selection Comparison script
To calculate the top feature selections for each of the 10 methods at a target subset size:
```bash
python solve_boston_housing_feature_selection.py
```

### 3. Generate the Stepwise Feature Selection Comparison Plot
To fit and compare the feature selection paths for all 9 algorithms as subset size $k$ goes from 1 to 5:
```bash
python plot_boston_housing_allinone.py
```

### 4. Run the Interactive Web Dashboard
To launch the interactive Streamlit web dashboard (which executes all feature selection methods, shows consensus charts, and includes a live house price simulator):
```bash
streamlit run solve_boston_housing_dashboard.py
```
Or simply double-click **`run_dashboard.bat`** in Windows Explorer.

---

## 📈 Summary of Results

### 1. Final Model Performance
The optimal model selected is **Model 4: All Features**. Its test and cross-validated performance metrics are:
- **Test $R^2$ Score**: `0.668759`
- **Test RMSE**: `$4.93k`
- **5-Fold CV $R^2$ Mean**: `0.715222` ($\pm$ `0.037467`)
- **5-Fold CV RMSE Mean**: `$4.84k`

### 2. Business & Feature Interpretation
- **rm (Average Rooms)** has the largest positive impact on housing value: more rooms reflect structural size and premium pricing.
- **lstat (% Lower Status of Population)** has a strong negative impact: higher lower-status concentration is associated with lower home prices.
- **ptratio (Pupil-Teacher Ratio)** has a significant negative impact: higher ratios (crowded schools) depress valuation.
- Features are fully standardized using `StandardScaler` to prevent feature scales (e.g. tax rate) from distorting coefficient magnitudes, ensuring reliable and interpretable modeling.

### 3. Deployed Simulation Prediction
For a new neighborhood profile with crime rate = 0.1, rooms = 6.5, lower status = 12.0%, pupil-teacher ratio = 18.0, and distance to employment centers = 4.0 miles, the deployed model predicts a median house value of **`$27,139.66`** (Scale: `27.14k`).
