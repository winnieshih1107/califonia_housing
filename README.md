# L7-Multiple Linear Regression Workflow

This repository implements a modular Scikit-learn regression solution to predict California median house values based on geographic, demographic, and structural indicators, adhering strictly to the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology.

---

## 📂 Project Structure & Deliverables

- **[solve_linear_regression_crispdm.py](solve_linear_regression_crispdm.py)**: The main CRISP-DM pipeline script containing modular functions representing Business Understanding, Data Understanding (EDA), Data Preparation (Pipeline with `ColumnTransformer` + `StandardScaler`), Modeling, Evaluation (5-Fold Cross Validation), and Deployment.
- **[califonia_housing.csv](califonia_housing.csv)**: The primary California Housing dataset containing 20,000+ rows and 9 columns.
- **[linear_regression_model.pkl](linear_regression_model.pkl)**: Serialized final pipeline model (refitted on all samples using the optimal features).
- **[solve_linear_regression_feature_selection.py](solve_linear_regression_feature_selection.py)**: Script to calculate and compare 9 top feature selection algorithms.
- **[plot_linear_regression_allinone.py](plot_linear_regression_allinone.py)**: Generates a unified stepwise feature selection comparison plot.
- **[solve_linear_regression_dashboard.py](solve_linear_regression_dashboard.py)**: Interactive Streamlit web dashboard with feature selection charts and live house price simulator.
- **[feature_selection_performance_allinone.png](feature_selection_performance_allinone.png)**: Comparative visual comparing 9 top feature selection algorithms (Pearson, Spearman, F-test, Mutual Info, RFE, SFS, SBS, Lasso, and Random Forest).
- **[run_workflow.py](run_workflow.py)**: General-purpose automated pipeline for feature selection analysis on any tabular dataset.

---

## 🚀 Execution & Running Instructions

Make sure the required libraries are installed:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn streamlit altair
```

### 1. Run the Main Machine Learning Pipeline
Executes the full 6-step CRISP-DM pipeline, evaluates feature-set models, prints selection justifications, and runs the deployment simulation:
```bash
python solve_linear_regression_crispdm.py
```

### 2. Run the Feature Selection Comparison Script
Calculates the top feature selections for each of the 9 methods at a target subset size:
```bash
python solve_linear_regression_feature_selection.py
```

### 3. Generate the Stepwise Feature Selection Comparison Plot
Fits and compares the feature selection paths for all 9 algorithms as subset size k goes from 1 to all features:
```bash
python plot_linear_regression_allinone.py
```

### 4. Run the Interactive Web Dashboard
Launches the interactive Streamlit web dashboard (executes all feature selection methods, shows consensus charts, and includes a live house price simulator):
```bash
streamlit run solve_linear_regression_dashboard.py
```
Or simply double-click **`run_dashboard.bat`** in Windows Explorer.

---

## 📈 Summary of Results

### 1. Dataset
The California Housing dataset contains **20,640 rows** and the following features:

| Feature | Description |
|---|---|
| `MedInc` | Median income in block group |
| `HouseAge` | Median house age in block group |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Block group population |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude |
| `Longitude` | Block group longitude |
| `MedHouseVal` | **Target**: Median house value (in $100,000s) |

### 2. Business & Feature Interpretation
- **MedInc (Median Income)** has the largest positive impact: higher income neighborhoods command significantly higher house values.
- **Latitude / Longitude** capture geographic effects, reflecting California's coastal premium and regional price disparities.
- **AveOccup (Average Occupancy)** has a negative impact: higher occupancy per household is associated with lower property values.
- Features are fully standardized using `StandardScaler` to ensure reliable and interpretable coefficient magnitudes.
