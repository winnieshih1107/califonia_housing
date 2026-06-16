# ============================================================
# CRISP-DM Step 4: Modeling with Feature Selection Algorithms
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr


# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("califonia_housing.csv")
df.dropna(inplace=True)

X = df.drop(columns=['median_house_value', 'ocean_proximity'])
y = df['median_house_value']

feature_cols = list(X.columns)

# ============================================================
# Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols)
    ]
)


# ============================================================
# Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# Transform Data First
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = np.array(feature_cols)

print("Processed Feature Names:")
print(feature_names)


# ============================================================
# Helper Function for Evaluation
# ============================================================

def evaluate_model(name, X_train_selected, X_test_selected, selected_features):
    model = LinearRegression()
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_test_selected)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return {
        "Method": name,
        "Selected Features": list(selected_features),
        "Number of Features": len(selected_features),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


results = []
top_k = 3

# ============================================================
# Method 1: Pearson Correlation
# ============================================================

pearson_scores = []

for i in range(X_train_processed.shape[1]):
    corr = np.corrcoef(X_train_processed[:, i], y_train)[0, 1]
    pearson_scores.append(abs(corr))

pearson_scores = np.array(pearson_scores)
pearson_indices = np.argsort(pearson_scores)[-top_k:]

results.append(
    evaluate_model(
        "Pearson Correlation",
        X_train_processed[:, pearson_indices],
        X_test_processed[:, pearson_indices],
        feature_names[pearson_indices]
    )
)


# ============================================================
# Method 2: Spearman Correlation
# ============================================================

spearman_scores = []

for i in range(X_train_processed.shape[1]):
    corr, _ = spearmanr(X_train_processed[:, i], y_train)
    spearman_scores.append(abs(corr))

spearman_scores = np.array(spearman_scores)
spearman_indices = np.argsort(spearman_scores)[-top_k:]

results.append(
    evaluate_model(
        "Spearman Correlation",
        X_train_processed[:, spearman_indices],
        X_test_processed[:, spearman_indices],
        feature_names[spearman_indices]
    )
)


# ============================================================
# Method 3: Variance Threshold
# ============================================================

# Note: since we scaled inputs, variance is 1.0. Let's apply VarianceThreshold
# on raw training features and then transform/evaluate.
variance_selector = VarianceThreshold(threshold=5.0) # filters columns with low variance in raw data
X_train_var = variance_selector.fit_transform(X_train)
X_test_var = variance_selector.transform(X_test)
var_features = feature_names[variance_selector.get_support()]

# Standardize the selected subset for model evaluation
sub_scaler = StandardScaler()
X_train_var_scaled = sub_scaler.fit_transform(X_train_var)
X_test_var_scaled = sub_scaler.transform(X_test_var)

results.append(
    evaluate_model(
        "Variance Threshold",
        X_train_var_scaled,
        X_test_var_scaled,
        var_features
    )
)


# ============================================================
# Method 4: F-test for Regression
# ============================================================

f_selector = SelectKBest(score_func=f_regression, k=top_k)

X_train_f = f_selector.fit_transform(X_train_processed, y_train)
X_test_f = f_selector.transform(X_test_processed)

f_features = feature_names[f_selector.get_support()]

results.append(
    evaluate_model(
        "F-test Regression",
        X_train_f,
        X_test_f,
        f_features
    )
)


# ============================================================
# Method 5: Mutual Information Regression
# ============================================================

mi_selector = SelectKBest(score_func=mutual_info_regression, k=top_k)

# Using random_state=42 for reproducibility
# Note: mutual_info_regression might be stochastic depending on internal seed.
np.random.seed(42)
X_train_mi = mi_selector.fit_transform(X_train_processed, y_train)
X_test_mi = mi_selector.transform(X_test_processed)

mi_features = feature_names[mi_selector.get_support()]

results.append(
    evaluate_model(
        "Mutual Information",
        X_train_mi,
        X_test_mi,
        mi_features
    )
)


# ============================================================
# Method 6: Recursive Feature Elimination, RFE
# ============================================================

rfe_selector = RFE(
    estimator=LinearRegression(),
    n_features_to_select=top_k
)

X_train_rfe = rfe_selector.fit_transform(X_train_processed, y_train)
X_test_rfe = rfe_selector.transform(X_test_processed)

rfe_features = feature_names[rfe_selector.get_support()]

results.append(
    evaluate_model(
        "RFE Linear Regression",
        X_train_rfe,
        X_test_rfe,
        rfe_features
    )
)


# ============================================================
# Method 7: Sequential Forward Selection
# ============================================================

sfs_forward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select=top_k,
    direction="forward",
    scoring="r2",
    cv=5
)

X_train_sfs_forward = sfs_forward.fit_transform(X_train_processed, y_train)
X_test_sfs_forward = sfs_forward.transform(X_test_processed)

sfs_forward_features = feature_names[sfs_forward.get_support()]

results.append(
    evaluate_model(
        "Sequential Forward Selection",
        X_train_sfs_forward,
        X_test_sfs_forward,
        sfs_forward_features
    )
)


# ============================================================
# Method 8: Sequential Backward Selection
# ============================================================

sfs_backward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    n_features_to_select=top_k,
    direction="backward",
    scoring="r2",
    cv=5
)

X_train_sfs_backward = sfs_backward.fit_transform(X_train_processed, y_train)
X_test_sfs_backward = sfs_backward.transform(X_test_processed)

sfs_backward_features = feature_names[sfs_backward.get_support()]

results.append(
    evaluate_model(
        "Sequential Backward Selection",
        X_train_sfs_backward,
        X_test_sfs_backward,
        sfs_backward_features
    )
)


# ============================================================
# Method 9: Lasso Feature Selection
# ============================================================

lasso_selector = SelectFromModel(
    estimator=LassoCV(cv=5, random_state=42),
    threshold="mean"
)

X_train_lasso = lasso_selector.fit_transform(X_train_processed, y_train)
X_test_lasso = lasso_selector.transform(X_test_processed)

lasso_features = feature_names[lasso_selector.get_support()]

results.append(
    evaluate_model(
        "Lasso L1 Selection",
        X_train_lasso,
        X_test_lasso,
        lasso_features
    )
)


# ============================================================
# Method 10: Tree-Based Feature Importance
# ============================================================

tree_selector = SelectFromModel(
    estimator=RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),
    threshold="mean"
)

X_train_tree = tree_selector.fit_transform(X_train_processed, y_train)
X_test_tree = tree_selector.transform(X_test_processed)

tree_features = feature_names[tree_selector.get_support()]

results.append(
    evaluate_model(
        "Random Forest Importance",
        X_train_tree,
        X_test_tree,
        tree_features
    )
)


# ============================================================
# Compare All Feature Selection Methods
# ============================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False)

print("\nFeature Selection Comparison:")
print(results_df)


# ============================================================
# Best Method
# ============================================================

best_method = results_df.iloc[0]

print("\nBest Feature Selection Method:")
print("Method:", best_method["Method"])
print("Selected Features:", best_method["Selected Features"])
print("R2:", best_method["R2"])
print("RMSE:", best_method["RMSE"])
