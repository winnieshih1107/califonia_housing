import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

def main():
    for candidate in ["califonia_housing.csv", "data.csv"]:
        if os.path.exists(candidate):
            dataset_file = candidate
            break
    else:
        print("Error: No dataset found. Please place 'califonia_housing.csv' or 'data.csv' in the current directory.")
        sys.exit(1)

    print(f"Loading dataset '{dataset_file}'...")
    df = pd.read_csv(dataset_file)
    df.dropna(inplace=True)
    print(f"Dataset loaded successfully. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

    # 1. Detect target and features
    if "medv" in df.columns:
        target_col = "medv"
    elif "median_house_value" in df.columns:
        target_col = "median_house_value"
    elif "Profit" in df.columns:
        target_col = "Profit"
    else:
        target_col = df.columns[-1]
    
    print(f"Target column detected: '{target_col}'")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numerical features: {numeric_cols}")
    print(f"Categorical features: {categorical_cols}")

    # 3. Preprocessing Pipeline
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # 4. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform features
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Extract feature names after preprocessing
    feature_names = []
    if numeric_cols:
        feature_names.extend(numeric_cols)
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        cat_names = cat_encoder.get_feature_names_out(categorical_cols).tolist()
        feature_names.extend(cat_names)
        
    feature_names = np.array(feature_names)
    num_features = X_train_processed.shape[1]
    print(f"Total processed features: {num_features}")
    print(f"Processed feature list: {list(feature_names)}")

    # 5. Compute Rankings
    print("\nComputing feature rankings for 9 algorithms...")
    
    # 1. Pearson
    pearson_scores = [abs(np.corrcoef(X_train_processed[:, i], y_train)[0, 1]) for i in range(num_features)]
    pearson_rank_order = np.argsort(pearson_scores)[::-1]

    # 2. Spearman
    spearman_scores = [abs(spearmanr(X_train_processed[:, i], y_train)[0]) for i in range(num_features)]
    spearman_rank_order = np.argsort(spearman_scores)[::-1]

    # 3. F-test
    f_selector = SelectKBest(score_func=f_regression, k='all')
    f_selector.fit(X_train_processed, y_train)
    f_rank_order = np.argsort(f_selector.scores_)[::-1]

    # 4. Mutual Info
    mi_scores = mutual_info_regression(X_train_processed, y_train, random_state=42)
    mi_rank_order = np.argsort(mi_scores)[::-1]

    # 5. RFE (with LinearRegression)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
    rfe.fit(X_train_processed, y_train)
    rfe_rank_order = np.argsort(rfe.ranking_)

    # 6. SFS (Forward)
    sfs_forward_features = []
    for k in range(1, num_features + 1):
        if k == num_features:
            sfs_forward_features.append(np.arange(num_features))
        else:
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="forward", cv=3)
            sfs.fit(X_train_processed, y_train)
            sfs_forward_features.append(np.where(sfs.get_support())[0])

    sfs_f_rank_order = []
    for k in range(num_features):
        for idx in sfs_forward_features[k]:
            if idx not in sfs_f_rank_order:
                sfs_f_rank_order.append(idx)
                break
    for i in range(num_features):
        if i not in sfs_f_rank_order:
            sfs_f_rank_order.append(i)

    # 7. SBS (Backward)
    sfs_backward_features = []
    for k in range(1, num_features + 1):
        if k == num_features:
            sfs_backward_features.append(np.arange(num_features))
        else:
            sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction="backward", cv=3)
            sfs.fit(X_train_processed, y_train)
            sfs_backward_features.append(np.where(sfs.get_support())[0])

    sfs_b_rank_order = []
    sfs_b_rank_order.append(sfs_backward_features[0][0])
    for k in range(1, num_features):
        for idx in sfs_backward_features[k]:
            if idx not in sfs_b_rank_order:
                sfs_b_rank_order.append(idx)
                break
    for i in range(num_features):
        if i not in sfs_b_rank_order:
            sfs_b_rank_order.append(i)

    # 8. Lasso (L1)
    lasso = LassoCV(cv=3, random_state=42)
    lasso.fit(X_train_processed, y_train)
    lasso_rank_order = np.argsort(abs(lasso.coef_))[::-1]

    # 9. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_processed, y_train)
    rf_rank_order = np.argsort(rf.feature_importances_)[::-1]

    # 6. Evaluate Stepwise Models
    print("Evaluating models stepwise...")
    methods = {
        "Pearson Corr": lambda k: pearson_rank_order[:k],
        "Spearman Corr": lambda k: spearman_rank_order[:k],
        "F-test Reg": lambda k: f_rank_order[:k],
        "Mutual Info": lambda k: mi_rank_order[:k],
        "RFE": lambda k: rfe_rank_order[:k],
        "SFS (Forward)": lambda k: sfs_f_rank_order[:k],
        "SBS (Backward)": lambda k: sfs_b_rank_order[:k],
        "Lasso (L1)": lambda k: lasso_rank_order[:k],
        "Random Forest": lambda k: rf_rank_order[:k]
    }

    eval_results = {name: {"R2": [], "MSE": []} for name in methods}
    ks = list(range(1, num_features + 1))

    for m_name, get_indices in methods.items():
        for k in ks:
            indices = get_indices(k)
            X_tr = X_train_processed[:, indices]
            X_te = X_test_processed[:, indices]
            
            model = LinearRegression()
            model.fit(X_tr, y_train)
            y_pred = model.predict(X_te)
            
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            eval_results[m_name]["R2"].append(r2)
            eval_results[m_name]["MSE"].append(mse)

    # 7. Generate Plot
    print("Generating publication-quality visualization...")
    sns.set_theme(style="whitegrid")
    
    fig = plt.figure(figsize=(16, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.65], hspace=0.35, wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])  # Top Left: R2
    ax2 = fig.add_subplot(gs[0, 1])  # Top Right: MSE
    ax3 = fig.add_subplot(gs[1, :])   # Bottom: Table

    import matplotlib
    color_palette = matplotlib.colormaps['tab10']
    colors = {name: color_palette(i) for i, name in enumerate(methods)}

    markers = {
        "Pearson Corr": "o", "Spearman Corr": "v", "F-test Reg": "<", "Mutual Info": ">",
        "RFE": "s", "SFS (Forward)": "p", "SBS (Backward)": "*", "Lasso (L1)": "D", "Random Forest": "h"
    }

    # Plot lines
    for m_name in methods:
        ax1.plot(ks, eval_results[m_name]["R2"], label=m_name, color=colors[m_name], 
                 marker=markers[m_name], markersize=5, linewidth=1.8, alpha=0.85)
        ax2.plot(ks, eval_results[m_name]["MSE"], label=m_name, color=colors[m_name], 
                 marker=markers[m_name], markersize=5, linewidth=1.8, alpha=0.85)

    # Format Subplot 1
    ax1.set_title("Test R-squared Score by Feature Subset Size", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Number of Features in Model", fontsize=11, fontweight='semibold', labelpad=8)
    ax1.set_ylabel("Test R-squared ($R^2$)", fontsize=10, fontweight='semibold')
    ax1.set_xticks(ks)
    ax1.set_xlim(1, num_features)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Format Subplot 2
    ax2.set_title("Test MSE by Feature Subset Size", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Number of Features in Model", fontsize=11, fontweight='semibold', labelpad=8)
    ax2.set_ylabel("Test Mean Squared Error (MSE)", fontsize=10, fontweight='semibold')
    ax2.set_xticks(ks)
    ax2.set_xlim(1, num_features)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Performance Frontier
    best_r2_per_k = []
    best_mse_per_k = []
    for k in ks:
        r2_vals = [eval_results[m]["R2"][k-1] for m in methods]
        mse_vals = [eval_results[m]["MSE"][k-1] for m in methods]
        best_r2_per_k.append(max(r2_vals))
        best_mse_per_k.append(min(mse_vals))

    ax1.plot(ks, best_r2_per_k, color="#f59e0b", linewidth=2.5, linestyle="--",
             marker="D", markersize=6, zorder=10, label="Best (Frontier)", alpha=0.95)
    ax2.plot(ks, best_mse_per_k, color="#f59e0b", linewidth=2.5, linestyle="--",
             marker="D", markersize=6, zorder=10, label="Best (Frontier)", alpha=0.95)

    ax1.legend(loc='lower right', ncol=2, frameon=True, facecolor='white', edgecolor='lightgray', fontsize=8.5)
    ax2.legend(loc='upper right', ncol=2, frameon=True, facecolor='white', edgecolor='lightgray', fontsize=8.5)

    # Bottom table details (only print top 10 ranks for readability if features are many)
    table_ranks = min(num_features, 10)
    
    # Table headers
    headers = ["Rank", "Pearson", "Spearman", "F-test", "Mutual Info", "RFE", "SFS (Fwd)", "SBS (Bwd)", "Lasso (L1)", "Random Forest"]
    row_data = []
    for r in range(table_ranks):
        row_data.append([
            f"Rank {r+1}",
            feature_names[pearson_rank_order[r]] if r < len(pearson_rank_order) else "",
            feature_names[spearman_rank_order[r]] if r < len(spearman_rank_order) else "",
            feature_names[f_rank_order[r]] if r < len(f_rank_order) else "",
            feature_names[mi_rank_order[r]] if r < len(mi_rank_order) else "",
            feature_names[rfe_rank_order[r]] if r < len(rfe_rank_order) else "",
            feature_names[sfs_f_rank_order[r]] if r < len(sfs_f_rank_order) else "",
            feature_names[sfs_b_rank_order[r]] if r < len(sfs_b_rank_order) else "",
            feature_names[lasso_rank_order[r]] if r < len(lasso_rank_order) else "",
            feature_names[rf_rank_order[r]] if r < len(rf_rank_order) else ""
        ])

    ax3.axis('off')
    table = ax3.table(cellText=row_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.25)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#1e293b')
        else:
            if row in [1, 2, 3]:
                cell.set_text_props(weight='bold', color='#1e293b')
                cell.set_facecolor('#f1f5f9')
            else:
                cell.set_facecolor('#ffffff')
        cell.set_linewidth(0.8)
        cell.set_edgecolor('#cbd5e1')

    plt.suptitle("General CRISP-DM Feature Selection Stepwise Evaluation Plot", fontsize=15, fontweight='bold', y=0.96)
    
    plot_path = "feature_selection_performance_allinone.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully generated and saved plot to '{plot_path}'!")
    print(f"Visual shows R2 and MSE metrics for features 1 to {num_features}.")

if __name__ == "__main__":
    main()
