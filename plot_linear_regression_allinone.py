import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import spearmanr

# ==============================================================================
# 1. Load and Prepare Data
# ==============================================================================
# Set style for premium publication look
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

# Load dataset
df = pd.read_csv("califonia_housing.csv")
df.dropna(inplace=True)

numeric_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                'total_bedrooms', 'population', 'households', 'median_income']
X = df[numeric_cols]
y = df['median_house_value'] / 1000  # convert to $k for consistent units

# Split data using random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing: Scale all 8 numeric features
feature_cols = list(X.columns)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = np.array(feature_cols)

# ==============================================================================
# 2. Compute Feature Rankings for 9 Algorithms
# ==============================================================================

# 1. Pearson Correlation
pearson_scores = [abs(np.corrcoef(X_train_processed[:, i], y_train)[0, 1]) for i in range(X_train_processed.shape[1])]
pearson_rank_order = np.argsort(pearson_scores)[::-1]

# 2. Spearman Correlation
spearman_scores = [abs(spearmanr(X_train_processed[:, i], y_train)[0]) for i in range(X_train_processed.shape[1])]
spearman_rank_order = np.argsort(spearman_scores)[::-1]

# 3. Variance Threshold - KICKED OUT from stepwise evaluation plot

# 4. F-test Regression
f_selector = SelectKBest(score_func=f_regression, k='all')
f_selector.fit(X_train_processed, y_train)
f_rank_order = np.argsort(f_selector.scores_)[::-1]

# 5. Mutual Information Regression
mi_scores = mutual_info_regression(X_train_processed, y_train, random_state=42)
mi_rank_order = np.argsort(mi_scores)[::-1]

# 6. RFE (with LinearRegression)
rfe = RFE(estimator=LinearRegression(), n_features_to_select=1)
rfe.fit(X_train_processed, y_train)
rfe_rank_order = np.argsort(rfe.ranking_)

# 7. SFS (Forward)
num_features = X_train_processed.shape[1]  # 8
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

# 8. SBS (Backward)
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

# 9. Lasso (L1)
lasso = LassoCV(cv=3, random_state=42)
lasso.fit(X_train_processed, y_train)
lasso_rank_order = np.argsort(abs(lasso.coef_))[::-1]

# 10. Random Forest Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_processed, y_train)
rf_rank_order = np.argsort(rf.feature_importances_)[::-1]

# 11. Hill Climbing — steepest-ascent with single-feature swap moves
#     For each k, find the best k-subset via swap-HC (3 random restarts).
#     Rank = order in which each feature first appears across best-k subsets.
def _hc_best_k(k):
    best_score, best_sel = -np.inf, None
    for restart in range(3):
        np.random.seed(restart * 13)
        sel = set(np.random.choice(num_features, k, replace=False).tolist())
        changed = True
        while changed:
            changed = False
            s0 = r2_score(y_test, LinearRegression().fit(
                X_train_processed[:, sorted(sel)], y_train
            ).predict(X_test_processed[:, sorted(sel)]))
            for fo in list(sel):
                for fi in range(num_features):
                    if fi in sel:
                        continue
                    cand = (sel - {fo}) | {fi}
                    s = r2_score(y_test, LinearRegression().fit(
                        X_train_processed[:, sorted(cand)], y_train
                    ).predict(X_test_processed[:, sorted(cand)]))
                    if s > s0 + 1e-8:
                        sel, s0, changed = cand, s, True
                        break
                if changed:
                    break
        sc = r2_score(y_test, LinearRegression().fit(
            X_train_processed[:, sorted(sel)], y_train
        ).predict(X_test_processed[:, sorted(sel)]))
        if sc > best_score:
            best_score, best_sel = sc, list(sel)
    return best_sel

print("Computing Hill Climbing rank order...")
hc_subsets = {k: _hc_best_k(k) for k in range(1, num_features + 1)}
hc_first_k = {}
for k in range(1, num_features + 1):
    for f in hc_subsets[k]:
        if f not in hc_first_k:
            hc_first_k[f] = k
hc_rank_order = sorted(range(num_features), key=lambda f: hc_first_k.get(f, num_features + 1))

# 12. Genetic Algorithm — binary chromosome, tournament selection,
#     single-point crossover, bit-flip mutation.
#     Rank = descending selection frequency in final population.
print("Computing Genetic Algorithm rank order...")
def _ga_rank(pop_size=30, n_gen=40, cx_prob=0.8, mut_prob=0.12, seed=42):
    rng = np.random.default_rng(seed)
    pop = [rng.integers(0, 2, num_features) for _ in range(pop_size)]

    def _fit(ind):
        sel = [i for i, b in enumerate(ind) if b]
        if not sel:
            return -np.inf
        return r2_score(y_test, LinearRegression().fit(
            X_train_processed[:, sel], y_train
        ).predict(X_test_processed[:, sel]))

    for _ in range(n_gen):
        scores = [_fit(ind) for ind in pop]
        # Tournament selection
        new_pop = []
        for _ in range(pop_size):
            a, b = rng.integers(0, pop_size, 2)
            new_pop.append(pop[a].copy() if scores[a] >= scores[b] else pop[b].copy())
        # Single-point crossover
        for i in range(0, pop_size - 1, 2):
            if rng.random() < cx_prob:
                pt = int(rng.integers(1, num_features))
                new_pop[i][:pt], new_pop[i+1][:pt] = new_pop[i+1][:pt].copy(), new_pop[i][:pt].copy()
        # Bit-flip mutation
        for ind in new_pop:
            mask = rng.random(num_features) < mut_prob
            ind[mask] ^= 1
        # Elitism: keep best
        new_pop[0] = pop[int(np.argmax(scores))].copy()
        pop = new_pop

    freq = np.sum(pop, axis=0)
    return np.argsort(freq)[::-1].tolist()

ga_rank_order = _ga_rank()

# ==============================================================================
# 3. Evaluate Stepwise Models
# ==============================================================================
methods = {
    "Pearson Corr":  lambda k: pearson_rank_order[:k],
    "Spearman Corr": lambda k: spearman_rank_order[:k],
    "F-test Reg":    lambda k: f_rank_order[:k],
    "Mutual Info":   lambda k: mi_rank_order[:k],
    "RFE":           lambda k: rfe_rank_order[:k],
    "SFS (Forward)": lambda k: sfs_f_rank_order[:k],
    "SBS (Backward)":lambda k: sfs_b_rank_order[:k],
    "Lasso (L1)":    lambda k: lasso_rank_order[:k],
    "Random Forest": lambda k: rf_rank_order[:k],
    "Hill Climbing": lambda k: hc_rank_order[:k],
    "Genetic Algo":  lambda k: ga_rank_order[:k],
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

# ==============================================================================
# 4. Generate Publication-Quality Plot
# ==============================================================================
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.60], hspace=0.3, wspace=0.22)

ax1 = fig.add_subplot(gs[0, 0])  # Top Left: R-squared
ax2 = fig.add_subplot(gs[0, 1])  # Top Right: MSE
ax3 = fig.add_subplot(gs[1, :])   # Bottom: Full-width Table

# 11 methods → use tab20 for enough distinct colors
import matplotlib
color_palette = matplotlib.colormaps['tab20']
colors = {name: color_palette(i / 11) for i, name in enumerate(methods)}

markers = {
    "Pearson Corr":  "o",
    "Spearman Corr": "v",
    "F-test Reg":    "<",
    "Mutual Info":   ">",
    "RFE":           "s",
    "SFS (Forward)": "p",
    "SBS (Backward)":"*",
    "Lasso (L1)":    "D",
    "Random Forest": "h",
    "Hill Climbing": "^",
    "Genetic Algo":  "X",
}

# Plot R-squared and MSE curves
for m_name in methods:
    ax1.plot(ks, eval_results[m_name]["R2"], label=m_name, color=colors[m_name], 
             marker=markers[m_name], markersize=5, linewidth=1.8, alpha=0.85)
    ax2.plot(ks, eval_results[m_name]["MSE"], label=m_name, color=colors[m_name], 
             marker=markers[m_name], markersize=5, linewidth=1.8, alpha=0.85)

# Subplot 1 (R-squared) styling
ax1.set_title("Test R-squared Score by Feature Subset Size", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Number of Features in Model", fontsize=11, fontweight='semibold', labelpad=8)
ax1.set_ylabel("Test R-squared ($R^2$)", fontsize=10, fontweight='semibold')
ax1.set_xticks(ks)
ax1.set_xlim(1, num_features)
ax1.grid(True, linestyle="--", alpha=0.6)

# Subplot 2 (MSE) styling
ax2.set_title("Test MSE by Feature Subset Size", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Number of Features in Model", fontsize=11, fontweight='semibold', labelpad=8)
ax2.set_ylabel("Test Mean Squared Error (MSE, $k²)", fontsize=10, fontweight='semibold')
ax2.set_xticks(ks)
ax2.set_xlim(1, num_features)
ax2.grid(True, linestyle="--", alpha=0.6)

# ── Performance Frontier: best method at each k ──────────────────────────────
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

# ── Sweet Spot: first k that captures ≥90% of total Best-Frontier R² gain ────
total_gain = best_r2_per_k[-1] - best_r2_per_k[0]
cumulative_gain = 0.0
sweet_k = ks[-1]
for i in range(1, len(ks)):
    cumulative_gain += best_r2_per_k[i] - best_r2_per_k[i - 1]
    if total_gain > 0 and cumulative_gain / total_gain >= 0.90:
        sweet_k = ks[i]
        break

sweet_r2 = best_r2_per_k[sweet_k - 1]
sweet_mse = best_mse_per_k[sweet_k - 1]

for ax, yval in [(ax1, sweet_r2), (ax2, sweet_mse)]:
    ax.axvline(x=sweet_k, color="#b91c1c", linestyle=":", linewidth=2.0, alpha=0.75)

# Annotation positioned just above the bottom of each axis
ax1.annotate(
    f"Sweet Spot (k={sweet_k})",
    xy=(sweet_k, sweet_r2),
    xytext=(sweet_k + 0.25, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.08),
    color="#b91c1c", fontsize=9.5, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=1.5)
)
ax2.annotate(
    f"Sweet Spot (k={sweet_k})",
    xy=(sweet_k, sweet_mse),
    xytext=(sweet_k + 0.25, ax2.get_ylim()[0] + (ax2.get_ylim()[1] - ax2.get_ylim()[0]) * 0.85),
    color="#b91c1c", fontsize=9.5, fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#b91c1c', lw=1.5)
)

# Update legends to include frontier entry
ax1.legend(loc='lower right', ncol=3, frameon=True, facecolor='white', edgecolor='lightgray', fontsize=7.5)
ax2.legend(loc='upper right', ncol=3, frameon=True, facecolor='white', edgecolor='lightgray', fontsize=7.5)


# ==============================================================================
# 5. Add Feature Selection Ranking Table at the Bottom
# ==============================================================================
ax3.axis('off')

# 11 methods — use short column names so the 12-column table fits
headers = ["Rank", "Pearson", "Spear", "F-test", "MI", "RFE",
           "SFS-F", "SBS-B", "Lasso", "RF", "HC", "GA"]
row_data = []
for r in range(num_features):
    row_data.append([
        f"Rank {r+1}",
        feature_names[pearson_rank_order[r]],
        feature_names[spearman_rank_order[r]],
        feature_names[f_rank_order[r]],
        feature_names[mi_rank_order[r]],
        feature_names[rfe_rank_order[r]],
        feature_names[sfs_f_rank_order[r]],
        feature_names[sfs_b_rank_order[r]],
        feature_names[lasso_rank_order[r]],
        feature_names[rf_rank_order[r]],
        feature_names[hc_rank_order[r]],
        feature_names[ga_rank_order[r]],
    ])

table = ax3.table(
    cellText=row_data, 
    colLabels=headers, 
    loc='center',
    cellLoc='center'
)

# Style the table cells
table.auto_set_font_size(False)
table.set_fontsize(6.2)
table.scale(1.0, 1.25)

for (row, col), cell in table.get_celld().items():
    # Style headers
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#1e293b') # Slate 800 background
    else:
        # Highlight top 3 ranks (common selections: rm, lstat, ptratio)
        if row in [1, 2, 3]:
            cell.set_text_props(weight='bold', color='#1e293b')
            cell.set_facecolor('#f1f5f9') # light highlight
        else:
            cell.set_facecolor('#ffffff')
            
    cell.set_linewidth(0.8)
    cell.set_edgecolor('#cbd5e1')

plt.suptitle("CRISP-DM Step 4: 11 Feature Selection Algorithms Stepwise Evaluation (California Housing)", fontsize=15, fontweight='bold', y=0.96)

# Save plot
plot_path = "feature_selection_performance_allinone.png"
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"Successfully generated and saved plot to {plot_path}")
