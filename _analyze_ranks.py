import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, SequentialFeatureSelector
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

df = pd.read_csv('califonia_housing.csv'); df.dropna(inplace=True)
numeric_cols = ['longitude','latitude','housing_median_age','total_rooms',
                'total_bedrooms','population','households','median_income']
X = df[numeric_cols]; y = df['median_house_value'] / 1000
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
pre = ColumnTransformer([('num',StandardScaler(),numeric_cols)])
Xtr = pre.fit_transform(X_train); Xte = pre.transform(X_test)
fn = np.array(numeric_cols); n = len(fn)

# 1. Pearson
pearson = np.argsort([abs(np.corrcoef(Xtr[:,i],y_train)[0,1]) for i in range(n)])[::-1]
# 2. Spearman
spearman = np.argsort([abs(spearmanr(Xtr[:,i],y_train)[0]) for i in range(n)])[::-1]
# 3. F-test
fsel = SelectKBest(f_regression, k='all').fit(Xtr,y_train)
ftest = np.argsort(fsel.scores_)[::-1]
# 4. MI
mi = np.argsort(mutual_info_regression(Xtr,y_train,random_state=42))[::-1]
# 5. RFE
rfe = RFE(LinearRegression(),n_features_to_select=1).fit(Xtr,y_train)
rfe_r = np.argsort(rfe.ranking_)
# 6. Lasso
las = LassoCV(cv=3,random_state=42).fit(Xtr,y_train)
lasso_r = np.argsort(abs(las.coef_))[::-1]
# 7. RF
rf = RandomForestRegressor(100,random_state=42).fit(Xtr,y_train)
rf_r = np.argsort(rf.feature_importances_)[::-1]

# 8. SFS Forward
sfs_f_feats = []
for k in range(1, n+1):
    if k == n:
        sfs_f_feats.append(np.arange(n))
    else:
        s = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='forward', cv=3).fit(Xtr, y_train)
        sfs_f_feats.append(np.where(s.get_support())[0])
sfs_f = []
for k in range(n):
    for idx in sfs_f_feats[k]:
        if idx not in sfs_f:
            sfs_f.append(idx); break
for i in range(n):
    if i not in sfs_f: sfs_f.append(i)

# 9. SBS Backward
sfs_b_feats = []
for k in range(1, n+1):
    if k == n:
        sfs_b_feats.append(np.arange(n))
    else:
        s = SequentialFeatureSelector(LinearRegression(), n_features_to_select=k, direction='backward', cv=3).fit(Xtr, y_train)
        sfs_b_feats.append(np.where(sfs_b_feats[-1].get_support() if False else s.get_support())[0])
sfs_b = [sfs_b_feats[0][0]]
for k in range(1, n):
    for idx in sfs_b_feats[k]:
        if idx not in sfs_b:
            sfs_b.append(idx); break
for i in range(n):
    if i not in sfs_b: sfs_b.append(i)

# 10. Hill Climbing
def hc_best_k(k):
    best_s, best_sel = -np.inf, None
    for restart in range(3):
        np.random.seed(restart * 13)
        sel = set(np.random.choice(n, k, replace=False).tolist())
        changed = True
        while changed:
            changed = False
            s0 = r2_score(y_test, LinearRegression().fit(Xtr[:,sorted(sel)], y_train).predict(Xte[:,sorted(sel)]))
            for fo in list(sel):
                for fi in range(n):
                    if fi in sel: continue
                    cand = (sel - {fo}) | {fi}
                    s = r2_score(y_test, LinearRegression().fit(Xtr[:,sorted(cand)], y_train).predict(Xte[:,sorted(cand)]))
                    if s > s0 + 1e-8:
                        sel, s0, changed = cand, s, True; break
                if changed: break
        sc = r2_score(y_test, LinearRegression().fit(Xtr[:,sorted(sel)], y_train).predict(Xte[:,sorted(sel)]))
        if sc > best_s: best_s, best_sel = sc, list(sel)
    return best_sel

hc_sub = {k: hc_best_k(k) for k in range(1, n+1)}
hc_fk = {}
for k in range(1, n+1):
    for f in hc_sub[k]:
        if f not in hc_fk: hc_fk[f] = k
hc_r = sorted(range(n), key=lambda f: hc_fk.get(f, n+1))

# 11. GA
def ga_rank():
    rng = np.random.default_rng(42)
    pop = [rng.integers(0, 2, n) for _ in range(30)]
    def fit(ind):
        sel = [i for i, b in enumerate(ind) if b]
        if not sel: return -np.inf
        return r2_score(y_test, LinearRegression().fit(Xtr[:,sel], y_train).predict(Xte[:,sel]))
    for _ in range(40):
        scores = [fit(ind) for ind in pop]
        new_pop = []
        for _ in range(30):
            a, b = rng.integers(0, 30, 2)
            new_pop.append(pop[a].copy() if scores[a] >= scores[b] else pop[b].copy())
        for i in range(0, 29, 2):
            if rng.random() < 0.8:
                pt = int(rng.integers(1, n))
                new_pop[i][:pt], new_pop[i+1][:pt] = new_pop[i+1][:pt].copy(), new_pop[i][:pt].copy()
        for ind in new_pop:
            mask = rng.random(n) < 0.12; ind[mask] ^= 1
        new_pop[0] = pop[int(np.argmax(scores))].copy()
        pop = new_pop
    freq = np.sum(pop, axis=0)
    return np.argsort(freq)[::-1].tolist()

ga_r = ga_rank()

# Compile all ranks
all_ranks = {
    'Pearson': pearson, 'Spearman': spearman, 'F-test': ftest, 'MI': mi,
    'RFE': rfe_r, 'SFS-F': sfs_f, 'SBS-B': sfs_b, 'Lasso': lasso_r,
    'RF': rf_r, 'HC': hc_r, 'GA': ga_r
}

# Print rank table
print("\n=== FEATURE RANK TABLE (all 11 algorithms) ===")
header = f"{'Feature':<22}"
for m in all_ranks: header += f"{m:>9}"
header += f"  {'AvgRank':>8}  {'Votes@k<=3':>10}  {'Votes@k<=5':>10}"
print(header)
print('-' * 125)
summary_rows = []
for fi in range(n):
    ranks_for_feat = [list(v).index(fi)+1 for v in all_ranks.values()]
    avg = np.mean(ranks_for_feat)
    top3 = sum(1 for r in ranks_for_feat if r <= 3)
    top5 = sum(1 for r in ranks_for_feat if r <= 5)
    row = f"{fn[fi]:<22}"
    for r in ranks_for_feat: row += f"{r:>9}"
    row += f"  {avg:>8.2f}  {top3:>10}  {top5:>10}"
    print(row)
    summary_rows.append((fn[fi], avg, top3, top5, ranks_for_feat))

# Best R2 per k (frontier)
print("\n=== STEPWISE FRONTIER R2 ===")
methods_list = list(all_ranks.items())
for k in range(1, n+1):
    best_r2 = -np.inf; best_m = ''
    for m, order in methods_list:
        idx = list(order)[:k]
        r2 = r2_score(y_test, LinearRegression().fit(Xtr[:,idx], y_train).predict(Xte[:,idx]))
        if r2 > best_r2: best_r2 = r2; best_m = m
    print(f"k={k}: Best R2={best_r2:.4f}  (best method: {best_m})")

# Individual feature R2 when used alone
print("\n=== SINGLE FEATURE R2 ===")
for fi in range(n):
    r2 = r2_score(y_test, LinearRegression().fit(Xtr[:,[fi]], y_train).predict(Xte[:,[fi]]))
    print(f"  {fn[fi]:<25} R2={r2:.4f}")
