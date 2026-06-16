import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SequentialFeatureSelector,
    SelectFromModel
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# ==============================================================================
# Page Configuration & Styling
# ==============================================================================
st.set_page_config(
    page_title="L7-Multiple Linear Regression Workflow",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Google Fonts & Custom CSS Injection for Modern Dark Mode Aesthetics
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Main body and app container alignment */
    .stApp {
        background: radial-gradient(circle at 90% 10%, #1e1b4b 0%, #0f172a 60%, #090d16 100%);
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #f1f5f9;
    }
    
    /* Headers & Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }
    
    .gradient-title {
        background: linear-gradient(135deg, #c084fc 0%, #818cf8 50%, #38bdf8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .gradient-subtitle {
        color: #94a3b8;
        font-size: 1.1rem !important;
        font-weight: 400 !important;
        margin-bottom: 2rem !important;
    }
    
    /* Custom Card Design (Glassmorphism) */
    .glass-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.3);
        box-shadow: 0 12px 40px 0 rgba(129, 140, 248, 0.15);
    }
    
    /* KPI Metric styling overrides */
    div[data-testid="stMetricValue"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #94a3b8 !important;
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Custom badges */
    .feature-badge {
        display: inline-block;
        background: rgba(129, 140, 248, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 4px;
        font-weight: 600;
    }
    
    .feature-badge-gold {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 4px;
        font-weight: 600;
    }
    
    /* Interactive Element Enhancements */
    .stSlider > div {
        padding-top: 1rem;
    }
    
    /* Footer Style */
    .footer-text {
        text-align: center;
        color: #64748b;
        font-size: 0.85rem;
        margin-top: 4rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# Helper Data Sourcing & Preprocessing
# ==============================================================================
@st.cache_data
def load_and_preprocess_data(sample_size: int = 5000):
    for path in ["califonia_housing.csv", "data.csv"]:
        try:
            data = pd.read_csv(path)
            data.dropna(inplace=True)
            if 'median_house_value' in data.columns:
                data['median_house_value'] = data['median_house_value'] / 1000
            # Sample for dashboard responsiveness on large datasets
            if len(data) > sample_size:
                data = data.sample(n=sample_size, random_state=42).reset_index(drop=True)
            return data
        except FileNotFoundError:
            continue
    raise FileNotFoundError("No dataset found. Please provide 'califonia_housing.csv'.")

df = load_and_preprocess_data()

# ==============================================================================
# Sidebar - Configuration Panel
# ==============================================================================
st.sidebar.markdown("### ⚙️ Split & Selection Settings")
test_size = st.sidebar.slider("Test Split Size (%)", min_value=10, max_value=40, value=20, step=5) / 100.0
random_seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=1000, value=42)
top_k = st.sidebar.slider("Target Number of Features (k)", min_value=1, max_value=8, value=3)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 Model Architecture Selection")
model_type = st.sidebar.selectbox(
    "Choose Evaluator Model:",
    ["Linear Regression", "Lasso (L1 Regularized)", "Random Forest Regressor", "Decision Tree Regressor"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 View Navigator")
view_options = [
    "📋 Complete Comparison Table",
    "🔍 Detailed Method Inspector",
    "🔮 Live House Price Simulator",
    "📈 Stepwise Curves (All-in-One)",
    "🏆 Best Method per k",
]
selected_view = st.sidebar.radio(
    "Select Analysis View:",
    view_options,
    index=0,
    label_visibility="collapsed"
)
st.sidebar.markdown("---")

# Setup data partitions — numeric features only
numeric_feature_cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                        'total_bedrooms', 'population', 'households', 'median_income']
feature_cols = [c for c in numeric_feature_cols if c in df.columns]
X = df[feature_cols]
y = df['median_house_value']

# Preprocessing Pipeline (Standardize all numeric features)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), feature_cols)
    ]
)

# Train-Test Split based on inputs
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_seed
)

# Transform Data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

clean_feature_names = np.array([
    name.replace("num__", "") for name in feature_names
])

# Helper to instantiate selected model architecture
def get_model_instance(model_name):
    if model_name == "Linear Regression":
        return LinearRegression()
    elif model_name == "Lasso (L1 Regularized)":
        return Lasso(alpha=0.1, random_state=random_seed)
    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor(n_estimators=100, random_state=random_seed)
    elif model_name == "Decision Tree Regressor":
        return DecisionTreeRegressor(max_depth=4, random_state=random_seed)

# ==============================================================================
# Header Section
# ==============================================================================
st.markdown('<div class="gradient-title">L7-Multiple Linear Regression Workflow</div>', unsafe_allow_html=True)
st.markdown(f'<div class="gradient-subtitle">CRISP-DM Step 4: California Housing — Feature Selectors Evaluated via <b>{model_type}</b></div>', unsafe_allow_html=True)

# ==============================================================================
# Run All Feature Selection Algorithms (Full Ranking Computations)
# ==============================================================================
results = []
feature_selections = {}

# Evaluation Helper
def evaluate_selected(name, indices):
    X_tr = X_train_processed[:, indices]
    X_te = X_test_processed[:, indices]
    
    model = get_model_instance(model_type)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    selected_feats = clean_feature_names[indices]
    feature_selections[name] = selected_feats
    
    return {
        "Method": name,
        "Selected Features": list(selected_feats),
        "Number of Features": len(indices),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Model": model,
        "Indices": indices
    }

# 1. Pearson Correlation
pearson_scores = [abs(np.corrcoef(X_train_processed[:, i], y_train)[0, 1]) for i in range(X_train_processed.shape[1])]
pearson_all_ranks = np.argsort(pearson_scores)[::-1]
results.append(evaluate_selected("Pearson Correlation", pearson_all_ranks[:top_k]))

# 2. Spearman Correlation
spearman_scores = [abs(spearmanr(X_train_processed[:, i], y_train)[0]) for i in range(X_train_processed.shape[1])]
spearman_all_ranks = np.argsort(spearman_scores)[::-1]
results.append(evaluate_selected("Spearman Correlation", spearman_all_ranks[:top_k]))

# 3. F-test Regression
f_selector = SelectKBest(score_func=f_regression, k='all')
f_selector.fit(X_train_processed, y_train)
f_all_ranks = np.argsort(f_selector.scores_)[::-1]
results.append(evaluate_selected("F-test Regression", f_all_ranks[:top_k]))

# 4. Mutual Information Regression
mi_scores = mutual_info_regression(X_train_processed, y_train, random_state=random_seed)
mi_all_ranks = np.argsort(mi_scores)[::-1]
results.append(evaluate_selected("Mutual Information", mi_all_ranks[:top_k]))

# 5. RFE
if "Forest" in model_type:
    rfe_est = RandomForestRegressor(n_estimators=50, random_state=random_seed)
elif "Tree" in model_type:
    rfe_est = DecisionTreeRegressor(max_depth=4, random_state=random_seed)
else:
    rfe_est = LinearRegression()
rfe_selector = RFE(estimator=rfe_est, n_features_to_select=1)
rfe_selector.fit(X_train_processed, y_train)
rfe_all_ranks = np.argsort(rfe_selector.ranking_)
results.append(evaluate_selected("RFE Selection Path", rfe_all_ranks[:top_k]))

# 6. SFS Forward
sfs_forward_features = []
max_eval_k = min(8, X_train_processed.shape[1])
for k in range(1, max_eval_k + 1):
    if k == X_train_processed.shape[1]:
        sfs_forward_features.append(np.arange(X_train_processed.shape[1]))
    else:
        sfs = SequentialFeatureSelector(get_model_instance(model_type), n_features_to_select=k, direction="forward", cv=3)
        sfs.fit(X_train_processed, y_train)
        sfs_forward_features.append(np.where(sfs.get_support())[0])

sfs_f_rank_order = []
for k in range(max_eval_k):
    for idx in sfs_forward_features[k]:
        if idx not in sfs_f_rank_order:
            sfs_f_rank_order.append(idx)
            break
# Fill remaining if any
for i in range(X_train_processed.shape[1]):
    if i not in sfs_f_rank_order:
        sfs_f_rank_order.append(i)
results.append(evaluate_selected("Sequential Forward Selection", sfs_f_rank_order[:top_k]))

# 7. SBS Backward
sfs_backward_features = []
for k in range(1, max_eval_k + 1):
    if k == X_train_processed.shape[1]:
        sfs_backward_features.append(np.arange(X_train_processed.shape[1]))
    else:
        sfs = SequentialFeatureSelector(get_model_instance(model_type), n_features_to_select=k, direction="backward", cv=3)
        sfs.fit(X_train_processed, y_train)
        sfs_backward_features.append(np.where(sfs.get_support())[0])

sfs_b_rank_order = []
# greedy extraction
for k in range(max_eval_k):
    for idx in sfs_backward_features[k]:
        if idx not in sfs_b_rank_order:
            sfs_b_rank_order.append(idx)
            break
for i in range(X_train_processed.shape[1]):
    if i not in sfs_b_rank_order:
        sfs_b_rank_order.append(i)
results.append(evaluate_selected("Sequential Backward Selection", sfs_b_rank_order[:top_k]))

# 8. Lasso Selection Model
lasso = LassoCV(cv=3, random_state=random_seed)
lasso.fit(X_train_processed, y_train)
lasso_all_ranks = np.argsort(abs(lasso.coef_))[::-1]
results.append(evaluate_selected("Lasso L1 Selection", lasso_all_ranks[:top_k]))

# 9. Random Forest Selection Model
rf = RandomForestRegressor(n_estimators=100, random_state=random_seed)
rf.fit(X_train_processed, y_train)
rf_all_ranks = np.argsort(rf.feature_importances_)[::-1]
results.append(evaluate_selected("Random Forest Importance", rf_all_ranks[:top_k]))

# Convert to DataFrame
results_df = pd.DataFrame(results)
results_df_sorted = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

# Find the best method
best_method_row = results_df_sorted.iloc[0]
sorted_methods = results_df_sorted['Method'].tolist()

# ==============================================================================
# Overview Metrics Section (KPI Row)
# ==============================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="glass-card">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Best Selection Strategy</div>
            <div style="font-size: 1.8rem; font-family: 'Outfit'; font-weight: 700; color: #f59e0b; margin-top: 8px;">{best_method_row['Method']}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric(
        label=f"Max {model_type} R²",
        value=f"{best_method_row['R2']:.4f}",
        delta=f"{(best_method_row['R2'] - results_df['R2'].mean()):.4f} vs Avg"
    )

with col3:
    st.metric(
        label="Lowest Test RMSE",
        value=f"${best_method_row['RMSE']:.2f}k",
        delta=f"-${(results_df['RMSE'].mean() - best_method_row['RMSE']):.2f}k",
        delta_color="inverse"
    )

with col4:
    st.markdown(f"""
        <div class="glass-card" style="padding: 20px;">
            <div style="font-size: 0.9rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Optimal Feature Count</div>
            <div style="font-size: 2.2rem; font-family: 'Outfit'; font-weight: 700; color: #10b981; margin-top: 4px;">{best_method_row['Number of Features']}</div>
        </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# Charts & Comparative Visualizations
# ==============================================================================
st.markdown("### 📊 Performance Analysis")

left_chart_col, right_chart_col = st.columns([1.1, 0.9])

with left_chart_col:
    # 1. Altair Bar Chart of R-squared Scores
    chart_data = results_df.drop(columns=['Model', 'Indices']).copy()
    chart_data['R2_display'] = chart_data['R2'].round(4)
    
    r2_chart = alt.Chart(chart_data).mark_bar(
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6
    ).encode(
        y=alt.Y('Method:N', sort=sorted_methods, title=None),
        x=alt.X('R2:Q', title=f"Test R-squared Score ({model_type})", scale=alt.Scale(domain=[max(0.0, results_df['R2'].min() - 0.05), min(1.0, results_df['R2'].max() + 0.02)])),
        color=alt.Color(
            'R2:Q', 
            scale=alt.Scale(scheme='viridis'),
            legend=None
        ),
        tooltip=['Method', alt.Tooltip('R2:Q', format='.4f'), alt.Tooltip('RMSE:Q', format='.2f'), 'Number of Features']
    ).properties(
        title=f"Test R-squared Score by Feature Selection Method ({model_type})",
        height=320
    )
    
    # Add text labels on bars
    text = r2_chart.mark_text(
        align='left',
        baseline='middle',
        dx=5,
        color='white',
        fontWeight=600
    ).encode(
        text=alt.Text('R2_display:Q', format='.4f')
    )
    
    st.altair_chart((r2_chart + text), use_container_width=True)

with right_chart_col:
    # 2. Consensus Plot - Which features are selected most often?
    feature_counts = {}
    for method_feats in feature_selections.values():
        for feat in method_feats:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1
            
    consensus_df = pd.DataFrame([
        {"Feature": k, "Votes": v} for k, v in feature_counts.items()
    ]).sort_values(by="Votes", ascending=False)
    
    consensus_chart = alt.Chart(consensus_df).mark_bar(
        color='#818cf8',
        size=20,
        cornerRadiusTopRight=6,
        cornerRadiusBottomRight=6
    ).encode(
        x=alt.X('Votes:Q', title="Number of Algorithms Selecting Feature", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y('Feature:N', sort='-x', title=None),
        tooltip=['Feature', 'Votes']
    ).properties(
        title="Feature Selection Consensus (Out of 9 Algorithms)",
        height=320
    )
    
    st.altair_chart(consensus_chart, use_container_width=True)

# ==============================================================================
# Stepwise Evaluation
# ==============================================================================
stepwise_methods = {
    "Pearson Corr": lambda k: pearson_all_ranks[:k],
    "Spearman Corr": lambda k: spearman_all_ranks[:k],
    "F-test Reg": lambda k: f_all_ranks[:k],
    "Mutual Info": lambda k: mi_all_ranks[:k],
    "RFE": lambda k: rfe_all_ranks[:k],
    "SFS (Forward)": lambda k: sfs_f_rank_order[:k],
    "SBS (Backward)": lambda k: sfs_b_rank_order[:k],
    "Lasso (L1)": lambda k: lasso_all_ranks[:k],
    "Random Forest": lambda k: rf_all_ranks[:k]
}

stepwise_records = []
for m_name, get_indices in stepwise_methods.items():
    for k in range(1, 6):
        indices = get_indices(k)
        X_tr = X_train_processed[:, indices]
        X_te = X_test_processed[:, indices]
        mdl = get_model_instance(model_type)
        mdl.fit(X_tr, y_train)
        y_pred = mdl.predict(X_te)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        stepwise_records.append({
            "Method": m_name,
            "Number of Features (k)": k,
            "R2": r2,
            "RMSE": rmse
        })
stepwise_df = pd.DataFrame(stepwise_records)

# ==============================================================================
# Main Content View Dispatch
# ==============================================================================
st.markdown(f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:1.5rem;">
        <div style="font-size:1.6rem;">{selected_view.split()[0]}</div>
        <div style="font-size:1.3rem; font-family:'Outfit',sans-serif; font-weight:600; color:#e2e8f0;">
            {' '.join(selected_view.split()[1:])}
        </div>
    </div>
""", unsafe_allow_html=True)

show_comparison  = selected_view == "📋 Complete Comparison Table"
show_inspector   = selected_view == "🔍 Detailed Method Inspector"
show_simulator   = selected_view == "🔮 Live House Price Simulator"
show_stepwise    = selected_view == "📈 Stepwise Curves (All-in-One)"
show_frontier    = selected_view == "🏆 Best Method per k"

if show_comparison:
    st.markdown(f"#### 📋 Performance Metrics Comparison (Evaluated via {model_type})")
    
    # Display formatted DataFrame
    display_df = results_df_sorted.copy()
    display_df['MAE'] = display_df['MAE'].map('${:,.3f}k'.format)
    display_df['RMSE'] = display_df['RMSE'].map('${:,.3f}k'.format)
    display_df['R2'] = display_df['R2'].map('{:.5f}'.format)
    
    # Style features as badge lists
    def format_badges(feat_list):
        return " ".join([f'<span class="feature-badge">{f}</span>' for f in feat_list])
        
    display_df['Selected Features'] = display_df['Selected Features'].apply(format_badges)
    
    st.write(
        display_df[['Method', 'Selected Features', 'Number of Features', 'MAE', 'RMSE', 'R2']]
        .to_html(escape=False, index=False, classes="table table-hover"), 
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)

elif show_inspector:
    st.markdown("#### 🔍 Model Weights & Feature Importance Inspector")
    selected_method_name = st.selectbox("Select a Selection Strategy to Inspect:", sorted_methods)
    
    method_data = next(item for item in results if item["Method"] == selected_method_name)
    selected_model = method_data["Model"]
    indices = method_data["Indices"]
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown(f"##### Selected Feature Set ({len(indices)} features)")
        for f in clean_feature_names[indices]:
            st.markdown(f'<span class="feature-badge-gold">{f}</span>', unsafe_allow_html=True)
            
        st.markdown(f"##### Model Internal Weights ({model_type})")
        
        # Display coefficients for linear models, or importances for tree models
        if hasattr(selected_model, "coef_"):
            st.markdown(f"**Intercept**: `${selected_model.intercept_:.2f}k`")
            for coef, idx in zip(selected_model.coef_, indices):
                st.markdown(f"* **{clean_feature_names[idx]} (Weight)**: `{coef:.4f}`")
        elif hasattr(selected_model, "feature_importances_"):
            for imp, idx in zip(selected_model.feature_importances_, indices):
                st.markdown(f"* **{clean_feature_names[idx]} (Tree Importance)**: `{imp:.4f}`")
        else:
            st.write("This model does not expose feature coefficients or importances directly.")
            
    with col_right:
        st.markdown("##### Performance Metrics")
        st.write(f"- **R-squared Score**: `{method_data['R2']:.6f}`")
        st.write(f"- **Root Mean Squared Error (RMSE)**: `${method_data['RMSE']:.2f}k`")
        st.write(f"- **Mean Absolute Error (MAE)**: `${method_data['MAE']:.2f}k`")
        
        # Educational Insight Box
        if "Linear" in model_type or "Lasso" in model_type:
            st.markdown("""
                <div style="background-color: rgba(129, 140, 248, 0.08); border-left: 4px solid #818cf8; padding: 15px; border-radius: 4px; margin-top: 15px;">
                    <h6 style="color:#a5b4fc; margin-top:0;">💡 Linear Model Interpretation:</h6>
                    Coefficients represent the change in target value (median_house_value in $1000s) per standard deviation change of the feature.
                    Standardization prevents columns with larger scales (like total_rooms) from artificially dominating weights.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background-color: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; padding: 15px; border-radius: 4px; margin-top: 15px;">
                    <h6 style="color:#a7f3d0; margin-top:0;">🌲 Tree Model Interpretation:</h6>
                    Tree-based models calculate feature importance based on split variance reduction (MSE reduction). 
                    These importances sum to 1.0, representing each feature's contribution to tree splits.
                </div>
            """, unsafe_allow_html=True)

elif show_simulator:
    st.markdown("#### 🔮 Live Price Simulator")
    st.markdown("Predict the expected price of a house using any of the fitted configurations.")
    
    sim_col_left, sim_col_mid, sim_col_right = st.columns([1.1, 0.9, 1])
    
    with sim_col_left:
        st.markdown("##### 1. Input District Block Specifications")
        longitude = st.slider("longitude", min_value=-124.5, max_value=-114.0, value=-118.25, step=0.1)
        latitude = st.slider("latitude", min_value=32.0, max_value=42.0, value=34.05, step=0.1)
        housing_median_age = st.slider("housing_median_age (years)", min_value=1, max_value=52, value=25, step=1)
        total_rooms = st.number_input("total_rooms", min_value=100.0, max_value=10000.0, value=2000.0, step=100.0)
        total_bedrooms = st.number_input("total_bedrooms", min_value=10.0, max_value=2000.0, value=400.0, step=10.0)
        population = st.number_input("population", min_value=50.0, max_value=5000.0, value=800.0, step=50.0)
        households = st.number_input("households", min_value=10.0, max_value=2000.0, value=350.0, step=10.0)
        median_income = st.slider("median_income (in $10k units)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
        
    with sim_col_mid:
        st.markdown("##### 2. Choose Model Configuration")
        sim_method_name = st.selectbox("Model Feature Selection:", sorted_methods, key="sim_method")
        
        sim_method_data = next(item for item in results if item["Method"] == sim_method_name)
        sim_model = sim_method_data["Model"]
        sim_indices = sim_method_data["Indices"]
        
        st.markdown("##### Model Architecture Active:")
        st.markdown(f'<span class="feature-badge-gold">{model_type}</span>', unsafe_allow_html=True)
        
        st.markdown("##### Features Used by Model:")
        for f in clean_feature_names[sim_indices]:
            st.markdown(f'<span class="feature-badge">{f}</span>', unsafe_allow_html=True)

    with sim_col_right:
        st.markdown("##### 3. Live Prediction Results")
        
        # Prepare inputs in dataframe structure
        input_dict = {
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Transform inputs using the fitted preprocessor
        input_processed = preprocessor.transform(input_df)
        
        # Filter processed inputs to only columns that the selected model expects
        input_filtered = input_processed[:, sim_indices]
        
        # Predict price (median_house_value in $1000s)
        predicted_price = sim_model.predict(input_filtered)[0]
        
        # Clamp prediction to avoid negative values
        predicted_price = max(0.0, predicted_price)
        
        # Display visually
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%); 
                        border: 1px solid rgba(129, 140, 248, 0.2); 
                        border-radius: 12px; 
                        padding: 30px; 
                        text-align: center;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
                        margin-top: 20px;">
                <div style="font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; font-weight:600;">Predicted House Value</div>
                <div style="font-size: 2.8rem; font-family: 'Outfit'; font-weight: 800; color: #10b981; margin: 15px 0;">${predicted_price * 1000:,.2f}</div>
                <div style="font-size: 0.85rem; color: #64748b; font-style: italic;">
                    Fitted model: {model_type} | selection: {sim_method_name} (Scale: {predicted_price:.2f}k)
                </div>
            </div>
        """, unsafe_allow_html=True)

elif show_stepwise:
    st.markdown("#### 📈 Stepwise Performance Curves (9 Algorithms)")
    st.markdown(f"Compare how different feature selection paths perform under **{model_type}** as the feature count $k$ increases from 1 to 5.")

    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        # R2 Line Chart
        r2_line_chart = alt.Chart(stepwise_df).mark_line(point=True).encode(
            x=alt.X('Number of Features (k):O', title="Number of Features (k)"),
            y=alt.Y('R2:Q', title="Test R-squared Score", scale=alt.Scale(domain=[max(0.0, stepwise_df['R2'].min() - 0.05), min(1.0, stepwise_df['R2'].max() + 0.05)])),
            color=alt.Color('Method:N', title="Algorithm"),
            tooltip=['Method', 'Number of Features (k)', alt.Tooltip('R2:Q', format='.5f'), alt.Tooltip('RMSE:Q', format='.2f')]
        ).properties(
            title=f"Test R-squared ($R^2$) vs. Subset Size ({model_type})",
            height=380
        ).interactive()
        
        st.altair_chart(r2_line_chart, use_container_width=True)
        
    with col_c2:
        # RMSE Line Chart
        rmse_line_chart = alt.Chart(stepwise_df).mark_line(point=True).encode(
            x=alt.X('Number of Features (k):O', title="Number of Features (k)"),
            y=alt.Y('RMSE:Q', title="Test RMSE ($k)", scale=alt.Scale(domain=[max(0.0, stepwise_df['RMSE'].min() - 0.5), stepwise_df['RMSE'].max() + 0.5])),
            color=alt.Color('Method:N', title="Algorithm"),
            tooltip=['Method', 'Number of Features (k)', alt.Tooltip('R2:Q', format='.5f'), alt.Tooltip('RMSE:Q', format='.2f')]
        ).properties(
            title=f"Test RMSE ($k) vs. Subset Size ({model_type})",
            height=380
        ).interactive()
        
        st.altair_chart(rmse_line_chart, use_container_width=True)
        
    st.markdown("##### 📋 Stepwise Feature Selections (Rank 1 to 5)")
    row_data = []
    for r in range(5):
        row_data.append({
            "Rank": f"Rank {r+1}",
            "Pearson": clean_feature_names[pearson_all_ranks[r]],
            "Spearman": clean_feature_names[spearman_all_ranks[r]],
            "F-test": clean_feature_names[f_all_ranks[r]],
            "Mutual Info": clean_feature_names[mi_all_ranks[r]],
            "RFE": clean_feature_names[rfe_all_ranks[r]],
            "SFS (Fwd)": clean_feature_names[sfs_f_rank_order[r]],
            "SBS (Bwd)": clean_feature_names[sfs_b_rank_order[r]],
            "Lasso (L1)": clean_feature_names[lasso_all_ranks[r]],
            "Random Forest": clean_feature_names[rf_all_ranks[r]]
        })
    table_df = pd.DataFrame(row_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

elif show_frontier:
    st.markdown("#### 🏆 Best Feature Selection Method at Each Feature Count (k)")
    st.markdown(
        f"For each value of **k** (1 to 5 features), this shows the **best-performing method** "
        f"evaluated via **{model_type}**. This is the performance frontier — the ceiling you can "
        f"expect when you know exactly how many features to use and pick optimally."
    )

    # Build the best-per-k table
    best_per_k_rows = []
    for k_val in range(1, 6):
        subset = stepwise_df[stepwise_df["Number of Features (k)"] == k_val]
        best_idx = subset["R2"].idxmax()
        worst_idx = subset["R2"].idxmin()
        best_row = subset.loc[best_idx]
        worst_row = subset.loc[worst_idx]

        best_method_name = best_row["Method"]
        best_indices = stepwise_methods[best_method_name](k_val)
        chosen_features = ", ".join(clean_feature_names[best_indices])

        best_per_k_rows.append({
            "Number of Features (k)": k_val,
            "Best Method": best_method_name,
            "R2": round(float(best_row["R2"]), 5),
            "RMSE": round(float(best_row["RMSE"]), 2),
            "Worst R2": round(float(worst_row["R2"]), 5),
            "R2 Spread": round(float(best_row["R2"]) - float(worst_row["R2"]), 5),
            "Features Selected": chosen_features,
        })

    best_per_k_df = pd.DataFrame(best_per_k_rows)

    # Performance Frontier charts
    frontier_col1, frontier_col2 = st.columns(2)

    r2_min = float(stepwise_df["R2"].min()) - 0.05
    r2_max = float(stepwise_df["R2"].max()) + 0.05
    rmse_min = float(stepwise_df["RMSE"].min()) - 0.5
    rmse_max = float(stepwise_df["RMSE"].max()) + 0.5

    with frontier_col1:
        # Faded background
        bg_r2 = alt.Chart(stepwise_df).mark_line(
            opacity=0.18, strokeWidth=1.5
        ).encode(
            x=alt.X("Number of Features (k):O", title="Number of Features (k)"),
            y=alt.Y("R2:Q", scale=alt.Scale(domain=[max(0.0, r2_min), min(1.0, r2_max)]),
                    title="Test R²"),
            color=alt.Color("Method:N", legend=None)
        )

        # Gold frontier line
        frontier_r2 = alt.Chart(best_per_k_df).mark_line(
            strokeWidth=3, color="#f59e0b",
            point=alt.OverlayMarkDef(filled=True, size=120, color="#f59e0b")
        ).encode(
            x=alt.X("Number of Features (k):O"),
            y=alt.Y("R2:Q", scale=alt.Scale(domain=[max(0.0, r2_min), min(1.0, r2_max)])),
            tooltip=[
                alt.Tooltip("Number of Features (k):O", title="k"),
                alt.Tooltip("Best Method:N"),
                alt.Tooltip("R2:Q", format=".5f", title="Best R²"),
                alt.Tooltip("Features Selected:N"),
            ]
        )

        label_r2 = alt.Chart(best_per_k_df).mark_text(
            dy=-14, fontSize=9, fontWeight="bold", color="#fbbf24"
        ).encode(
            x=alt.X("Number of Features (k):O"),
            y=alt.Y("R2:Q"),
            text=alt.Text("Best Method:N")
        )

        st.altair_chart(
            (bg_r2 + frontier_r2 + label_r2)
            .resolve_scale(y="shared")
            .properties(title=f"R² Frontier — Best Method per k ({model_type})", height=380)
            .interactive(),
            use_container_width=True
        )

    with frontier_col2:
        # Faded background
        bg_rmse = alt.Chart(stepwise_df).mark_line(
            opacity=0.18, strokeWidth=1.5
        ).encode(
            x=alt.X("Number of Features (k):O", title="Number of Features (k)"),
            y=alt.Y("RMSE:Q", scale=alt.Scale(domain=[rmse_min, rmse_max]),
                    title="Test RMSE ($k)"),
            color=alt.Color("Method:N", legend=None)
        )

        # Green frontier line
        frontier_rmse = alt.Chart(best_per_k_df).mark_line(
            strokeWidth=3, color="#10b981",
            point=alt.OverlayMarkDef(filled=True, size=120, color="#10b981")
        ).encode(
            x=alt.X("Number of Features (k):O"),
            y=alt.Y("RMSE:Q", scale=alt.Scale(domain=[rmse_min, rmse_max])),
            tooltip=[
                alt.Tooltip("Number of Features (k):O", title="k"),
                alt.Tooltip("Best Method:N"),
                alt.Tooltip("RMSE:Q", format=".2f", title="Best RMSE"),
                alt.Tooltip("Features Selected:N"),
            ]
        )

        label_rmse = alt.Chart(best_per_k_df).mark_text(
            dy=-14, fontSize=9, fontWeight="bold", color="#34d399"
        ).encode(
            x=alt.X("Number of Features (k):O"),
            y=alt.Y("RMSE:Q"),
            text=alt.Text("Best Method:N")
        )

        st.altair_chart(
            (bg_rmse + frontier_rmse + label_rmse)
            .resolve_scale(y="shared")
            .properties(title=f"RMSE Frontier — Best Method per k ({model_type})", height=380)
            .interactive(),
            use_container_width=True
        )

    st.markdown("##### 📏 Impact of Method Choice at Each k (R² Spread)")
    st.markdown(
        "A taller bar = bigger penalty for picking the wrong method at that k."
    )
    range_chart = alt.Chart(best_per_k_df).mark_bar(
        cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color="#818cf8"
    ).encode(
        x=alt.X("Number of Features (k):O", title="Number of Features (k)"),
        y=alt.Y("R2 Spread:Q", title="R² Spread (Best − Worst)"),
        tooltip=[
            alt.Tooltip("Number of Features (k):O", title="k"),
            alt.Tooltip("Best Method:N"),
            alt.Tooltip("R2:Q", format=".5f", title="Best R²"),
            alt.Tooltip("Worst R2:Q", format=".5f", title="Worst R²"),
            alt.Tooltip("R2 Spread:Q", format=".5f"),
        ]
    ).properties(title="R² Spread Between Best and Worst Method at Each k", height=220)
    st.altair_chart(range_chart, use_container_width=True)

    st.markdown("##### 🥇 Winner Summary Table")
    display_best = best_per_k_df.rename(columns={
        "Number of Features (k)": "k",
        "R2": "Best R²",
        "RMSE": "Best RMSE ($k)",
        "Worst R2": "Worst R² (same k)",
        "R2 Spread": "Range (R²)"
    })

    def highlight_winner(row):
        if row["Best R²"] == display_best["Best R²"].max():
            return ["background-color: rgba(245,158,11,0.18); font-weight:bold"] * len(row)
        return [""] * len(row)

    st.dataframe(
        display_best.style
            .apply(highlight_winner, axis=1)
            .format({
                "Best R²": "{:.5f}",
                "Best RMSE ($k)": "${:,.2f}k",
                "Worst R² (same k)": "{:.5f}",
                "Range (R²)": "{:.5f}",
            }),
        use_container_width=True, hide_index=True
    )

    best_k_overall = best_per_k_df.loc[best_per_k_df["R2"].idxmax()]
    st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(245,158,11,0.10), rgba(251,191,36,0.04));
                    border-left: 4px solid #f59e0b; padding: 18px 22px;
                    border-radius: 8px; margin-top: 16px;">
            <h6 style="color:#fbbf24; margin:0 0 8px 0;">💡 Optimal Configuration</h6>
            <p style="margin:0; color:#cbd5e1;">
                Best overall: <b style="color:#f59e0b;">k = {int(best_k_overall['Number of Features (k)'])} features</b>
                using <b style="color:#f59e0b;">{best_k_overall['Best Method']}</b> —
                R² = <b>{best_k_overall['R2']:.5f}</b>,
                RMSE = <b>${best_k_overall['RMSE']:.2f}k</b><br>
                <span style="color:#94a3b8;">Features: <i>{best_k_overall['Features Selected']}</i></span>
            </p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer-text">
        L7-Multiple Linear Regression Workflow &copy; 2026. Made with 💜 using Streamlit and Altair.
    </div>
""", unsafe_allow_html=True)
