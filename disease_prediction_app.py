import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
from xgboost import XGBClassifier
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disease Prediction ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: rgba(255,255,255,0.05); border-radius: 10px; padding: 10px; border: 1px solid rgba(255,255,255,0.1); }
    .stSelectbox label, .stSlider label { color: #94a3b8 !important; font-size: 13px !important; }
    div[data-testid="metric-container"] { background: rgba(255,255,255,0.04); border-radius: 10px; padding: 12px; border: 1px solid rgba(255,255,255,0.08); }
    h1, h2, h3 { color: #f1f5f9 !important; }
    .prediction-box { padding: 20px; border-radius: 12px; text-align: center; margin: 10px 0; }
    .high-risk { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4); }
    .low-risk  { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); }
</style>
""", unsafe_allow_html=True)

# ─── DATA LOADERS ────────────────────────────────────────────────────────────
@st.cache_data
def load_heart_data():
    url = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart_disease.csv"
    try:
        df = pd.read_csv(url)
        df.columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
                      "thalach","exang","oldpeak","slope","ca","thal","target"]
        df["target"] = (df["target"] > 0).astype(int)
        return df
    except:
        # Fallback synthetic data
        np.random.seed(42)
        n = 303
        df = pd.DataFrame({
            "age":      np.random.randint(29, 77, n),
            "sex":      np.random.randint(0, 2, n),
            "cp":       np.random.randint(0, 4, n),
            "trestbps": np.random.randint(94, 200, n),
            "chol":     np.random.randint(126, 564, n),
            "fbs":      np.random.randint(0, 2, n),
            "restecg":  np.random.randint(0, 3, n),
            "thalach":  np.random.randint(71, 202, n),
            "exang":    np.random.randint(0, 2, n),
            "oldpeak":  np.round(np.random.uniform(0, 6.2, n), 1),
            "slope":    np.random.randint(0, 3, n),
            "ca":       np.random.randint(0, 4, n),
            "thal":     np.random.randint(0, 3, n),
            "target":   np.random.randint(0, 2, n),
        })
        return df

@st.cache_data
def load_diabetes_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    cols = ["pregnancies","glucose","bloodpressure","skinthickness",
            "insulin","bmi","dpf","age","target"]
    try:
        df = pd.read_csv(url, header=None, names=cols)
        return df
    except:
        np.random.seed(42)
        n = 768
        df = pd.DataFrame({
            "pregnancies":   np.random.randint(0, 17, n),
            "glucose":       np.random.randint(44, 199, n),
            "bloodpressure": np.random.randint(24, 122, n),
            "skinthickness": np.random.randint(7, 99, n),
            "insulin":       np.random.randint(14, 846, n),
            "bmi":           np.round(np.random.uniform(18, 67, n), 1),
            "dpf":           np.round(np.random.uniform(0.08, 2.42, n), 3),
            "age":           np.random.randint(21, 81, n),
            "target":        np.random.randint(0, 2, n),
        })
        return df

@st.cache_data
def load_cancer_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df

# ─── MODEL TRAINING ──────────────────────────────────────────────────────────
@st.cache_data
def train_models(dataset_name):
    if dataset_name == "🫀 Heart Disease":
        df = load_heart_data()
        feature_cols = [c for c in df.columns if c != "target"]
    elif dataset_name == "🩸 Diabetes":
        df = load_diabetes_data()
        feature_cols = [c for c in df.columns if c != "target"]
    else:
        df = load_cancer_data()
        feature_cols = [c for c in df.columns if c != "target"]

    X = df[feature_cols]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models = {
        "SVM":               SVC(probability=True, kernel="rbf", C=1.0, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost":           XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
    }

    results = {}
    trained  = {}

    for name, model in models.items():
        use_scaled = name in ["SVM", "Logistic Regression"]
        Xtr = X_train_sc if use_scaled else X_train
        Xte = X_test_sc  if use_scaled else X_test

        model.fit(Xtr, y_train)
        y_pred = model.predict(Xte)
        y_prob = model.predict_proba(Xte)[:, 1]

        cv_scores = cross_val_score(model, Xtr, y_train, cv=5, scoring="accuracy")

        results[name] = {
            "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 2),
            "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 2),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 2),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 2),
            "auc":       round(roc_auc_score(y_test, y_prob), 4),
            "cv_mean":   round(cv_scores.mean() * 100, 2),
            "cv_std":    round(cv_scores.std() * 100, 2),
            "cv_scores": cv_scores * 100,
            "cm":        confusion_matrix(y_test, y_pred),
            "fpr":       roc_curve(y_test, y_prob)[0],
            "tpr":       roc_curve(y_test, y_prob)[1],
        }
        trained[name] = (model, scaler if use_scaled else None)

    feature_importances = None
    rf_model = trained["Random Forest"][0]
    if hasattr(rf_model, "feature_importances_"):
        feature_importances = pd.Series(
            rf_model.feature_importances_, index=feature_cols
        ).sort_values(ascending=False)

    return results, trained, feature_cols, scaler, df

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Disease Prediction")
    st.markdown("---")

    dataset_name = st.selectbox(
        "Select Dataset",
        ["🫀 Heart Disease", "🩸 Diabetes", "🎗️ Breast Cancer"]
    )

    algo_name = st.selectbox(
        "Select Algorithm",
        ["XGBoost", "Random Forest", "SVM", "Logistic Regression"]
    )

    st.markdown("---")
    st.markdown("### 📊 Dataset Info")
    info = {
        "🫀 Heart Disease":  ("303 samples", "13 features", "UCI Cleveland"),
        "🩸 Diabetes":       ("768 samples", "8 features",  "Pima Indians"),
        "🎗️ Breast Cancer": ("569 samples", "30 features", "UCI Wisconsin"),
    }
    s, f, src = info[dataset_name]
    st.markdown(f"- **Samples:** {s}\n- **Features:** {f}\n- **Source:** {src}")

    st.markdown("---")
    train_btn = st.button("🚀 Train All Models", use_container_width=True, type="primary")

# ─── MAIN ────────────────────────────────────────────────────────────────────
st.title("🏥 Disease Prediction from Medical Data")
st.markdown(f"**Dataset:** {dataset_name} &nbsp;|&nbsp; **Algorithm:** {algo_name}")
st.markdown("---")

if "trained_results" not in st.session_state or train_btn or \
   st.session_state.get("last_dataset") != dataset_name:

    with st.spinner("⏳ Training models... please wait"):
        results, trained_models, feature_cols, scaler, df = train_models(dataset_name)
        st.session_state.trained_results  = results
        st.session_state.trained_models   = trained_models
        st.session_state.feature_cols     = feature_cols
        st.session_state.scaler           = scaler
        st.session_state.df               = df
        st.session_state.last_dataset     = dataset_name
    st.success("✅ All 4 models trained successfully!")

results       = st.session_state.trained_results
trained_models= st.session_state.trained_models
feature_cols  = st.session_state.feature_cols
df            = st.session_state.df

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "📈 Model Metrics", "🔍 Feature Analysis", "🔬 Predict"])

# ══ TAB 1: OVERVIEW ══════════════════════════════════════════════════════════
with tab1:
    st.subheader("📊 Performance Comparison — All Algorithms")

    # Metric cards for selected algo
    r = results[algo_name]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{r['accuracy']}%")
    c2.metric("Precision", f"{r['precision']}%")
    c3.metric("Recall",    f"{r['recall']}%")
    c4.metric("F1-Score",  f"{r['f1']}%")
    c5.metric("AUC-ROC",   f"{r['auc']}")

    st.markdown("")

    # Grouped bar chart
    algos   = list(results.keys())
    metrics_list = ["accuracy", "precision", "recall", "f1"]
    colors  = ["#6366f1", "#10b981", "#f59e0b", "#ef4444"]

    fig = go.Figure()
    for i, metric in enumerate(metrics_list):
        fig.add_trace(go.Bar(
            name=metric.capitalize(),
            x=algos,
            y=[results[a][metric] for a in algos],
            marker_color=colors[i],
            text=[f"{results[a][metric]}%" for a in algos],
            textposition="outside",
        ))

    fig.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.12),
        yaxis=dict(range=[60, 105], title="Score (%)"),
        height=380, margin=dict(t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("📋 Full Results Table")
    table_data = []
    for name, r2 in results.items():
        table_data.append({
            "Algorithm": name,
            "Accuracy (%)":  r2["accuracy"],
            "Precision (%)": r2["precision"],
            "Recall (%)":    r2["recall"],
            "F1-Score (%)":  r2["f1"],
            "AUC-ROC":       r2["auc"],
            "CV Mean (%)":   r2["cv_mean"],
            "CV Std":        f"±{r2['cv_std']}%"
        })
    tdf = pd.DataFrame(table_data).set_index("Algorithm")
    st.dataframe(tdf.style.highlight_max(
        subset=["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "AUC-ROC"],
        color="#1e3a2e"), use_container_width=True)

# ══ TAB 2: MODEL METRICS ═════════════════════════════════════════════════════
with tab2:
    st.subheader(f"📈 Detailed Metrics — {algo_name}")
    r = results[algo_name]

    col1, col2 = st.columns(2)

    with col1:
        # Confusion Matrix
        cm = r["cm"]
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Predicted: No", "Predicted: Yes"],
            y=["Actual: No", "Actual: Yes"],
            colorscale=[[0, "#0f172a"], [1, "#6366f1"]],
            text=cm, texttemplate="%{text}",
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=320, margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        # ROC Curve
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines",
            name=f"{algo_name} (AUC={r['auc']})",
            line=dict(color="#6366f1", width=2.5)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random", line=dict(color="#475569", dash="dash", width=1.5)
        ))
        fig_roc.update_layout(
            title=f"ROC Curve (AUC = {r['auc']})", template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=320, margin=dict(t=50, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Cross Validation
    st.subheader("🔄 5-Fold Cross Validation")
    cv_df = pd.DataFrame({
        "Fold": [f"Fold {i+1}" for i in range(5)],
        "Accuracy (%)": r["cv_scores"]
    })
    fig_cv = px.bar(cv_df, x="Fold", y="Accuracy (%)",
                    color_discrete_sequence=["#10b981"],
                    text="Accuracy (%)", template="plotly_dark")
    fig_cv.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_cv.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[60, 105]), height=300,
        title=f"Mean: {r['cv_mean']}% ± {r['cv_std']}%"
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    # All ROC curves together
    st.subheader("📉 ROC Curves — All Algorithms")
    fig_all_roc = go.Figure()
    roc_colors = {"SVM": "#6366f1", "Logistic Regression": "#10b981",
                  "Random Forest": "#f59e0b", "XGBoost": "#ef4444"}
    for name, r2 in results.items():
        fig_all_roc.add_trace(go.Scatter(
            x=r2["fpr"], y=r2["tpr"], mode="lines",
            name=f"{name} (AUC={r2['auc']})",
            line=dict(color=roc_colors[name], width=2)
        ))
    fig_all_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(color="#475569", dash="dash")
    ))
    fig_all_roc.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="FPR", yaxis_title="TPR", height=380
    )
    st.plotly_chart(fig_all_roc, use_container_width=True)

# ══ TAB 3: FEATURE ANALYSIS ══════════════════════════════════════════════════
with tab3:
    st.subheader("🔍 Feature Importance (Random Forest)")

    rf_model = trained_models["Random Forest"][0]
    fi = pd.Series(rf_model.feature_importances_, index=feature_cols)\
           .sort_values(ascending=True).tail(15)

    fig_fi = go.Figure(go.Bar(
        x=fi.values * 100, y=fi.index,
        orientation="h", marker_color="#f59e0b",
        text=[f"{v*100:.1f}%" for v in fi.values],
        textposition="outside"
    ))
    fig_fi.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Importance (%)", height=480,
        margin=dict(l=20, r=60)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Correlation Heatmap
    st.subheader("🌡️ Feature Correlation Heatmap")
    top_feats = pd.Series(rf_model.feature_importances_, index=feature_cols)\
                  .sort_values(ascending=False).head(10).index.tolist()
    corr = df[top_feats + ["target"]].corr()

    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        colorscale="RdBu", zmid=0,
        text=np.round(corr.values, 2), texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=440, margin=dict(t=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Class Distribution
    st.subheader("📊 Class Distribution")
    class_counts = df["target"].value_counts().reset_index()
    class_counts.columns = ["Class", "Count"]
    class_counts["Class"] = class_counts["Class"].map({0: "Negative (No Disease)", 1: "Positive (Disease)"})
    fig_pie = px.pie(class_counts, values="Count", names="Class",
                     color_discrete_sequence=["#10b981", "#ef4444"],
                     template="plotly_dark")
    fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=320)
    st.plotly_chart(fig_pie, use_container_width=True)

# ══ TAB 4: PREDICT ═══════════════════════════════════════════════════════════
with tab4:
    st.subheader(f"🔬 Predict with {algo_name}")
    st.markdown("Adjust the patient parameters below and click **Predict**.")

    model_obj, sc = trained_models[algo_name]

    # Build sliders dynamically from data stats
    input_vals = {}
    num_cols = min(3, len(feature_cols))
    cols = st.columns(num_cols)

    for i, feat in enumerate(feature_cols[:12]):  # show top 12
        col = cols[i % num_cols]
        mn  = float(df[feat].min())
        mx  = float(df[feat].max())
        med = float(df[feat].median())
        if df[feat].nunique() <= 4:
            input_vals[feat] = col.selectbox(feat, sorted(df[feat].unique()), index=0)
        else:
            step = 1.0 if mx - mn > 10 else 0.01
            input_vals[feat] = col.slider(feat, mn, mx, med, step=step)

    # Fill remaining features with median
    for feat in feature_cols[12:]:
        input_vals[feat] = float(df[feat].median())

    st.markdown("")
    pred_btn = st.button("🔬 Run Prediction", type="primary", use_container_width=False)

    if pred_btn:
        input_df = pd.DataFrame([input_vals])[feature_cols]

        if sc is not None:
            input_scaled = sc.transform(input_df)
        else:
            input_scaled = input_df

        prob = model_obj.predict_proba(input_scaled)[0][1]
        pred = model_obj.predict(input_scaled)[0]
        pct  = prob * 100

        st.markdown("---")
        st.subheader("📋 Prediction Result")

        risk_class = "high-risk" if pred == 1 else "low-risk"
        risk_label = "⚠️ HIGH RISK — Disease Detected" if pred == 1 else "✅ LOW RISK — No Disease Detected"
        risk_color = "#ef4444" if pred == 1 else "#10b981"

        st.markdown(f"""
        <div class="prediction-box {risk_class}">
            <h2 style="color:{risk_color}; margin:0">{risk_label}</h2>
            <h3 style="color:{risk_color}; margin:8px 0">Probability: {pct:.1f}%</h3>
            <p style="color:#94a3b8; margin:0">Algorithm: {algo_name} | Dataset: {dataset_name}</p>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pct,
            number={"suffix": "%", "font": {"size": 32, "color": risk_color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": risk_color},
                "steps": [
                    {"range": [0, 30],  "color": "rgba(16,185,129,0.2)"},
                    {"range": [30, 60], "color": "rgba(245,158,11,0.2)"},
                    {"range": [60, 100],"color": "rgba(239,68,68,0.2)"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 50}
            },
            title={"text": "Disease Probability", "font": {"color": "#94a3b8"}}
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", height=320
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("""
        <div style="background:rgba(245,158,11,0.1); border:1px solid rgba(245,158,11,0.3);
             border-radius:8px; padding:12px; margin-top:10px; font-size:13px; color:#94a3b8">
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational/research purposes only.
        It does not constitute medical advice. Always consult a qualified healthcare professional.
        </div>
        """, unsafe_allow_html=True)
