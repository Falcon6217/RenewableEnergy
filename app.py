import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Renewable Energy AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Set global Matplotlib styling for dark theme readability
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#cbd5e1",
    "ytick.color": "#cbd5e1",
    "axes.edgecolor": "#475569",
    "grid.color": "#334155",
})


# ============================================================
# CUSTOM CSS — HIGH CONTRAST DARK / BRIGHT ACCENTS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(34,197,94,.14), transparent 28%),
            radial-gradient(circle at 90% 15%, rgba(59,130,246,.14), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(168,85,247,.12), transparent 32%),
            linear-gradient(135deg, #07111f 0%, #0b1728 45%, #071827 100%);
        color: #f8fafc;
    }

    .main .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Hero Banner */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.4rem 2.5rem;
        border-radius: 28px;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(15,118,110,.32), rgba(30,64,175,.28)), rgba(15,23,42,.78);
        border: 1px solid rgba(148,163,184,.18);
        box-shadow: 0 25px 70px rgba(0,0,0,.30);
        backdrop-filter: blur(18px);
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 5vw, 3.5rem);
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(90deg, #86efac, #67e8f9, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 850px;
        line-height: 1.6;
        margin-top: .8rem;
    }

    .pill {
        display: inline-block;
        padding: .35rem .8rem;
        margin-bottom: .7rem;
        border-radius: 999px;
        background: rgba(34,197,94,.15);
        border: 1px solid rgba(74,222,128,.4);
        color: #86efac;
        font-size: .8rem;
        font-weight: 700;
        letter-spacing: .5px;
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(15,23,42,.68);
        border: 1px solid rgba(148,163,184,.16);
        border-radius: 22px;
        padding: 1.25rem;
        box-shadow: 0 15px 45px rgba(0,0,0,.20);
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: .35rem;
    }

    .section-subtitle {
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }

    /* Metric Visual Overrides */
    div[data-testid="stMetric"] {
        background: rgba(30,41,59,.78) !important;
        border: 1px solid rgba(148,163,184,.2) !important;
        border-radius: 18px !important;
        padding: 1rem 1.15rem !important;
    }

    div[data-testid="stMetricLabel"] > label {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #4ade80 !important;
        font-weight: 800 !important;
    }

    /* Tabs Styling */
    button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #4ade80 !important;
    }

    /* Status Cards */
    .prediction-adoption {
        padding: 1.5rem;
        border-radius: 20px;
        background: rgba(22,163,74,.22);
        border: 1px solid rgba(74,222,128,.45);
        text-align: center;
    }

    .prediction-non {
        padding: 1.5rem;
        border-radius: 20px;
        background: rgba(239,68,68,.22);
        border: 1px solid rgba(248,113,113,.45);
        text-align: center;
    }

    .prediction-icon {
        font-size: 3rem;
        margin-bottom: .4rem;
    }

    .prediction-text {
        font-size: 1.65rem;
        font-weight: 800;
        color: #ffffff;
    }

    .feature-chip {
        display: inline-block;
        padding: .55rem .8rem;
        margin: .25rem;
        border-radius: 12px;
        background: rgba(51,65,85,.75);
        border: 1px solid rgba(148,163,184,.25);
        color: #f1f5f9;
        font-size: .85rem;
        font-weight: 500;
    }

    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 2rem 0 .5rem;
        font-size: .85rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS & SETUP
# ============================================================

FEATURES = [
    "carbon_emissions",
    "energy_output",
    "renewability_index",
    "cost_efficiency",
]

TARGET = "adoption"
CLASS_NAMES = ["Non-Adoption", "Adoption"]

FEATURE_BOUNDS = {
    "carbon_emissions": (50.0, 400.0, "Lower emissions generally favor adoption."),
    "energy_output": (100.0, 1000.0, "Energy produced in hypothetical units."),
    "renewability_index": (0.0, 1.0, "0 = non-renewable, 1 = fully renewable."),
    "cost_efficiency": (0.5, 5.0, "Lower score means more cost efficient."),
}


# ============================================================
# DATA PIPELINE
# ============================================================

@st.cache_data
def generate_demo_data(num_samples=200, seed=42):
    rng = np.random.default_rng(seed)
    carbon_emissions = rng.uniform(50, 400, num_samples)
    energy_output = rng.uniform(100, 1000, num_samples)
    renewability_index = rng.uniform(0, 1, num_samples)
    cost_efficiency = rng.uniform(0.5, 5, num_samples)

    score = (
        -0.01 * carbon_emissions
        + 0.002 * energy_output
        + 3.0 * renewability_index
        - 0.8 * cost_efficiency
    )

    std = score.std()
    probability = 0.5 if std == 0 else 1 / (1 + np.exp(-(score - score.mean()) / std))
    adoption = (rng.random(num_samples) < probability).astype(int)

    return pd.DataFrame({
        "carbon_emissions": carbon_emissions,
        "energy_output": energy_output,
        "renewability_index": renewability_index,
        "cost_efficiency": cost_efficiency,
        "adoption": adoption,
    })


def validate_dataset(df):
    required = FEATURES + [TARGET]
    missing = [col for col in required if col not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(missing)}"
    if df.empty:
        return False, "Dataset is empty."

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df.dropna(subset=required, inplace=True)

    if df.empty:
        return False, "No valid rows remain after cleaning."
    if not set(df[TARGET].unique()).issubset({0, 1}):
        return False, "'adoption' target column must contain only values 0 and 1."
    if df[TARGET].nunique() < 2:
        return False, "Target must contain both 0 and 1 classes."

    return True, "Valid dataset."


@st.cache_resource
def train_model(df, max_depth, test_size, random_state):
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    class_counts = y.value_counts()
    use_stratify = len(class_counts) == 2 and class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if use_stratify else None,
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, list(model.classes_).index(1)]
        if 1 in model.classes_
        else np.zeros(len(X_test))
    )

    return model, X_train, X_test, y_train, y_test, y_pred, y_proba


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="
            padding:1rem;
            border-radius:18px;
            background:linear-gradient(135deg, rgba(34,197,94,.16), rgba(6,182,212,.10));
            border:1px solid rgba(74,222,128,.2);
            margin-bottom:1rem;
        ">
            <div style="font-size:2rem;">🌱</div>
            <div style="font-size:1.2rem;font-weight:800;color:#f8fafc;">Green AI</div>
            <div style="color:#94a3b8;font-size:.8rem;">Renewable Energy Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Controls")
    data_source = st.radio(
        "Dataset Source",
        ["Use Built-in Demo Data", "Upload Renewable_Energy_Adoption.csv"],
    )

    if data_source == "Upload Renewable_Energy_Adoption.csv":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                valid, message = validate_dataset(data)
                if not valid:
                    st.error(message)
                    st.stop()
                st.success(f"✓ {len(data)} rows loaded")
            except Exception as error:
                st.error(f"CSV error: {error}")
                st.stop()
        else:
            data = generate_demo_data()
            st.info("Using demo data until CSV is uploaded.")
    else:
        num_samples = st.slider("Demo Samples", 50, 1000, 200, 50)
        seed = st.number_input("Random Seed", 0, 999999, 42, 1)
        data = generate_demo_data(num_samples=num_samples, seed=int(seed))
        st.caption("Synthetic dataset loaded.")

    st.markdown("---")
    max_depth = st.slider("🌲 Tree Depth", 1, 10, 3)
    test_size = st.slider("🧪 Test Size", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("🎲 Random State", 0, 999999, 42, 1)


# ============================================================
# TRAIN MODEL
# ============================================================

try:
    model, X_train, X_test, y_train, y_test, y_pred, y_proba = train_model(
        data, max_depth, test_size, int(random_state)
    )
except Exception as error:
    st.error(f"Model training failed: {error}")
    st.stop()


# ============================================================
# HERO HEADER & TOP METRICS
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="pill">● AI POWERED ENERGY ANALYTICS</div>
        <h1>Renewable Energy AI</h1>
        <p>
            Predict renewable-energy adoption using an interactive Decision Tree
            machine-learning system. Explore your data, understand decision boundaries, and test new energy scenarios.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

accuracy = accuracy_score(y_test, y_pred)
adoption_rate = data[TARGET].mean() * 100
tree_depth = model.get_depth()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎯 Accuracy", f"{accuracy * 100:.1f}%")
c2.metric("📊 Dataset Rows", f"{len(data):,}")
c3.metric("🌱 Adoption Rate", f"{adoption_rate:.1f}%")
c4.metric("🌲 Tree Depth", tree_depth)


# ============================================================
# TABS INTERFACE
# ============================================================

tab_predict, tab_explore, tab_tree, tab_performance, tab_about = st.tabs(
    [
        "🔮 Predict",
        "📊 Explore Data",
        "🌲 Tree Viewer",
        "📈 Performance",
        "ℹ️ About",
    ]
)


# ------------------------------------------------------------
# 1. PREDICT TAB
# ------------------------------------------------------------

with tab_predict:
    st.markdown('<div class="section-title">🔮 Make a Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Adjust scenario parameters to generate real-time inferences.</div>', unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large")
    input_values = {}

    with left:
        for feature in FEATURES:
            low, high, help_text = FEATURE_BOUNDS[feature]
            default = float(np.clip(data[feature].mean(), low, high))
            step = (high - low) / 100
            input_values[feature] = st.slider(
                feature.replace("_", " ").title(),
                min_value=float(low),
                max_value=float(high),
                value=default,
                step=float(step),
                help=help_text,
            )

    input_df = pd.DataFrame([input_values])[FEATURES]
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    with right:
        if prediction == 1:
            st.markdown(
                """
                <div class="prediction-adoption">
                    <div class="prediction-icon">🌱</div>
                    <div class="prediction-text">ADOPTION</div>
                    <div style="color:#86efac;font-weight:600;">Renewable energy adoption is likely.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="prediction-non">
                    <div class="prediction-icon">⚠️</div>
                    <div class="prediction-text">NON-ADOPTION</div>
                    <div style="color:#fca5a5;font-weight:600;">Lower adoption likelihood predicted.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if 1 in model.classes_:
            idx = list(model.classes_).index(1)
            adoption_probability = probabilities[idx]
            st.markdown("<h4 style='color:#f8fafc;margin-top:1rem;'>Probability of Adoption</h4>", unsafe_allow_html=True)
            st.progress(float(adoption_probability))
            st.markdown(
                f"<div style='text-align:center;font-size:2rem;font-weight:800;color:#4ade80;'>{adoption_probability * 100:.1f}%</div>",
                unsafe_allow_html=True,
            )

        with st.expander("📋 View Input Summary"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

        with st.expander("🌲 Explain the Decision"):
            path = model.decision_path(input_df)
            leaf = model.apply(input_df)[0]
            nodes = path.indices[path.indptr[0]:path.indptr[1]]

            for node in nodes:
                if node == leaf:
                    st.success(f"🍃 Leaf Node {node}: Prediction determined here.")
                    continue
                feature_index = model.tree_.feature[node]
                if feature_index < 0:
                    continue
                f_name = FEATURES[feature_index]
                val = input_df.iloc[0][f_name]
                thresh = model.tree_.threshold[node]
                op = "<=" if val <= thresh else ">"
                st.write(f"**Node {node}:** `{f_name}` = `{val:.2f}` {op} `{thresh:.2f}`")

    st.markdown("---")
    st.markdown('<div class="section-title">📂 Batch Prediction</div>', unsafe_allow_html=True)
    batch_file = st.file_uploader("Upload prediction CSV", type=["csv"], key="batch_upload")

    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            missing = [f for f in FEATURES if f not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                for f in FEATURES:
                    batch_df[f] = pd.to_numeric(batch_df[f], errors="coerce")

                if batch_df[FEATURES].isna().any().any():
                    st.error("Invalid numeric values found in file.")
                else:
                    batch_df["predicted_adoption"] = model.predict(batch_df[FEATURES])
                    if 1 in model.classes_:
                        idx = list(model.classes_).index(1)
                        batch_df["probability_adoption"] = model.predict_proba(batch_df[FEATURES])[:, idx]

                    st.dataframe(batch_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download Predictions",
                        batch_df.to_csv(index=False).encode("utf-8"),
                        "predictions.csv",
                        "text/csv",
                    )
        except Exception as error:
            st.error(f"Batch processing failed: {error}")


# ------------------------------------------------------------
# 2. EXPLORE TAB
# ------------------------------------------------------------

with tab_explore:
    st.markdown('<div class="section-title">📊 Explore Dataset</div>', unsafe_allow_html=True)
    st.dataframe(data.head(20), use_container_width=True)
    st.caption(f"{len(data):,} rows × {len(data.columns)} columns")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h4 style='color:#f8fafc;'>🌱 Class Balance</h4>", unsafe_allow_html=True)
        counts = data[TARGET].value_counts().reindex([0, 1], fill_value=0)

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(["Non-Adoption", "Adoption"], counts.values, color=["#ef4444", "#22c55e"])
        ax.set_ylabel("Count")
        ax.bar_label(bars, color="#f8fafc", padding=3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        st.markdown("<h4 style='color:#f8fafc;'>📈 Feature Distribution</h4>", unsafe_allow_html=True)
        selected_feature = st.selectbox("Choose Feature", FEATURES)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        for val, label, col in [(0, "Non-Adoption", "#ef4444"), (1, "Adoption", "#22c55e")]:
            vals = data[data[TARGET] == val][selected_feature]
            ax2.hist(vals, bins=20, alpha=0.6, label=label, color=col)
        ax2.set_xlabel(selected_feature.replace("_", " ").title())
        ax2.set_ylabel("Frequency")
        ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.markdown("---")
    st.markdown("<h4 style='color:#f8fafc;'>🔗 Correlation Matrix</h4>", unsafe_allow_html=True)
    st.dataframe(data[FEATURES + [TARGET]].corr().round(3), use_container_width=True)


# ------------------------------------------------------------
# 3. TREE VIEWER TAB
# ------------------------------------------------------------

with tab_tree:
    st.markdown('<div class="section-title">🌲 Decision Tree Visualizer</div>', unsafe_allow_html=True)
    st.caption(f"Current Max Depth: {max_depth}")

    fig, ax = plt.subplots(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=FEATURES,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax,
    )

    st.pyplot(fig, use_container_width=True)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=180, facecolor="#0f172a")
    buffer.seek(0)

    st.download_button(
        "⬇️ Download Decision Tree PNG",
        buffer.getvalue(),
        "decision_tree.png",
        "image/png",
    )
    plt.close(fig)


# ------------------------------------------------------------
# 4. PERFORMANCE TAB
# ------------------------------------------------------------

with tab_performance:
    st.markdown('<div class="section-title">📈 Performance Metrics</div>', unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    p1.metric("🎯 Accuracy", f"{accuracy * 100:.2f}%")
    p2.metric("🧠 Train Count", len(X_train))
    p3.metric("🧪 Test Count", len(X_test))

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("<h4 style='color:#f8fafc;'>🔲 Confusion Matrix</h4>", unsafe_allow_html=True)
        matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, matrix[i, j], ha="center", va="center", color="#f8fafc", fontsize=14, fontweight="bold")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("<h4 style='color:#f8fafc;'>📉 ROC Curve</h4>", unsafe_allow_html=True)
        if y_test.nunique() == 2 and len(np.unique(y_proba)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.plot(fpr, tpr, color="#22c55e", linewidth=2, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("ROC calculation requires presence of both binary classes in the test sample.")

    st.markdown("---")
    st.markdown("<h4 style='color:#f8fafc;'>📋 Classification Report</h4>", unsafe_allow_html=True)
    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    st.markdown("---")
    st.markdown("<h4 style='color:#f8fafc;'>⭐ Feature Importance</h4>", unsafe_allow_html=True)
    importance = pd.DataFrame({
        "Feature": [f.replace("_", " ").title() for f in FEATURES],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.barh(importance["Feature"], importance["Importance"], color="#38bdf8")
    ax.set_xlabel("Importance Metric")
    ax.invert_yaxis()
    ax.bar_label(bars, fmt="%.3f", color="#f8fafc", padding=3)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ------------------------------------------------------------
# 5. ABOUT TAB
# ------------------------------------------------------------

with tab_about:
    st.markdown('<div class="section-title">ℹ️ About Green AI</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card">
            <h3 style="color:#f8fafc;margin-top:0;">🌱 Renewable Energy Adoption Predictor</h3>
            <p style="color:#cbd5e1;line-height:1.7;">
                This application applies Supervised Machine Learning (Decision Tree Classification) to model and predict
                renewable-energy adoption probabilities based on key economic and technical parameters.
            </p>
            <h4 style="color:#f8fafc;margin-top:1.2rem;">Pipeline Overview</h4>
            <div class="feature-chip">📥 Data Ingestion</div>
            <div class="feature-chip">🧹 Validation</div>
            <div class="feature-chip">✂️ Train/Test Stratification</div>
            <div class="feature-chip">🌲 Decision Tree Model</div>
            <div class="feature-chip">📈 Metrics Evaluation</div>
            <div class="feature-chip">🔍 Explainability Engine</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        🌱 Renewable Energy AI &nbsp;•&nbsp; Decision Tree Machine Learning Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)