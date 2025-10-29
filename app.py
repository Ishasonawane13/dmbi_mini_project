import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage


# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Credit Card Default Dashboard",
    page_icon="💳",
    layout="wide",
)

sns.set_theme(style="whitegrid")


# -----------------------------
# Utilities
# -----------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        (
            c.strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace("/", "_")
        )
        for c in df.columns
    ]
    return df


def try_map_known_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Map known UCI-style encodings to readable categories when detected."""
    df = df.copy()

    # SEX: 1=male, 2=female
    for cand in ["sex", "gender"]:
        if cand in df.columns and df[cand].dropna().astype(str).str.isnumeric().all():
            df[cand] = df[cand].map({1: "Male", 2: "Female"}).fillna(df[cand])

    # EDUCATION: 1=graduate school, 2=university, 3=high school, 4=others
    if "education" in df.columns:
        edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
        df["education"] = df["education"].replace(edu_map)

    # MARRIAGE: 1=married, 2=single, 3=others
    if "marriage" in df.columns:
        mar_map = {1: "Married", 2: "Single", 3: "Others"}
        df["marriage"] = df["marriage"].replace(mar_map)

    return df


def detect_target_column(columns: List[str]) -> Optional[str]:
    target_aliases = [
        "default", "default_status", "is_default",
        "default_payment_next_month", "default.payment.next.month",
    ]
    for alias in target_aliases:
        if alias in columns:
            return alias
    return None


def detect_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def coerce_numeric_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that look numeric into numeric dtype.
    A column is converted if at least 80% of its non-null values parse as numbers.
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        series = df[col].astype(str).str.replace(",", "", regex=False).str.strip()
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = series.notna().sum()
        parsed_ratio = numeric.notna().sum() / max(non_null, 1)
        if parsed_ratio >= 0.8:
            df[col] = numeric
    return df


@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if isinstance(file, str):
        df = pd.read_csv(file)
    else:
        df = pd.read_csv(file)
    df = standardize_columns(df)
    df = try_map_known_categories(df)
    df = coerce_numeric_like_columns(df)
    return df


def get_numeric_and_categorical_columns(df: pd.DataFrame, target: Optional[str] = None) -> Tuple[List[str], List[str]]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if target and target in numeric_cols:
        numeric_cols.remove(target)
    if target and target in categorical_cols:
        categorical_cols.remove(target)
    return numeric_cols, categorical_cols


def plot_missing_values(df: pd.DataFrame):
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        st.info("No missing values detected.")
        return
    miss_df = missing.reset_index()
    miss_df.columns = ["column", "missing_count"]
    st.bar_chart(miss_df.set_index("column"))


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation heatmap.")
        return
    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (numeric features)")
    st.pyplot(fig, clear_figure=True)


# -----------------------------
# Sidebar: Data Source and Filters
# -----------------------------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"]) 

default_path = "credit_card_default.csv"
data_status = ""

if uploaded_file is not None:
    df = load_data(uploaded_file)
    data_status = "Using uploaded file"
elif os.path.exists(default_path):
    df = load_data(default_path)
    data_status = f"Loaded default file: {default_path}"
else:
    df = None


st.title("💳 Credit Card Default Prediction Dashboard")
st.caption("Analyze, visualize, and model credit card default data. Upload your own CSV or use the default if present.")

if df is None:
    st.warning(
        "No dataset found. Please upload a CSV using the sidebar."
    )
    st.stop()


# Detect key columns for filters
cols = df.columns.tolist()
target_col = detect_target_column(cols)

gender_col = detect_column(cols, ["sex", "gender"])  # categorical
education_col = detect_column(cols, ["education"])     # categorical
marriage_col = detect_column(cols, ["marriage", "marital_status"])  # categorical
age_col = detect_column(cols, ["age"])                 # numeric
limit_col = detect_column(cols, ["limit_balance", "balance_limit", "credit_limit", "limit_bal", "limit"])

numeric_cols, categorical_cols = get_numeric_and_categorical_columns(df, target=target_col)


# Sidebar Filters
st.sidebar.subheader("Filters")
filtered_df = df.copy()

if gender_col and filtered_df[gender_col].notna().any():
    genders = ["All"] + sorted(filtered_df[gender_col].dropna().astype(str).unique().tolist())
    gsel = st.sidebar.selectbox("Gender", genders, index=0)
    if gsel != "All":
        filtered_df = filtered_df[filtered_df[gender_col].astype(str) == gsel]

if education_col and filtered_df[education_col].notna().any():
    edus = ["All"] + sorted(filtered_df[education_col].dropna().astype(str).unique().tolist())
    esel = st.sidebar.selectbox("Education", edus, index=0)
    if esel != "All":
        filtered_df = filtered_df[filtered_df[education_col].astype(str) == esel]

if marriage_col and filtered_df[marriage_col].notna().any():
    mars = ["All"] + sorted(filtered_df[marriage_col].dropna().astype(str).unique().tolist())
    msel = st.sidebar.selectbox("Marital status", mars, index=0)
    if msel != "All":
        filtered_df = filtered_df[filtered_df[marriage_col].astype(str) == msel]

if age_col and filtered_df[age_col].notna().any():
    min_age, max_age = int(filtered_df[age_col].min()), int(filtered_df[age_col].max())
    age_range = st.sidebar.slider("Age range", min_age, max_age, (min_age, max_age))
    filtered_df = filtered_df[(filtered_df[age_col] >= age_range[0]) & (filtered_df[age_col] <= age_range[1])]

if limit_col and filtered_df[limit_col].notna().any():
    min_l, max_l = float(filtered_df[limit_col].min()), float(filtered_df[limit_col].max())
    lb_range = st.sidebar.slider("Credit limit range", min_l, max_l, (min_l, max_l))
    filtered_df = filtered_df[(filtered_df[limit_col] >= lb_range[0]) & (filtered_df[limit_col] <= lb_range[1])]


# -----------------------------
# Tabs: Overview, EDA, Model, Insights
# -----------------------------
tab_overview, tab_eda, tab_model, tab_cluster, tab_insights = st.tabs([
    "📂 Data Overview", "🔍 EDA & Visualization", "🧠 Model (Optional)", "🧩 Clustering", "📊 Insights"
])


with tab_overview:
    st.subheader("Dataset Overview")
    st.caption(data_status)
    st.write(f"Rows: {filtered_df.shape[0]:,} | Columns: {filtered_df.shape[1]:,}")

    st.dataframe(filtered_df.head(50), use_container_width=True)

    st.markdown("---")
    st.subheader("Summary Statistics")
    # Use broad describe() without datetime_is_numeric for wider pandas compatibility
    st.dataframe(filtered_df.describe(include="all").transpose(), use_container_width=True)

    st.markdown("---")
    st.subheader("Missing Values")
    plot_missing_values(filtered_df)

    st.markdown("---")
    st.subheader("Correlation Heatmap")
    plot_correlation_heatmap(filtered_df)

    # Additional quick charts under overview
    st.markdown("---")
    st.subheader("Quick Visuals")

    # Default by Gender / Education / Marital status
    if target_col and target_col in filtered_df.columns:
        ov_cols = []
        if gender_col and gender_col in filtered_df.columns:
            ov_cols.append((gender_col, "Default by Gender"))
        if education_col and education_col in filtered_df.columns:
            ov_cols.append((education_col, "Default by Education"))
        if marriage_col and marriage_col in filtered_df.columns:
            ov_cols.append((marriage_col, "Default by Marital Status"))
        
        if ov_cols:
            for cat, title in ov_cols:
                ctab = pd.crosstab(filtered_df[cat].astype(str), filtered_df[target_col].astype(str))
                col1, col2 = st.columns(2)
                with col1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ctab.plot(kind='bar', ax=ax)
                    ax.set_title(title)
                    ax.set_xlabel(cat)
                    ax.set_ylabel('Count')
                    ax.legend(title=str(target_col))
                    ax.tick_params(axis='x', rotation=30)
                    st.pyplot(fig, clear_figure=True)
                with col2:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    (ctab / ctab.sum(axis=1).replace(0, np.nan)).plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(title + " (Proportion)")
                    ax.set_xlabel(cat)
                    ax.set_ylabel('Proportion')
                    ax.legend(title=str(target_col))
                    ax.tick_params(axis='x', rotation=30)
                    st.pyplot(fig, clear_figure=True)

    # Top numeric distributions (histogram + box)
    if numeric_cols:
        top_nums = numeric_cols[: min(3, len(numeric_cols))]
        for num in top_nums:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(filtered_df[num].dropna(), bins=30, kde=True, color="#2a9d8f", ax=ax)
                ax.set_title(f"Distribution: {num}")
                st.pyplot(fig, clear_figure=True)
            with c2:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=filtered_df, y=num, color="#e76f51", ax=ax)
                ax.set_title(f"Box Plot: {num}")
                st.pyplot(fig, clear_figure=True)

    # Optional pair plot on a small sample for performance
    if len(numeric_cols) >= 3 and len(filtered_df) <= 2000:
        try:
            st.markdown("**Pair Plot (sample)**")
            sample_df = filtered_df[numeric_cols[:5]].dropna().sample(frac=1.0, random_state=42)
            pair_fig = sns.pairplot(sample_df, corner=True)
            st.pyplot(pair_fig)
        except Exception:
            pass


with tab_eda:
    st.subheader("🔍 Exploratory Data Analysis & Visualization")
    
    # Quick EDA Grid (shows multiple charts by default)
    show_quick_grid = st.checkbox("Show Quick EDA Grid", value=True, help="Displays a small set of common plots automatically.")
    if show_quick_grid:
        qc1, qc2 = st.columns(2)
        # Histogram (first numeric)
        if numeric_cols:
            with qc1:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(filtered_df[col].dropna(), bins=30, kde=True, color="#2a9d8f", ax=ax)
                ax.set_title(f"Histogram: {col}")
                st.pyplot(fig, clear_figure=True)
        # Count plot (first categorical)
        if categorical_cols:
            with qc2:
                col = categorical_cols[0]
                order = filtered_df[col].astype(str).value_counts().index
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.countplot(data=filtered_df, x=col, order=order, color="#264653", ax=ax)
                ax.set_title(f"Counts: {col}")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
                st.pyplot(fig, clear_figure=True)
        
        qc3, qc4 = st.columns(2)
        # Box plot (second numeric if exists)
        if len(numeric_cols) >= 1:
            with qc3:
                col = numeric_cols[min(1, len(numeric_cols)-1)]
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=filtered_df, y=col, color="#e76f51", ax=ax)
                ax.set_title(f"Box Plot: {col}")
                st.pyplot(fig, clear_figure=True)
        # Pie on target
        if target_col and target_col in filtered_df.columns:
            with qc4:
                tgt_counts = filtered_df[target_col].astype(str).value_counts(dropna=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = ["#e76f51", "#2a9d8f", "#f4a261", "#457b9d", "#a8dadc"]
                ax.pie(tgt_counts.values, labels=tgt_counts.index, autopct="%1.1f%%", startangle=90, colors=colors[:len(tgt_counts)])
                ax.set_title("Target Distribution")
                ax.axis("equal")
                st.pyplot(fig, clear_figure=True)
        
        # Scatter + Violin quick views
        qc5, qc6 = st.columns(2)
        if len(numeric_cols) >= 2:
            with qc5:
                xcol = numeric_cols[0]
                ycol = numeric_cols[1]
                fig, ax = plt.subplots(figsize=(6, 4))
                if target_col and target_col in filtered_df.columns:
                    sns.scatterplot(data=filtered_df, x=xcol, y=ycol, hue=target_col, palette="Set2", ax=ax, s=35, alpha=0.7)
                    ax.legend(loc="best", fontsize="x-small")
                else:
                    sns.scatterplot(data=filtered_df, x=xcol, y=ycol, color="#1d3557", ax=ax, s=35, alpha=0.7)
                ax.set_title(f"Scatter: {ycol} vs {xcol}")
                st.pyplot(fig, clear_figure=True)
        if numeric_cols and target_col and target_col in filtered_df.columns:
            with qc6:
                col = numeric_cols[0]
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.violinplot(data=filtered_df, x=target_col, y=col, palette="Set2", ax=ax)
                ax.set_title(f"Violin: {col} by {target_col}")
                st.pyplot(fig, clear_figure=True)
        
        # Grouped/Stacked quick bars
        if target_col and target_col in filtered_df.columns:
            if education_col and education_col in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                pd.crosstab(filtered_df[education_col], filtered_df[target_col]).plot(kind='bar', ax=ax, color=['#e76f51', '#2a9d8f'])
                ax.set_title(f"Grouped: {education_col} vs {target_col}")
                ax.tick_params(axis='x', rotation=30)
                st.pyplot(fig, clear_figure=True)
            if marriage_col and marriage_col in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                crosstab = pd.crosstab(filtered_df[marriage_col], filtered_df[target_col], normalize='index')
                crosstab.plot(kind='bar', stacked=True, ax=ax, color=['#e76f51', '#2a9d8f'])
                ax.set_title(f"Stacked: {marriage_col} vs {target_col}")
                ax.tick_params(axis='x', rotation=30)
                st.pyplot(fig, clear_figure=True)
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "🟢 Univariate Analysis",
            "🔵 Bivariate Analysis", 
            "🟣 Advanced/Interactive Visuals"
        ]
    )
    
    if viz_type == "🟢 Univariate Analysis":
        st.markdown("### Univariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Histogram for Numeric Features**")
            if numeric_cols:
                num_col = st.selectbox("Select numeric feature", numeric_cols, key="hist_num")
                bins = st.slider("Number of bins", 5, 100, 30, key="hist_bins")
                kde = st.checkbox("Show KDE overlay", value=True, key="hist_kde")
                
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(filtered_df[num_col].dropna(), bins=bins, kde=kde, color="#2a9d8f", ax=ax)
                ax.set_xlabel(num_col)
                ax.set_title(f"Distribution of {num_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No numeric columns detected.")
        
        with col2:
            st.markdown("**📦 Box Plot for Outlier Detection**")
            if numeric_cols:
                box_col = st.selectbox("Select feature for box plot", numeric_cols, key="box_col")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.boxplot(data=filtered_df, y=box_col, color="#e76f51", ax=ax)
                ax.set_title(f"Box Plot of {box_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No numeric columns detected.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**📈 Density Plot (KDE)**")
            if numeric_cols:
                kde_col = st.selectbox("Select feature for density plot", numeric_cols, key="kde_col")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.kdeplot(data=filtered_df, x=kde_col, fill=True, color="#f4a261", ax=ax)
                ax.set_title(f"Density Plot of {kde_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No numeric columns detected.")
        
        with col4:
            st.markdown("**📊 Bar Chart for Categorical Features**")
            if categorical_cols:
                cat_col = st.selectbox("Select categorical feature", categorical_cols, key="cat_bar")
                order = filtered_df[cat_col].astype(str).value_counts().index
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=filtered_df, x=cat_col, order=order, color="#264653", ax=ax)
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
                ax.set_title(f"Counts of {cat_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("No categorical columns detected.")
        
        # Target distribution pie chart
        if target_col and target_col in df.columns:
            st.markdown("**🥧 Target Distribution (Pie Chart)**")
            tgt_counts = filtered_df[target_col].astype(str).value_counts(dropna=False)
            fig, ax = plt.subplots(figsize=(8, 6))
            colors = ["#e76f51", "#2a9d8f", "#f4a261", "#457b9d", "#a8dadc"]
            wedges, texts, autotexts = ax.pie(
                tgt_counts.values,
                labels=tgt_counts.index,
                autopct="%1.1f%%",
                startangle=90,
                colors=colors[: len(tgt_counts)],
                explode=[0.05] * len(tgt_counts)
            )
            ax.set_title("Target Distribution", fontsize=14, fontweight='bold')
            st.pyplot(fig, clear_figure=True)
        else:
            st.info("Target column not detected. Add 'default' column to view pie chart.")
    
    elif viz_type == "🔵 Bivariate Analysis":
        st.markdown("### Bivariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔍 Scatter Plot with Target Coloring**")
            if len(numeric_cols) >= 2 and target_col:
                xcol = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                ycol = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="scatter_y")
                fig, ax = plt.subplots(figsize=(8, 6))
                if target_col in filtered_df.columns:
                    sns.scatterplot(data=filtered_df, x=xcol, y=ycol, hue=target_col, palette="Set2", ax=ax, s=50, alpha=0.7)
                    ax.legend(title=target_col, loc="best")
                else:
                    sns.scatterplot(data=filtered_df, x=xcol, y=ycol, color="#1d3557", ax=ax, s=50, alpha=0.7)
                ax.set_title(f"{ycol} vs {xcol}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need at least two numeric columns and target column for scatter plot.")
        
        with col2:
            st.markdown("**🎻 Violin Plot by Target**")
            if numeric_cols and target_col and target_col in filtered_df.columns:
                violin_col = st.selectbox("Select feature for violin plot", numeric_cols, key="violin_col")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.violinplot(data=filtered_df, x=target_col, y=violin_col, palette="Set2", ax=ax)
                ax.set_title(f"Distribution of {violin_col} by {target_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need numeric columns and target column for violin plot.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**📊 Grouped Bar Chart**")
            if categorical_cols and target_col and target_col in filtered_df.columns:
                group_col = st.selectbox("Select categorical feature", categorical_cols, key="grouped_bar")
                fig, ax = plt.subplots(figsize=(8, 6))
                pd.crosstab(filtered_df[group_col], filtered_df[target_col]).plot(kind='bar', ax=ax, color=['#e76f51', '#2a9d8f'])
                ax.set_title(f"{group_col} vs {target_col}")
                ax.set_xlabel(group_col)
                ax.set_ylabel("Count")
                ax.legend(title=target_col)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need categorical columns and target column for grouped bar chart.")
        
        with col4:
            st.markdown("**📊 Stacked Bar Chart**")
            if categorical_cols and target_col and target_col in filtered_df.columns:
                stack_col = st.selectbox("Select feature for stacked chart", categorical_cols, key="stacked_bar")
                fig, ax = plt.subplots(figsize=(8, 6))
                crosstab = pd.crosstab(filtered_df[stack_col], filtered_df[target_col], normalize='index')
                crosstab.plot(kind='bar', stacked=True, ax=ax, color=['#e76f51', '#2a9d8f'])
                ax.set_title(f"{stack_col} vs {target_col} (Proportions)")
                ax.set_xlabel(stack_col)
                ax.set_ylabel("Proportion")
                ax.legend(title=target_col)
                ax.tick_params(axis='x', rotation=45)
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need categorical columns and target column for stacked bar chart.")
        
        # Correlation heatmap
        st.markdown("**🔥 Correlation Heatmap**")
        plot_correlation_heatmap(filtered_df)
    
    elif viz_type == "🟣 Advanced/Interactive Visuals":
        st.markdown("### Advanced/Interactive Visuals")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📈 Interactive Line Chart (Plotly)**")
            if numeric_cols:
                line_col = st.selectbox("Select feature for line chart", numeric_cols, key="line_col")
                fig = px.line(filtered_df, y=line_col, title=f"Trend of {line_col}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for line chart.")
        
        with col2:
            st.markdown("**📊 Area Chart (Plotly)**")
            if numeric_cols:
                area_col = st.selectbox("Select feature for area chart", numeric_cols, key="area_col")
                fig = px.area(filtered_df, y=area_col, title=f"Area Chart of {area_col}")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric columns for area chart.")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**📦 Boxen Plot (Detailed Distribution)**")
            if numeric_cols and target_col and target_col in filtered_df.columns:
                boxen_col = st.selectbox("Select feature for boxen plot", numeric_cols, key="boxen_col")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxenplot(data=filtered_df, x=target_col, y=boxen_col, palette="Set2", ax=ax)
                ax.set_title(f"Boxen Plot of {boxen_col} by {target_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need numeric columns and target for boxen plot.")
        
        with col4:
            st.markdown("**🐝 Swarm Plot (Detailed Distribution)**")
            if numeric_cols and target_col and target_col in filtered_df.columns and len(filtered_df) < 1000:
                swarm_col = st.selectbox("Select feature for swarm plot", numeric_cols, key="swarm_col")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.swarmplot(data=filtered_df, x=target_col, y=swarm_col, palette="Set2", ax=ax)
                ax.set_title(f"Swarm Plot of {swarm_col} by {target_col}")
                st.pyplot(fig, clear_figure=True)
            else:
                st.info("Need numeric columns, target, and <1000 rows for swarm plot.")
        
        # 3D Scatter Plot
        if len(numeric_cols) >= 3 and target_col and target_col in filtered_df.columns:
            st.markdown("**🌐 3D Scatter Plot (Plotly)**")
            col_3d1, col_3d2, col_3d3 = st.columns(3)
            with col_3d1:
                x_3d = st.selectbox("X-axis", numeric_cols, key="3d_x")
            with col_3d2:
                y_3d = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1), key="3d_y")
            with col_3d3:
                z_3d = st.selectbox("Z-axis", numeric_cols, index=min(2, len(numeric_cols)-1), key="3d_z")
            
            fig = px.scatter_3d(
                filtered_df, 
                x=x_3d, y=y_3d, z=z_3d, 
                color=target_col,
                title=f"3D Scatter: {x_3d} vs {y_3d} vs {z_3d}",
                opacity=0.7
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Treemap
        if categorical_cols and target_col and target_col in filtered_df.columns:
            st.markdown("**🌳 Treemap (Hierarchical Data)**")
            treemap_col = st.selectbox("Select categorical feature for treemap", categorical_cols, key="treemap_col")
            
            # Create treemap data
            treemap_data = filtered_df.groupby([treemap_col, target_col]).size().reset_index(name='count')
            treemap_data['path'] = treemap_data[treemap_col] + ' - ' + treemap_data[target_col].astype(str)
            
            fig = px.treemap(
                treemap_data, 
                path=['path'], 
                values='count',
                title=f"Treemap: {treemap_col} vs {target_col}"
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        # Radar Chart
        if numeric_cols and target_col and target_col in filtered_df.columns:
            st.markdown("**🕸️ Radar Chart (Feature Comparison)**")
            radar_cols = st.multiselect("Select features for radar chart", numeric_cols, default=numeric_cols[:5], key="radar_cols")
            
            if len(radar_cols) >= 3:
                # Calculate means by target
                radar_data = []
                for target_val in sorted(filtered_df[target_col].unique()):
                    subset = filtered_df[filtered_df[target_col] == target_val]
                    means = [subset[col].mean() for col in radar_cols]
                    radar_data.append(go.Scatterpolar(
                        r=means,
                        theta=radar_cols,
                        fill='toself',
                        name=f'{target_col} = {target_val}'
                    ))
                
                fig = go.Figure(data=radar_data)
                
                # Calculate max value for radial axis
                max_val = 0
                for target_val in sorted(filtered_df[target_col].unique()):
                    subset = filtered_df[filtered_df[target_col] == target_val]
                    means = [subset[col].mean() for col in radar_cols]
                    max_val = max(max_val, max(means))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, max_val * 1.1]),
                    ),
                    showlegend=True,
                    title="Radar Chart: Feature Comparison by Target"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least 3 numeric features for radar chart.")


with tab_model:
    st.subheader("Train a Simple Model")
    if not target_col:
        st.info("Target column not detected. The target column should be one of: 'default', 'default_status', 'is_default', 'default_payment_next_month', or 'default.payment.next.month'.")
    else:
        st.write(f"Detected target column: `{target_col}`")
        features = [c for c in df.columns if c != target_col]

        model_type = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=0)
        test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)

        # Prepare features and target
        X = df[features]
        y = df[target_col]

        # Ensure target is binary/int if possible
        if y.dtype == "O" or str(y.dtype).startswith("category"):
            # Try to map typical values to 0/1
            y = y.astype(str).str.strip().str.lower().map({"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1}).fillna(y)
        try:
            y = y.astype(int)
        except Exception:
            st.error("Target column could not be coerced to integers (0/1). Please ensure it's binary.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
        )

        num_cols, cat_cols = get_numeric_and_categorical_columns(X_train)

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_cols),
                ("cat", categorical_transformer, cat_cols),
            ],
            remainder="drop",
        )

        if model_type == "Logistic Regression":
            model = LogisticRegression(max_iter=200, n_jobs=None)
        else:
            model = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")

        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)
                acc = accuracy_score(y_test, preds)

            st.success(f"Accuracy: {acc:.3f}")

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig, clear_figure=True)

            # Classification Report
            st.markdown("**Classification Report**")
            st.text(classification_report(y_test, preds, digits=3))

            # Feature Importance
            st.markdown("**Feature Importance**")
            if hasattr(clf.named_steps['model'], 'feature_importances_'):
                # Random Forest feature importance
                feature_names = []
                if len(num_cols) > 0:
                    feature_names.extend(num_cols)
                if len(cat_cols) > 0:
                    # Get one-hot encoded feature names
                    cat_features = clf.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
                    feature_names.extend(cat_features)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': clf.named_steps['model'].feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(15), x='importance', y='feature', ax=ax, color="#2a9d8f")
                ax.set_title("Top 15 Feature Importances")
                ax.set_xlabel("Importance")
                st.pyplot(fig, clear_figure=True)
                
                st.dataframe(importance_df, use_container_width=True)
            elif hasattr(clf.named_steps['model'], 'coef_'):
                # Logistic Regression coefficients
                feature_names = []
                if len(num_cols) > 0:
                    feature_names.extend(num_cols)
                if len(cat_cols) > 0:
                    cat_features = clf.named_steps['preprocess'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols)
                    feature_names.extend(cat_features)
                
                coef_df = pd.DataFrame({
                    'feature': feature_names,
                    'coefficient': clf.named_steps['model'].coef_[0]
                }).sort_values('coefficient', key=abs, ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#e76f51' if x < 0 else '#2a9d8f' for x in coef_df.head(15)['coefficient']]
                sns.barplot(data=coef_df.head(15), x='coefficient', y='feature', ax=ax, palette=colors)
                ax.set_title("Top 15 Feature Coefficients (Logistic Regression)")
                ax.set_xlabel("Coefficient")
                st.pyplot(fig, clear_figure=True)
                
                st.dataframe(coef_df, use_container_width=True)
            else:
                st.info("Feature importance not available for this model type.")


with tab_cluster:
    st.subheader("🧩 Clustering Analysis")
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric columns for clustering.")
    else:
        # Controls
        default_features = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols[:2]
        selected_features = st.multiselect(
            "Select features for clustering (2-3 recommended)",
            numeric_cols,
            default=default_features,
        )
        if len(selected_features) < 2:
            st.warning("Select at least 2 numeric features.")
        else:
            sample_n = st.slider("Sample size for speed", min_value=500, max_value=min(10000, len(filtered_df)), value=min(5000, len(filtered_df)), step=500)
            use_scaler = st.checkbox("Standardize features", value=True)
            df_sample = filtered_df[selected_features].dropna().sample(n=sample_n, random_state=42) if len(filtered_df) > sample_n else filtered_df[selected_features].dropna()
            X = df_sample.values
            if use_scaler:
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            else:
                X_scaled = X

            # --- KMeans ---
            st.markdown("### K-Means Clustering")
            km_col1, km_col2 = st.columns([1,1])
            with km_col1:
                k_min, k_max = st.slider("K range (elbow)", 1, 12, (1, 10))
                wcss = []
                for k in range(k_min, k_max + 1):
                    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42)
                    kmeans.fit(X_scaled)
                    wcss.append(kmeans.inertia_)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(range(k_min, k_max + 1), wcss, marker='o')
                ax.set_title('Elbow Method for K-Means')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('WCSS')
                st.pyplot(fig, clear_figure=True)
            with km_col2:
                k = st.number_input("K (clusters)", min_value=2, max_value=12, value=min(3, k_max))
                kmeans = KMeans(n_clusters=int(k), init='k-means++', max_iter=300, n_init=10, random_state=42)
                labels = kmeans.fit_predict(X_scaled)
                # 2D scatter using first two selected features
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=labels, palette='viridis', legend='full', ax=ax)
                ax.set_title(f'K-Means Clustering (k={int(k)})')
                ax.set_xlabel(f'Scaled {selected_features[0]}')
                ax.set_ylabel(f'Scaled {selected_features[1]}')
                st.pyplot(fig, clear_figure=True)

            st.markdown("---")
            # --- Agglomerative ---
            st.markdown("### Agglomerative Clustering")
            agg_c1, agg_c2 = st.columns([1,1])
            with agg_c1:
                # Dendrogram (Ward)
                linked = linkage(X_scaled, method='ward')
                fig, ax = plt.subplots(figsize=(7, 4))
                dendrogram(linked, orientation='top', p=12, truncate_mode='lastp', distance_sort='descending', show_leaf_counts=True, ax=ax)
                ax.set_title('Dendrogram (Ward linkage)')
                ax.set_xlabel('Sample index / Cluster size')
                ax.set_ylabel('Distance')
                st.pyplot(fig, clear_figure=True)
            with agg_c2:
                n_clusters = st.number_input("Agglomerative clusters", min_value=2, max_value=12, value=3)
                agg = AgglomerativeClustering(n_clusters=int(n_clusters), linkage='ward')
                agg_labels = agg.fit_predict(X_scaled)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=agg_labels, palette='viridis', legend='full', ax=ax)
                ax.set_title(f'Agglomerative (n={int(n_clusters)})')
                ax.set_xlabel(f'Scaled {selected_features[0]}')
                ax.set_ylabel(f'Scaled {selected_features[1]}')
                st.pyplot(fig, clear_figure=True)

            st.markdown("---")
            # --- DBSCAN ---
            st.markdown("### DBSCAN Clustering")
            db_c1, db_c2 = st.columns([1,1])
            with db_c1:
                k_neighbors = max(2, 2 * len(selected_features))
                neighbors = NearestNeighbors(n_neighbors=k_neighbors)
                neighbors_fit = neighbors.fit(X_scaled)
                distances, indices = neighbors_fit.kneighbors(X_scaled)
                distances = np.sort(distances[:, -1], axis=0)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(distances)
                ax.set_title(f'K-distance Graph (k={k_neighbors})')
                ax.set_xlabel('Points sorted by distance')
                ax.set_ylabel('Distance to k-th neighbor (eps)')
                st.pyplot(fig, clear_figure=True)
            with db_c2:
                eps = st.number_input("DBSCAN eps", min_value=0.05, max_value=5.0, value=0.5, step=0.05)
                min_samples = st.number_input("DBSCAN min_samples", min_value=2, max_value=200, value=k_neighbors)
                db = DBSCAN(eps=float(eps), min_samples=int(min_samples))
                db_labels = db.fit_predict(X_scaled)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=db_labels, palette='viridis', legend='full', ax=ax)
                ax.set_title('DBSCAN Clustering')
                ax.set_xlabel(f'Scaled {selected_features[0]}')
                ax.set_ylabel(f'Scaled {selected_features[1]}')
                st.pyplot(fig, clear_figure=True)

            # DBSCAN summary
            n_clusters_ = len(set(db_labels)) - (1 if -1 in db_labels else 0)
            n_noise_ = list(db_labels).count(-1)
            st.caption(f"DBSCAN clusters: {n_clusters_} | Noise points: {n_noise_}")


with tab_insights:
    st.subheader("Key Insights")
    insights = []

    # Default rate
    if target_col and target_col in df.columns:
        default_rate = (df[target_col] == 1).mean() if pd.api.types.is_integer_dtype(df[target_col]) else (df[target_col].astype(str).str.lower().isin(["1", "yes", "true"]).mean())
        insights.append(f"Estimated default rate: {default_rate:.1%}")

    # Correlation with target (numeric only)
    if target_col and target_col in df.columns:
        try:
            y_bin = df[target_col]
            if y_bin.dtype == "O" or str(y_bin.dtype).startswith("category"):
                y_bin = y_bin.astype(str).str.strip().str.lower().map({"0": 0, "1": 1, "no": 0, "yes": 1, "false": 0, "true": 1}).astype(float)
            corr_with_target = df.select_dtypes(include=[np.number]).corr(numeric_only=True)[target_col].drop(target_col).sort_values(ascending=False)
            top_corr = corr_with_target.head(3).index.tolist()
            if len(top_corr) > 0:
                insights.append("Top correlated numeric features with target: " + ", ".join(top_corr))
        except Exception:
            pass

    # Grouped default rate by key categories
    for c in [gender_col, education_col, marriage_col]:
        if c and target_col and c in df.columns and target_col in df.columns:
            try:
                grp = df.groupby(c)[target_col]
                rate = grp.apply(lambda s: (s == 1).mean() if pd.api.types.is_integer_dtype(s) else s.astype(str).str.lower().isin(["1", "yes", "true"]).mean())
                top = rate.sort_values(ascending=False).head(3)
                insights.append(f"Highest default rates by {c}: " + ", ".join([f"{idx} ({val:.1%})" for idx, val in top.items()]))
            except Exception:
                continue

    if insights:
        for i in insights:
            st.markdown(f"- **{i}**")
    else:
        st.info("Insights will appear here once a valid target and features are detected.")


# Footer
st.markdown("---")
st.caption("Built with Streamlit · Upload a CSV to explore and model your data.")


