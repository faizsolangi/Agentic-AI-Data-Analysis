# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    classification_report, mean_squared_error, r2_score)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional XGBoost (if installed)
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False

# OpenAI
import openai
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

st.set_page_config(page_title="Agentic Data Analysis Bot", layout="wide")
st.title("🤖 Agentic AI Data Analysis Bot — Streamlit (LLM selects model)")

# Utility functions ----------------------------------------------------------------
@st.cache_data
def load_dataframe(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip column names and whitespace in string cells
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None})
    # Normalize common country abbreviations (example)
    if "Country" in df.columns:
        df["Country"] = df["Country"].replace({"Pak": "Pakistan", "pak": "Pakistan", "UAE": "UAE"})
    return df

def generate_metadata(df: pd.DataFrame, target_col: str = None) -> dict:
    meta = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_columns": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
    }
    if target_col:
        meta["target"] = {
            "name": target_col,
            "unique_values": int(df[target_col].nunique()),
            "dtype": str(df[target_col].dtype)
        }
    return meta

def llm_select_model(metadata: dict, sample_df: pd.DataFrame) -> str:
    """
    Ask the LLM to pick a suitable model based on metadata.
    Returns the model string (e.g., 'RandomForest', 'XGBoost', 'LogisticRegression', 'LinearRegression').
    If OpenAI key is unavailable or LLM fails, uses fallback heuristic.
    """
    prompt = f"""
You are an expert ML engineer. Given the dataset metadata and a small sample, recommend ONE best model to train.
Return only the model name from this list if appropriate:
['LogisticRegression','RandomForest','XGBoost','LinearRegression','Lasso','Ridge'].

Metadata:
{metadata}

Sample (first 8 rows):
{sample_df.to_string()}

Consider: dataset size, datatypes, target type (classification vs regression), class imbalance, non-linearity. Reply with model name only.
"""
    # If API key is available try LLM
    if OPENAI_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",  # use a modern model; replace if you prefer gpt-4.1-mini
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message["content"].strip()
            # sanitize to known names
            for candidate in ["LogisticRegression","RandomForest","XGBoost","LinearRegression","Lasso","Ridge"]:
                if candidate.lower() in content.lower():
                    return candidate
            # fallback to first word
            return content.split()[0]
        except Exception as e:
            st.warning(f"LLM model selection failed, using heuristic. ({e})")
    # Fallback heuristic
    return heuristic_select_model(metadata)

def heuristic_select_model(metadata: dict) -> str:
    # If target present
    target = metadata.get("target")
    if not target:
        return "RandomForest"
    dtype = target.get("dtype", "")
    unique = target.get("unique_values", 0)
    rows = metadata.get("rows", 0)
    numeric_cols = len(metadata.get("numeric_columns", []))
    # classification if dtype not numeric or few uniques
    if dtype.startswith("int") or dtype.startswith("float"):
        # treat as regression if many unique
        if unique > 20:
            if numeric_cols > 10:
                return "Lasso"
            return "RandomForest"
        else:
            # numeric but few unique => classification
            return "RandomForest"
    else:
        # categorical target
        if rows < 2000:
            return "RandomForest"
        else:
            if numeric_cols > 20:
                return "LogisticRegression"
            return "RandomForest"

def build_pipeline(df: pd.DataFrame, target_col: str):
    # Determine numeric and categorical features
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # Transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]) if len(categorical_cols) > 0 else None

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

# UI ------------------------------------------------------------------------------
st.sidebar.header("Options")
st.sidebar.markdown("This app runs entirely in Streamlit. Optional: set OPENAI_API_KEY as an environment variable to enable LLM model selection.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv", "xlsx", "xls"])
if not uploaded_file:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

# Load
with st.spinner("Loading data..."):
    df = load_dataframe(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head())

# Cleaning
if st.checkbox("Run basic cleaning (strip whitespace, normalize columns)", value=True):
    df = basic_clean(df)
    st.success("Basic cleaning applied.")
    st.dataframe(df.head())

# Choose target column
st.subheader("Target selection")
cols = df.columns.tolist()
target_col = st.selectbox("Select target column (for modeling). Leave blank to skip modeling.", options=[""] + cols)
if target_col == "":
    target_col = None

# Show metadata
metadata = generate_metadata(df, target_col)
st.subheader("Dataset metadata")
st.json(metadata)

# EDA
if st.checkbox("Show EDA (summary, histograms, correlation)"):
    st.subheader("Summary statistics")
    st.write(df.describe(include='all').T)

    st.subheader("Histograms (numeric cols)")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        st.write("No numeric columns to plot.")
    else:
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Correlation heatmap (numeric columns)")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="vlag")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Not enough numeric columns for correlation matrix.")

# Ask LLM (or heuristic) to select model
st.subheader("Model selection (LLM-powered)")
if target_col is None:
    st.info("No target selected — skipping model selection and training.")
else:
    sample = df.head(8)
    selected_model = llm_select_model(metadata, sample)
    st.success(f"Selected model (LLM or heuristic): **{selected_model}**")

    # Map model name to actual estimator
    def get_estimator(name, task_type):
        name = name.lower()
        if "xgboost" in name or "xgb" in name:
            if HAS_XGBOOST:
                return XGBClassifier(eval_metric="logloss") if task_type=="classification" else XGBRegressor()
            else:
                st.warning("XGBoost not installed — falling back to RandomForest.")
                return RandomForestClassifier(n_estimators=200, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=200, random_state=42)
        if "randomforest" in name:
            return RandomForestClassifier(n_estimators=200, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=200, random_state=42)
        if "logistic" in name:
            return LogisticRegression(max_iter=1000)
        if "lasso" in name:
            return Lasso()
        if "ridge" in name:
            return Ridge()
        if "linear" in name:
            return LinearRegression()
        # default
        return RandomForestClassifier(n_estimators=200, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=200, random_state=42)

    # Determine task type
    target_series = df[target_col]
    is_regression = pd.api.types.is_float_dtype(target_series) or pd.api.types.is_integer_dtype(target_series) and target_series.nunique() > 20
    task = "regression" if is_regression else "classification"
    st.write(f"Detected task type: **{task}**")

    # Prepare data: basic encode target if categorical
    model_df = df.copy()
    if task == "classification" and not pd.api.types.is_numeric_dtype(model_df[target_col]):
        model_df[target_col] = model_df[target_col].astype('category').cat.codes

    # Build preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = build_pipeline(model_df, target_col)
    st.write(f"Numeric features: {numeric_cols}")
    st.write(f"Categorical features: {categorical_cols}")

    # Split data
    X = model_df.drop(columns=[target_col])
    y = model_df[target_col]
    test_size = st.slider("Test set size (%)", min_value=10, max_value=40, value=20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=42, stratify=y if task=="classification" else None)

    # Build final estimator
    estimator = get_estimator(selected_model, "classification" if task=="classification" else "regression")
    # Full pipeline
    pipeline = Pipeline(steps=[('pre', preprocessor), ('est', estimator)])

    # Train
    if st.button("Train selected model"):
        with st.spinner("Training..."):
            pipeline.fit(X_train, y_train)
        st.success("Model trained!")

        # Predict and evaluate
        preds = pipeline.predict(X_test)
        if task == "classification":
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='weighted', zero_division=0)
            rec = recall_score(y_test, preds, average='weighted', zero_division=0)
            f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
            st.subheader("Classification metrics")
            st.write(f"Accuracy: {acc:.4f}")
            st.write(f"Precision (weighted): {prec:.4f}")
            st.write(f"Recall (weighted): {rec:.4f}")
            st.write(f"F1 (weighted): {f1:.4f}")
            st.text("Classification report:")
            st.text(classification_report(y_test, preds))

            # Confusion matrix
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close(fig)
        else:
            rmse = mean_squared_error(y_test, preds, squared=False)
            r2 = r2_score(y_test, preds)
            st.subheader("Regression metrics")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"R²: {r2:.4f}")

        # Feature importances (if available)
        st.subheader("Feature importances / coefficients")
        try:
            est = pipeline.named_steps['est']
            # If tree-based with feature_importances_
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                # Need feature names after preprocessing
                # Get preprocessor output feature names
                ohe_cols = []
                if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] is not None:
                    # ColumnTransformer -> OneHotEncoder inside
                    cat_pipe = preprocessor.named_transformers_.get('cat')
                    if cat_pipe is not None:
                        try:
                            ohe = cat_pipe.named_steps['onehot']
                            cat_cols = categorical_cols
                            ohe_cols = list(ohe.get_feature_names_out(cat_cols))
                        except Exception:
                            ohe_cols = categorical_cols
                feature_names = numeric_cols + ohe_cols
                # if lengths mismatch, fallback to numeric columns
                if len(importances) == len(feature_names):
                    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                else:
                    # fallback: show top numeric importances
                    feat_imp = pd.Series(importances[:len(numeric_cols)], index=numeric_cols).sort_values(ascending=False)
                st.dataframe(feat_imp.head(20).to_frame("importance"))
                fig, ax = plt.subplots()
                feat_imp.head(20).plot.bar(ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            elif hasattr(est, "coef_"):
                coefs = est.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]
                # approximate feature names as numeric_cols + categorical encoded names
                feature_names = numeric_cols
                coefs_series = pd.Series(coefs[:len(feature_names)], index=feature_names).sort_values(key=abs, ascending=False)
                st.dataframe(coefs_series.head(20).to_frame("coef"))
            else:
                st.write("Estimator does not expose importances/coefficients.")
        except Exception as e:
            st.warning(f"Could not compute feature importances: {e}")

        # Allow downloading cleaned CSV and model
        st.subheader("Download artifacts")
        # Cleaned CSV
        csv_bytes = io.BytesIO()
        df.to_csv(csv_bytes, index=False)
        csv_bytes.seek(0)
        st.download_button("Download cleaned CSV", csv_bytes, file_name="cleaned_data.csv", mime="text/csv")

        # Joblib model
        model_bytes = io.BytesIO()
        joblib.dump(pipeline, model_bytes)
        model_bytes.seek(0)
        st.download_button("Download trained model (joblib)", model_bytes, file_name="trained_model.joblib", mime="application/octet-stream")
