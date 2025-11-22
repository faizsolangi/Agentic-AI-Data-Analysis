# app.py
import os
import io
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score
)

# Models
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Optional: XGBoost
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
st.title("🤖 Agentic AI Data Analysis Bot — Streamlit (OpenAI model selection)")

# ---------------------- Utilities ----------------------
@st.cache_data
def load_dataframe(uploaded_file):
    # Try CSV then Excel
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Strip column names & whitespace string values
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": None})
    # Example normalization
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

def heuristic_select_model(metadata: dict) -> str:
    """
    A robust fallback heuristic to select a model name when LLM isn't available.
    Returns one of: LogisticRegression, RandomForest, XGBoost, LinearRegression, Lasso, Ridge
    """
    target = metadata.get("target")
    if not target:
        return "RandomForest"
    dtype = target.get("dtype", "")
    unique = target.get("unique_values", 0)
    rows = metadata.get("rows", 0)
    numeric_cols = len(metadata.get("numeric_columns", []))

    # Simple heuristic:
    if dtype.startswith("int") or dtype.startswith("float"):
        # If many unique values -> regression
        if unique > 20:
            if numeric_cols > 15 and rows > 2000:
                return "Lasso"
            return "RandomForest"
        else:
            # numeric but few unique -> classification problem
            return "RandomForest"
    else:
        # categorical target
        if rows < 2000:
            return "RandomForest"
        else:
            if numeric_cols > 20:
                return "LogisticRegression"
            return "RandomForest"

def llm_select_model(metadata: dict, sample_df: pd.DataFrame) -> str:
    """
    Ask OpenAI to pick a model. If OpenAI key missing or fails, use heuristic.
    Returns model string.
    """
    allowed = ["LogisticRegression", "RandomForest", "XGBoost", "LinearRegression", "Lasso", "Ridge"]
    prompt = f"""
You are an expert ML engineer. Choose ONE best model from this list:
{allowed}

Given the dataset metadata and example rows, reply with ONLY the model name from the list (no explanation).

Metadata:
{metadata}

Sample rows (first 8):
{sample_df.to_string()}

Consider dataset size, feature types, whether target is regression/classification, class imbalance, and simple generalization tradeoffs.
"""
    if OPENAI_KEY:
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4.1-mini",  # change if you prefer another model
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            content = resp.choices[0].message["content"].strip()
            # sanitize
            for cand in allowed:
                if cand.lower() in content.lower():
                    return cand
            # attempt to extract a word that matches
            for cand in allowed:
                if cand.split()[0].lower() in content.lower():
                    return cand
            return content.split()[0]
        except Exception as e:
            st.warning(f"OpenAI selection failed. Using heuristic fallback. ({e})")
    # fallback
    return heuristic_select_model(metadata)

def build_pipeline(df: pd.DataFrame, target_col: str):
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = None
    if categorical_cols:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])

    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    return preprocessor, numeric_cols, categorical_cols

def get_estimator(name: str, task_type: str):
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
    # default fallback
    return RandomForestClassifier(n_estimators=200, random_state=42) if task_type=="classification" else RandomForestRegressor(n_estimators=200, random_state=42)

# ---------------------- UI ----------------------
st.sidebar.header("Options")
st.sidebar.write("Set OPENAI_API_KEY in environment to enable LLM model selection.")

uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx", "xls"])
if not uploaded_file:
    st.info("Please upload a CSV or Excel file to begin.")
    st.stop()

with st.spinner("Loading data..."):
    df = load_dataframe(uploaded_file)

st.subheader("Raw data preview")
st.dataframe(df.head())

# Basic cleaning
if st.checkbox("Run basic cleaning (strip whitespace, normalize strings)", value=True):
    df = basic_clean(df)
    st.success("Basic cleaning applied.")
    st.dataframe(df.head())

# ------------------ Target auto-detection ------------------
st.subheader("Target column (auto-detect or choose manually)")
lower_cols = [c.lower() for c in df.columns]
automatic_candidates = ["target","label","class","churn","y","outcome","response"]
detected = None
for candidate in automatic_candidates:
    if candidate in lower_cols:
        detected = df.columns[lower_cols.index(candidate)]
        break

# also detect if last column looks like a target heuristically
if detected is None:
    last_col = df.columns[-1]
    # if last col isn't unique per row (more than 2 unique values) consider it
    if df[last_col].nunique() > 1 and df[last_col].nunique() < df.shape[0]:
        detected = last_col

# Let user confirm or override
selected_target = st.selectbox("Detected target (you can override)", options=["(none)"] + df.columns.tolist(), index=(df.columns.get_loc(detected)+1 if detected else 0))
if selected_target == "(none)":
    target_col = None
else:
    target_col = selected_target
    st.success(f"Using target column: {target_col}")

metadata = generate_metadata(df, target_col)
st.subheader("Dataset metadata")
st.json(metadata)

# EDA
if st.checkbox("Show EDA (summary, histograms, correlation)"):
    st.subheader("Summary statistics")
    st.write(df.describe(include='all').T)

    st.subheader("Histograms")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if num_cols:
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Histogram: {col}")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.write("No numeric columns.")

    st.subheader("Correlation heatmap")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="vlag")
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.write("Not enough numeric columns.")

# Model selection & training
if target_col is None:
    st.info("No target selected — model selection and training are skipped. Select a target to enable modeling.")
else:
    st.subheader("Model selection (OpenAI LLM or heuristic fallback)")
    sample = df.head(8)
    # Refresh metadata with target info
    metadata = generate_metadata(df, target_col)
    model_name = llm_select_model(metadata, sample)
    st.success(f"Selected model: **{model_name}**")

    # Determine task
    target_series = df[target_col]
    # classification if categorical or low unique counts
    is_regression = pd.api.types.is_numeric_dtype(target_series) and (target_series.nunique() > 20)
    task = "regression" if is_regression else "classification"
    st.write(f"Detected task type: **{task}**")

    # Prepare dataset
    df_model = df.copy()
    if task == "classification" and not pd.api.types.is_numeric_dtype(df_model[target_col]):
        df_model[target_col] = df_model[target_col].astype('category').cat.codes

    # Build preprocessing pipeline
    preprocessor, numeric_cols, categorical_cols = build_pipeline(df_model, target_col)
    st.write(f"Numeric features: {numeric_cols}")
    st.write(f"Categorical features: {categorical_cols}")

    X = df_model.drop(columns=[target_col])
    y = df_model[target_col]

    test_pct = st.slider("Test set size (%)", min_value=10, max_value=40, value=20)
    stratify = y if task=="classification" else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100.0, random_state=42, stratify=stratify)

    estimator = get_estimator(model_name, "classification" if task=="classification" else "regression")
    pipeline = Pipeline(steps=[('pre', preprocessor), ('est', estimator)])

    if st.button("Train model"):
        with st.spinner("Training..."):
            pipeline.fit(X_train, y_train)
        st.success("Training completed.")

        # Predict & evaluate
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

        # Feature importance
        st.subheader("Feature importances / coefficients")
        try:
            est = pipeline.named_steps['est']
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
                # Get preprocessed feature names
                ohe_cols = []
                try:
                    if 'cat' in preprocessor.named_transformers_ and preprocessor.named_transformers_['cat'] is not None:
                        cat_pipe = preprocessor.named_transformers_['cat']
                        ohe = cat_pipe.named_steps.get('onehot', None)
                        if ohe is not None:
                            cat_cols = categorical_cols
                            ohe_cols = list(ohe.get_feature_names_out(cat_cols))
                except Exception:
                    ohe_cols = categorical_cols

                feature_names = numeric_cols + ohe_cols
                if len(importances) == len(feature_names):
                    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                else:
                    # fallback: align to numeric cols
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
                feature_names = numeric_cols
                coefs_series = pd.Series(coefs[:len(feature_names)], index=feature_names).sort_values(key=abs, ascending=False)
                st.dataframe(coefs_series.head(20).to_frame("coef"))
            else:
                st.write("No feature importances available for this estimator.")
        except Exception as e:
            st.warning(f"Could not compute importances: {e}")

        # Download artifacts
        st.subheader("Download artifacts")
        csv_buf = io.BytesIO()
        df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        st.download_button("Download cleaned CSV", csv_buf, file_name="cleaned_data.csv", mime="text/csv")

        model_buf = io.BytesIO()
        joblib.dump(pipeline, model_buf)
        model_buf.seek(0)
        st.download_button("Download trained model (joblib)", model_buf, file_name="trained_model.joblib", mime="application/octet-stream")

# ------------- Optional: Ask LLM for explanation of selection -------------
if target_col is not None:
    if st.checkbox("Explain model choice (ask OpenAI)"):
        if OPENAI_KEY:
            explain_prompt = f"""
You are an ML expert. Given the dataset metadata:
{metadata}

And that the selected model is: {model_name}

Explain in 3 short bullet points why that model is appropriate for this dataset, and any caveats.
"""
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-4.1-mini",
                    messages=[{"role": "user", "content": explain_prompt}],
                    temperature=0
                )
                explanation = resp.choices[0].message["content"].strip()
                st.markdown(explanation)
            except Exception as e:
                st.error(f"OpenAI failed for explanation: {e}")
        else:
            st.info("Set OPENAI_API_KEY in environment to enable explanation.")
