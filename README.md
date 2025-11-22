# Agentic-AI-Data-Analysis

An interactive, fully automated **Agentic AI Data Analysis Bot** built
using **Streamlit**, capable of:

-   Data preprocessing\
-   Automatic feature engineering\
-   Automatic model selection using LLM\
-   Model training & evaluation\
-   Dashboard-style visualization\
-   One-click deployment on Render

------------------------------------------------------------------------

## 🚀 Features

### ✅ Upload CSV

Simply upload a dataset, and the bot will handle:

-   Data cleaning\
-   Data transformation\
-   Data reduction\
-   Discretization\
-   Feature engineering

### 🤖 LLM‑Based Model Selector

The LLM automatically decides the best ML algorithm based on dataset
type and shape:

-   Classification → Logistic Regression, Random Forest, XGBoost\
-   Regression → Linear Regression, Random Forest Regressor\
-   Mixed → Handles preprocessing intelligently

### 📊 Visual Dashboard

Streamlit automatically visualizes:

-   Summary statistics\
-   Correlation heatmap\
-   Histograms & box plots\
-   Missing values report

### 🧠 Model Training + Metrics

Outputs:

-   Accuracy / RMSE\
-   Confusion matrix\
-   Feature importance

------------------------------------------------------------------------

## 🛠️ Installation

``` bash
git clone https://github.com/YOUR_USERNAME/agentic-ai-data-analysis-bot.git
cd agentic-ai-data-analysis-bot
pip install -r requirements.txt
streamlit run app.py
```

------------------------------------------------------------------------

## 📂 Project Structure

    ├── app.py
    ├── dummy_data.csv
    ├── README.md
    ├── requirements.txt

------------------------------------------------------------------------

## 🚀 Deploy on Render

1.  Push this repo to GitHub.\

2.  Go to **Render → New Web Service**.\

3.  Choose environment: **Python 3.x**\

4.  Add build command:

        pip install -r requirements.txt

5.  Start command:

        streamlit run app.py --server.port $PORT --server.address 0.0.0.0

------------------------------------------------------------------------

## 📌 License

MIT License.
