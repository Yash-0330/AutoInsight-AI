# AutoInsight AI — Autonomous Data Science Agent

An interactive Python-based data science application that automates dataset profiling, exploratory data analysis, anomaly detection, machine learning model comparison, SQL exploration, and analytical report generation.

## Overview

AutoInsight AI is designed to simplify repetitive data analysis tasks by bringing essential data science workflows into one web application. Users can upload a CSV dataset, inspect data quality, explore patterns, identify unusual observations, train machine learning models, and generate downloadable reports.

The application is built with Python and Streamlit, making data analysis accessible through a simple browser-based interface.

## Features

- **Automated Data Profiling:** Summarizes dataset dimensions, data types, missing values, duplicate rows, unique values, and memory usage.
- **Exploratory Data Analysis (EDA):** Displays statistical summaries, distributions, categorical patterns, and correlations.
- **Data Quality Inspection:** Highlights missing data and other quality issues for further investigation.
- **Anomaly Detection:** Uses Isolation Forest to flag potentially unusual records.
- **Machine Learning:** Trains and compares classification or regression models based on a selected target column.
- **Model Evaluation:** Reports relevant metrics such as accuracy, precision, recall, F1-score, MAE, RMSE, and R², depending on the task.
- **SQL Explorer:** Supports read-only SQL `SELECT` queries against data stored in SQLite.
- **Automated Reporting:** Generates a downloadable Markdown report containing data-quality findings, summaries, and model results.
- **Optional AI Explanations:** Can generate natural-language explanations through a configured compatible LLM API.
- **Automated Testing:** Includes Pytest tests for core application logic.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application logic |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and anomaly detection |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Streamlit | Interactive web interface |
| SQLite | Local data storage and SQL queries |
| Requests | Optional LLM API integration |
| python-dotenv | Environment variable management |
| Pytest | Automated testing |

## Project Structure

```text
AutoInsight-AI/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── data/
│   └── sample_data.csv
├── src/
│   ├── __init__.py
│   ├── profiling.py
│   ├── anomaly_detection.py
│   ├── modeling.py
│   └── reporting.py
├── tests/
│   └── test_app_logic.py
└── outputs/
    └── .gitkeep
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- Git
- Visual Studio Code or another Python editor
- Internet access for initial package installation

### 1. Clone the repository

```bash
git clone https://github.com/Yash-0330/AutoInsight-AI.git
cd AutoInsight-AI
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run automated tests

```bash
python -m pytest -q
```

### 5. Launch the application

```bash
python -m streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser.

## How to Use

1. Launch the application.
2. Load the included sample dataset or upload your own CSV file.
3. Review the dataset overview and column-quality information.
4. Explore numerical summaries, categorical distributions, and correlations.
5. Run anomaly detection on selected numeric features.
6. Select a target column and compare available machine learning models.
7. Use the SQL Explorer to query the supported local dataset.
8. Generate and download an analytical report.
9. Optionally configure an LLM API to obtain natural-language explanations.

## Sample Dataset

The repository includes `data/sample_data.csv`, a small synthetic retail dataset containing example fields such as:

- Order ID
- Region
- Sales channel
- Units
- Unit price
- Discount rate
- Order value
- Delivery time
- Returned-order indicator

The sample data provides a starting point for exploring the application's profiling, visualization, anomaly detection, and classification workflows.

**Note:** The included dataset is for demonstration and testing. It does not represent verified real-world business results.

## Optional LLM Configuration

Core functionality does not require an LLM API key.

To enable the optional AI explanation feature:

1. Copy `.env.example` to `.env`.
2. Set the compatible API's configuration values in `.env`.

Example:

```dotenv
LLM_API_KEY=your_api_key
LLM_API_URL=your_compatible_api_endpoint
LLM_MODEL=your_model_name
```

Use the endpoint and model name specified by your chosen provider. Never commit `.env` or expose API credentials in public repositories.

## Machine Learning Approach

The application supports classification and regression workflows.

**Classification examples**
- Predicting a returned-order indicator
- Evaluating categorical outcomes

**Regression examples**
- Predicting a continuous numerical value
- Estimating a target such as order value

Depending on the inferred task type, the application compares supported algorithms using a train-test split and appropriate evaluation metrics.

Anomaly detection uses Isolation Forest to identify observations that differ from common patterns in selected numeric features.

## Limitations

- Model quality depends on dataset size, data quality, feature selection, and target suitability.
- Automatically inferred task types may require user review, especially for numerical classification labels.
- Anomaly flags indicate unusual observations, not confirmed fraud or data errors.
- High-cardinality categorical features may be excluded from model training.
- Evaluation results from a single train-test split may differ from future real-world performance.
- Optional AI explanations depend on a correctly configured API and provider availability.
- SQL access is intended for local analytical exploration and is not a replacement for production database security controls.

## Future Improvements

- Add interactive feature selection and configurable preprocessing.
- Support time-series forecasting and more advanced model evaluation.
- Introduce cross-validation and hyperparameter tuning.
- Add explainability features such as SHAP.
- Expand support for additional file formats and larger datasets.
- Improve report generation with PDF and HTML exports.
- Add more comprehensive automated tests and continuous integration.
- Develop a more autonomous, tool-using analysis workflow with transparent recommendations.

## Skills Demonstrated

This project demonstrates practical work with:

- Python programming
- Data cleaning and preprocessing
- Exploratory data analysis
- Statistical summaries and visualization
- Anomaly detection
- Supervised machine learning
- Model evaluation
- SQL and SQLite
- Streamlit application development
- API integration and environment configuration
- Software testing and Git version control

## Author

**Yash Rakhunde**

- GitHub: [Yash-0330](https://github.com/Yash-0330)
- LinkedIn: [yashrakhunde](https://www.linkedin.com/in/yashrakhunde/)

## License

This project currently has no explicit license. Add a `LICENSE` file if you intend to grant others permission to reuse, modify, or distribute the code.

---

*Built as a practical data science project to explore an integrated workflow from raw data inspection to machine learning evaluation and report generation.*
