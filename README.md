# Air Quality Forecasting using Machine Learning

This project applies **machine learning techniques** to forecast air pollutant concentrations using the [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality).  
The dataset contains hourly averaged responses of gas sensors and true pollutant concentrations, along with meteorological features such as temperature and humidity.

The goal is to:
- Predict pollutant levels (e.g., **CO(GT)**, **NO₂(GT)**, **O₃(GT)**) from sensor and weather data
- Compare different **missing value imputation** strategies
- Benchmark multiple **regression models** using cross-validation
- Generate a **leaderboard of results** and profiling reports

---

## 📂 Project Structure
air-quality-forecasting/
│── forecast_eval1.py # Main training & evaluation script
│── AirQualityUCI_train.csv # Cleaned dataset (after dropping NaN targets)
│── results_forecasting.csv # Leaderboard of imputer + model combinations
│── README.md # Project description
│── .gitignore


---

## Features
- Handles **missing data** using several imputation methods:
  - Mean, Median, KNN, Iterative (MICE)
- Benchmarks multiple regressors:
  - Ridge Regression, Random Forest, Gradient Boosting
- Uses **Repeated K-Fold CV** for robust evaluation
- Outputs:
  - `results_forecasting.csv` → leaderboard with MAE, RMSE, R²
  - `air_quality_profile_report.html` → optional profiling report
- Supports **blind predictions** on datasets without target labels

---

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- ydata-profiling (optional, for EDA report)
- matplotlib (optional, for visualization)

Install dependencies with:

```bash
pip install -r requirements.txt

Usage
1. Clean the dataset
import pandas as pd
import numpy as np

df = pd.read_csv("AirQualityUCI_clean.csv")
target = "CO(GT)"   # change if predicting NO2(GT) or O3(GT)
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df[df[target].notna()]
df.to_csv("AirQualityUCI_train.csv", index=False)

2. Train & evaluate models

Run the main script:

python3 forecast_eval1.py --data AirQualityUCI_train.csv --target "CO(GT)" --profile


--data → dataset path

--target → pollutant column to predict ("CO(GT)", "NO2(GT)", "O3(GT)", etc.)

--profile → (optional) generate profiling HTML report

| imputer   | model | MAE  | RMSE | R²   |
| --------- | ----- | ---- | ---- | ---- |
| mean      | ridge | 1.23 | 2.10 | 0.78 |
| knn3      | rf    | 0.98 | 1.95 | 0.82 |
| iterative | gbr   | 1.10 | 2.05 | 0.80 |



