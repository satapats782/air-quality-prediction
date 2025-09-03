#!/usr/bin/env python3
"""
Air Quality Forecasting
-----------------------
Train/evaluate regressors to forecast a pollutant (e.g., PM2.5) from weather/traffic features.
- Compares multiple imputation strategies
- Repeated CV and prints a leaderboard
- (Optional) Predicts on a blind set
- (Optional) Generates a ydata-profiling HTML report

Examples:
  python3 forecast_eval.py --data AirQualityUCI_clean.csv --target "CO(GT)" --profile
  python3 forecast_eval.py --data air_quality.csv --target "PM2.5" --blind air_quality_blind.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
np.random.seed(42)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_numeric_imputers():
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn3": KNNImputer(n_neighbors=3, weights="uniform"),
        "iterative": IterativeImputer(random_state=42, max_iter=15, sample_posterior=False),
    }


def build_models():
    return {
        "ridge": Ridge(alpha=1.0, random_state=42),
        "rf": RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=42),
        "gbr": GradientBoostingRegressor(random_state=42),
    }


def detect_target(df, arg_target):
    if arg_target:
        if arg_target not in df.columns:
            raise ValueError(f"--target '{arg_target}' not found in columns.")
        return arg_target
    # Heuristic fallback
    pollutants = {"PM2.5", "PM10", "NO2", "O3", "SO2", "CO", "pm25", "pm10", "no2", "o3", "so2", "co"}
    for col in df.columns:
        if col in pollutants or col.lower() in pollutants:
            return col
    return df.columns[-1]


def build_preprocess(X_df, numeric_imputer):
    num_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_df.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", numeric_imputer),
        ("scale", StandardScaler(with_mean=True, with_std=True)),
    ])

    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def evaluate(X, y, preprocess, model, splits=5, repeats=2):
    rkf = RepeatedKFold(n_splits=splits, n_repeats=repeats, random_state=42)
    reg = Pipeline(steps=[("pre", preprocess), ("model", model)])
    scoring = {
        "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
        "RMSE": make_scorer(lambda yt, yp: np.sqrt(mean_squared_error(yt, yp)), greater_is_better=False),
        "R2": make_scorer(r2_score),
    }
    cv = cross_validate(reg, X, y, cv=rkf, scoring=scoring, n_jobs=-1, error_score="raise")
    mae = -np.mean(cv["test_MAE"])
    rmse_val = -np.mean(cv["test_RMSE"])
    r2 = np.mean(cv["test_R2"])
    return mae, rmse_val, r2


def maybe_profile(df, out_html="air_quality_profile_report.html"):
    try:
        from ydata_profiling import ProfileReport
        report = ProfileReport(df, title="Air Quality Profiling Report", minimal=True)
        report.to_file(out_html)
        print(f"[INFO] Profiling report saved to: {out_html}")
    except Exception as e:
        print(f"[WARN] Profiling skipped/not installed: {e}")


def train_on_full_and_predict(best_conf, X, y, blind_df, out_csv):
    numeric_imputer = build_numeric_imputers()[best_conf["imputer"]]
    model = build_models()[best_conf["model"]]
    preprocess = build_preprocess(X, numeric_imputer)
    reg = Pipeline(steps=[("pre", preprocess), ("model", model)])
    reg.fit(X, y)
    preds = reg.predict(blind_df)
    pd.DataFrame({"prediction": preds}).to_csv(out_csv, index=False)
    print(f"[INFO] Blind predictions saved to: {out_csv}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="CSV with features + target")
    p.add_argument("--target", type=str, default=None, help="Target column (pollutant). If omitted, auto-detect")
    p.add_argument("--blind", type=str, default=None, help="Optional blind CSV without target for inference")
    p.add_argument("--splits", type=int, default=5, help="CV splits")
    p.add_argument("--repeats", type=int, default=2, help="CV repeats (total folds = splits*repeats)")
    p.add_argument("--profile", action="store_true", help="Generate ydata-profiling report on the training CSV")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    if args.profile:
        maybe_profile(df)

    target = detect_target(df, args.target)
    print(f"[INFO] Using target: {target}")

    y = df[target].astype(float).values
    X = df.drop(columns=[target])

    imputers = build_numeric_imputers()
    models = build_models()

    rows = []
    for imp_name, imp in imputers.items():
        preprocess = build_preprocess(X, imp)
        for model_name, model in models.items():
            mae, rmse_val, r2 = evaluate(X, y, preprocess, model, splits=args.splits, repeats=args.repeats)
            rows.append({"imputer": imp_name, "model": model_name, "MAE": mae, "RMSE": rmse_val, "R2": r2})
            print(f"[CV] {imp_name:9s} + {model_name:4s} -> MAE={mae:.3f} RMSE={rmse_val:.3f} R2={r2:.3f}")

    leaderboard = pd.DataFrame(rows).sort_values(by=["MAE", "RMSE"], ascending=[True, True]).reset_index(drop=True)
    print("\n=== Leaderboard (lower MAE/RMSE is better) ===")
    print(leaderboard.to_string(index=False))
    leaderboard.to_csv("results_forecasting.csv", index=False)
    print("[INFO] Saved leaderboard to results_forecasting.csv")

    if args.blind:
        blind_df = pd.read_csv(args.blind)
        blind_df = blind_df[X.columns]  # align columns
        best = leaderboard.iloc[0].to_dict()
        train_on_full_and_predict(best, X, y, blind_df, Path("blind_predictions_forecasting.csv"))


if __name__ == "__main__":
    main()