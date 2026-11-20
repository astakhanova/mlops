"""
train_linear.py — обучение линейной регрессии на фичах из all_features.csv
с автоматическим трекингом экспериментов в MLflow.
"""

import os
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ----------------------------- Конфигурация -----------------------------
DATA_PATH = "features/all_features.csv"
TARGET_COL = "price"

# Признаки, которые реально есть в CSV (id / source_file — не признаки)
FEATURE_COLS = ["rooms", "area", "floor", "total_floors"]

# Гиперпараметры (для Ridge; для чистого LinearRegression игнорируются)
PARAMS = {
    "model": "ridge",          # "linear" | "ridge"
    "alpha": 1.0,              # только для ridge
    "test_size": 0.2,
    "random_state": 42,
    "log_target": True,        # обучать на log(price) — цена скошена вправо
}

EXPERIMENT_NAME = "domclick_price_regression"


# ----------------------------- Загрузка данных -----------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл с фичами: {path}. Сначала запусти парсер.")

    df = pd.read_csv(path)
    print(f"Загружено строк: {len(df)}")
    print(f"Колонки: {list(df.columns)}")

    # Оставляем только нужные колонки и чистим пропуски/выбросы
    df = df[FEATURE_COLS + [TARGET_COL]].copy()
    before = len(df)
    df = df.dropna()
    df = df[(df[TARGET_COL] > 0) & (df["area"] > 0) & (df["rooms"] >= 0)]
    # Отсекаем явные выбросы по цене (1%–99% квантили)
    q_low, q_high = df[TARGET_COL].quantile([0.01, 0.99])
    df = df[(df[TARGET_COL] >= q_low) & (df[TARGET_COL] <= q_high)]
    print(f"После чистки: {len(df)} (удалено {before - len(df)})")

    if len(df) < 50:
        raise ValueError("Слишком мало данных для обучения (<50 строк).")

    return df


# ----------------------------- Модель -----------------------------
def build_model(params: dict) -> Pipeline:
    if params["model"] == "ridge":
        reg = Ridge(alpha=params["alpha"], random_state=params["random_state"])
    else:
        reg = LinearRegression()

    return Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", reg),
    ])


# ----------------------------- Метрики -----------------------------
def evaluate(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# ----------------------------- Обучение -----------------------------
def train():
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_data(DATA_PATH)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    y_train_raw = y
    if PARAMS["log_target"]:
        y = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=PARAMS["test_size"],
        random_state=PARAMS["random_state"],
    )

    model = build_model(PARAMS)

    with mlflow.start_run(run_name=f"{PARAMS['model']}_alpha={PARAMS['alpha']}"):
        # Логируем параметры
        mlflow.log_params({
            "model": PARAMS["model"],
            "alpha": PARAMS["alpha"],
            "test_size": PARAMS["test_size"],
            "random_state": PARAMS["random_state"],
            "log_target": PARAMS["log_target"],
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": ",".join(FEATURE_COLS),
        })

        # Обучение
        model.fit(X_train, y_train)

        # Предсказания (возвращаем в исходный масштаб цены)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        if PARAMS["log_target"]:
            y_pred_train = np.expm1(y_pred_train)
            y_pred_test = np.expm1(y_pred_test)
            y_train_eval = np.expm1(y_train)
            y_test_eval = np.expm1(y_test)
        else:
            y_train_eval, y_test_eval = y_train, y_test

        train_metrics = evaluate(y_train_eval, y_pred_train)
        test_metrics = evaluate(y_test_eval, y_pred_test)

        mlflow.log_metrics({f"train_{k}": v for k, v in train_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})

        print("\nМетрики на train:", train_metrics)
        print("Метрики на test: ", test_metrics)

        # Важность признаков (для линейной модели — коэффициенты)
        reg = model.named_steps["regressor"]
        if hasattr(reg, "coef_"):
            coefs = dict(zip(FEATURE_COLS, map(float, reg.coef_)))
            mlflow.log_dict(coefs, "coefficients.json")
            print("Коэффициенты (на стандартизованных признаках):", coefs)

        # Логируем модель (со схемой входных данных)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_test.head(3),
        )

        # Сохраняем предсказания как артефакт
        with tempfile.TemporaryDirectory() as tmp:
            pred_path = os.path.join(tmp, "test_predictions.csv")
            pd.DataFrame({
                **{c: X_test[c].values for c in FEATURE_COLS},
                "y_true": y_test_eval.values,
                "y_pred": y_pred_test,
                "abs_error": np.abs(y_test_eval.values - y_pred_test),
            }).to_csv(pred_path, index=False)
            mlflow.log_artifact(pred_path, artifact_path="predictions")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run_id: {run_id}")
        print("Запустить UI: mlflow ui --host 127.0.0.1 --port 5000")


if __name__ == "__main__":
    train()
    