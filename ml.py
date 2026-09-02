"""Synthetic dataset generation + ML training pipeline for StudentLens AI."""
import random
import string
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FEATURES = [
    "study_hours", "attendance", "previous_score", "assignment_score",
    "internal_marks", "sleep_hours", "participation",
]

DEPARTMENTS = ["Computer Science", "Electronics", "Mechanical", "Business Administration", "Data Science"]

FIRST_NAMES = [
    "Aarav", "Diya", "Rohan", "Ishita", "Kabir", "Meera", "Aditya", "Sneha", "Vikram", "Ananya",
    "Arjun", "Priya", "Sameer", "Kavya", "Nikhil", "Tanya", "Rahul", "Pooja", "Karan", "Neha",
    "Devika", "Yash", "Riya", "Sanjay", "Alia", "Farhan", "Lakshmi", "Manav", "Zara", "Om",
]
LAST_NAMES = [
    "Sharma", "Patel", "Verma", "Iyer", "Nair", "Reddy", "Gupta", "Menon", "Rao", "Kapoor",
    "Singh", "Joshi", "Chatterjee", "Bose", "Malhotra", "Desai", "Pillai", "Chauhan", "Bhatt", "Agarwal",
]


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def generate_dataset(n: int = 1000, seed: int | None = None) -> pd.DataFrame:
    """Generate n synthetic student records with realistic feature correlations."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        study_hours = _clamp(rng.normal(5.5, 1.8), 1, 10)
        attendance = _clamp(rng.normal(80, 12), 40, 100)
        previous_score = _clamp(rng.normal(70, 13), 30, 100)
        assignment_score = _clamp(rng.normal(72, 12), 30, 100)
        internal_marks = _clamp(rng.normal(70, 11), 30, 100)
        sleep_hours = _clamp(rng.normal(6.5, 1.2), 3, 10)
        participation = _clamp(rng.normal(6, 2), 1, 10)

        final_score = (
            0.25 * attendance + 0.20 * previous_score + 0.20 * assignment_score +
            0.15 * internal_marks + 0.10 * (study_hours / 10 * 100) +
            0.05 * (sleep_hours / 9 * 100) + 0.05 * (participation / 10 * 100) +
            rng.normal(0, 6)
        )
        final_score = _clamp(final_score, 0, 100)

        rows.append({
            "id": f"STU{i + 1:04d}",
            "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "department": random.choice(DEPARTMENTS),
            "semester": random.randint(1, 8),
            "study_hours": round(study_hours, 1),
            "attendance": round(attendance, 1),
            "previous_score": round(previous_score, 1),
            "assignment_score": round(assignment_score, 1),
            "internal_marks": round(internal_marks, 1),
            "sleep_hours": round(sleep_hours, 1),
            "participation": round(participation, 1),
            "final_score": round(final_score, 1),
        })
    return pd.DataFrame(rows)


def assign_risk_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile-based risk bands: bottom 15% High, next 30% Medium, top 55% Low.
    This keeps risk buckets meaningfully populated regardless of score distribution."""
    scores = df["final_score"]
    high_cut = scores.quantile(0.15)
    low_cut = scores.quantile(0.45)

    def label(s):
        if s <= high_cut:
            return "High"
        if s <= low_cut:
            return "Medium"
        return "Low"

    df = df.copy()
    df["risk"] = scores.apply(label)
    return df


class ModelState:
    """Holds the currently trained models, metrics, and evaluation data."""

    def __init__(self):
        self.lr_model: LinearRegression | None = None
        self.rf_model: RandomForestRegressor | None = None
        self.metrics = {"lr": None, "rf": None}
        self.best_model = "lr"
        self.importance: dict[str, float] = {}
        self.eval_points: list[dict] = []
        self.train_count = 0
        self.test_count = 0
        self.trained_at: str | None = None

    def train(self, df: pd.DataFrame):
        X = df[FEATURES].values
        y = df["final_score"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random.randint(0, 10_000)
        )

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        lr_pred = np.clip(lr.predict(X_test), 0, 100)

        rf = RandomForestRegressor(n_estimators=150, max_depth=6, min_samples_leaf=8, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = np.clip(rf.predict(X_test), 0, 100)

        def metrics_for(y_true, y_pred):
            return {
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "r2": float(r2_score(y_true, y_pred)),
            }

        lr_metrics = metrics_for(y_test, lr_pred)
        rf_metrics = metrics_for(y_test, rf_pred)

        self.lr_model = lr
        self.rf_model = rf
        self.metrics = {"lr": lr_metrics, "rf": rf_metrics}
        self.best_model = "rf" if rf_metrics["r2"] >= lr_metrics["r2"] else "lr"
        self.importance = dict(zip(FEATURES, rf.feature_importances_.tolist()))
        self.train_count = len(X_train)
        self.test_count = len(X_test)

        best_pred = rf_pred if self.best_model == "rf" else lr_pred
        self.eval_points = [
            {"actual": float(a), "predicted": float(p), "residual": float(a - p)}
            for a, p in zip(y_test, best_pred)
        ]
        self.trained_at = datetime.now(timezone.utc).isoformat()

    def predict(self, feature_vector: list[float]) -> tuple[float, str]:
        model = self.rf_model if self.best_model == "rf" else self.lr_model
        pred = float(np.clip(model.predict([feature_vector])[0], 0, 100))
        return pred, self.best_model

    def best_metrics(self) -> dict:
        return self.metrics[self.best_model]
