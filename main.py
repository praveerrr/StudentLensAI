"""StudentLens AI backend — FastAPI + SQLite + scikit-learn.

Run with:  uvicorn main:app --reload
Docs at:   http://localhost:8000/docs
"""
import csv
import io
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import asc, desc

from database import Base, engine, SessionLocal
from models import Student
import schemas
from ml import (
    FEATURES, generate_dataset, assign_risk_labels, ModelState,
)

model_state = ModelState()
RISK_THRESHOLDS = {"high_cut": 50.0, "low_cut": 75.0}
REQUIRED_CSV_COLUMNS = [
    "Student_ID", "Study_Hours", "Attendance", "Previous_Score", "Assignment_Score",
    "Internal_Marks", "Sleep_Hours", "Participation", "Final_Score",
]


def _students_df(db: Session) -> pd.DataFrame:
    rows = db.query(Student).all()
    return pd.DataFrame([r.to_dict() for r in rows])


def _refresh_risk_thresholds(df: pd.DataFrame):
    scores = df["final_score"]
    RISK_THRESHOLDS["high_cut"] = float(scores.quantile(0.15))
    RISK_THRESHOLDS["low_cut"] = float(scores.quantile(0.45))


def _risk_for_score(score: float) -> str:
    if score <= RISK_THRESHOLDS["high_cut"]:
        return "High"
    if score <= RISK_THRESHOLDS["low_cut"]:
        return "Medium"
    return "Low"


def _seed_and_train(db: Session, n: int = 1000):
    df = generate_dataset(n)
    df = assign_risk_labels(df)
    db.query(Student).delete()
    db.bulk_insert_mappings(Student, df.to_dict(orient="records"))
    db.commit()
    _refresh_risk_thresholds(df)
    model_state.train(df)


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        count = db.query(Student).count()
        if count == 0:
            _seed_and_train(db, 1000)
        else:
            df = _students_df(db)
            _refresh_risk_thresholds(df)
            model_state.train(df)
    finally:
        db.close()
    yield


app = FastAPI(title="StudentLens AI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------- health --
@app.get("/api/health")
def health():
    return {"status": "ok", "trained_at": model_state.trained_at, "best_model": model_state.best_model}


# ------------------------------------------------------------- students --
@app.get("/api/students", response_model=schemas.StudentListOut)
def list_students(
    search: str = "",
    department: str = "All",
    risk: str = "All",
    semester: Optional[int] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    sort_by: str = "final_score",
    sort_dir: str = "desc",
):
    db = SessionLocal()
    try:
        q = db.query(Student)
        if search:
            like = f"%{search}%"
            q = q.filter((Student.name.ilike(like)) | (Student.id.ilike(like)))
        if department != "All":
            q = q.filter(Student.department == department)
        if risk != "All":
            q = q.filter(Student.risk == risk)
        if semester is not None:
            q = q.filter(Student.semester == semester)

        sort_col = getattr(Student, sort_by, Student.final_score)
        q = q.order_by(asc(sort_col) if sort_dir == "asc" else desc(sort_col))

        total = q.count()
        items = q.offset((page - 1) * page_size).limit(page_size).all()
        return {
            "total": total, "page": page, "page_size": page_size,
            "items": [s.to_dict() for s in items],
        }
    finally:
        db.close()


@app.get("/api/students/{student_id}", response_model=schemas.StudentDetailOut)
def get_student(student_id: str):
    db = SessionLocal()
    try:
        s = db.query(Student).filter(Student.id == student_id).first()
        if not s:
            raise HTTPException(status_code=404, detail="Student not found")
        pred, _ = model_state.predict(s.as_feature_vector())
        return {**s.to_dict(), "predicted_score": round(pred, 1)}
    finally:
        db.close()


# ------------------------------------------------------------- dashboard --
@app.get("/api/dashboard/summary", response_model=schemas.DashboardSummaryOut)
def dashboard_summary():
    db = SessionLocal()
    try:
        df = _students_df(db)
        if df.empty:
            raise HTTPException(status_code=503, detail="No data loaded yet")
        return {
            "total_students": len(df),
            "average_score": round(float(df["final_score"].mean()), 1),
            "high_risk_count": int((df["risk"] == "High").sum()),
            "model_accuracy_r2": round(model_state.best_metrics()["r2"], 4),
            "best_model": "Random Forest" if model_state.best_model == "rf" else "Linear Regression",
        }
    finally:
        db.close()


@app.get("/api/dashboard/trends", response_model=list[schemas.TrendPointOut])
def dashboard_trends():
    db = SessionLocal()
    try:
        df = _students_df(db)
        if df.empty:
            return []
        grouped = df.groupby("semester")["final_score"].agg(["mean", "count"]).reindex(range(1, 9))
        out = []
        for sem, row in grouped.iterrows():
            out.append({
                "semester": int(sem),
                "average_score": round(float(row["mean"]), 1) if pd.notna(row["mean"]) else 0.0,
                "student_count": int(row["count"]) if pd.notna(row["count"]) else 0,
            })
        return out
    finally:
        db.close()


# ------------------------------------------------------------- analytics --
@app.get("/api/analytics/correlation", response_model=schemas.CorrelationOut)
def analytics_correlation(department: str = "All", semester: Optional[int] = None):
    db = SessionLocal()
    try:
        df = _students_df(db)
        if department != "All":
            df = df[df["department"] == department]
        if semester is not None:
            df = df[df["semester"] == semester]
        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Not enough rows for correlation with these filters")
        cols = FEATURES + ["final_score"]
        corr = df[cols].corr().round(4)
        return {"columns": cols, "matrix": corr.values.tolist()}
    finally:
        db.close()


@app.get("/api/analytics/feature-importance", response_model=list[schemas.FeatureImportanceOut])
def analytics_feature_importance():
    if not model_state.importance:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    items = sorted(model_state.importance.items(), key=lambda kv: kv[1], reverse=True)
    return [{"feature": f, "importance": round(v, 4)} for f, v in items]


# ------------------------------------------------------------- predict --
@app.post("/api/predict", response_model=schemas.PredictOut)
def predict(payload: schemas.PredictIn):
    if model_state.lr_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    feature_vector = [getattr(payload, f) for f in FEATURES]
    pred, used_model = model_state.predict(feature_vector)
    risk = _risk_for_score(pred)
    rmse = model_state.best_metrics()["rmse"]
    confidence = max(0.55, min(0.97, 1 - (rmse / 45)))

    contributions = {}
    if used_model == "rf" and model_state.importance:
        for f, imp in model_state.importance.items():
            contributions[f] = round(imp, 4)
    else:
        coefs = model_state.lr_model.coef_
        total = sum(abs(c) for c in coefs) or 1.0
        for f, c in zip(FEATURES, coefs):
            contributions[f] = round(abs(c) / total, 4)

    return {
        "predicted_score": round(pred, 1),
        "risk_level": risk,
        "confidence": round(confidence, 3),
        "model_used": "Random Forest" if used_model == "rf" else "Linear Regression",
        "feature_contributions": contributions,
    }


# --------------------------------------------------------- model perf --
def _performance_payload() -> dict:
    return {
        "best_model": "Random Forest" if model_state.best_model == "rf" else "Linear Regression",
        "linear_regression": model_state.metrics["lr"],
        "random_forest": model_state.metrics["rf"],
        "training_records": model_state.train_count,
        "test_records": model_state.test_count,
        "trained_at": model_state.trained_at,
        "eval_points": model_state.eval_points,
    }


@app.get("/api/model/performance", response_model=schemas.ModelPerformanceOut)
def model_performance():
    if model_state.lr_model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet")
    return _performance_payload()


@app.post("/api/model/retrain", response_model=schemas.RetrainOut)
def model_retrain():
    db = SessionLocal()
    try:
        df = _students_df(db)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data to train on")
        model_state.train(df)
        return {"message": f"Retrained on {len(df)} records", "metrics": _performance_payload()}
    finally:
        db.close()


# ------------------------------------------------------------- dataset --
@app.post("/api/dataset/regenerate", response_model=schemas.RegenerateOut)
def dataset_regenerate(n: int = Query(1000, ge=50, le=5000)):
    db = SessionLocal()
    try:
        _seed_and_train(db, n)
        return {"message": f"Generated {n} new records and retrained", "records_generated": n, "metrics": _performance_payload()}
    finally:
        db.close()


@app.post("/api/dataset/upload", response_model=schemas.DatasetUploadOut)
async def dataset_upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    raw = (await file.read()).decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    fieldnames = reader.fieldnames or []
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in fieldnames]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {', '.join(missing)}")

    rows = []
    for i, row in enumerate(reader):
        try:
            final_score = max(0.0, min(100.0, float(row["Final_Score"])))
            rows.append({
                "id": row["Student_ID"] or f"STU{i + 1:04d}",
                "name": row["Student_ID"] or f"Student {i + 1}",
                "department": "Uploaded",
                "semester": (i % 8) + 1,
                "study_hours": max(0.0, min(12.0, float(row["Study_Hours"]))),
                "attendance": max(0.0, min(100.0, float(row["Attendance"]))),
                "previous_score": max(0.0, min(100.0, float(row["Previous_Score"]))),
                "assignment_score": max(0.0, min(100.0, float(row["Assignment_Score"]))),
                "internal_marks": max(0.0, min(100.0, float(row["Internal_Marks"]))),
                "sleep_hours": max(0.0, min(12.0, float(row["Sleep_Hours"]))),
                "participation": max(0.0, min(10.0, float(row["Participation"]))),
                "final_score": final_score,
            })
        except (ValueError, KeyError):
            continue

    if len(rows) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 valid rows after parsing")

    df = pd.DataFrame(rows)
    df = assign_risk_labels(df)

    db = SessionLocal()
    try:
        db.query(Student).delete()
        db.bulk_insert_mappings(Student, df.to_dict(orient="records"))
        db.commit()
        _refresh_risk_thresholds(df)
        model_state.train(df)
        return {"message": "Dataset uploaded and model retrained", "records_loaded": len(df), "metrics": _performance_payload()}
    finally:
        db.close()


@app.get("/api/predictions/export")
def export_predictions():
    db = SessionLocal()
    try:
        students = db.query(Student).all()
        if not students:
            raise HTTPException(status_code=404, detail="No students to export")

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Student_ID", "Name", "Department", "Semester", "Actual_Score", "Predicted_Score", "Risk"])
        for s in students:
            pred, _ = model_state.predict(s.as_feature_vector())
            writer.writerow([s.id, s.name, s.department, s.semester, s.final_score, round(pred, 1), s.risk])
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=studentlens_predictions.csv"},
        )
    finally:
        db.close()
