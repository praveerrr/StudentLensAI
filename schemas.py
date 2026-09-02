"""Pydantic schemas for request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict


class StudentOut(BaseModel):
    id: str
    name: str
    department: str
    semester: int
    study_hours: float
    attendance: float
    previous_score: float
    assignment_score: float
    internal_marks: float
    sleep_hours: float
    participation: float
    final_score: float
    risk: str


class StudentDetailOut(StudentOut):
    predicted_score: float


class StudentListOut(BaseModel):
    total: int
    page: int
    page_size: int
    items: List[StudentOut]


class DashboardSummaryOut(BaseModel):
    total_students: int
    average_score: float
    high_risk_count: int
    model_accuracy_r2: float
    best_model: str


class TrendPointOut(BaseModel):
    semester: int
    average_score: float
    student_count: int


class CorrelationOut(BaseModel):
    columns: List[str]
    matrix: List[List[float]]


class FeatureImportanceOut(BaseModel):
    feature: str
    importance: float


class PredictIn(BaseModel):
    study_hours: float = Field(ge=0, le=12)
    attendance: float = Field(ge=0, le=100)
    previous_score: float = Field(ge=0, le=100)
    assignment_score: float = Field(ge=0, le=100)
    internal_marks: float = Field(ge=0, le=100)
    sleep_hours: float = Field(ge=0, le=12)
    participation: float = Field(ge=0, le=10)


class PredictOut(BaseModel):
    predicted_score: float
    risk_level: str
    confidence: float
    model_used: str
    feature_contributions: Dict[str, float]


class ModelMetricsOut(BaseModel):
    mae: float
    rmse: float
    r2: float


class ModelPerformanceOut(BaseModel):
    best_model: str
    linear_regression: ModelMetricsOut
    random_forest: ModelMetricsOut
    training_records: int
    test_records: int
    trained_at: Optional[str]
    eval_points: List[Dict[str, float]]


class RetrainOut(BaseModel):
    message: str
    metrics: ModelPerformanceOut


class DatasetUploadOut(BaseModel):
    message: str
    records_loaded: int
    metrics: ModelPerformanceOut


class RegenerateOut(BaseModel):
    message: str
    records_generated: int
    metrics: ModelPerformanceOut
