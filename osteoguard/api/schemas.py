"""
OsteoGuard AI - API Data Contracts and Pydantic Schemas.
Mirrors the OpenAPI 3.1 specification and frontend TypeScript interfaces.
"""

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

from osteoguard.constants import (
    BonePathologyClass,
    ModalityType,
)


class ClassProbabilities(BaseModel):
    metastasis: float = Field(..., ge=0.0, le=1.0)
    paget: float = Field(..., ge=0.0, le=1.0)
    oi: float = Field(..., ge=0.0, le=1.0)
    healthy: float = Field(..., ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    topClass: Literal["metastasis", "paget", "oi", "healthy"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ClassProbabilities
    uncertainty: Literal["low", "medium", "high"]


class CalibrationBin(BaseModel):
    predicted: float
    observed: float


class UncertaintyDetail(BaseModel):
    entropy: float = Field(..., ge=0.0, le=1.0, description="Normalized Shannon entropy")
    confidenceInterval: Tuple[float, float] = Field(..., description="95% CI [lower, upper]")
    oodFlag: bool = Field(..., description="Out-of-Distribution trigger")
    calibrationBin: CalibrationBin


class ShapFeature(BaseModel):
    name: str
    value: float = Field(..., ge=-1.0, le=1.0)


class Counterfactual(BaseModel):
    feature: str
    deltaDirection: Literal["increase", "decrease"]
    percentChange: float
    flipTo: Literal["metastasis", "paget", "oi", "healthy"]
    probabilityShift: float


class FractureRiskResult(BaseModel):
    level: Literal["low", "medium", "high"]
    percent: float = Field(..., ge=0.0, le=100.0)


class AnalyzeResponse(BaseModel):
    requestId: str
    timestamp: str
    modality: Literal["xray", "ct", "bone_scan"]
    imageUrl: str
    gradcamUrl: str
    prediction: PredictionResult
    uncertaintyDetail: UncertaintyDetail
    shap: List[ShapFeature]
    counterfactuals: List[Counterfactual]
    fractureRisk: FractureRiskResult
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    mode: Literal["simulation", "production"]

