from pydantic import BaseModel


class EmotionPrediction(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


class AnalysisResponse(BaseModel):
    transcript: str
    prediction: EmotionPrediction
    spoken_response: str


class HealthResponse(BaseModel):
    status: str


class ErrorResponse(BaseModel):
    detail: str
