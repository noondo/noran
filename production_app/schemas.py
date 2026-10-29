from pydantic import BaseModel


class PredictionRequest(BaseModel):
    mean_perimeter: float
    mean_concave_points: float
    worst_radius: float
    worst_perimeter: float
    worst_concave_points: float


class PredictionResponse(BaseModel):
    predicted_class: int
    label: str
    confidence_score: float | None = None
