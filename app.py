
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "model.joblib")

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational ML model for classifying diagnostic measurements.",
    version="1.0.0",
)

FEATURES = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


class CancerInput(BaseModel):
    radius_mean: float = Field(gt=0)
    texture_mean: float = Field(gt=0)
    perimeter_mean: float = Field(gt=0)
    area_mean: float = Field(gt=0)
    smoothness_mean: float = Field(gt=0)
    compactness_mean: float = Field(gt=0)
    concavity_mean: float = Field(ge=0)
    concave_points_mean: float = Field(ge=0)
    symmetry_mean: float = Field(gt=0)
    fractal_dimension_mean: float = Field(gt=0)

    radius_se: float = Field(gt=0)
    texture_se: float = Field(gt=0)
    perimeter_se: float = Field(gt=0)
    area_se: float = Field(gt=0)
    smoothness_se: float = Field(gt=0)
    compactness_se: float = Field(ge=0)
    concavity_se: float = Field(ge=0)
    concave_points_se: float = Field(ge=0)
    symmetry_se: float = Field(gt=0)
    fractal_dimension_se: float = Field(gt=0)

    radius_worst: float = Field(gt=0)
    texture_worst: float = Field(gt=0)
    perimeter_worst: float = Field(gt=0)
    area_worst: float = Field(gt=0)
    smoothness_worst: float = Field(gt=0)
    compactness_worst: float = Field(ge=0)
    concavity_worst: float = Field(ge=0)
    concave_points_worst: float = Field(ge=0)
    symmetry_worst: float = Field(gt=0)
    fractal_dimension_worst: float = Field(gt=0)


@app.get("/")
def home():
    return {"message": "Breast Cancer Classification API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: CancerInput):
    values = data.model_dump()

    # Match the feature names and order used during training
    values["concave points_mean"] = values.pop("concave_points_mean")
    values["concave points_se"] = values.pop("concave_points_se")
    values["concave points_worst"] = values.pop("concave_points_worst")

    input_df = pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES,
    )

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    return {
        "prediction": "malignant" if prediction == 1 else "benign",
        "class_id": prediction,
        "class_probabilities": {
            "benign": round(float(probabilities[0]), 4),
            "malignant": round(float(probabilities[1]), 4),
        },
        "note": "Educational prediction only; not for medical diagnosis.",
    }
