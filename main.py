import os
import io
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any

from model_utils import (
    SIZE_MODEL_PATH,
    SIZE_ENCODER_PATH,
    QUALITY_MODEL_PATH,
    QUALITY_ENCODER_PATH,
    PRICE_MODEL_PATH,
    DISEASE_MODEL_PATH,
    DISEASE_ENCODER_PATH,
    extract_image_features,
    get_remedy
)
from ap_locations import (
    get_all_ap_mandis,
    get_all_districts,
    get_mandis_by_district,
    get_mandi_details
)

app = FastAPI(
    title="Cauliflower AI Prediction API",
    description="Comprehensive API for cauliflower size grading, quality assessment, Andhra Pradesh mandi price estimation, and computer-vision disease diagnosis.",
    version="2.1"
)

MODEL_PATH = "models/cauliflower_size_model.pkl"
ENCODER_PATH = "models/size_encoder.pkl"

# Global model references
model = None
encoder = None
quality_model = None
quality_encoder = None
price_model = None
disease_model = None
disease_encoder = None

def load_models():
    global model, encoder, quality_model, quality_encoder, price_model, disease_model, disease_encoder

    if os.path.exists(SIZE_MODEL_PATH) and os.path.exists(SIZE_ENCODER_PATH):
        model = joblib.load(SIZE_MODEL_PATH)
        encoder = joblib.load(SIZE_ENCODER_PATH)

    if os.path.exists(QUALITY_MODEL_PATH) and os.path.exists(QUALITY_ENCODER_PATH):
        quality_model = joblib.load(QUALITY_MODEL_PATH)
        quality_encoder = joblib.load(QUALITY_ENCODER_PATH)

    if os.path.exists(PRICE_MODEL_PATH):
        price_model = joblib.load(PRICE_MODEL_PATH)

    if os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_ENCODER_PATH):
        disease_model = joblib.load(DISEASE_MODEL_PATH)
        disease_encoder = joblib.load(DISEASE_ENCODER_PATH)

load_models()


# Pydantic Schemas
class CauliflowerData(BaseModel):
    weight_g: float = Field(..., description="Weight of the cauliflower curd in grams (e.g., 850.0)")
    diameter_cm: float = Field(..., description="Diameter of the curd in centimeters (e.g., 16.5)")
    height_cm: float = Field(..., description="Height of the curd in centimeters (e.g., 13.0)")


class CauliflowerQualityData(BaseModel):
    color_score_1_10: float = Field(..., ge=1.0, le=10.0, description="Color score (1-10, higher is whiter/cleaner)")
    firmness_score_1_10: float = Field(..., ge=1.0, le=10.0, description="Firmness score (1-10, higher is more compact)")
    spots_count: int = Field(..., ge=0, description="Number of visible spots or blemishes")
    leaf_condition_score_1_10: float = Field(..., ge=1.0, le=10.0, description="Jacket leaf freshness (1-10)")


class CauliflowerPriceData(BaseModel):
    quantity_sold_kg: float = Field(..., gt=0, description="Batch quantity sold in kg")
    mandi_modal_rs_per_kg: float = Field(..., gt=0, description="Mandi modal/average price in Rs/kg")
    mandi_min_rs_per_kg: float = Field(..., gt=0, description="Mandi minimum price in Rs/kg")
    mandi_max_rs_per_kg: float = Field(..., gt=0, description="Mandi maximum price in Rs/kg")


# Routes
@app.get("/")
def home():
    return {
        "message": "Cauliflower Produce Intelligence & Diagnosis API is running",
        "docs_url": "/docs",
        "available_endpoints": [
            "POST /predict (Size classification)",
            "POST /predict/size (Detailed size grading)",
            "POST /predict/quality (Quality assessment)",
            "POST /predict/price (Mandi price estimation)",
            "POST /predict/image (Computer-vision disease diagnosis)",
            "GET /locations/andhra-pradesh (All AP Mandis & Rythu Bazars)",
            "GET /locations/andhra-pradesh/districts (All 26 AP Districts)",
            "GET /locations/andhra-pradesh/{district} (Mandis in specific AP District)"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "models_status": {
            "size_model": model is not None,
            "quality_model": quality_model is not None,
            "price_model": price_model is not None,
            "disease_model": disease_model is not None
        }
    }


@app.get("/locations/andhra-pradesh")
def list_andhra_pradesh_mandis():
    """Returns all Andhra Pradesh Mandis and Rythu Bazars with price benchmarks and district info."""
    mandis = get_all_ap_mandis()
    return {
        "state": "Andhra Pradesh",
        "total_markets": len(mandis),
        "mandis": mandis
    }


@app.get("/locations/andhra-pradesh/districts")
def list_andhra_pradesh_districts():
    """Returns the list of all covered Andhra Pradesh districts."""
    districts = get_all_districts()
    return {
        "state": "Andhra Pradesh",
        "total_districts": len(districts),
        "districts": districts
    }


@app.get("/locations/andhra-pradesh/{district}")
def get_district_mandis(district: str):
    """Returns mandis located in a given Andhra Pradesh district."""
    results = get_mandis_by_district(district)
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No mandis found for district '{district}'. Use /locations/andhra-pradesh/districts to view valid districts."
        )
    return {
        "district": district,
        "count": len(results),
        "mandis": results
    }


# Original size prediction endpoint (maintained for 100% backward compatibility)
@app.post("/predict")
def predict(data: CauliflowerData):
    if model is None or encoder is None:
        return {
            "error": "Model is not trained yet. Run train_model.py first."
        }

    input_df = pd.DataFrame(
        [[data.weight_g, data.diameter_cm, data.height_cm]],
        columns=["weight_g", "diameter_cm", "height_cm"]
    )
    prediction = model.predict(input_df)
    predicted_size = encoder.inverse_transform(prediction)[0]

    return {
        "weight_g": data.weight_g,
        "diameter_cm": data.diameter_cm,
        "height_cm": data.height_cm,
        "predicted_size": predicted_size
    }


# Detailed size endpoint with probabilities
@app.post("/predict/size")
def predict_size_detailed(data: CauliflowerData):
    if model is None or encoder is None:
        raise HTTPException(status_code=503, detail="Size model is not loaded. Run train_model.py first.")

    input_df = pd.DataFrame(
        [[data.weight_g, data.diameter_cm, data.height_cm]],
        columns=["weight_g", "diameter_cm", "height_cm"]
    )
    pred_idx = model.predict(input_df)[0]
    predicted_size = encoder.inverse_transform([pred_idx])[0]
    probabilities = model.predict_proba(input_df)[0]

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(encoder.classes_, probabilities)
    }

    return {
        "weight_g": data.weight_g,
        "diameter_cm": data.diameter_cm,
        "height_cm": data.height_cm,
        "predicted_size": predicted_size,
        "confidence": round(float(max(probabilities)), 4),
        "class_probabilities": prob_dict
    }


# Quality prediction endpoint
@app.post("/predict/quality")
def predict_quality(data: CauliflowerQualityData):
    if quality_model is None or quality_encoder is None:
        raise HTTPException(status_code=503, detail="Quality model is not loaded. Run train_model.py first.")

    input_df = pd.DataFrame(
        [[data.color_score_1_10, data.firmness_score_1_10, data.spots_count, data.leaf_condition_score_1_10]],
        columns=["color_score_1_10", "firmness_score_1_10", "spots_count", "leaf_condition_score_1_10"]
    )
    pred_idx = quality_model.predict(input_df)[0]
    predicted_grade = quality_encoder.inverse_transform([pred_idx])[0]
    probabilities = quality_model.predict_proba(input_df)[0]

    return {
        "input_scores": data.dict(),
        "predicted_grade": predicted_grade,
        "confidence": round(float(max(probabilities)), 4),
        "class_probabilities": {
            cls: round(float(p), 4) for cls, p in zip(quality_encoder.classes_, probabilities)
        }
    }


# Mandi price prediction endpoint
@app.post("/predict/price")
def predict_price(data: CauliflowerPriceData):
    if price_model is None:
        raise HTTPException(status_code=503, detail="Price model is not loaded. Run train_model.py first.")

    input_df = pd.DataFrame(
        [[data.quantity_sold_kg, data.mandi_modal_rs_per_kg, data.mandi_min_rs_per_kg, data.mandi_max_rs_per_kg]],
        columns=["quantity_sold_kg", "mandi_modal_rs_per_kg", "mandi_min_rs_per_kg", "mandi_max_rs_per_kg"]
    )
    predicted_price = float(price_model.predict(input_df)[0])

    return {
        "mandi_modal_rs_per_kg": data.mandi_modal_rs_per_kg,
        "quantity_sold_kg": data.quantity_sold_kg,
        "estimated_selling_price_rs_per_kg": round(predicted_price, 2),
        "estimated_total_value_rs": round(predicted_price * data.quantity_sold_kg, 2)
    }


# Computer vision disease diagnosis endpoint
@app.post("/predict/image")
async def predict_image_endpoint(file: UploadFile = File(...)):
    if disease_model is None or disease_encoder is None:
        raise HTTPException(status_code=503, detail="Disease vision model is not loaded. Run train_model.py first.")

    try:
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    feat = extract_image_features(pil_img).reshape(1, -1)
    pred_idx = disease_model.predict(feat)[0]
    diagnosis = disease_encoder.inverse_transform([pred_idx])[0]
    probabilities = disease_model.predict_proba(feat)[0]

    remedy = get_remedy(diagnosis)

    prob_dict = {
        cls_name: round(float(prob), 4)
        for cls_name, prob in zip(disease_encoder.classes_, probabilities)
    }

    return {
        "filename": file.filename,
        "diagnosis": diagnosis,
        "confidence": round(float(max(probabilities)), 4),
        "condition": remedy["condition"],
        "description": remedy["description"],
        "recommended_management": remedy["management"],
        "probabilities": prob_dict
    }