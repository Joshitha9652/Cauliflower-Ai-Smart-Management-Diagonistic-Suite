"""
model_utils.py
Common utilities for feature extraction, model loading, and predictions.
"""

import os
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model paths
SIZE_MODEL_PATH = os.path.join(MODELS_DIR, "cauliflower_size_model.pkl")
SIZE_ENCODER_PATH = os.path.join(MODELS_DIR, "size_encoder.pkl")
QUALITY_MODEL_PATH = os.path.join(MODELS_DIR, "cauliflower_quality_model.pkl")
QUALITY_ENCODER_PATH = os.path.join(MODELS_DIR, "quality_encoder.pkl")
PRICE_MODEL_PATH = os.path.join(MODELS_DIR, "cauliflower_price_model.pkl")
DISEASE_MODEL_PATH = os.path.join(MODELS_DIR, "cauliflower_disease_model.pkl")
DISEASE_ENCODER_PATH = os.path.join(MODELS_DIR, "disease_encoder.pkl")

DISEASE_REMEDIES = {
    "Healthy Fruit": {
        "condition": "Optimal / Healthy",
        "description": "Clean, dense white-to-cream curd with firm compact florets and healthy protective jacket leaves.",
        "management": "Maintain adequate moisture, proper ventilation, and harvest at peak firmness for top market value."
    },
    "Bacterial Spot Rot": {
        "condition": "Infected (Bacterial)",
        "description": "Water-soaked lesions on curd and leaves, developing into brownish-black sunken spots.",
        "management": "Apply copper oxychloride (2.5 g/L) or Streptocycline spray; avoid overhead sprinkler irrigation; remove severely infected heads."
    },
    "Black Rot": {
        "condition": "Infected (Xanthomonas campestris)",
        "description": "V-shaped yellow chlorotic lesions along leaf margins progressing inward with blackened veins.",
        "management": "Practice 3-year crop rotation; use disease-free certified seeds; spray Mancozeb or Streptocycline sulfate."
    },
    "Downy Mildew": {
        "condition": "Infected (Hyaloperonospora parasitica)",
        "description": "Fluffy white-to-greyish fungal growth on underside of leaves with corresponding yellow patches on upper surface.",
        "management": "Apply Metalaxyl + Mancozeb (Ridomil MZ) at 2 g/L; ensure wide row spacing for air circulation; avoid excess soil humidity."
    }
}

def extract_image_features(image_input):
    """
    Extracts unified HOG texture and RGB color histogram features from an image.
    image_input can be a filepath (str), file-like object, or PIL.Image instance.
    Returns: 1D numpy array of shape (536,)
    """
    if isinstance(image_input, (str, bytes, os.PathLike)):
        img = Image.open(image_input)
    elif isinstance(image_input, Image.Image):
        img = image_input
    else:
        img = Image.open(image_input)

    # Standardize image to 128x128 RGB
    img = img.convert("RGB").resize((128, 128))
    arr = np.array(img)

    # Grayscale for HOG
    gray = np.mean(arr, axis=2) / 255.0
    h_feats = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

    # Color histograms (8 bins per channel = 24 bins)
    r_hist, _ = np.histogram(arr[:, :, 0], bins=8, range=(0, 256), density=True)
    g_hist, _ = np.histogram(arr[:, :, 1], bins=8, range=(0, 256), density=True)
    b_hist, _ = np.histogram(arr[:, :, 2], bins=8, range=(0, 256), density=True)

    features = np.hstack([h_feats, r_hist, g_hist, b_hist])
    return features

def get_remedy(disease_name):
    """Returns remedy and agronomic management instructions for a diagnosed disease."""
    return DISEASE_REMEDIES.get(disease_name, {
        "condition": "Unknown",
        "description": "No specific profile available.",
        "management": "Consult local agricultural extension officer."
    })
