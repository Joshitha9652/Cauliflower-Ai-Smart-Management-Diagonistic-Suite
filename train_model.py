"""
train_model.py
End-to-end training pipeline for Cauliflower AI models:
1. Size Classification Model (Small, Medium, Large)
2. Quality Assessment Model (Grade A, Grade B, Grade C)
3. Mandi Price Prediction Model (Selling Price Regressor)
4. Computer Vision Disease Classifier (Bacterial Spot Rot, Black Rot, Downy Mildew, Healthy Fruit)
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, r2_score, mean_absolute_error

from model_utils import (
    MODELS_DIR,
    SIZE_MODEL_PATH,
    SIZE_ENCODER_PATH,
    QUALITY_MODEL_PATH,
    QUALITY_ENCODER_PATH,
    PRICE_MODEL_PATH,
    DISEASE_MODEL_PATH,
    DISEASE_ENCODER_PATH,
    extract_image_features
)

def train_size_model():
    print("=" * 60)
    print("1. TRAINING CAULIFLOWER SIZE CLASSIFIER")
    print("=" * 60)

    csv_path = "01_cauliflower_size_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run generate_datasets.py first.")

    df = pd.read_csv(csv_path)
    X = df[["weight_g", "diameter_cm", "height_cm"]]
    y = df["size"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Size Model Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(clf, SIZE_MODEL_PATH)
    joblib.dump(encoder, SIZE_ENCODER_PATH)
    print(f"Saved: {SIZE_MODEL_PATH}")
    print(f"Saved: {SIZE_ENCODER_PATH}\n")
    return acc

def train_quality_model():
    print("=" * 60)
    print("2. TRAINING CAULIFLOWER QUALITY ASSESSMENT MODEL")
    print("=" * 60)

    csv_path = "03_cauliflower_quality_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run generate_datasets.py first.")

    df = pd.read_csv(csv_path)
    features = ["color_score_1_10", "firmness_score_1_10", "spots_count", "leaf_condition_score_1_10"]
    X = df[features]
    y = df["quality_grade"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Quality Model Test Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(clf, QUALITY_MODEL_PATH)
    joblib.dump(encoder, QUALITY_ENCODER_PATH)
    print(f"Saved: {QUALITY_MODEL_PATH}")
    print(f"Saved: {QUALITY_ENCODER_PATH}\n")
    return acc

def train_price_model():
    print("=" * 60)
    print("3. TRAINING CAULIFLOWER MANDI PRICE REGRESSOR")
    print("=" * 60)

    csv_path = "04_cauliflower_price_dataset.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run generate_datasets.py first.")

    df = pd.read_csv(csv_path)
    features = ["quantity_sold_kg", "mandi_modal_rs_per_kg", "mandi_min_rs_per_kg", "mandi_max_rs_per_kg"]
    X = df[features]
    y = df["actual_selling_price_rs_per_kg"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Price Model R2 Score: {r2:.3f}")
    print(f"Price Model Mean Absolute Error: Rs {mae:.2f} per kg\n")

    joblib.dump(reg, PRICE_MODEL_PATH)
    print(f"Saved: {PRICE_MODEL_PATH}\n")
    return r2

def train_disease_vision_model():
    print("=" * 60)
    print("4. TRAINING COMPUTER VISION DISEASE CLASSIFIER")
    print("=" * 60)

    classes = ["Bacterial Spot Rot", "Black Rot", "Downy Mildew", "Healthy Fruit"]
    base_dir = os.path.join("Cauliflower_256x256", "Original Cauliflower Dataset")

    X = []
    y = []

    print(f"Extracting texture (HOG) & color histogram features from {base_dir}...")
    for c_name in classes:
        folder = os.path.join(base_dir, c_name)
        files = glob.glob(os.path.join(folder, "*.png"))
        print(f"  -> Processing class '{c_name}': {len(files)} images")
        for f in files:
            feat = extract_image_features(f)
            X.append(feat)
            y.append(c_name)

    X = np.array(X)
    y = np.array(y)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=150, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nDisease Vision Classifier Accuracy: {acc * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    joblib.dump(clf, DISEASE_MODEL_PATH)
    joblib.dump(encoder, DISEASE_ENCODER_PATH)
    print(f"Saved: {DISEASE_MODEL_PATH}")
    print(f"Saved: {DISEASE_ENCODER_PATH}\n")
    return acc

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("\nStarting Cauliflower ML & CV Training Pipeline...\n")

    acc_size = train_size_model()
    acc_quality = train_quality_model()
    r2_price = train_price_model()
    acc_disease = train_disease_vision_model()

    print("=" * 60)
    print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
    print("=" * 60)
    print(f"1. Size Classifier Accuracy:    {acc_size * 100:.2f}%")
    print(f"2. Quality Classifier Accuracy: {acc_quality * 100:.2f}%")
    print(f"3. Price Regressor R2 Score:    {r2_price:.3f}")
    print(f"4. Disease Vision Accuracy:     {acc_disease * 100:.2f}%")
    print(f"Target directory: {MODELS_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
