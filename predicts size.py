"""
predicts size.py
CLI Prediction tool for Cauliflower Size classification and produce assessment.
Supports:
1. Argument-based prediction: python "predicts size.py" --weight 850 --diameter 16 --height 13
2. Interactive mode (if no args provided): python "predicts size.py"
3. Batch CSV mode: python "predicts size.py" --csv 01_cauliflower_size_dataset.csv
4. Image diagnosis mode: python "predicts size.py" --image path/to/image.png
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from model_utils import (
    SIZE_MODEL_PATH,
    SIZE_ENCODER_PATH,
    DISEASE_MODEL_PATH,
    DISEASE_ENCODER_PATH,
    extract_image_features,
    get_remedy
)

def load_size_model():
    if not os.path.exists(SIZE_MODEL_PATH) or not os.path.exists(SIZE_ENCODER_PATH):
        print(f"Error: Model files not found in models/ directory.")
        print("Please run 'python train_model.py' first.")
        sys.exit(1)
    model = joblib.load(SIZE_MODEL_PATH)
    encoder = joblib.load(SIZE_ENCODER_PATH)
    return model, encoder

def predict_single_size(weight_g, diameter_cm, height_cm, model, encoder):
    input_data = pd.DataFrame(
        [[weight_g, diameter_cm, height_cm]],
        columns=["weight_g", "diameter_cm", "height_cm"]
    )
    pred_idx = model.predict(input_data)[0]
    predicted_label = encoder.inverse_transform([pred_idx])[0]
    probabilities = model.predict_proba(input_data)[0]

    print("\n" + "=" * 50)
    print("      CAULIFLOWER SIZE PREDICTION RESULT")
    print("=" * 50)
    print(f"  Input Features:")
    print(f"    - Weight:        {weight_g:.1f} g")
    print(f"    - Curd Diameter: {diameter_cm:.1f} cm")
    print(f"    - Curd Height:   {height_cm:.1f} cm")
    print("-" * 50)
    print(f"  PREDICTED SIZE:   >>>  {predicted_label.upper()}  <<<")
    print("-" * 50)
    print("  Confidence Probabilities:")
    for cls_name, prob in zip(encoder.classes_, probabilities):
        bar = "#" * int(prob * 20)
        print(f"    {cls_name:<8}: {prob * 100:5.1f}% | {bar}")
    print("=" * 50 + "\n")
    return predicted_label

def predict_batch_csv(csv_path, model, encoder, output_csv=None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    required = ["weight_g", "diameter_cm", "height_cm"]
    for col in required:
        if col not in df.columns:
            print(f"Error: CSV missing required column: {col}")
            return

    X = df[required]
    preds = model.predict(X)
    labels = encoder.inverse_transform(preds)
    df["predicted_size"] = labels

    print(f"\nBatch predictions completed for {len(df)} samples.")
    print("\nSample Preview (First 5):")
    cols_to_show = [c for c in ["sample_id", "weight_g", "diameter_cm", "height_cm", "size", "predicted_size"] if c in df.columns]
    print(df[cols_to_show].head())

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

def predict_image(image_path):
    if not os.path.exists(DISEASE_MODEL_PATH) or not os.path.exists(DISEASE_ENCODER_PATH):
        print("Disease model not found. Run 'python train_model.py' first.")
        return

    model = joblib.load(DISEASE_MODEL_PATH)
    encoder = joblib.load(DISEASE_ENCODER_PATH)

    feat = extract_image_features(image_path).reshape(1, -1)
    pred_idx = model.predict(feat)[0]
    diagnosis = encoder.inverse_transform([pred_idx])[0]
    probs = model.predict_proba(feat)[0]

    remedy = get_remedy(diagnosis)

    print("\n" + "=" * 55)
    print("      CAULIFLOWER IMAGE DISEASE DIAGNOSIS")
    print("=" * 55)
    print(f"  Image: {image_path}")
    print(f"  DIAGNOSIS:       >>>  {diagnosis.upper()}  <<<")
    print(f"  Condition:       {remedy['condition']}")
    print("-" * 55)
    print("  Confidence Breakdown:")
    for cls_name, prob in zip(encoder.classes_, probs):
        bar = "#" * int(prob * 20)
        print(f"    {cls_name:<20}: {prob * 100:5.1f}% | {bar}")
    print("-" * 55)
    print(f"  Description: {remedy['description']}")
    print(f"  Management:  {remedy['management']}")
    print("=" * 55 + "\n")

def interactive_mode(model, encoder):
    print("\n" + "=" * 50)
    print("    CAULIFLOWER SIZE PREDICTOR - INTERACTIVE CLI")
    print("=" * 50)
    print("Enter the physical measurements of the cauliflower.")
    print("(Press Ctrl+C at any time to exit)\n")

    while True:
        try:
            w_str = input("Enter weight in grams (e.g., 850): ").strip()
            if not w_str:
                continue
            weight = float(w_str)

            d_str = input("Enter diameter in cm (e.g., 16.5): ").strip()
            diameter = float(d_str)

            h_str = input("Enter height in cm (e.g., 13.0): ").strip()
            height = float(h_str)

            predict_single_size(weight, diameter, height, model, encoder)

            again = input("Predict another? (y/n, default: y): ").strip().lower()
            if again == "n":
                print("Exiting predictor. Have a great day!")
                break
        except ValueError:
            print("Invalid number entered. Please enter valid numeric values.\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting predictor.")
            break

def main():
    parser = argparse.ArgumentParser(description="Predict Cauliflower Size and Quality")
    parser.add_argument("--weight", type=float, help="Curd weight in grams")
    parser.add_argument("--diameter", type=float, help="Curd diameter in centimeters")
    parser.add_argument("--height", type=float, help="Curd height in centimeters")
    parser.add_argument("--csv", type=str, help="Path to CSV file for batch predictions")
    parser.add_argument("--out", type=str, help="Path to save batch predictions CSV")
    parser.add_argument("--image", type=str, help="Path to image file for disease diagnosis")

    args = parser.parse_args()

    if args.image:
        predict_image(args.image)
        return

    model, encoder = load_size_model()

    if args.csv:
        predict_batch_csv(args.csv, model, encoder, args.out)
    elif args.weight is not None and args.diameter is not None and args.height is not None:
        predict_single_size(args.weight, args.diameter, args.height, model, encoder)
    else:
        interactive_mode(model, encoder)

if __name__ == "__main__":
    main()
