"""
live_camera_stream.py
Real-Time Live Camera Feed & Agricultural Computer Vision Diagnosis Suite
for Cauliflower Grading, Disease Detection, and Andhra Pradesh Mandi Pricing.

Controls:
  - [SPACE] : Capture current frame & run AI Disease Diagnosis + Mandi Price Estimation
  - [M]     : Cycle through Andhra Pradesh Mandis (Madanapalle, Vijayawada, Guntur, Araku, etc.)
  - [S]     : Save captured annotated diagnosis snapshot to disk
  - [R]     : Reset / Return to continuous live camera scan
  - [Q/ESC] : Quit camera feed
"""

import os
import sys
import time
import joblib
import numpy as np
import cv2
import pandas as pd
from PIL import Image

from model_utils import (
    DISEASE_MODEL_PATH,
    DISEASE_ENCODER_PATH,
    PRICE_MODEL_PATH,
    extract_image_features,
    get_remedy
)
from ap_locations import get_all_ap_mandis

def run_live_camera():
    print("=" * 70)
    print("  🥦 CAULIFLOWER AI - REAL-TIME LIVE CAMERA INSPECTION SUITE")
    print("=" * 70)

    # 1. Load ML models
    disease_model = None
    disease_encoder = None
    price_model = None

    if os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_ENCODER_PATH):
        try:
            disease_model = joblib.load(DISEASE_MODEL_PATH)
            disease_encoder = joblib.load(DISEASE_ENCODER_PATH)
            print("✓ Disease Vision Model loaded successfully.")
        except Exception as e:
            print(f"⚠ Could not load disease model: {e}")

    if os.path.exists(PRICE_MODEL_PATH):
        try:
            price_model = joblib.load(PRICE_MODEL_PATH)
            print("✓ APMC Mandi Price Model loaded successfully.")
        except Exception as e:
            print(f"⚠ Could not load price model: {e}")

    # 2. Load Andhra Pradesh Mandi list
    ap_mandis = get_all_ap_mandis()
    current_mandi_idx = 0
    print(f"✓ Loaded {len(ap_mandis)} Andhra Pradesh Mandis and Rythu Bazars.")

    # 3. Initialize camera
    print("\nAttempting to connect to live webcam (Index 0)...")
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    use_fallback = False
    if not cap.isOpened():
        print("⚠ Live camera (Index 0) could not be opened automatically.")
        print("  Trying camera index 1...")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("⚠ No hardware webcam detected. Running in SIMULATED CAMERA MODE with sample image.")
            use_fallback = True

    output_dir = "camera_captures"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "-" * 70)
    print("LIVE CAMERA ACTIVE! Keyboard Commands:")
    print("  [SPACE] : Diagnose Curd (Run AI Disease & AP Mandi Price)")
    print("  [M]     : Switch Andhra Pradesh Mandi")
    print("  [S]     : Save Diagnosed Image to disk")
    print("  [R]     : Resume live scan")
    print("  [Q/ESC] : Exit")
    print("-" * 70 + "\n")

    latest_diagnosis = None
    latest_confidence = None
    latest_remedy = None
    latest_price = None
    last_action_text = "Live Scanning... Position cauliflower inside the central frame"

    # Fallback image if no physical webcam
    fallback_img = None
    if use_fallback:
        sample_path = os.path.join("Cauliflower_256x256", "Original Cauliflower Dataset", "Healthy Fruit", "Healthy Fruit (1).png")
        if os.path.exists(sample_path):
            fallback_img = cv2.imread(sample_path)
            fallback_img = cv2.resize(fallback_img, (640, 480))
        else:
            fallback_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(fallback_img, "Camera Not Connected", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    try:
        while True:
            if not use_fallback:
                ret, frame = cap.read()
                if not ret:
                    print("⚠ Failed to grab frame from camera.")
                    break
            else:
                frame = fallback_img.copy()
                time.sleep(0.03)

            h, w = frame.shape[:2]

            # Central ROI box for cauliflower placement
            box_size = int(min(h, w) * 0.65)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size

            # Current Andhra Pradesh Mandi
            mandi = ap_mandis[current_mandi_idx]
            mandi_name = mandi["mandi_name"]
            mandi_dist = mandi["district"]
            m_modal = mandi["modal_price"]
            m_min = mandi["min_price"]
            m_max = mandi["max_price"]

            # Draw UI Overlays
            # Top Banner (Dark Bar)
            cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 20), -1)
            cv2.putText(frame, "🥦 Cauliflower AI Live Camera Vision", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 127), 2)
            cv2.putText(frame, f"AP Mandi [{current_mandi_idx + 1}/{len(ap_mandis)}]: {mandi_name}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            # Central Targeting Box
            box_color = (0, 255, 0) if latest_diagnosis == "Healthy Fruit" else (0, 140, 255) if latest_diagnosis is not None else (0, 220, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, "Target: Align Curd Inside Box", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)

            # Bottom Status & Info Bar
            cv2.rectangle(frame, (0, h - 90), (w, h), (25, 25, 25), -1)

            if latest_diagnosis is not None:
                # Show AI diagnosis result
                diag_color = (0, 255, 0) if latest_diagnosis == "Healthy Fruit" else (0, 70, 255)
                cv2.putText(frame, f"DIAGNOSIS: {latest_diagnosis.upper()} ({latest_confidence*100:.1f}%)", (15, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, diag_color, 2)
                cv2.putText(frame, f"Mandi Modal: Rs {m_modal:.1f}/kg | Est. Value: Rs {latest_price:.1f}/kg ({mandi_dist} Dist)", (15, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1)
                cv2.putText(frame, f"Condition: {latest_remedy['condition']}", (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            else:
                cv2.putText(frame, f"Status: {last_action_text}", (15, h - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
                cv2.putText(frame, f"Mandi Rate ({mandi_dist}): Min Rs {m_min:.1f} | Modal Rs {m_modal:.1f} | Max Rs {m_max:.1f}", (15, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
                cv2.putText(frame, "[SPACE]: Diagnose | [M]: Next AP Mandi | [S]: Save | [Q]: Quit", (15, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

            cv2.imshow("Cauliflower AI - Live Camera & AP Mandi Pricing", frame)
            key = cv2.waitKey(1) & 0xFF

            # Key handlers
            if key in [ord('q'), ord('Q'), 27]: # Quit
                print("Exiting live camera...")
                break

            elif key in [ord('m'), ord('M')]: # Cycle Andhra Pradesh Mandi
                current_mandi_idx = (current_mandi_idx + 1) % len(ap_mandis)
                sel_mandi = ap_mandis[current_mandi_idx]
                print(f"Selected AP Mandi: {sel_mandi['mandi_name']} ({sel_mandi['district']} District)")

            elif key == 32: # SPACE: Run Diagnosis on ROI
                print("Capturing frame for AI Diagnosis...")
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0 and disease_model is not None and disease_encoder is not None:
                    # Convert BGR to RGB PIL
                    rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    pil_roi = Image.fromarray(rgb_roi)

                    # Extract features and predict
                    feat = extract_image_features(pil_roi).reshape(1, -1)
                    pred_idx = disease_model.predict(feat)[0]
                    latest_diagnosis = disease_encoder.inverse_transform([pred_idx])[0]
                    probs = disease_model.predict_proba(feat)[0]
                    latest_confidence = float(max(probs))
                    latest_remedy = get_remedy(latest_diagnosis)

                    # Estimate price in current AP Mandi
                    if price_model is not None:
                        # Baseline batch 500kg
                        p_df = pd.DataFrame(
                            [[500.0, m_modal, m_min, m_max]],
                            columns=["quantity_sold_kg", "mandi_modal_rs_per_kg", "mandi_min_rs_per_kg", "mandi_max_rs_per_kg"]
                        )
                        base_p = float(price_model.predict(p_df)[0])
                        if latest_diagnosis == "Healthy Fruit":
                            latest_price = round(base_p + 2.0, 2)
                        else:
                            latest_price = round(max(8.0, base_p - 4.0), 2)
                    else:
                        latest_price = m_modal

                    print(f"→ Result: {latest_diagnosis} (Confidence: {latest_confidence*100:.1f}%)")
                    print(f"→ AP Market: {mandi_name}")
                    print(f"→ Est. Selling Price: Rs {latest_price:.2f} / kg")
                    print(f"→ Recommended Action: {latest_remedy['management']}")
                    last_action_text = f"Diagnosed: {latest_diagnosis}"

            elif key in [ord('s'), ord('S')]: # Save snapshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"cauliflower_scan_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"✓ Saved annotated diagnosis to {filename}")
                last_action_text = f"Saved snapshot to {filename}"

            elif key in [ord('r'), ord('R')]: # Reset
                latest_diagnosis = None
                latest_confidence = None
                latest_remedy = None
                latest_price = None
                last_action_text = "Live Scanning... Position cauliflower inside the central frame"
                print("Reset to live scanning.")

    finally:
        if not use_fallback:
            cap.release()
        cv2.destroyAllWindows()
        print("Live camera session ended.")

if __name__ == "__main__":
    run_live_camera()
