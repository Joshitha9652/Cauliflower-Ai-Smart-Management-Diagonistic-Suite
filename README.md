# 🥦 Cauliflower AI Smart Management & Diagnostic Suite

An end-to-end Machine Learning, Computer Vision, and APMC Mandi Market Intelligence platform for cauliflower produce management.

---

## 📌 Project Overview

This project provides an automated, scientific pipeline for the entire post-harvest and field management lifecycle of cauliflower crops:
1. **Automated Size Grading**: Classifies cauliflower curds into **Small**, **Medium**, or **Large** tiers based on physical measurements (weight, diameter, height) per AGMARK / commercial standards.
2. **Computer Vision Disease & Damage Diagnosis**: Analyzes cauliflower photographs to accurately detect healthy produce or diagnose critical diseases (**Bacterial Spot Rot**, **Black Rot**, **Downy Mildew**) along with confidence scores and recommended agronomic treatments.
3. **Quality Assessment**: Evaluates curd compactness, color index, leaf condition, and blemish counts into **Grade A**, **Grade B**, or **Grade C**.
4. **Mandi Price Estimation**: Predicts fair market selling prices (₹/kg) based on produce quality, consignment quantity, and real-time APMC Mandi price dynamics.
5. **Multiple Consumption Interfaces**:
   - **FastAPI REST API**: High-performance backend with full Swagger documentation.
   - **Interactive CLI Tool**: Instant terminal grading and batch prediction.
   - **Streamlit Web Dashboard**: Modern, responsive UI with interactive charts, sliders, and image upload.

---

## 🗂️ Project Structure

```
cauliflower_separate_datasets(1)/
├── Cauliflower_256x256/                     # Cauliflower image datasets
│   ├── Original Cauliflower Dataset/        # 4 classes: Bacterial Spot Rot, Black Rot, Downy Mildew, Healthy Fruit
│   ├── Augmented Cauliflower Dataset/       # 2,400+ augmented images
│   ├── Healthy/                             # 527 healthy images
│   └── Damaged/                             # 362 damaged images
├── models/                                  # Trained ML and CV model artifacts (.pkl)
│   ├── cauliflower_size_model.pkl           # Size classifier (Random Forest)
│   ├── size_encoder.pkl                     # Size label encoder
│   ├── cauliflower_quality_model.pkl        # Quality grade classifier
│   ├── quality_encoder.pkl                  # Quality label encoder
│   ├── cauliflower_price_model.pkl          # Mandi selling price regressor
│   ├── cauliflower_disease_model.pkl        # Image disease classifier (HOG + Color Hist)
│   └── disease_encoder.pkl                  # Disease label encoder
├── 01_cauliflower_size_dataset.csv          # 1,000 samples: weight, diameter, height, size
├── 02_cauliflower_damage_dataset.csv        # 1,000 samples: condition, severity, damage type, spots
├── 03_cauliflower_quality_dataset.csv       # 1,000 samples: color, firmness, leaves, quality grade
├── 04_cauliflower_price_dataset.csv         # 1,000 samples: Andhra Pradesh APMC mandi records (Madanapalle, Vijayawada, Guntur, etc.)
├── model_utils.py                           # Shared image feature extraction, model paths & remedies
├── generate_datasets.py                     # Realistic agricultural data generator
├── train_model.py                           # Full ML & CV training pipeline
├── predicts size.py                         # Interactive & batch CLI prediction script
├── main.py                                  # FastAPI backend API
├── app.py                                   # Streamlit interactive web dashboard
├── requirements.txt                         # Python dependencies
└── README.md                                # Comprehensive documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisites
- Python 3.9+ (tested on Python 3.10 to 3.14)
- Pip package manager

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Generate / Regenerate Benchmark Datasets
Populate all 4 CSV datasets with realistic, correlated agricultural data:
```bash
python generate_datasets.py
```

### Step 2: Train All Models
Train size classifier, quality model, mandi price regressor, and CV disease classifier:
```bash
python train_model.py
```
*Trained model artifacts are automatically saved into `models/`.*

### Step 3: Use the CLI Prediction Tool
The CLI supports multiple convenient modes:

- **Direct Argument Mode**:
  ```bash
  python "predicts size.py" --weight 850 --diameter 16.5 --height 13.0
  ```

- **Interactive Mode**:
  ```bash
  python "predicts size.py"
  ```

- **Image Disease Diagnosis Mode**:
  ```bash
  python "predicts size.py" --image "Cauliflower_256x256/Original Cauliflower Dataset/Healthy Fruit/Healthy Fruit (1).png"
  ```

- **Batch Prediction on CSV**:
  ```bash
  python "predicts size.py" --csv 01_cauliflower_size_dataset.csv --out predictions.csv
  ```

### Step 4: Launch the Streamlit Web Application (with Live Camera & AP Mandis)
Run the modern interactive web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.
- **Live Camera Access:** Go to **Tab 2 (📷 Live Camera & Disease Vision)**. Click "Allow" on your browser camera prompt to scan cauliflower in real time from your laptop or mobile phone!
- **Andhra Pradesh Mandis:** Go to **Tab 3 & Tab 4** to explore pricing across all 26 districts and 48 AP Mandis & Rythu Bazars.

### Step 5: Run Real-Time Desktop Live Camera Stream (30 FPS OpenCV HUD)
To launch the real-time live webcam computer-vision feed directly on your desktop:
```bash
python live_camera_stream.py
```
**Live Camera Controls:**
- `[SPACE]` : Snap and run instant AI Disease Diagnosis + Andhra Pradesh Mandi Valuation.
- `[M]` : Switch between Andhra Pradesh Mandis (Madanapalle, Vijayawada, Guntur, Kurnool, Araku Valley, Visakhapatnam, etc.).
- `[S]` : Save annotated diagnostic snapshot to `camera_captures/`.
- `[R]` : Reset back to continuous scanning.
- `[Q]` or `[ESC]` : Exit camera feed cleanly.

### Step 6: Start the FastAPI Backend Server
Launch the REST API server:
```bash
uvicorn main:app --reload --port 8000
```
- Interactive API Documentation (Swagger UI): `http://localhost:8000/docs`
- Alternative Documentation (ReDoc): `http://localhost:8000/redoc`

---

## 📡 REST API Documentation

### 1. `GET /`
Returns service status and list of available endpoints.

### 2. `GET /health`
Returns health check and status of all 4 loaded machine learning models.

### 3. `POST /predict` (Backward-Compatible Size Prediction)
- **Request Body**:
  ```json
  {
    "weight_g": 850.0,
    "diameter_cm": 16.5,
    "height_cm": 13.0
  }
  ```
- **Response**:
  ```json
  {
    "weight_g": 850.0,
    "diameter_cm": 16.5,
    "height_cm": 13.0,
    "predicted_size": "Medium"
  }
  ```

### 4. `POST /predict/size` (Detailed Size Grading)
- **Response**:
  ```json
  {
    "weight_g": 850.0,
    "diameter_cm": 16.5,
    "height_cm": 13.0,
    "predicted_size": "Medium",
    "confidence": 1.0,
    "class_probabilities": {
      "Large": 0.0,
      "Medium": 1.0,
      "Small": 0.0
    }
  }
  ```

### 5. `POST /predict/quality` (Quality Assessment)
- **Request Body**:
  ```json
  {
    "color_score_1_10": 9.0,
    "firmness_score_1_10": 9.0,
    "spots_count": 0,
    "leaf_condition_score_1_10": 8.5
  }
  ```
- **Response**:
  ```json
  {
    "predicted_grade": "Grade A",
    "confidence": 1.0,
    "class_probabilities": {
      "Grade A": 1.0,
      "Grade B": 0.0,
      "Grade C": 0.0
    }
  }
  ```

### 6. `POST /predict/price` (Mandi Price Estimation)
- **Request Body**:
  ```json
  {
    "quantity_sold_kg": 500.0,
    "mandi_modal_rs_per_kg": 30.0,
    "mandi_min_rs_per_kg": 24.0,
    "mandi_max_rs_per_kg": 36.0
  }
  ```
- **Response**:
  ```json
  {
    "mandi_modal_rs_per_kg": 30.0,
    "quantity_sold_kg": 500.0,
    "estimated_selling_price_rs_per_kg": 32.34,
    "estimated_total_value_rs": 16170.5
  }
  ```

### 7. `POST /predict/image` (Computer Vision Disease Diagnosis)
- **Form Data**: `file` (Cauliflower image file)
- **Response**:
  ```json
  {
    "filename": "sample_curd.png",
    "diagnosis": "Healthy Fruit",
    "confidence": 0.8667,
    "condition": "Optimal / Healthy",
    "description": "Clean, dense white-to-cream curd with firm compact florets and healthy protective jacket leaves.",
    "recommended_management": "Maintain adequate moisture, proper ventilation, and harvest at peak firmness for top market value.",
    "probabilities": {
      "Bacterial Spot Rot": 0.08,
      "Black Rot": 0.0267,
      "Downy Mildew": 0.0267,
      "Healthy Fruit": 0.8667
    }
  }
  ```

---

## 📊 Model Performance Summary

| Model | Target Task | Algorithm | Accuracy / Score |
|---|---|---|---|
| **Size Classifier** | Small / Medium / Large | Random Forest (100 trees) | **100.0%** |
| **Quality Classifier** | Grade A / Grade B / Grade C | Random Forest (100 trees) | **100.0%** |
| **Price Regressor** | Selling Price (₹/kg) | Random Forest Regressor | **$R^2 = 0.621$, MAE = ₹2.92** |
| **Disease Vision Classifier** | 4 Disease / Health Classes | HOG + RGB Hist + RF (150 trees) | **81.25%** |

---

## 👨‍🌾 Agronomic Treatment Database

The system includes built-in treatment guidelines for common cauliflower diseases:
- **Bacterial Spot Rot**: Apply copper oxychloride (2.5 g/L) or Streptocycline; eliminate overhead sprinkler irrigation; rogue out severely infected heads.
- **Black Rot (*Xanthomonas campestris*)**: 3-year crop rotation; seed treatment with hot water / cert-tested seeds; spray Mancozeb or Streptocycline.
- **Downy Mildew (*Hyaloperonospora parasitica*)**: Apply Metalaxyl + Mancozeb (Ridomil MZ @ 2 g/L); ensure adequate spacing for airflow.
