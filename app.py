"""
app.py
Streamlit Web Application for Cauliflower Produce Intelligence & Diagnosis:
1. Physical Size Grading & Recommendation
2. Live Camera Access & Computer Vision Disease Diagnosis with Treatment Guide
3. Quality & Comprehensive Andhra Pradesh Mandi Price Estimation (All 26 Districts)
4. Andhra Pradesh Mandis & Rythu Bazars Directory
5. Dataset & Agricultural Analytics Explorer
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
    get_mandi_names,
    get_mandi_price_dict,
    get_mandi_details,
    ANDHRA_PRADESH_MANDIS
)

# Page configuration
st.set_page_config(
    page_title="Cauliflower AI Suite - Andhra Pradesh Mandis",
    page_icon="🥦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #2E7D32;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F1F8E9;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
    .status-healthy {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-warning {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .status-danger {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 0.4rem 0.8rem;
        border-radius: 4px;
        font-weight: 600;
    }
    .mandi-card {
        background-color: #f9fbf9;
        border: 1px solid #c8e6c9;
        border-left: 6px solid #2e7d32;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Cache model loading
@st.cache_resource
def load_all_models():
    models = {}
    if os.path.exists(SIZE_MODEL_PATH) and os.path.exists(SIZE_ENCODER_PATH):
        models["size_model"] = joblib.load(SIZE_MODEL_PATH)
        models["size_encoder"] = joblib.load(SIZE_ENCODER_PATH)
    if os.path.exists(QUALITY_MODEL_PATH) and os.path.exists(QUALITY_ENCODER_PATH):
        models["quality_model"] = joblib.load(QUALITY_MODEL_PATH)
        models["quality_encoder"] = joblib.load(QUALITY_ENCODER_PATH)
    if os.path.exists(PRICE_MODEL_PATH):
        models["price_model"] = joblib.load(PRICE_MODEL_PATH)
    if os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_ENCODER_PATH):
        models["disease_model"] = joblib.load(DISEASE_MODEL_PATH)
        models["disease_encoder"] = joblib.load(DISEASE_ENCODER_PATH)
    return models

models = load_all_models()

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1568584711075-3d021a7c3ca3?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("Cauliflower AI System")
    st.markdown("**Precision agriculture tool for grading, disease diagnosis, and Andhra Pradesh mandi market intelligence.**")

    st.markdown("---")
    st.subheader("Model Status")
    for name, key in [
        ("Size Classifier", "size_model"),
        ("Quality Classifier", "quality_model"),
        ("Price Regressor", "price_model"),
        ("Vision Disease Classifier", "disease_model")
    ]:
        if key in models and models[key] is not None:
            st.markdown(f"🟢 **{name}**: Ready")
        else:
            st.markdown(f"🔴 **{name}**: Missing (run `train_model.py`)")

    st.markdown("---")
    st.subheader("📷 Live Camera Quick Launcher")
    st.info("You can scan curds directly via browser camera in **Tab 2**, or run the 30 FPS OpenCV desktop live HUD:")
    st.code("python live_camera_stream.py", language="bash")

    st.markdown("---")
    st.subheader("📍 Andhra Pradesh Coverage")
    st.markdown(f"• **{len(get_all_districts())} Districts Covered**")
    st.markdown(f"• **{len(ANDHRA_PRADESH_MANDIS)} APMC Mandis & Rythu Bazars**")
    st.caption("Cauliflower AI Suite v2.1 • Powered by scikit-learn, OpenCV & FastAPI")

# Main Header
st.markdown('<div class="main-title">🥦 Cauliflower AI Smart Management Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent physical grading, live camera disease diagnosis, and Andhra Pradesh mandi market pricing.</div>', unsafe_allow_html=True)

# Top KPI metrics
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Master Samples", "1,000", "4 Datasets")
with kpi2:
    st.metric("Size Model Accuracy", "100.0%", "Random Forest")
with kpi3:
    st.metric("Disease Vision Accuracy", "81.25%", "4 Pathogen Classes")
with kpi4:
    st.metric("AP Mandis & Bazars", f"{len(ANDHRA_PRADESH_MANDIS)} Markets", f"{len(get_all_districts())} AP Districts")

st.markdown("---")

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📏 Size Grading",
    "📷 Live Camera & Disease Vision",
    "💰 Quality & AP Mandi Price",
    "📍 Andhra Pradesh Mandi Directory",
    "📊 Datasets & Analytics"
])

# ==========================================
# TAB 1: SIZE PREDICTION
# ==========================================
with tab1:
    st.header("Cauliflower Size Grading")
    st.write("Enter physical curd dimensions to classify produce into Small, Medium, or Large grades per agricultural standards.")

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("Physical Measurements")
        weight = st.slider("Weight (grams)", min_value=300.0, max_value=2500.0, value=850.0, step=10.0, help="Weight of trimmed curd in grams")
        diameter = st.slider("Curd Diameter (cm)", min_value=8.0, max_value=26.0, value=16.5, step=0.5, help="Maximum horizontal curd diameter")
        height = st.slider("Curd Height (cm)", min_value=7.0, max_value=23.0, value=13.0, step=0.5, help="Vertical curd height from base to top")

        predict_btn = st.button("Grade Cauliflower Size", type="primary", use_container_width=True)

    with c2:
        st.subheader("Classification Outcome")
        if "size_model" in models and models["size_model"] is not None:
            input_df = pd.DataFrame([[weight, diameter, height]], columns=["weight_g", "diameter_cm", "height_cm"])
            model = models["size_model"]
            encoder = models["size_encoder"]

            pred_idx = model.predict(input_df)[0]
            predicted_size = encoder.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(input_df)[0]

            color_badge = "#2E7D32" if predicted_size == "Large" else "#1976D2" if predicted_size == "Medium" else "#E65100"

            st.markdown(f"""
            <div style="background-color: {color_badge}15; border: 2px solid {color_badge}; border-radius: 8px; padding: 1.5rem; text-align: center;">
                <div style="font-size: 1rem; color: #666; text-transform: uppercase; letter-spacing: 1px;">Standard Produce Grade</div>
                <div style="font-size: 2.4rem; font-weight: 800; color: {color_badge}; margin: 0.4rem 0;">{predicted_size.upper()}</div>
                <div style="font-size: 0.95rem; color: #444;">Confidence: <b>{max(probs)*100:.1f}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Probability Distribution")
            for cls_name, prob in zip(encoder.classes_, probs):
                st.write(f"**{cls_name}** ({prob*100:.1f}%)")
                st.progress(float(prob))

            st.markdown("#### Market Recommendation")
            if predicted_size == "Small":
                st.info("💡 **Small Grade (<700g, <13.5cm):** Best suited for direct retail packs, fresh salad preparations, and nuclear family consumption.")
            elif predicted_size == "Medium":
                st.success("🌟 **Medium Grade (700g–1350g, 13.5–19cm):** The highest commercial demand tier. Ideal for general retail and wholesale supermarkets.")
            else:
                st.success("🏆 **Large Grade (>1350g, >19cm):** Premium grade. Preferred for food processing, restaurant catering, and frozen florets.")
        else:
            st.warning("Size model not loaded. Run train_model.py to enable predictions.")

# ==========================================
# TAB 2: LIVE CAMERA & DISEASE DIAGNOSIS
# ==========================================
with tab2:
    st.header("📷 Live Camera Scan & Computer Vision Diagnosis")
    st.write("Scan cauliflower directly using your camera/webcam, or upload an image to inspect for bacterial rots, downy mildew, and curd defects.")

    with st.expander("ℹ️ How to Allow Camera Access & Troubleshooting Tips", expanded=False):
        st.markdown("""
        - **Browser Permission:** When prompted, click **'Allow'** to let your browser access your camera.
        - **Chrome / Edge:** Click the lock or camera icon on the left of the browser address bar, then switch **Camera** to **Allow**.
        - **Mobile / Phone:** Works seamlessly on Android Chrome and iPhone Safari! Hold your phone over the cauliflower curd.
        - **Optimal Capture:** Center the curd in good, natural lighting without harsh shadows or glares.
        - **Continuous Desktop Stream (30 FPS):** For real-time continuous video feed with bounding reticle and AP mandi overlay, run:
          `python live_camera_stream.py` in your terminal!
        """)

    img_col1, img_col2 = st.columns([1.1, 0.9])

    with img_col1:
        upload_mode = st.radio(
            "Select Camera or Image Input Mode:",
            ["📷 Live Camera (Webcam / Phone)", "📁 Upload Image File", "🧪 Select Benchmark Sample Image"],
            horizontal=True
        )

        selected_image = None
        if upload_mode == "📷 Live Camera (Webcam / Phone)":
            st.markdown("##### 📸 Live Camera Capture")
            camera_file = st.camera_input("Point camera at cauliflower curd and take a snapshot:")
            if camera_file is not None:
                selected_image = Image.open(camera_file)
                st.success("✓ Live camera photo captured successfully!")
        elif upload_mode == "📁 Upload Image File":
            uploaded_file = st.file_uploader("Upload Cauliflower Image (from camera, phone, or disk)", type=["png", "jpg", "jpeg", "webp"])
            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
        else:
            sample_options = {
                "Sample: Healthy Fruit": "Cauliflower_256x256/Original Cauliflower Dataset/Healthy Fruit/Healthy Fruit (1).png",
                "Sample: Bacterial Spot Rot": "Cauliflower_256x256/Original Cauliflower Dataset/Bacterial Spot Rot/Bacterial Spot Rot (1).png",
                "Sample: Black Rot": "Cauliflower_256x256/Original Cauliflower Dataset/Black Rot/Black Rot (1).png",
                "Sample: Downy Mildew": "Cauliflower_256x256/Original Cauliflower Dataset/Downy Mildew/Downy Mildew (1).png"
            }
            sample_choice = st.selectbox("Select a benchmark sample to test:", list(sample_options.keys()))
            sample_path = sample_options[sample_choice]
            if os.path.exists(sample_path):
                selected_image = Image.open(sample_path)

        if selected_image is not None and upload_mode != "📷 Live Camera (Webcam / Phone)":
            st.image(selected_image, caption="Cauliflower Image for Analysis", use_container_width=True)

    with img_col2:
        st.subheader("Diagnostic Results & Remedies")
        if selected_image is not None and "disease_model" in models and models["disease_model"] is not None:
            model = models["disease_model"]
            encoder = models["disease_encoder"]

            feat = extract_image_features(selected_image).reshape(1, -1)
            pred_idx = model.predict(feat)[0]
            diagnosis = encoder.inverse_transform([pred_idx])[0]
            probs = model.predict_proba(feat)[0]
            confidence = max(probs)

            remedy = get_remedy(diagnosis)

            status_style = "status-healthy" if diagnosis == "Healthy Fruit" else "status-danger"
            st.markdown(f"""
            <div style="background-color: #fafafa; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem; margin-bottom: 1rem;">
                <span class="{status_style}">{remedy['condition']}</span>
                <h2 style="margin: 0.5rem 0; color: #222;">{diagnosis}</h2>
                <p style="color: #666; margin-bottom: 0.2rem;">Diagnostic Confidence: <b>{confidence*100:.1f}%</b></p>
            </div>
            """, unsafe_allow_html=True)

            # Instant Andhra Pradesh Mandi Valuation
            st.markdown("#### 📍 Instant Andhra Pradesh Mandi Valuation")
            quick_mandi = st.selectbox(
                "Evaluate fair selling price for this curd in AP Mandi:",
                [
                    "Madanapalle Mandi (Annamayya / Chittoor)",
                    "Vijayawada Gollapudi APMC (NTR / Krishna)",
                    "Guntur APMC Market (Guntur)",
                    "Kurnool APMC Market (C-Camp)",
                    "Visakhapatnam MVP Rythu Bazar (Visakhapatnam)",
                    "Tirupati RC Road Rythu Bazar",
                    "Araku Valley Horticulture Hub (ASR District)",
                    "Rajahmundry APMC (East Godavari)"
                ]
            )
            mandi_info = get_mandi_details(quick_mandi)
            if mandi_info:
                base_modal = mandi_info["modal_price"]
                if diagnosis == "Healthy Fruit":
                    est_price = base_modal + 2.5
                    badge = "Grade A Premium"
                    val_color = "#2E7D32"
                elif diagnosis == "Downy Mildew":
                    est_price = max(8.0, base_modal - 3.0)
                    badge = "Grade B (Minor Rot Discount)"
                    val_color = "#FB8C00"
                else:
                    est_price = max(6.0, base_modal - 5.5)
                    badge = "Grade C (Severe Rot Discount)"
                    val_color = "#D32F2F"

                st.markdown(f"""
                <div style="background-color: {val_color}10; border-left: 4px solid {val_color}; padding: 0.8rem 1rem; border-radius: 4px; margin-bottom: 1rem;">
                    <b>Estimated Selling Price:</b> <span style="font-size: 1.4rem; font-weight: 800; color: {val_color};">₹ {est_price:.2f} / kg</span><br/>
                    <small>Benchmark Modal Price in {mandi_info['district']} Dist: ₹ {base_modal:.2f}/kg ({badge})</small>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("#### Probability Analysis")
            for cls_name, prob in zip(encoder.classes_, probs):
                st.write(f"**{cls_name}**: {prob*100:.1f}%")
                st.progress(float(prob))

            st.markdown("#### Pathological Profile")
            st.info(f"**Symptoms:** {remedy['description']}")
            st.warning(f"**Recommended Agronomic Action:**\n\n{remedy['management']}")
        elif selected_image is None:
            st.info("👆 Please allow camera access and click 'Take Photo' or select an image on the left.")
        else:
            st.warning("Disease vision model not loaded. Run train_model.py first.")

# ==========================================
# TAB 3: QUALITY & PRICE PREDICTION (ANDHRA PRADESH)
# ==========================================
with tab3:
    st.header("Quality Grading & Andhra Pradesh Mandi Price Calculator")
    st.write("Estimate fair market selling price and consignment revenue across all major Andhra Pradesh APMC mandis and Rythu Bazars.")

    q_col1, q_col2 = st.columns([1, 1])

    with q_col1:
        st.subheader("1. Curd Quality Indicators")
        color_val = st.slider("Color Whiteness Score (1 = Brown/Yellowish, 10 = Pure Snow White)", 1.0, 10.0, 8.5, 0.5)
        firmness_val = st.slider("Firmness Score (1 = Loose/Ricy florets, 10 = Very Compact/Solid)", 1.0, 10.0, 8.5, 0.5)
        leaf_val = st.slider("Jacket Leaf Freshness (1 = Dry/Wilting, 10 = Turgid & Fresh Deep Green)", 1.0, 10.0, 8.0, 0.5)
        spots_val = st.number_input("Blemish / Pest Spot Count", min_value=0, max_value=60, value=2, step=1)

        st.subheader("2. Andhra Pradesh Location & Mandi Selection")
        
        # District filter
        all_districts = ["All Andhra Pradesh Districts"] + get_all_districts()
        selected_district = st.selectbox("Filter by Andhra Pradesh District:", all_districts)

        if selected_district == "All Andhra Pradesh Districts":
            available_mandis = get_all_ap_mandis()
        else:
            available_mandis = get_mandis_by_district(selected_district)

        mandi_names_list = [m["mandi_name"] for m in available_mandis]
        market_selected = st.selectbox("Select Target APMC Mandi or Rythu Bazar:", mandi_names_list)

        # Retrieve selected mandi metadata
        mandi_details = get_mandi_details(market_selected)
        if mandi_details:
            st.markdown(f"""
            <div class="mandi-card">
                <b>District:</b> {mandi_details['district']} | <b>Region:</b> {mandi_details['region']}<br/>
                <b>Market Type:</b> {mandi_details['market_type']}<br/>
                <b>Cauliflower Supply:</b> {mandi_details['cauliflower_production']}<br/>
                <b>Peak Season:</b> {mandi_details['peak_months']}<br/>
                <small style="color: #555;">{mandi_details['notes']}</small>
            </div>
            """, unsafe_allow_html=True)
            min_p = mandi_details["min_price"]
            modal_p = mandi_details["modal_price"]
            max_p = mandi_details["max_price"]
        else:
            min_p, modal_p, max_p = 20.0, 26.0, 32.0

        p1, p2, p3 = st.columns(3)
        with p1:
            mandi_min = st.number_input("Mandi Minimum (₹/kg)", value=min_p, step=1.0)
        with p2:
            mandi_modal = st.number_input("Mandi Modal (₹/kg)", value=modal_p, step=1.0)
        with p3:
            mandi_max = st.number_input("Mandi Maximum (₹/kg)", value=max_p, step=1.0)

        batch_qty = st.number_input("Consignment Quantity (kg)", min_value=50.0, max_value=25000.0, value=500.0, step=50.0)

    with q_col2:
        st.subheader("Price & Grade Evaluation")
        if "quality_model" in models and models["quality_model"] is not None and "price_model" in models and models["price_model"] is not None:
            # Quality prediction
            q_df = pd.DataFrame([[color_val, firmness_val, spots_val, leaf_val]], columns=["color_score_1_10", "firmness_score_1_10", "spots_count", "leaf_condition_score_1_10"])
            q_model = models["quality_model"]
            q_encoder = models["quality_encoder"]
            grade = q_encoder.inverse_transform(q_model.predict(q_df))[0]

            # Price prediction
            p_df = pd.DataFrame([[batch_qty, mandi_modal, mandi_min, mandi_max]], columns=["quantity_sold_kg", "mandi_modal_rs_per_kg", "mandi_min_rs_per_kg", "mandi_max_rs_per_kg"])
            base_pred_price = float(models["price_model"].predict(p_df)[0])

            # Apply quality adjustment
            if grade == "Grade A":
                final_price = base_pred_price + 2.0
            elif grade == "Grade C":
                final_price = base_pred_price - 3.5
            else:
                final_price = base_pred_price

            final_price = max(8.0, round(final_price, 2))
            total_value = round(final_price * batch_qty, 2)

            grade_color = "#2E7D32" if grade == "Grade A" else "#FB8C00" if grade == "Grade B" else "#D32F2F"

            st.markdown(f"""
            <div style="background-color: {grade_color}10; border: 2px solid {grade_color}; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;">
                <div style="font-size: 0.95rem; color: #555; text-transform: uppercase;">Evaluated Produce Quality</div>
                <div style="font-size: 2.2rem; font-weight: 800; color: {grade_color}; margin: 0.3rem 0;">{grade}</div>
                <div style="font-size: 1rem; color: #333;">Recommended Fair Selling Price:</div>
                <div style="font-size: 2.8rem; font-weight: 900; color: #1B5E20; margin: 0.3rem 0;">₹ {final_price:.2f} <span style="font-size: 1.1rem; color: #666;">/ kg</span></div>
                <div style="font-size: 1.2rem; font-weight: 600; color: #333; margin-top: 0.5rem;">Total Consignment Value ({batch_qty:,.0f} kg): ₹ {total_value:,.2f}</div>
                <div style="margin-top: 0.5rem; font-size: 0.9rem; color: #555;">Market: <b>{market_selected}</b></div>
            </div>
            """, unsafe_allow_html=True)

            diff = final_price - mandi_modal
            if diff >= 0:
                st.success(f"📈 Premium of ₹ {diff:.2f}/kg over current mandi modal price (₹ {mandi_modal:.2f}/kg).")
            else:
                st.warning(f"📉 Discount of ₹ {abs(diff):.2f}/kg below current mandi modal price due to quality score.")

            st.markdown("#### Quality Grade Breakdown")
            st.write("• **Grade A (Premium):** Pure snow-white curd, compact florets, spotless, fresh green jacket leaves. Commands top APMC prices.")
            st.write("• **Grade B (Commercial):** Minor curd yellowing or slight superficial spots; firm curd. Standard market pricing.")
            st.write("• **Grade C (Sub-Standard):** Loose florets, prominent spots, wilting jacket leaves. Discounted sale or industrial processing.")
        else:
            st.warning("Quality or price model not loaded. Run train_model.py.")

# ==========================================
# TAB 4: ANDHRA PRADESH MANDI DIRECTORY
# ==========================================
with tab4:
    st.header("📍 Andhra Pradesh Mandi Directory & Agricultural Map")
    st.write(f"Comprehensive directory of **{len(ANDHRA_PRADESH_MANDIS)} agricultural markets** across all **{len(get_all_districts())} districts** of Andhra Pradesh.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        reg_filter = st.selectbox("Filter by Region:", ["All Regions", "Rayalaseema", "Coastal Andhra", "North Coastal"])
    with f_col2:
        dist_options = ["All Districts"]
        if reg_filter != "All Regions":
            dist_options += sorted(list(set(m["district"] for m in ANDHRA_PRADESH_MANDIS if m["region"] == reg_filter)))
        else:
            dist_options += get_all_districts()
        sel_dist = st.selectbox("Filter by District:", dist_options)
    with f_col3:
        search_query = st.text_input("Search Mandi Name or Town:", "")

    # Filter mandis
    filtered_mandis = ANDHRA_PRADESH_MANDIS
    if reg_filter != "All Regions":
        filtered_mandis = [m for m in filtered_mandis if m["region"] == reg_filter]
    if sel_dist != "All Districts":
        filtered_mandis = [m for m in filtered_mandis if m["district"] == sel_dist]
    if search_query:
        filtered_mandis = [m for m in filtered_mandis if search_query.lower() in m["mandi_name"].lower() or search_query.lower() in m["district"].lower()]

    st.write(f"Showing **{len(filtered_mandis)} mandis** matching filters:")

    # Table view
    mandi_table_data = []
    for m in filtered_mandis:
        mandi_table_data.append({
            "District": m["district"],
            "Region": m["region"],
            "Mandi / Market Name": m["mandi_name"],
            "Market Type": m["market_type"],
            "Min Price (₹/kg)": m["min_price"],
            "Modal Price (₹/kg)": m["modal_price"],
            "Max Price (₹/kg)": m["max_price"],
            "Peak Supply Months": m["peak_months"],
            "Cauliflower Cultivation": m["cauliflower_production"]
        })
    df_mandi_table = pd.DataFrame(mandi_table_data)
    st.dataframe(df_mandi_table, use_container_width=True)

    st.markdown("### Regional Price Comparison across Andhra Pradesh")
    price_by_reg = pd.DataFrame(ANDHRA_PRADESH_MANDIS).groupby("region")[["min_price", "modal_price", "max_price"]].mean().reset_index()
    
    fig_reg, ax_reg = plt.subplots(figsize=(10, 4))
    x = np.arange(len(price_by_reg))
    width = 0.25
    ax_reg.bar(x - width, price_by_reg["min_price"], width, label="Avg Min Price (₹)", color="#81c784")
    ax_reg.bar(x, price_by_reg["modal_price"], width, label="Avg Modal Price (₹)", color="#2e7d32")
    ax_reg.bar(x + width, price_by_reg["max_price"], width, label="Avg Max Price (₹)", color="#1b5e20")
    ax_reg.set_xticks(x)
    ax_reg.set_xticklabels(price_by_reg["region"])
    ax_reg.set_ylabel("Price (₹ per kg)")
    ax_reg.set_title("Average Cauliflower Price Benchmarks by Andhra Pradesh Region")
    ax_reg.legend()
    ax_reg.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig_reg)

    st.markdown("### 🌾 Key Cauliflower Agricultural Zones in Andhra Pradesh")
    z1, z2, z3 = st.columns(3)
    with z1:
        st.markdown("""
        **1. Madanapalle & Chittoor Belt (Annamayya / Chittoor)**
        - High-elevation plateau (700m AMSL) with mild winter climate.
        - One of the largest vegetable assembly and dispatch markets in southern India.
        - Cauliflower harvest from September to February.
        """)
    with z2:
        st.markdown("""
        **2. Araku Valley & Paderu Hills (Alluri Sitharama Raju)**
        - Eastern Ghats hill range (900–1,200m elevation).
        - Pristine microclimate producing ultra-dense, snow-white curds.
        - Extended cool growing season (August to March).
        """)
    with z3:
        st.markdown("""
        **3. Krishna & Godavari Delta Belts (Guntur, Vijayawada, Rajahmundry)**
        - Rich alluvial soils along river basins.
        - Intensive irrigated winter farming with large head weights (1.5kg - 2.2kg).
        - Direct connectivity to mega wholesale markets in Vijayawada & Visakhapatnam.
        """)

# ==========================================
# TAB 5: DATASET & ANALYTICS
# ==========================================
with tab5:
    st.header("Datasets & Agricultural Analytics")
    st.write("Explore the populated master datasets, correlations, and distributions.")

    dataset_choice = st.selectbox("Select Dataset to Inspect:", [
        "01: Size Dataset (Physical Attributes)",
        "02: Damage Dataset (Pathology & Defects)",
        "03: Quality Dataset (Grading Scores)",
        "04: Price Dataset (APMC Mandi Records)"
    ])

    file_mapping = {
        "01: Size Dataset (Physical Attributes)": "01_cauliflower_size_dataset.csv",
        "02: Damage Dataset (Pathology & Defects)": "02_cauliflower_damage_dataset.csv",
        "03: Quality Dataset (Grading Scores)": "03_cauliflower_quality_dataset.csv",
        "04: Price Dataset (APMC Mandi Records)": "04_cauliflower_price_dataset.csv"
    }

    selected_csv = file_mapping[dataset_choice]
    if os.path.exists(selected_csv):
        df_view = pd.read_csv(selected_csv)
        st.write(f"Showing **{len(df_view)} records** and **{len(df_view.columns)} features**:")
        st.dataframe(df_view.head(25), use_container_width=True)

        st.markdown("### Agricultural Visualizations")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if "01:" in dataset_choice:
            sns.scatterplot(data=df_view, x="diameter_cm", y="weight_g", hue="size", ax=axes[0], palette="viridis")
            axes[0].set_title("Curd Diameter vs Weight by Size Category")
            axes[0].grid(True, linestyle="--", alpha=0.6)

            df_view["size"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=axes[1], colors=["#66bb6a", "#42a5f5", "#ffa726"])
            axes[1].set_ylabel("")
            axes[1].set_title("Size Category Distribution")

        elif "02:" in dataset_choice:
            df_view["condition"].value_counts().plot(kind="bar", ax=axes[0], color=["#4caf50", "#e53935"])
            axes[0].set_title("Healthy vs Damaged Cauliflower Count")
            axes[0].set_ylabel("Count")

            damaged_only = df_view[df_view["condition"] == "Damaged"]
            damaged_only["damage_type"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=axes[1])
            axes[1].set_ylabel("")
            axes[1].set_title("Damage Type Breakdown")

        elif "03:" in dataset_choice:
            sns.histplot(data=df_view, x="color_score_1_10", kde=True, ax=axes[0], color="#2e7d32")
            axes[0].set_title("Distribution of Color Scores (1-10)")

            df_view["quality_grade"].value_counts().plot(kind="bar", ax=axes[1], color=["#388e3c", "#fbc02d", "#d32f2f"])
            axes[1].set_title("Quality Grade Proportions")

        else:
            sns.boxplot(data=df_view, x="market_or_village", y="actual_selling_price_rs_per_kg", ax=axes[0], palette="Set2")
            axes[0].set_title("Selling Price Variation Across Mandis")
            axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)

            sns.histplot(data=df_view, x="actual_selling_price_rs_per_kg", kde=True, ax=axes[1], color="#0288d1")
            axes[1].set_title("Selling Price (Rs/kg) Histogram")

        st.pyplot(fig)
    else:
        st.warning(f"File {selected_csv} not found.")
