"""
generate_datasets.py
Populates the 4 blank cauliflower dataset CSV templates with realistic, correlated agricultural data.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ap_locations import ANDHRA_PRADESH_MANDIS

def generate_datasets(random_seed=42):
    np.random.seed(random_seed)
    random.seed(random_seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    f_size = os.path.join(base_dir, "01_cauliflower_size_dataset.csv")
    f_damage = os.path.join(base_dir, "02_cauliflower_damage_dataset.csv")
    f_quality = os.path.join(base_dir, "03_cauliflower_quality_dataset.csv")
    f_price = os.path.join(base_dir, "04_cauliflower_price_dataset.csv")

    n_samples = 1000
    sample_ids = [f"CF{i:04d}" for i in range(1, n_samples + 1)]
    image_names = [f"CF{i:04d}.jpg" for i in range(1, n_samples + 1)]

    farmers = [f"FARM_{i:03d}" for i in range(1, 41)]
    start_date = datetime(2024, 1, 10)

    # Pick random AP Mandis from comprehensive AP database
    sampled_mandis = [random.choice(ANDHRA_PRADESH_MANDIS) for _ in range(n_samples)]
    market_list = [m["mandi_name"] for m in sampled_mandis]
    farmer_ids = [random.choice(farmers) for _ in range(n_samples)]
    collection_dates = [(start_date + timedelta(days=random.randint(0, 75))).strftime("%Y-%m-%d") for _ in range(n_samples)]

    diameters = []
    heights = []
    weights = []
    sizes = []

    for _ in range(n_samples):
        tier = np.random.choice(["Small", "Medium", "Large"], p=[0.28, 0.47, 0.25])
        if tier == "Small":
            d = round(float(np.random.normal(11.5, 1.2)), 1)
            d = max(8.5, min(13.4, d))
            h = round(float(np.random.normal(10.0, 1.0)), 1)
            h = max(7.0, min(12.5, h))
            w = round(float(d * h * np.random.uniform(4.0, 5.0) + np.random.normal(0, 30)), 1)
            w = max(300.0, min(695.0, w))
            size_label = "Small"
        elif tier == "Medium":
            d = round(float(np.random.normal(16.0, 1.4)), 1)
            d = max(13.5, min(18.9, d))
            h = round(float(np.random.normal(14.0, 1.2)), 1)
            h = max(11.0, min(17.0, h))
            w = round(float(d * h * np.random.uniform(4.2, 5.2) + np.random.normal(0, 45)), 1)
            w = max(700.0, min(1345.0, w))
            size_label = "Medium"
        else: # Large
            d = round(float(np.random.normal(21.0, 1.5)), 1)
            d = max(19.0, min(26.0, d))
            h = round(float(np.random.normal(17.5, 1.3)), 1)
            h = max(15.0, min(22.5, h))
            w = round(float(d * h * np.random.uniform(4.4, 5.5) + np.random.normal(0, 60)), 1)
            w = max(1350.0, min(2400.0, w))
            size_label = "Large"

        diameters.append(d)
        heights.append(h)
        weights.append(w)
        sizes.append(size_label)

    # 2. Damage & Condition
    conditions = []
    severities = []
    damage_types = []
    spots_counts = []

    disease_choices = ["Bacterial Spot Rot", "Black Rot", "Downy Mildew", "Physical Damage"]

    for _ in range(n_samples):
        is_healthy = np.random.rand() > 0.38
        if is_healthy:
            conditions.append("Healthy")
            severities.append("None")
            damage_types.append("None")
            spots_counts.append(int(np.random.choice([0, 1, 2], p=[0.75, 0.20, 0.05])))
        else:
            conditions.append("Damaged")
            sev = np.random.choice(["Low", "Medium", "High"], p=[0.40, 0.35, 0.25])
            severities.append(sev)
            damage_types.append(random.choice(disease_choices))
            if sev == "Low":
                spots = int(np.random.randint(3, 10))
            elif sev == "Medium":
                spots = int(np.random.randint(10, 25))
            else:
                spots = int(np.random.randint(25, 60))
            spots_counts.append(spots)

    # 3. Quality scores (1 to 10)
    color_scores = []
    firmness_scores = []
    leaf_scores = []
    quality_grades = []

    for i in range(n_samples):
        cond = conditions[i]
        sev = severities[i]
        if cond == "Healthy":
            c = round(float(np.random.uniform(8.0, 10.0)), 1)
            f = round(float(np.random.uniform(8.0, 10.0)), 1)
            l = round(float(np.random.uniform(7.5, 10.0)), 1)
        else:
            if sev == "Low":
                c = round(float(np.random.uniform(6.0, 7.8)), 1)
                f = round(float(np.random.uniform(6.0, 7.8)), 1)
                l = round(float(np.random.uniform(5.5, 7.5)), 1)
            elif sev == "Medium":
                c = round(float(np.random.uniform(4.0, 6.2)), 1)
                f = round(float(np.random.uniform(4.0, 6.2)), 1)
                l = round(float(np.random.uniform(3.5, 5.8)), 1)
            else: # High
                c = round(float(np.random.uniform(1.5, 4.2)), 1)
                f = round(float(np.random.uniform(1.5, 4.2)), 1)
                l = round(float(np.random.uniform(1.0, 3.8)), 1)

        color_scores.append(c)
        firmness_scores.append(f)
        leaf_scores.append(l)

        composite = 0.4 * c + 0.35 * f + 0.25 * l - (0.05 * spots_counts[i])
        if composite >= 7.5 and cond == "Healthy":
            quality_grades.append("Grade A")
        elif composite >= 5.0:
            quality_grades.append("Grade B")
        else:
            quality_grades.append("Grade C")

    # 4. Mandi & Market Prices
    quantities = []
    mandi_mins = []
    mandi_modals = []
    mandi_maxs = []
    actual_prices = []
    price_sources = []
    notes_list = []

    for i in range(n_samples):
        m_item = sampled_mandis[i]
        base_modal = m_item["modal_price"] + np.random.normal(0, 1.5)
        base_modal = max(16.0, round(float(base_modal), 1))

        p_min = round(float(base_modal - np.random.uniform(4.0, 6.0)), 1)
        p_min = max(10.0, p_min)

        p_max = round(float(base_modal + np.random.uniform(4.0, 7.0)), 1)

        grade = quality_grades[i]
        sz = sizes[i]

        premium = 0.0
        if grade == "Grade A":
            premium += np.random.uniform(2.0, 4.5)
        elif grade == "Grade B":
            premium += np.random.uniform(-1.0, 1.5)
        else:
            premium += np.random.uniform(-5.0, -2.0)

        if sz == "Large":
            premium += 1.5
        elif sz == "Small":
            premium -= 1.5

        actual_p = round(float(base_modal + premium + np.random.normal(0, 0.8)), 1)
        actual_p = max(8.0, actual_p)

        qty = int(np.random.randint(150, 2500))

        quantities.append(qty)
        mandi_mins.append(p_min)
        mandi_modals.append(base_modal)
        mandi_maxs.append(p_max)
        actual_prices.append(actual_p)
        price_sources.append("Andhra Pradesh Marketing Dept / Agmarknet")
        notes_list.append(f"{grade} cauliflower batch from {farmer_ids[i]}")

    df_size = pd.DataFrame({
        "sample_id": sample_ids,
        "image_name": image_names,
        "farmer_or_batch_id": farmer_ids,
        "collection_date": collection_dates,
        "market_or_village": market_list,
        "weight_g": weights,
        "diameter_cm": diameters,
        "height_cm": heights,
        "size": sizes
    })

    df_damage = pd.DataFrame({
        "sample_id": sample_ids,
        "image_name": image_names,
        "farmer_or_batch_id": farmer_ids,
        "collection_date": collection_dates,
        "condition": conditions,
        "damage_severity": severities,
        "damage_type": damage_types,
        "spots_count": spots_counts
    })

    df_quality = pd.DataFrame({
        "sample_id": sample_ids,
        "image_name": image_names,
        "farmer_or_batch_id": farmer_ids,
        "collection_date": collection_dates,
        "condition": conditions,
        "color_score_1_10": color_scores,
        "firmness_score_1_10": firmness_scores,
        "spots_count": spots_counts,
        "leaf_condition_score_1_10": leaf_scores,
        "quality_grade": quality_grades
    })

    df_price = pd.DataFrame({
        "sample_id": sample_ids,
        "image_name": image_names,
        "farmer_or_batch_id": farmer_ids,
        "collection_date": collection_dates,
        "market_or_village": market_list,
        "quantity_sold_kg": quantities,
        "actual_selling_price_rs_per_kg": actual_prices,
        "mandi": market_list,
        "mandi_date": collection_dates,
        "mandi_min_rs_per_kg": mandi_mins,
        "mandi_modal_rs_per_kg": mandi_modals,
        "mandi_max_rs_per_kg": mandi_maxs,
        "price_source": price_sources,
        "notes": notes_list
    })

    df_size.to_csv(f_size, index=False)
    df_damage.to_csv(f_damage, index=False)
    df_quality.to_csv(f_quality, index=False)
    df_price.to_csv(f_price, index=False)

    print("Successfully populated all 4 CSV datasets with 1,000 realistic agricultural records!")
    print(f"Size dataset: {df_size.shape}")
    print(f"Damage dataset: {df_damage.shape}")
    print(f"Quality dataset: {df_quality.shape}")
    print(f"Price dataset: {df_price.shape}")

if __name__ == "__main__":
    generate_datasets()
