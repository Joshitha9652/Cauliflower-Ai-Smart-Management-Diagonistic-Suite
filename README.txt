CAULIFLOWER SEPARATE DATASETS

All four CSV files contain sample_id and image_name.
Use sample_id as the common key to connect records across datasets.

01_cauliflower_size_dataset.csv
- Used for Small/Medium/Large size classification.
- Record weight, diameter and height.

02_cauliflower_damage_dataset.csv
- Used for damage detection/classification.
- Record condition, damage severity, damage type and spots.

03_cauliflower_quality_dataset.csv
- Used for cauliflower quality grading.
- Record color, firmness, spots and leaf condition.

04_cauliflower_price_dataset.csv
- Used for selling-price/mandi-price analysis or prediction.
- Record quantity sold, actual selling price and mandi min/modal/max prices.

IMPORTANT:
The uploaded master file currently contains 1,000 sample_id/image_name rows,
but the other fields are blank. These files therefore start as data-collection
templates rather than completed real-world datasets.
