"""
ap_locations.py
Comprehensive Directory of Andhra Pradesh Agricultural Mandis, Rythu Bazars, 
and Cauliflower Producing Districts.

Covers all 26 Districts of Andhra Pradesh with benchmark market rates,
market types, and regional agronomic details.
"""

from typing import Dict, List, Any, Optional

ANDHRA_PRADESH_MANDIS: List[Dict[str, Any]] = [
    # --- Rayalaseema Region ---
    {
        "district": "Annamayya",
        "region": "Rayalaseema",
        "mandi_name": "Madanapalle Mandi (Annamayya / Chittoor)",
        "market_type": "Major Wholesale APMC",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "High (Primary Cultivation Hub)",
        "peak_months": "September - February",
        "notes": "Major vegetable hub in South India with cool microclimate; large producer and dispatch hub."
    },
    {
        "district": "Annamayya",
        "region": "Rayalaseema",
        "mandi_name": "Rayachoti Vegetable Market",
        "market_type": "APMC Sub-Market",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - February",
        "notes": "District headquarters market catering to Rayachoti and Rajampet farmers."
    },
    {
        "district": "Chittoor",
        "region": "Rayalaseema",
        "mandi_name": "Chittoor Town APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate to High",
        "peak_months": "October - February",
        "notes": "Direct connectivity with Tamil Nadu borders; high daily trading volume."
    },
    {
        "district": "Chittoor",
        "region": "Rayalaseema",
        "mandi_name": "Palamaner Vegetable Mandi",
        "market_type": "Regional Hub",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "High",
        "peak_months": "September - January",
        "notes": "Major horticulture pocket with abundant cauliflower and cabbage acreage."
    },
    {
        "district": "Chittoor",
        "region": "Rayalaseema",
        "mandi_name": "Kuppam Horticulture Mandi",
        "market_type": "Border Market Yard",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "High",
        "peak_months": "October - March",
        "notes": "Supplies extensively to Bangalore and Chennai wholesale corridors."
    },
    {
        "district": "Tirupati",
        "region": "Rayalaseema",
        "mandi_name": "Tirupati RC Road Rythu Bazar",
        "market_type": "Government Rythu Bazar",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Consumer Hub (Sourced from Madanapalle/Palamaner)",
        "peak_months": "Year-round demand",
        "notes": "Direct farmer-to-consumer pricing with heavy pilgrim floating demand."
    },
    {
        "district": "Tirupati",
        "region": "Rayalaseema",
        "mandi_name": "Srikalahasti APMC Market",
        "market_type": "Semi-Wholesale",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "Low to Moderate",
        "peak_months": "November - February",
        "notes": "Key market along the Swarnamukhi river basin."
    },
    {
        "district": "Kurnool",
        "region": "Rayalaseema",
        "mandi_name": "Kurnool APMC Market (C-Camp)",
        "market_type": "Major Wholesale APMC",
        "min_price": 19.0,
        "modal_price": 25.0,
        "max_price": 31.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - January",
        "notes": "Major gateway for Rayalaseema and Hyderabad connecting supplies."
    },
    {
        "district": "Kurnool",
        "region": "Rayalaseema",
        "mandi_name": "Adoni Vegetable APMC",
        "market_type": "Wholesale APMC",
        "min_price": 18.0,
        "modal_price": 24.0,
        "max_price": 30.0,
        "cauliflower_production": "Low to Moderate",
        "peak_months": "November - January",
        "notes": "Large commercial mandi serving western Kurnool and border regions."
    },
    {
        "district": "Nandyal",
        "region": "Rayalaseema",
        "mandi_name": "Nandyal APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - February",
        "notes": "Fertile Kundu river valley; expanding vegetable cultivation."
    },
    {
        "district": "Nandyal",
        "region": "Rayalaseema",
        "mandi_name": "Allagadda Rythu Bazar",
        "market_type": "Rythu Bazar",
        "min_price": 19.0,
        "modal_price": 25.0,
        "max_price": 31.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Direct marketing facility for local vegetable growers."
    },
    {
        "district": "Ananthapuramu",
        "region": "Rayalaseema",
        "mandi_name": "Anantapur APMC Mandi",
        "market_type": "Major Wholesale APMC",
        "min_price": 18.0,
        "modal_price": 24.0,
        "max_price": 30.0,
        "cauliflower_production": "Moderate (Borewell Irrigated)",
        "peak_months": "October - January",
        "notes": "Central distribution market for dryland horticulture produce."
    },
    {
        "district": "Ananthapuramu",
        "region": "Rayalaseema",
        "mandi_name": "Dharmavaram Vegetable Market",
        "market_type": "Local APMC Yard",
        "min_price": 19.0,
        "modal_price": 25.0,
        "max_price": 31.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Active daily vegetable trading hub in southern Anantapur."
    },
    {
        "district": "Sri Sathya Sai",
        "region": "Rayalaseema",
        "mandi_name": "Hindupur APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate to High",
        "peak_months": "September - February",
        "notes": "Close proximity to Karnataka; high transit trade for cole crops."
    },
    {
        "district": "Sri Sathya Sai",
        "region": "Rayalaseema",
        "mandi_name": "Kadiri Vegetable Market",
        "market_type": "Sub-Market Yard",
        "min_price": 18.0,
        "modal_price": 24.0,
        "max_price": 30.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - January",
        "notes": "Serves eastern belts of Sri Sathya Sai district."
    },
    {
        "district": "YSR Kadapa",
        "region": "Rayalaseema",
        "mandi_name": "Kadapa APMC Mandi",
        "market_type": "Wholesale APMC",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - February",
        "notes": "Central Rayalaseema transit market with solid cold-storage facilities."
    },
    {
        "district": "YSR Kadapa",
        "region": "Rayalaseema",
        "mandi_name": "Proddatur Vegetable Market",
        "market_type": "Commercial Market",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Major consumer and commercial trading center in Kadapa basin."
    },

    # --- Coastal Andhra (Central & South) ---
    {
        "district": "NTR",
        "region": "Coastal Andhra",
        "mandi_name": "Vijayawada Gollapudi APMC (NTR / Krishna)",
        "market_type": "Super Wholesale APMC",
        "min_price": 25.0,
        "modal_price": 32.0,
        "max_price": 38.0,
        "cauliflower_production": "Massive Consumption & Re-distribution Hub",
        "peak_months": "Year-round (Peak Oct-Mar)",
        "notes": "One of AP's largest wholesale markets; sets benchmark pricing across coastal districts."
    },
    {
        "district": "NTR",
        "region": "Coastal Andhra",
        "mandi_name": "Vijayawada Swaraj Maidan Rythu Bazar",
        "market_type": "Model Rythu Bazar",
        "min_price": 26.0,
        "modal_price": 33.0,
        "max_price": 40.0,
        "cauliflower_production": "Consumer Retail Rythu Bazar",
        "peak_months": "Year-round",
        "notes": "Pioneer Rythu Bazar with computerized electronic weighing and regulated pricing."
    },
    {
        "district": "Krishna",
        "region": "Coastal Andhra",
        "mandi_name": "Machilipatnam APMC Market",
        "market_type": "Coastal APMC Yard",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Headquarters port city market receiving produce from Krishna deltas."
    },
    {
        "district": "Krishna",
        "region": "Coastal Andhra",
        "mandi_name": "Gudivada Vegetable Market",
        "market_type": "APMC Sub-Market",
        "min_price": 24.0,
        "modal_price": 30.0,
        "max_price": 36.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Active commercial market linking Vijayawada and Bhimavaram corridors."
    },
    {
        "district": "Guntur",
        "region": "Coastal Andhra",
        "mandi_name": "Guntur APMC Market (Guntur)",
        "market_type": "Super Wholesale APMC",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "High (Extensive River-bank cultivation)",
        "peak_months": "October - March",
        "notes": "Mega agricultural hub; Krishna river alluvium allows high yield cauliflower farming."
    },
    {
        "district": "Guntur",
        "region": "Coastal Andhra",
        "mandi_name": "Tenali Vegetable Rythu Bazar",
        "market_type": "Rythu Bazar",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "High (Canal irrigated)",
        "peak_months": "October - February",
        "notes": "Bustling delta market with dense surrounding farmer clusters."
    },
    {
        "district": "Palnadu",
        "region": "Coastal Andhra",
        "mandi_name": "Narasaraopet APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Commercial center for upland Palnadu farmers."
    },
    {
        "district": "Palnadu",
        "region": "Coastal Andhra",
        "mandi_name": "Piduguralla Vegetable Mandi",
        "market_type": "Local APMC Yard",
        "min_price": 19.0,
        "modal_price": 25.0,
        "max_price": 31.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - January",
        "notes": "Gateway trading town between Guntur and Telangana border."
    },
    {
        "district": "Bapatla",
        "region": "Coastal Andhra",
        "mandi_name": "Bapatla APMC Market",
        "market_type": "Coastal APMC",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Home to Agricultural College; adopts modern horticulture varieties."
    },
    {
        "district": "Bapatla",
        "region": "Coastal Andhra",
        "mandi_name": "Chirala Vegetable Market",
        "market_type": "Town Market",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "High local consumer demand in densely populated coastal belt."
    },
    {
        "district": "Prakasam",
        "region": "Coastal Andhra",
        "mandi_name": "Ongole APMC Market (Prakasam)",
        "market_type": "Wholesale APMC",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - February",
        "notes": "Headquarters market connecting Chennai-Kolkata NH-16 highway."
    },
    {
        "district": "Prakasam",
        "region": "Coastal Andhra",
        "mandi_name": "Markapur Vegetable Mandi",
        "market_type": "Semi-Wholesale",
        "min_price": 19.0,
        "modal_price": 25.0,
        "max_price": 31.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - January",
        "notes": "Serves western Prakasam foothills."
    },
    {
        "district": "SPSR Nellore",
        "region": "Coastal Andhra",
        "mandi_name": "Nellore Vegetable Market (SPSR Nellore)",
        "market_type": "Major Wholesale APMC",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Moderate to High",
        "peak_months": "October - February",
        "notes": "Large trade volume; Penna river delta cultivation."
    },
    {
        "district": "SPSR Nellore",
        "region": "Coastal Andhra",
        "mandi_name": "Gudur Vegetable Market",
        "market_type": "Sub-Market",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Key transport junction for South Coastal Andhra."
    },

    # --- Godavari & North Coastal Region ---
    {
        "district": "East Godavari",
        "region": "Coastal Andhra",
        "mandi_name": "Rajahmundry APMC (East Godavari)",
        "market_type": "Major Wholesale APMC",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "High (Godavari Delta & River Lanka Beds)",
        "peak_months": "October - March",
        "notes": "Famous fertile Godavari alluvial soils; intensive winter vegetable production."
    },
    {
        "district": "Kakinada",
        "region": "Coastal Andhra",
        "mandi_name": "Kakinada Main Rythu Bazar",
        "market_type": "Major Rythu Bazar",
        "min_price": 24.0,
        "modal_price": 31.0,
        "max_price": 37.0,
        "cauliflower_production": "Consumer Hub",
        "peak_months": "Year-round",
        "notes": "Port city market receiving cauliflower from Agency hills and Godavari plains."
    },
    {
        "district": "Dr. B.R. Ambedkar Konaseema",
        "region": "Coastal Andhra",
        "mandi_name": "Amalapuram APMC Yard",
        "market_type": "Delta APMC",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Core island delta market with rich organic soil conditions."
    },
    {
        "district": "West Godavari",
        "region": "Coastal Andhra",
        "mandi_name": "Bhimavaram APMC Market",
        "market_type": "Commercial APMC",
        "min_price": 24.0,
        "modal_price": 30.0,
        "max_price": 36.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "High commercial purchasing power; steady cold storage supply."
    },
    {
        "district": "West Godavari",
        "region": "Coastal Andhra",
        "mandi_name": "Tadepalligudem Vegetable Market",
        "market_type": "Major Wholesale Trading Hub",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "High (Horticulture belt)",
        "peak_months": "October - February",
        "notes": "One of AP's busiest transit mandis for onions and seasonal vegetables."
    },
    {
        "district": "Eluru",
        "region": "Coastal Andhra",
        "mandi_name": "Eluru APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 22.0,
        "modal_price": 28.0,
        "max_price": 34.0,
        "cauliflower_production": "Moderate to High",
        "peak_months": "October - February",
        "notes": "Centrally located market between Krishna and Godavari districts."
    },
    {
        "district": "Eluru",
        "region": "Coastal Andhra",
        "mandi_name": "Jangareddygudem Market",
        "market_type": "Upland Market Yard",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Key agency border trading town."
    },
    {
        "district": "Visakhapatnam",
        "region": "North Coastal",
        "mandi_name": "Visakhapatnam MVP Rythu Bazar (Visakhapatnam)",
        "market_type": "Flagship Rythu Bazar",
        "min_price": 26.0,
        "modal_price": 34.0,
        "max_price": 41.0,
        "cauliflower_production": "Major Metropolitan Consumer Market",
        "peak_months": "Year-round demand",
        "notes": "Highest premium realization in AP for Grade A curd; supplies drawn from Araku and Agency belts."
    },
    {
        "district": "Visakhapatnam",
        "region": "North Coastal",
        "mandi_name": "Gajuwaka Industrial Rythu Bazar",
        "market_type": "Rythu Bazar",
        "min_price": 25.0,
        "modal_price": 32.0,
        "max_price": 39.0,
        "cauliflower_production": "Urban Consumption",
        "peak_months": "Year-round",
        "notes": "Caters to steel plant and industrial worker townships."
    },
    {
        "district": "Alluri Sitharama Raju (ASR)",
        "region": "North Coastal",
        "mandi_name": "Araku Valley Horticulture Hub (ASR District)",
        "market_type": "Hill Station Production & Farmer Market",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 36.0,
        "cauliflower_production": "High (Pristine Hill Climate Cultivation)",
        "peak_months": "August - March (Extended season due to elevation)",
        "notes": "Cool climate 900m above sea level creates dense, ultra-white curds with minimal pesticide use."
    },
    {
        "district": "Alluri Sitharama Raju (ASR)",
        "region": "North Coastal",
        "mandi_name": "Paderu Agency Market",
        "market_type": "Tribal / Agency Market",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate to High",
        "peak_months": "September - February",
        "notes": "Primary collection center for tribal horticulture produce."
    },
    {
        "district": "Anakapalli",
        "region": "North Coastal",
        "mandi_name": "Anakapalle APMC Yard",
        "market_type": "Major APMC",
        "min_price": 23.0,
        "modal_price": 29.0,
        "max_price": 35.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Major commercial market just south of Visakhapatnam city."
    },
    {
        "district": "Vizianagaram",
        "region": "North Coastal",
        "mandi_name": "Vizianagaram APMC Market",
        "market_type": "Wholesale APMC",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate",
        "peak_months": "October - February",
        "notes": "Large vegetable supply base for northern coastal belt."
    },
    {
        "district": "Vizianagaram",
        "region": "North Coastal",
        "mandi_name": "Bobbili Vegetable Market",
        "market_type": "Sub-Market",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Northern interior trading center."
    },
    {
        "district": "Parvathipuram Manyam",
        "region": "North Coastal",
        "mandi_name": "Parvathipuram APMC Market",
        "market_type": "Semi-Wholesale",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Bordering Odisha; collects produce from Nagavali river plain."
    },
    {
        "district": "Srikakulam",
        "region": "North Coastal",
        "mandi_name": "Srikakulam APMC Market Yard",
        "market_type": "Wholesale APMC",
        "min_price": 21.0,
        "modal_price": 27.0,
        "max_price": 33.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Northernmost coastal district headquarters market."
    },
    {
        "district": "Srikakulam",
        "region": "North Coastal",
        "mandi_name": "Amadalavalasa Vegetable Market",
        "market_type": "Local APMC",
        "min_price": 20.0,
        "modal_price": 26.0,
        "max_price": 32.0,
        "cauliflower_production": "Moderate",
        "peak_months": "November - February",
        "notes": "Rail-connected trading town for North Coastal farmers."
    }
]

def get_all_ap_mandis() -> List[Dict[str, Any]]:
    """Return all Andhra Pradesh mandis with detailed attributes."""
    return ANDHRA_PRADESH_MANDIS

def get_all_districts() -> List[str]:
    """Return sorted unique list of all Andhra Pradesh districts covered."""
    return sorted(list(set(m["district"] for m in ANDHRA_PRADESH_MANDIS)))

def get_mandis_by_district(district_name: str) -> List[Dict[str, Any]]:
    """Filter mandis by a given Andhra Pradesh district."""
    return [m for m in ANDHRA_PRADESH_MANDIS if m["district"].lower() == district_name.lower()]

def get_mandi_names() -> List[str]:
    """Return formatted list of all mandi names."""
    return [m["mandi_name"] for m in ANDHRA_PRADESH_MANDIS]

def get_mandi_price_dict() -> Dict[str, tuple]:
    """Return lookup dictionary mapping Mandi Name -> (min_price, modal_price, max_price)."""
    return {
        m["mandi_name"]: (m["min_price"], m["modal_price"], m["max_price"])
        for m in ANDHRA_PRADESH_MANDIS
    }

def get_mandi_details(mandi_name: str) -> Optional[Dict[str, Any]]:
    """Get full details for a specific Andhra Pradesh mandi."""
    for m in ANDHRA_PRADESH_MANDIS:
        if m["mandi_name"] == mandi_name:
            return m
    return None
