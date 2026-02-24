import requests
import pandas as pd
import time

# =====================================================
# CONFIGURATION
# =====================================================

MIN_LAT, MIN_LON = 18.85, 72.75
MAX_LAT, MAX_LON = 19.30, 73.00

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# =====================================================
# OVERPASS QUERY FUNCTION
# =====================================================

def overpass_query(query):
    time.sleep(1)
    response = requests.post(OVERPASS_URL, data=query)
    response.raise_for_status()
    return response.json()

# =====================================================
# BUILD QUERY
# =====================================================

query = f"""
[out:json][timeout:120];
(
  node["tourism"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["tourism"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  relation["tourism"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["historic"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["historic"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["natural"="beach"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["natural"="beach"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["leisure"="park"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["leisure"="park"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["railway"="station"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["railway"="station"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["highway"="bus_stop"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  node["public_transport"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["aeroway"="terminal"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["aeroway"="terminal"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["amenity"~"hospital|university|place_of_worship|marketplace"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["amenity"~"hospital|university|place_of_worship|marketplace"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["shop"="mall"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["shop"="mall"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});

  node["office"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
  way["office"]({MIN_LAT},{MIN_LON},{MAX_LAT},{MAX_LON});
);
out center;
"""

print("Fetching places from OSM...")
data = overpass_query(query)
elements = data.get("elements", [])
print("Raw elements:", len(elements))

# =====================================================
# PARSE RESULTS
# =====================================================

places = []

for el in elements:
    tags = el.get("tags", {})
    name = tags.get("name")

    if not name:
        continue

    # Extract coordinates
    if el["type"] == "node":
        lat = el.get("lat")
        lon = el.get("lon")
    else:
        center = el.get("center")
        if not center:
            continue
        lat = center.get("lat")
        lon = center.get("lon")

    if lat is None or lon is None:
        continue

    # Determine primary tag
    tag = None
    for key in ["tourism", "historic", "natural", "leisure",
                "railway", "highway", "public_transport",
                "aeroway", "amenity", "shop", "office"]:
        if key in tags:
            tag = f"{key}:{tags[key]}"
            break

    if not tag:
        continue

    places.append({
        "name": name.strip(),
        "latitude": lat,
        "longitude": lon,
        "tag": tag
    })

df = pd.DataFrame(places)

# =====================================================
# CLEANING
# =====================================================

df = df.drop_duplicates(subset=["name", "tag"])
df = df[df["name"].str.len() > 2]

print("Cleaned places:", len(df))

df.to_csv("Mumbai Landmarks.csv", index=False)

print("Saved: mumbai_all_places.csv")