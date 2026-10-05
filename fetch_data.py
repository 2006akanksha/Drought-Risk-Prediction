from datetime import datetime
import pandas as pd
import requests

locations = {
    "Latur_Marathwada_High": {"lat": 18.40, "lon": 76.57, "drought_type": "High"},
    "Anantapur_Rayalaseema_High": {
        "lat": 14.68,
        "lon": 77.60,
        "drought_type": "High",
    },
    "Varanasi_UP_Moderate": {
        "lat": 25.33,
        "lon": 83.00,
        "drought_type": "Moderate",
    },
    "Bengaluru_KA_Moderate": {
        "lat": 12.97,
        "lon": 77.59,
        "drought_type": "Moderate",
    },
    "Dakshina_Kannada_Low": {
        "lat": 12.87,
        "lon": 74.88,
        "drought_type": "Low_Coastal",
    },
}

START_DATE = "20000101"
END_DATE = datetime.now().strftime("%Y%m%d")

all_location_data = []

for loc_name, details in locations.items():
  print(f"Fetching daily data for {loc_name} (2000-2026)...")

  # Use JSON format endpoint for clean parsing
  url = "https://power.larc.nasa.gov/api/temporal/daily/point"
  params = {
      "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,GWETROOT",
      "community": "AG",
      "longitude": details["lon"],
      "latitude": details["lat"],
      "start": START_DATE,
      "end": END_DATE,
      "format": "JSON",
  }

  response = requests.get(url, params=params)
  data = response.json()

  # Extract parameter dictionaries from JSON response
  parameter_data = data["properties"]["parameter"]

  # Convert dictionary output to DataFrame
  df = pd.DataFrame(parameter_data)

  # Reset index (which is YYYYMMDD string format) into date columns
  df.reset_index(inplace=True)
  df.rename(columns={"index": "DATE_STR"}, inplace=True)

  # Parse Date strings into YEAR, MO, DY
  df["DATE"] = pd.to_datetime(df["DATE_STR"], format="%Y%m%d")
  df["YEAR"] = df["DATE"].dt.year
  df["MO"] = df["DATE"].dt.month
  df["DY"] = df["DATE"].dt.day

  # Add Metadata tag columns
  df["Location"] = loc_name
  df["Drought_Risk_Category"] = details["drought_type"]

  # Reorder key columns
  cols_order = [
      "YEAR",
      "MO",
      "DY",
      "PRECTOTCORR",
      "T2M",
      "T2M_MAX",
      "T2M_MIN",
      "GWETROOT",
      "Location",
      "Drought_Risk_Category",
  ]
  df = df[cols_order]

  all_location_data.append(df)

# Concatenate all 5 locations into one Master DataFrame
master_df = pd.concat(all_location_data, ignore_index=True)

# Save Master CSV file
csv_filename = "india_drought_daily_5locations_2000_2026.csv"
master_df.to_csv(csv_filename, index=False)

print("\n" + "=" * 50)
print(f"SUCCESS! Master Dataset generated: {csv_filename}")
print(f"Total Rows Fetched: {len(master_df)}")
print("\nFirst 5 rows preview:")
print(master_df.head())
print("=" * 50)