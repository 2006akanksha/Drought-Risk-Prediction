import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ==========================================
# 1. LOAD DATA & SORT CHRONOLOGICALLY
# ==========================================
print("Loading dataset...")
df = pd.read_csv("india_drought_daily_5locations_2000_2026.csv")

# Create a combined datetime column
df["DATE"] = pd.to_datetime(
    df["YEAR"].astype(str)
    + "-"
    + df["MO"].astype(str)
    + "-"
    + df["DY"].astype(str)
)

# Sort chronologically for each location
df = df.sort_values(by=["Location", "DATE"]).reset_index(drop=True)

# ==========================================
# 2. FEATURE ENGINEERING (ROLLING METRICS)
# ==========================================
print("Computing 7-day and 30-day rolling weather features...")

df["rain_7d_sum"] = df.groupby("Location")["PRECTOTCORR"].transform(
    lambda x: x.rolling(7, min_periods=1).sum()
)
df["rain_30d_sum"] = df.groupby("Location")["PRECTOTCORR"].transform(
    lambda x: x.rolling(30, min_periods=1).sum()
)
df["temp_7d_avg"] = df.groupby("Location")["T2M"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
df["soil_7d_avg"] = df.groupby("Location")["GWETROOT"].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)

# ==========================================
# 3. DEFINE BASE DROUGHT RISK RULE
# 0 = Low Risk, 1 = Moderate Risk, 2 = High Risk
# ==========================================
conditions = [
    (df["GWETROOT"] < 0.35) & (df["rain_30d_sum"] < 10),  # High Risk
    (df["GWETROOT"] < 0.45) | (df["rain_30d_sum"] < 30),  # Moderate Risk
]
choices = [2, 1]
df["Drought_Risk_Current"] = np.select(conditions, choices, default=0)

# ==========================================
# 4. DEFINE INPUT FEATURES & FORECAST HORIZONS
# ==========================================
feature_cols = [
    "PRECTOTCORR",
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "GWETROOT",
    "rain_7d_sum",
    "rain_30d_sum",
    "temp_7d_avg",
    "soil_7d_avg",
]

# Forecast horizons (shift days into the future)
horizons = {
    "Current": 0,
    "7_Days": 7,
    "15_Days": 15,
    "30_Days": 30,
    "60_Days": 60,
}

# Create folder to store saved models for web deployment
os.makedirs("models", exist_ok=True)
target_names = ["Low Risk (0)", "Moderate Risk (1)", "High Risk (2)"]

# ==========================================
# 5. TRAIN & SAVE MODELS FOR EACH HORIZON
# ==========================================
for horizon_name, shift_days in horizons.items():
    print("\n" + "=" * 60)
    print(f" TRAINING MODEL FOR HORIZON: {horizon_name.upper()} ")
    print("=" * 60)

    if shift_days == 0:
        df[f"Target_{horizon_name}"] = df["Drought_Risk_Current"]
    else:
        # Shift target into the future per location
        df[f"Target_{horizon_name}"] = df.groupby("Location")[
            "Drought_Risk_Current"
        ].shift(-shift_days)

    # Clean missing values resulting from time-shifting at the end of the series
    clean_df = df.dropna(subset=[f"Target_{horizon_name}"]).copy()

    X = clean_df[feature_cols]
    y = clean_df[f"Target_{horizon_name}"]

    # 80% Train, 20% Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate Model
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"Model Accuracy ({horizon_name}): {accuracy:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Save trained model file to disk (.joblib)
    model_filename = f"models/drought_model_{horizon_name.lower()}.joblib"
    joblib.dump(rf, model_filename)
    print(f"Successfully saved model to '{model_filename}'")

print("\n" + "=" * 60)
print("ALL 5 MODELS TRAINED & SAVED SUCCESSFULLY IN 'models/' FOLDER!")
print("=" * 60)