import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")

# ================= OUTPUT FOLDER =================

OUTPUT_DIR = "graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= SOLAR DATA =================

print("Loading Solar Energy data...")
solar_chunks = []

for chunk in pd.read_csv("Solar_Energy_Generation.csv", chunksize=500_000):
    chunk['timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce')
    chunk = chunk[['timestamp', 'SolarGeneration']].dropna()

    chunk = (
        chunk.set_index('timestamp')
             .resample('1h')
             .mean()
             .dropna()
             .reset_index()
    )

    solar_chunks.append(chunk)

solar = pd.concat(solar_chunks, ignore_index=True)
solar = solar.groupby('timestamp').mean().reset_index()

print("Solar data after resampling:", solar.shape)

# ================= WEATHER DATA =================

print("Loading Weather data...")
weather_chunks = []

weather_cols = [
    'ApparentTemperature',
    'AirTemperature',
    'DewPointTemperature',
    'RelativeHumidity',
    'WindSpeed',
    'WindDirection'
]

for chunk in pd.read_csv("Weather_Data_reordered_all.csv", chunksize=300_000):
    chunk['timestamp'] = pd.to_datetime(chunk['Timestamp'], errors='coerce')
    chunk = chunk[['timestamp'] + weather_cols].dropna()

    chunk = (
        chunk.set_index('timestamp')
             .resample('1h')
             .mean()
             .dropna()
             .reset_index()
    )

    weather_chunks.append(chunk)

weather = pd.concat(weather_chunks, ignore_index=True)
weather = weather.groupby('timestamp').mean().reset_index()

print("Weather data after resampling:", weather.shape)

# ================= MERGE =================

df = pd.merge(solar, weather, on='timestamp', how='inner')
print("Merged dataset shape:", df.shape)

# ================= FEATURES =================

df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour

# ================= DESCRIPTIVE STATS =================

desc_table = df.describe().T.round(3)
print("\nDESCRIPTIVE STATISTICS\n")
print(desc_table)

desc_table.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics_table.csv"))
with open(os.path.join(OUTPUT_DIR, "descriptive_statistics_table.txt"), "w") as f:
    f.write(desc_table.to_string())

# ================= GRAPHS =================

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

df.hist(bins=30, figsize=(14,7))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "histograms.png"))
plt.close()

# ================= MODEL DATA =================

target = 'SolarGeneration'
X = df.drop(columns=[target, 'timestamp'])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# ================= LINEAR REGRESSION =================

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

# ================= RANDOM FOREST =================

rf = RandomForestRegressor(
    n_estimators=50,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

# ================= METRICS =================

metrics = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R2": [r2_score(y_test, lr_preds), r2_score(y_test, rf_preds)],
    "MAE": [mean_absolute_error(y_test, lr_preds), mean_absolute_error(y_test, rf_preds)],
    "MSE": [mean_squared_error(y_test, lr_preds), mean_squared_error(y_test, rf_preds)]
}).round(4)

print("\nMODEL PERFORMANCE\n")
print(metrics)

metrics.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

# ================= MODEL COMPARISON =================

plt.figure(figsize=(12,5))
plt.plot(y_test.values[:200], label="Actual")
plt.plot(lr_preds[:200], label="Linear Regression")
plt.plot(rf_preds[:200], label="Random Forest")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"))
plt.close()

# ================= JANUARY 2026 FORECAST =================

jan = df[df['month'] == 1].copy()
jan['year'] = 2026

X_jan = jan[X.columns]
jan['Forecast_Energy'] = rf.predict(X_jan)

jan_2026_avg = jan['Forecast_Energy'].mean()
print("\nAverage Solar Energy Forecast for January 2026:", jan_2026_avg)

comparison = (
    df[df['month'] == 1]
    .groupby('year')[target]
    .mean()
)

comparison.loc[2026] = jan_2026_avg
comparison.to_csv(os.path.join(OUTPUT_DIR, "january_comparison_table.csv"))

plt.figure(figsize=(8,5))
comparison.plot(marker='o')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "january_comparison.png"))
plt.close()
