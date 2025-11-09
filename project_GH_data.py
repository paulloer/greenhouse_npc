import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta

# === Parameters ===
CSV_PATH = "./online_optimization_and_learning/2025-06-09_GH_Data.csv"  # Change this to your real CSV file
OUTPUT_PATH = "./online_optimization_and_learning/full_hour_data.csv"
INTERVAL_SECONDS = 30
N_BACK = 120  # how many steps to extrapolate backward (1 hour at 30s intervals)
N_FIT = 20    # number of samples to use for linear fit

# === Load and preprocess ===
df = pd.read_csv(CSV_PATH, parse_dates=["Date"])
df.set_index("Date", inplace=True)
df = df.sort_index()

# Interpolate to uniform 30-second intervals
df = df.resample("30s").interpolate().ffill().bfill()

print(f"Available data time span: {(df.index[-1] - df.index[0]).total_seconds() / 60:.1f} minutes")

# If already 1 hour available, save directly
if (df.index[-1] - df.index[0]) >= timedelta(hours=1):
    df.to_csv(OUTPUT_PATH)
    print(f"Full hour of data already available. Saved to {OUTPUT_PATH}.")
    exit()

# === Extrapolation prep ===
projected_times = [df.index[0] - timedelta(seconds=i * INTERVAL_SECONDS) for i in range(N_BACK, 0, -1)]
projected_data = {}

for col in df.columns:
    y_fit_full = df[col].values[:N_FIT]

    # Handle NaNs by masking out invalid values
    mask = ~np.isnan(y_fit_full)
    if mask.sum() < 2:
        print(f"Skipping column '{col}' due to insufficient valid data.")
        projected_data[col] = [np.nan] * N_BACK
        continue

    x_fit = np.arange(N_FIT)[mask].reshape(-1, 1)
    y_fit = y_fit_full[mask]

    model = LinearRegression().fit(x_fit, y_fit)

    x_extrap = np.arange(-N_BACK, 0).reshape(-1, 1)
    y_extrap = model.predict(x_extrap)

    projected_data[col] = y_extrap

# Create projected DataFrame
df_projected = pd.DataFrame(projected_data, index=projected_times)

# Combine and save
df_full = pd.concat([df_projected, df])
df_full = df_full.sort_index()
df_full.to_csv(OUTPUT_PATH)

print(f"Saved extrapolated 1-hour dataset to: {OUTPUT_PATH}")