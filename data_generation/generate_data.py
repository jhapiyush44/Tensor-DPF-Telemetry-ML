import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path


# =================================================
# Config
# =================================================

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

N_VEHICLES = 30
DAYS = 90
FREQ = "1min"
TRIP_DURATION = 60  # minutes

rng = np.random.default_rng(42)  # modern reproducible RNG


# =================================================
# Helper: simulate one vehicle
# =================================================

def simulate_vehicle(vehicle_id: int, timestamps: pd.DatetimeIndex):

    n = len(timestamps)

    # -------------------------
    # Base correlated signals
    # -------------------------

    speed = np.clip(rng.normal(45, 20, n), 0, 110)
    rpm = speed * 35 + rng.normal(0, 200, n)
    engine_load = np.clip(speed / 110 + rng.normal(0, 0.1, n), 0, 1)

    ambient_temp = rng.normal(30, 5, n)

    exhaust_temp_pre = 200 + 300 * engine_load + rng.normal(0, 15, n)
    exhaust_temp_post = exhaust_temp_pre - rng.normal(15, 5, n)

    flow_rate = speed * 0.8 + rng.normal(0, 2, n)

    # -------------------------
    # Soot simulation
    # -------------------------

    soot = np.zeros(n)
    maintenance_events = []

    for i in range(1, n):

        delta = 0.02 * engine_load[i]

        if exhaust_temp_pre[i] > 500:  # passive regen
            delta -= 0.015

        soot[i] = max(0, soot[i - 1] + delta)

        if soot[i] > 1.0:  # active regen
            soot[i] *= 0.1
            maintenance_events.append(timestamps[i])

    diff_pressure = soot * 50 + rng.normal(0, 1, n)

    # =================================================
    # TELEMETRY
    # =================================================

    telemetry = pd.DataFrame({
        "vehicle_id": vehicle_id,
        "timestamp": timestamps,
        "engine_load": engine_load,
        "rpm": rpm,
        "speed": speed,
        "exhaust_temp_pre": exhaust_temp_pre,
        "exhaust_temp_post": exhaust_temp_post,
        "flow_rate": flow_rate,
        "ambient_temp": ambient_temp,
        "diff_pressure": diff_pressure,
        "soot_load": soot
    })

    float_cols = telemetry.select_dtypes("float64").columns
    telemetry[float_cols] = telemetry[float_cols].astype("float32")
    telemetry["vehicle_id"] = telemetry["vehicle_id"].astype("int16")

    # =================================================
    # MAINTENANCE
    # =================================================

    maintenance = pd.DataFrame({
        "vehicle_id": [vehicle_id] * len(maintenance_events),
        "maintenance_time": maintenance_events,
        "action_type": ["active_regen"] * len(maintenance_events)
    })

    # =================================================
    # TRIPS ⭐ (IMPORTANT FOR FEATURE PIPELINE)
    # =================================================

    trips = []

    for i in range(0, n, TRIP_DURATION):

        end_idx = min(i + TRIP_DURATION, n)
        chunk = slice(i, end_idx)

        trip_speed = speed[chunk]

        duration_min = end_idx - i
        distance_km = trip_speed.mean() * (duration_min / 60)

        stop_count = int((trip_speed < 5).sum())

        if stop_count > 20:
            pattern = "city"
        elif trip_speed.mean() > 70:
            pattern = "highway"
        else:
            pattern = "mixed"

        trips.append({
            "vehicle_id": vehicle_id,
            "trip_start": timestamps[i],
            "trip_end": timestamps[end_idx - 1],
            "distance_km": distance_km,      # ⭐ standardized name
            "stop_count": stop_count,
            "driving_pattern": pattern
        })

    trips_df = pd.DataFrame(trips)

    return telemetry, maintenance, trips_df


# =================================================
# Main
# =================================================

def main():

    print("Generating synthetic fleet data...\n")

    start = datetime(2025, 1, 1)

    timestamps = pd.date_range(
        start=start,
        periods=DAYS * 24 * 60,
        freq=FREQ
    )

    telemetry_all = []
    maint_all = []
    trip_all = []

    for vid in range(N_VEHICLES):
        print(f"Simulating vehicle {vid}...")

        t_df, m_df, trip_df = simulate_vehicle(vid, timestamps)

        telemetry_all.append(t_df)
        maint_all.append(m_df)
        trip_all.append(trip_df)

    telemetry = pd.concat(telemetry_all, ignore_index=True)
    maintenance = pd.concat(maint_all, ignore_index=True)
    trips = pd.concat(trip_all, ignore_index=True)

    telemetry.to_csv(OUTPUT_DIR / "telemetry.csv", index=False)
    maintenance.to_csv(OUTPUT_DIR / "maintenance.csv", index=False)
    trips.to_csv(OUTPUT_DIR / "trips.csv", index=False)

    print("\nSaved:")
    print(f" telemetry.csv   → {len(telemetry):,} rows")
    print(f" maintenance.csv → {len(maintenance):,} rows")
    print(f" trips.csv       → {len(trips):,} rows")


if __name__ == "__main__":
    main()
