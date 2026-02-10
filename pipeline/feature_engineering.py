import pandas as pd
from pathlib import Path


DATA_DIR = Path("data")
OUT_PATH = Path("data/features.parquet")


# =================================================
# Load
# =================================================

def load_data():
    """
    Load raw datasets.

    Returns:
        telemetry   : per-minute sensor data
        maintenance : regen events
        trips       : trip-level summaries
    """
    telemetry = pd.read_csv(DATA_DIR / "telemetry.csv", parse_dates=["timestamp"])
    maintenance = pd.read_csv(DATA_DIR / "maintenance.csv", parse_dates=["maintenance_time"])
    trips = pd.read_csv(DATA_DIR / "trips.csv", parse_dates=["trip_start", "trip_end"])

    return telemetry, maintenance, trips


# =================================================
# Clean
# =================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clip impossible values + forward fill."""
    df = df.copy()

    df["speed"] = df["speed"].clip(0, 120)
    df["rpm"] = df["rpm"].clip(500, 4000)
    df["engine_load"] = df["engine_load"].clip(0, 1)

    return df.ffill()


# =================================================
# Rolling features
# =================================================

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling temperature statistics."""
    df = df.sort_values(["vehicle_id", "timestamp"]).copy()

    for window in [10, 60]:
        df[f"temp_roll_mean_{window}"] = (
            df.groupby("vehicle_id")["exhaust_temp_pre"]
              .rolling(window, min_periods=1)
              .mean()
              .reset_index(level=0, drop=True)
        )

    df["temp_delta"] = df["exhaust_temp_pre"] - df["exhaust_temp_post"]

    return df


# =================================================
# Regen features
# =================================================

def add_regen_features(df: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    """Add regen flag + minutes since last regen."""
    df = df.copy()
    maintenance = maintenance.copy()

    maintenance["is_regen"] = 1

    df = df.merge(
        maintenance[["vehicle_id", "maintenance_time", "is_regen"]],
        how="left",
        left_on=["vehicle_id", "timestamp"],
        right_on=["vehicle_id", "maintenance_time"]
    )

    df["is_regen"] = df["is_regen"].fillna(0).astype(int)

    regen_block = df.groupby("vehicle_id")["is_regen"].cumsum()

    df["minutes_since_regen"] = (
        df.groupby(["vehicle_id", regen_block]).cumcount()
    )

    return df.drop(columns=["maintenance_time"])


# =================================================
# Driving behavior features
# =================================================

def add_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Idle/high-load driving patterns."""
    df = df.copy()

    df["idle"] = (df["speed"] < 5).astype(int)
    df["high_load"] = (df["engine_load"] > 0.7).astype(int)

    df["idle_ratio_30"] = (
        df.groupby("vehicle_id")["idle"]
          .rolling(30, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    df["high_load_ratio_30"] = (
        df.groupby("vehicle_id")["high_load"]
          .rolling(30, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )

    return df


# =================================================
# Trip features ⭐ (NEW + IMPORTANT)
# =================================================

def add_trip_features(df: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    """
    Merge trip-level statistics into telemetry.

    Adds:
        trip_duration_min
        trip_avg_speed
    """
    df = df.copy()
    trips = trips.copy()

    trips["trip_duration_min"] = (
        (trips["trip_end"] - trips["trip_start"]).dt.total_seconds() / 60
    )

    trips["trip_avg_speed"] = (
        trips["distance_km"] / (trips["trip_duration_min"] / 60 + 1e-6)
    )

    df = df.merge(
        trips[["vehicle_id", "trip_start", "trip_duration_min", "trip_avg_speed"]],
        how="left",
        left_on=["vehicle_id", "timestamp"],
        right_on=["vehicle_id", "trip_start"]
    )

    df[["trip_duration_min", "trip_avg_speed"]] = (
        df[["trip_duration_min", "trip_avg_speed"]].ffill()
    )

    return df.drop(columns=["trip_start"])


# =================================================
# Targets
# =================================================

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Classification label."""
    df = df.copy()
    df["regen_needed"] = (df["soot_load"] > 0.8).astype(int)
    return df


# =================================================
# Main pipeline
# =================================================

def main():

    telemetry, maintenance, trips = load_data()

    df = clean_data(telemetry)
    df = add_rolling_features(df)
    df = add_regen_features(df, maintenance)
    df = add_behavior_features(df)
    df = add_trip_features(df, trips)
    df = add_targets(df)

    # memory optimization (huge for 3–4M rows)
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    OUT_PATH.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Feature table saved → features.parquet")
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
