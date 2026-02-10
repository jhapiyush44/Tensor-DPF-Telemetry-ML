import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
OUT_PATH = Path("data/features.parquet")


# -------------------------------------------------
# Load
# -------------------------------------------------

def load_data():
    telemetry = pd.read_csv(DATA_DIR / "telemetry.csv", parse_dates=["timestamp"])
    maintenance = pd.read_csv(DATA_DIR / "maintenance.csv", parse_dates=["maintenance_time"])
    trips = pd.read_csv(DATA_DIR / "trips.csv", parse_dates=["trip_start", "trip_end"])
    return telemetry, maintenance, trips


# -------------------------------------------------
# Clean
# -------------------------------------------------

def clean_data(df):

    df["speed"] = df["speed"].clip(0, 120)
    df["rpm"] = df["rpm"].clip(500, 4000)
    df["engine_load"] = df["engine_load"].clip(0, 1)

    df = df.ffill()

    return df


# -------------------------------------------------
# Rolling features
# -------------------------------------------------

def add_rolling_features(df):

    df = df.sort_values(["vehicle_id", "timestamp"])

    for window in [10, 60]:
        df[f"temp_roll_mean_{window}"] = (
            df.groupby("vehicle_id")["exhaust_temp_pre"]
              .transform(lambda x: x.rolling(window, min_periods=1).mean())
        )

    df["temp_delta"] = df["exhaust_temp_pre"] - df["exhaust_temp_post"]

    return df


# -------------------------------------------------
# Regen features (FIXED ⭐)
# -------------------------------------------------

def add_regen_features(df, maintenance):

    maintenance["is_regen"] = 1

    df = df.merge(
        maintenance[["vehicle_id", "maintenance_time", "is_regen"]],
        how="left",
        left_on=["vehicle_id", "timestamp"],
        right_on=["vehicle_id", "maintenance_time"]
    )

    df["is_regen"] = df["is_regen"].fillna(0)

    df["minutes_since_regen"] = (
        df.groupby("vehicle_id")["is_regen"]
          .transform(lambda x: x.eq(1).cumsum())
    )

    df["minutes_since_regen"] = (
        df.groupby(["vehicle_id", "minutes_since_regen"]).cumcount()
    )

    return df.drop(columns=["maintenance_time"])


# -------------------------------------------------
# Behavior
# -------------------------------------------------

def add_behavior_features(df):

    df["idle"] = (df["speed"] < 5).astype(int)
    df["high_load"] = (df["engine_load"] > 0.7).astype(int)

    df["idle_ratio_30"] = (
        df.groupby("vehicle_id")["idle"]
        .transform(lambda x: x.rolling(30, 1).mean())
    )

    df["high_load_ratio_30"] = (
        df.groupby("vehicle_id")["high_load"]
        .transform(lambda x: x.rolling(30, 1).mean())
    )

    return df


# -------------------------------------------------
# Targets
# -------------------------------------------------

def add_targets(df):
    df["regen_needed"] = (df["soot_load"] > 0.8).astype(int)
    return df


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():

    telemetry, maintenance, trips = load_data()

    df = clean_data(telemetry)
    df = add_rolling_features(df)
    df = add_regen_features(df, maintenance)
    df = add_behavior_features(df)
    df = add_targets(df)

    # memory optimization
    float_cols = df.select_dtypes("float64").columns
    df[float_cols] = df[float_cols].astype("float32")

    df.to_parquet(OUT_PATH, index=False)

    print("Feature table saved → features.parquet")
    print("Shape:", df.shape)


if __name__ == "__main__":
    main()
