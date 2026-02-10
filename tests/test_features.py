import pandas as pd
from pipeline.feature_engineering import clean_data


def test_clean_data():

    df = pd.DataFrame({
        "speed": [-10, 50],
        "rpm": [100, 9000],
        "engine_load": [2, -1]
    })

    cleaned = clean_data(df)

    # speed clipped
    assert cleaned["speed"].min() >= 0

    # rpm clipped
    assert cleaned["rpm"].max() <= 4000

    # engine_load clipped
    assert cleaned["engine_load"].between(0, 1).all()
