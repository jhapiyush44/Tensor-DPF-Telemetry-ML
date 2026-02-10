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
    assert cleaned["speed"].tolist() == [0, 50]

    # rpm clipped
    assert cleaned["rpm"].tolist() == [500, 4000]

    # engine load clipped
    assert cleaned["engine_load"].tolist() == [1, 0]
