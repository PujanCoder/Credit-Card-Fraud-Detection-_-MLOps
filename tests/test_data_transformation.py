import pandas as pd

from src.components.data_transformation import DataTransformation


def test_feature_engineering_creates_expected_columns():
    sample = pd.DataFrame([{
        "trans_date_trans_time": "2020-06-21 12:14:25",
        "merchant": "fraud_Kirlin and Sons",
        "category": "personal_care",
        "amt": 25.50,
        "city": "Moravian Falls",
        "state": "NC",
        "city_pop": 3495,
        "job": "Psychologist",
        "dob": "1988-03-09",
        "lat": 36.0788,
        "long": -81.1781,
        "merch_lat": 36.0113,
        "merch_long": -82.0483,
    }])

    transformer = DataTransformation()
    result = transformer.create_features(sample)

    assert "transaction_hour" in result.columns
    assert "transaction_day_of_week" in result.columns
    assert "transaction_month" in result.columns
    assert "age" in result.columns
    assert "merchant_distance_km" in result.columns

    assert "dob" not in result.columns
    assert "lat" not in result.columns
    assert "merch_lat" not in result.columns
    