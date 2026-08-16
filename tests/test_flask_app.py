from flask_app.app import app


def test_prediction_requires_fields():
    client = app.test_client()

    response = client.post("/predict", json={})

    assert response.status_code == 400
    assert "Missing required fields" in response.get_json()["error"]


def test_home_endpoint_exists():
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200