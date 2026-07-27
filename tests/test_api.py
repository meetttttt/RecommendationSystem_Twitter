from fastapi.testclient import TestClient

from floter_ai.api import app, logger


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_recommendations_endpoint_returns_top_k():
    response = client.get("/recommendations/00029?top_k=5")

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "00029"
    assert payload["top_k"] == 5
    assert payload["candidate_pool_size"] == 100
    assert len(payload["recommendations"]) == 5
    assert "proxy labels" in payload["note"]


def test_recommendations_endpoint_uses_default_top_k():
    response = client.get("/recommendations/00029")

    assert response.status_code == 200
    payload = response.json()
    assert payload["top_k"] == 10
    assert len(payload["recommendations"]) == 10


def test_recommendations_endpoint_unknown_user_returns_404():
    response = client.get("/recommendations/999998?top_k=5")

    assert response.status_code == 404


def test_recommendations_endpoint_invalid_user_id_returns_422():
    response = client.get("/recommendations/not-a-user?top_k=5")

    assert response.status_code == 422


def test_recommendations_endpoint_invalid_top_k_returns_422():
    response = client.get("/recommendations/00029?top_k=0")

    assert response.status_code == 422


def test_recommendations_endpoint_non_integer_top_k_returns_422():
    response = client.get("/recommendations/00029?top_k=abc")

    assert response.status_code == 422


def test_sample_recommendations_endpoint_returns_top_k():
    response = client.post(
        "/recommendations/sample",
        json={
            "user_id": "new_user_001",
            "age": 28,
            "gender": "Female",
            "city": "Mumbai",
            "country": "India",
            "interests": ["Technology", "Travel", "Music"],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "new_user_001"
    assert payload["top_k"] == 5
    assert payload["candidate_pool_size"] == 100
    assert len(payload["recommendations"]) == 5
    assert "proxy labels" in payload["note"]


def test_sample_recommendations_endpoint_uses_default_top_k():
    response = client.post(
        "/recommendations/sample",
        json={
            "user_id": "new_user_001",
            "age": 28,
            "gender": "Female",
            "city": "Mumbai",
            "country": "India",
            "interests": ["Technology", "Travel", "Music"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["top_k"] == 10
    assert len(payload["recommendations"]) == 10


def test_sample_recommendations_endpoint_invalid_payload_returns_422():
    response = client.post(
        "/recommendations/sample",
        json={
            "user_id": "new_user_001",
            "age": 8,
            "gender": "Female",
            "city": "Mumbai",
            "country": "India",
            "interests": ["Technology"],
            "top_k": 5,
        },
    )

    assert response.status_code == 422


def test_sample_recommendations_endpoint_no_known_interests_returns_400():
    response = client.post(
        "/recommendations/sample",
        json={
            "user_id": "new_user_001",
            "age": 28,
            "gender": "Female",
            "city": "Mumbai",
            "country": "India",
            "interests": ["Unknown Interest"],
            "top_k": 5,
        },
    )

    assert response.status_code == 400
    assert "known interest" in response.json()["detail"]


def test_sample_recommendations_endpoint_note_includes_ignored_interests():
    response = client.post(
        "/recommendations/sample",
        json={
            "user_id": "new_user_001",
            "age": 28,
            "gender": "Female",
            "city": "Mumbai",
            "country": "India",
            "interests": ["Technology", "Unknown Interest"],
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert "Ignored interests" in response.json()["note"]
    assert "Unknown Interest" in response.json()["note"]


def test_recommendations_success_logs_event(caplog):
    with caplog.at_level("INFO", logger=logger.name):
        response = client.get("/recommendations/00029?top_k=3")

    assert response.status_code == 200
    assert '"event": "recommendations_request"' in caplog.text
    assert '"user_id": "00029"' in caplog.text
    assert '"top_k": 3' in caplog.text
    assert '"status_code": 200' in caplog.text
    assert '"recommendation_count": 3' in caplog.text
    assert '"duration_ms":' in caplog.text


def test_recommendations_error_logs_event(caplog):
    with caplog.at_level("INFO", logger=logger.name):
        response = client.get("/recommendations/999998?top_k=3")

    assert response.status_code == 404
    assert '"event": "recommendations_request"' in caplog.text
    assert '"user_id": "999998"' in caplog.text
    assert '"status_code": 404' in caplog.text
    assert '"recommendation_count": 0' in caplog.text
    assert '"error":' in caplog.text
