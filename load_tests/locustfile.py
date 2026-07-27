"""Locust load test for the recommendation API.

Run after starting the API:

    locust -f load_tests/locustfile.py --host http://localhost:8000

Headless example:

    locust -f load_tests/locustfile.py \
      --host http://localhost:8000 \
      --headless \
      --users 10 \
      --spawn-rate 2 \
      --run-time 1m
"""

from random import choice

from locust import HttpUser, between, task


SAMPLE_USER_IDS = [
    "00029",
    "00032",
    "00047",
    "00064",
    "00075",
]

SAMPLE_PROFILES = [
    {
        "user_id": "new_user_001",
        "age": 28,
        "gender": "Female",
        "city": "Mumbai",
        "country": "India",
        "interests": ["Technology", "Travel", "Music"],
        "top_k": 10,
    },
    {
        "user_id": "new_user_002",
        "age": 35,
        "gender": "Male",
        "city": "Austin",
        "country": "United States",
        "interests": ["Gaming", "Photography", "Movies"],
        "top_k": 10,
    },
]


class RecommendationApiUser(HttpUser):
    """Simulate users requesting recommendations from the API."""

    wait_time = between(1, 3)

    @task(1)
    def health_check(self) -> None:
        """Check API health."""
        self.client.get("/health", name="/health")

    @task(5)
    def get_top_10_recommendations(self) -> None:
        """Request top-10 recommendations for a sample user."""
        user_id = choice(SAMPLE_USER_IDS)
        self.client.get(
            f"/recommendations/{user_id}?top_k=10",
            name="/recommendations/{user_id}?top_k=10",
        )

    @task(2)
    def get_top_5_recommendations(self) -> None:
        """Request top-5 recommendations for a sample user."""
        user_id = choice(SAMPLE_USER_IDS)
        self.client.get(
            f"/recommendations/{user_id}?top_k=5",
            name="/recommendations/{user_id}?top_k=5",
        )

    @task(2)
    def get_sample_recommendations(self) -> None:
        """Request recommendations for a stateless new sample profile."""
        self.client.post(
            "/recommendations/sample",
            json=choice(SAMPLE_PROFILES),
            name="/recommendations/sample",
        )
