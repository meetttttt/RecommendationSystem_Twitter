"""Generate Stage 3 offline verification candidates through the API.

The assessment asks for a dedicated script that creates a new test user,
runs the recommendation API, and stores the returned candidates in CSV form.

Run after starting the API locally:

    PYTHONPATH=src uvicorn floter_ai.api:app --host 127.0.0.1 --port 8000
    PYTHONPATH=src python scripts/generate_test_candidates.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_PATH = Path("data/test_candidates.csv")

TEST_USER_PAYLOAD = {
    "user_id": "stage3_test_user",
    "age": 28,
    "gender": "Female",
    "city": "Mumbai",
    "country": "India",
    "interests": ["Technology", "Travel", "Music"],
    "top_k": 10,
}

OUTPUT_COLUMNS = [
    "target_user_id",
    "target_age",
    "target_gender",
    "target_city",
    "target_country",
    "target_interests",
    "rank",
    "candidate_user_id",
    "candidate_name",
    "candidate_city",
    "candidate_country",
    "candidate_interests",
    "predicted_relevance_score",
    "ann_similarity",
    "interest_jaccard_similarity",
    "same_country",
    "same_city_country_key",
    "age_difference",
    "api_note",
]


def call_sample_recommendation_endpoint(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Call POST /recommendations/sample and return the JSON response."""
    endpoint = f"{api_url.rstrip('/')}/recommendations/sample"
    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise RuntimeError(f"API returned HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not connect to API at {endpoint}: {exc.reason}") from exc


def flatten_recommendation_response(
    payload: dict[str, Any],
    test_user: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert API response JSON into CSV-ready candidate rows."""
    rows = []
    note = payload.get("note", "")

    for rank, recommendation in enumerate(payload["recommendations"], start=1):
        rows.append(
            {
                "target_user_id": test_user["user_id"],
                "target_age": test_user["age"],
                "target_gender": test_user["gender"],
                "target_city": test_user["city"],
                "target_country": test_user["country"],
                "target_interests": json.dumps(test_user["interests"]),
                "rank": rank,
                "candidate_user_id": recommendation["user_id"],
                "candidate_name": recommendation["name"],
                "candidate_city": recommendation["city"],
                "candidate_country": recommendation["country"],
                "candidate_interests": json.dumps(recommendation["interests"]),
                "predicted_relevance_score": recommendation["score"],
                "ann_similarity": recommendation["ann_similarity"],
                "interest_jaccard_similarity": recommendation["interest_jaccard_similarity"],
                "same_country": recommendation["same_country"],
                "same_city_country_key": recommendation["same_city_country_key"],
                "age_difference": recommendation["age_difference"],
                "api_note": note,
            }
        )

    return rows


def write_candidates_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write flattened recommendation rows to test_candidates.csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Generate Stage 3 test_candidates.csv")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the running API. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the Stage 3 candidate CSV."""
    args = parse_args()
    response_payload = call_sample_recommendation_endpoint(args.api_url, TEST_USER_PAYLOAD)
    rows = flatten_recommendation_response(response_payload, TEST_USER_PAYLOAD)
    write_candidates_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
