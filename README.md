# Floter AI Recommendation Backend

This project builds a simple two-stage recommendation system for Twitter-like user profiles.

The dataset contains user profile information such as interests, city, country, age, and gender. It does not contain real user interaction labels such as clicks, follows, likes, or messages. Because of that, the ranking model is trained using proxy relevance labels created from interest similarity, location match, and age closeness.

## Solution overview

The recommendation flow is:

- Stage 1 retrieves 100 candidates with cosine nearest-neighbor search over interest vectors.
- Stage 2 reranks those candidates with a PyTorch neural ranking model.
- The API returns the final top recommendations for an existing user or a new stateless sample profile.

High-level flow:

```text
raw data
  -> preprocessing
  -> candidate retrieval
  -> pairwise feature creation
  -> neural ranking model
  -> FastAPI recommendation endpoint
```

Important limitation:

- The model demonstrates the recommendation pipeline.
- It should not be treated as real production accuracy because the dataset has no real engagement labels.

## Folder structure

```text
EDA/
  01_data_overview_eda.ipynb
  02_preprocessing.ipynb
  03_candidate_generation_and_features.ipynb
  04_ann_candidate_generation.ipynb
  05_neural_ranking_model.ipynb

data/
  SocialMediaUsersDataset.csv
  processed_data.csv

models/
  neural_ranker.pt

src/floter_ai/
  api.py
  recommender.py
  retrieval.py
  ranker.py
  features.py
  data_loader.py
  schemas.py

tests/
  test_api.py
  test_recommender.py

postman/
  floter_ai_recommendation_api.postman_collection.json
```

Main folder purpose:

- `EDA/`: notebooks used for data understanding, preprocessing, candidate generation, ANN-style retrieval, and neural ranking.
- `data/`: raw and processed data files.
- `models/`: saved PyTorch ranking model.
- `src/floter_ai/`: modular backend source code.
- `tests/`: API and recommendation service tests.
- `postman/`: Postman collection for testing API requests.

## Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn floter_ai.api:app --host 0.0.0.0 --port 8000
```

## API

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/recommendations/00029?top_k=10"
```

Sample/new user recommendation:

```bash
curl -X POST "http://localhost:8000/recommendations/sample" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "new_user_001",
    "age": 28,
    "gender": "Female",
    "city": "Mumbai",
    "country": "India",
    "interests": ["Technology", "Travel", "Music"],
    "top_k": 10
  }'
```

## API validation

The recommendation endpoint uses Pydantic-based validation for request parameters.

Existing user endpoint:

```text
GET /recommendations/{user_id}?top_k=10
```

Validation rules:

- `user_id` is required in the path.
- `user_id` must be a numeric string with 1 to 6 digits.
- `top_k` is optional and defaults to `10`.
- `top_k` must be an integer between `1` and `100`.

Response behavior:

| Scenario | Request | Result |
| --- | --- | --- |
| Valid existing user | `/recommendations/00029?top_k=5` | `200 OK` |
| Valid existing user without `top_k` | `/recommendations/00029` | `200 OK`, uses default `top_k=10` |
| Invalid `user_id` format | `/recommendations/not-a-user?top_k=5` | `422`, Pydantic validation error |
| Invalid `top_k` value | `/recommendations/00029?top_k=0` | `422`, Pydantic validation error |
| Invalid `top_k` type | `/recommendations/00029?top_k=abc` | `422`, Pydantic validation error |
| Valid-looking user ID not found in data | `/recommendations/999998?top_k=5` | `404`, user not found |

New sample endpoint:

```text
POST /recommendations/sample
```

Request body:

```json
{
  "user_id": "new_user_001",
  "age": 28,
  "gender": "Female",
  "city": "Mumbai",
  "country": "India",
  "interests": ["Technology", "Travel", "Music"],
  "top_k": 10
}
```

Validation rules:

- The sample user is not saved into `processed_data.csv`.
- `user_id`, `gender`, `city`, and `country` are required non-empty strings.
- `age` must be between `13` and `100`.
- `interests` must contain at least one value.
- Duplicate interests are removed before recommendation.
- At least one interest must exist in the model vocabulary.
- Unknown interests are ignored and mentioned in the response note.
- `top_k` is optional and defaults to `10`.
- `top_k` must be an integer between `1` and `100`.

Response behavior:

| Scenario | Result |
| --- | --- |
| Valid sample with known interests | `200 OK` |
| Valid sample with some unknown interests | `200 OK`, unknown interests are ignored and listed in the response note |
| Valid sample with all unknown interests | `400`, no usable interests for recommendation |
| Invalid payload, such as missing field, invalid age, empty city, empty interests, or invalid `top_k` | `422`, Pydantic validation error |

## API response time

Local response-time check for:

```text
/recommendations/00029?top_k=10
```

Measured result:

```text
Cold start:    ~448 ms
Warm average:  ~29 ms
Warm median:   ~28 ms
Warm p95:      ~29 ms
```

The first request is slower because the service loads the processed data, model, and retrieval index. After startup, the normal warmed response time is around `28–30 ms` locally.

## API logging

The API uses Python's standard `logging` module.

Logs are written to:

- stdout, useful for Docker logs
- `logs/api.log`, useful for local file-based debugging

Recommendation request logs include:

- `event`
- `user_id`
- `top_k`
- `status_code`
- `duration_ms`
- `recommendation_count`
- `error`, when applicable
- `ignored_interests`, for new sample recommendations when applicable

Logged API events:

| Endpoint | Log event |
| --- | --- |
| `GET /health` | `health_request` |
| `GET /recommendations/{user_id}` | `recommendations_request` |
| `POST /recommendations/sample` | `sample_recommendations_request` |

Example:

```text
2026-07-27 19:11:17 INFO floter_ai {"event": "recommendations_request", "user_id": "00029", "top_k": 10, "status_code": 200, "duration_ms": 31.42, "recommendation_count": 10}
```

New sample recommendation example:

```text
2026-07-27 19:15:42 INFO floter_ai {"event": "sample_recommendations_request", "user_id": "new_user_001", "top_k": 10, "status_code": 200, "duration_ms": 34.87, "recommendation_count": 10, "ignored_interests": []}
```

View Docker logs:

```bash
docker compose logs -f floter-ai-recommender
```

View local log file:

```bash
tail -f logs/api.log
```

## Docker

```bash
docker compose up --build -d
```

Services:

- API: `http://localhost:8000`
- Locust dashboard: `http://localhost:8089`

Stop services:

```bash
docker compose down
```

## Stage 3 test candidates CSV

The assessment requires a dedicated script that creates a new test user, runs the recommendation API, and stores the result in `test_candidates.csv`.

Start the API first:

```bash
PYTHONPATH=src uvicorn floter_ai.api:app --host 127.0.0.1 --port 8000
```

In another terminal, run:

```bash
PYTHONPATH=src python scripts/generate_test_candidates.py
```

Generated output:

```text
data/test_candidates.csv
```

The script uses the `POST /recommendations/sample` endpoint, so the test user is not saved into `processed_data.csv`.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Current test status:

```text
19 passed, 1 warning
```

The warning is from FastAPI/Starlette's test client dependency and does not come from the application code.

## Load testing with Locust

Start the API and Locust dashboard:

```bash
docker compose up --build -d
```

Open the Locust dashboard:

```text
http://localhost:8089
```

In the dashboard, use:

```text
Host: http://floter-ai-recommender:8000
```

Local headless example without Docker dashboard:

```bash
locust -f load_tests/locustfile.py \
  --host http://localhost:8000 \
  --headless \
  --users 10 \
  --spawn-rate 2 \
  --run-time 1m
```

Short Locust smoke-test result:

```text
Users: 5
Spawn rate: 5 users/second
Duration: 20 seconds
Total requests: 46
Failures: 0
Failure rate: 0.00%
Requests/sec: 2.35
```

Endpoint result:

| Endpoint | Requests | Failures | Avg ms | Min ms | Max ms | Median ms | P90 ms | P95 ms | P99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `/health` | 4 | 0 | 1 | 1 | 2 | 1 | 2 | 2 | 2 |
| `/recommendations/sample` | 12 | 0 | 188 | 25 | 1889 | 34 | 40 | 1900 | 1900 |
| `/recommendations/{user_id}?top_k=10` | 21 | 0 | 315 | 33 | 2002 | 39 | 1900 | 2000 | 2000 |
| `/recommendations/{user_id}?top_k=5` | 9 | 0 | 255 | 31 | 2003 | 37 | 2000 | 2000 | 2000 |
| Aggregated | 46 | 0 | 242 | 1 | 2003 | 36 | 1900 | 2000 | 2000 |

The average and high percentiles are higher than the median because this short test includes cold-start/model-loading overhead. After the service is warm, recommendation requests are typically much faster.

## Postman

Import:

```text
postman/floter_ai_recommendation_api.postman_collection.json
```

## Next steps / improvements

This project is a working recommendation pipeline, but the current dataset limits how far model quality can be proven. Good next steps would be:

- Add real interaction data such as follows, profile views, likes, clicks, messages, or skips.
- Train with real labels instead of proxy labels.
- Find and Evaluate with recommendation metrics as per discussion with Domain Expert and Stakeholder.
- Replace brute-force nearest-neighbor search with a true ANN engine such as FAISS, HNSW, or Annoy for larger datasets.
- Add better cold-start handling for new users with unseen interests or sparse profiles.
- Add a controlled A/B testing setup once real user feedback or interaction events are available.
- Add model/version metadata to API responses so recommendations can be traced back to a specific model artifact.
- Add monitoring for latency, error rate, request volume, unknown interests, and empty-result cases.
- Add retraining pipeline support so preprocessing, candidate generation, model training, and deployment can be repeated safely.
