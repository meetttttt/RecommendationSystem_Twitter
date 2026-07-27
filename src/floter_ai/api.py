"""FastAPI application for the Floter AI recommender."""

from __future__ import annotations

from typing import Annotated
from functools import lru_cache
from time import perf_counter

from fastapi import Depends, FastAPI, HTTPException, Path, Query

from floter_ai.config import DEFAULT_CANDIDATE_POOL_SIZE, DEFAULT_TOP_K
from floter_ai.logging_config import configure_logging, log_event
from floter_ai.recommender import RecommendationService
from floter_ai.schemas import (
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SampleRecommendationRequest,
)


logger = configure_logging()

app = FastAPI(
    title="Floter AI Recommendation API",
    version="1.0.0",
    description="Two-stage user recommendation API using nearest-neighbor retrieval and a PyTorch neural ranker.",
)


@lru_cache(maxsize=1)
def get_service() -> RecommendationService:
    """Create the recommendation service once per process."""
    return RecommendationService()


def validate_recommendation_request(
    user_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=6,
            pattern=r"^\d{1,6}$",
            description="Numeric user ID from the processed dataset.",
        ),
    ],
    top_k: Annotated[
        int,
        Query(
            ge=1,
            le=DEFAULT_CANDIDATE_POOL_SIZE,
            description="Number of recommendations to return.",
        ),
    ] = DEFAULT_TOP_K,
) -> RecommendationRequest:
    """Validate recommendation request parameters with Pydantic."""
    return RecommendationRequest(user_id=user_id, top_k=top_k)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return basic service health."""
    start = perf_counter()
    duration_ms = round((perf_counter() - start) * 1000, 2)
    log_event(
        logger,
        "health_request",
        status_code=200,
        duration_ms=duration_ms,
    )
    return HealthResponse(status="ok")


@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def recommendations(
    request: Annotated[RecommendationRequest, Depends(validate_recommendation_request)],
) -> RecommendationResponse:
    """Return top-k recommendations for a user."""
    start = perf_counter()
    service = get_service()
    user_id = request.user_id
    top_k = request.top_k

    try:
        results = service.get_recommendations(user_id=user_id, top_k=top_k)
    except KeyError as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            "recommendations_request",
            user_id=user_id,
            top_k=top_k,
            status_code=404,
            duration_ms=duration_ms,
            recommendation_count=0,
            error=str(exc),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            "recommendations_request",
            user_id=user_id,
            top_k=top_k,
            status_code=400,
            duration_ms=duration_ms,
            recommendation_count=0,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration_ms = round((perf_counter() - start) * 1000, 2)
    log_event(
        logger,
        "recommendations_request",
        user_id=user_id,
        top_k=top_k,
        status_code=200,
        duration_ms=duration_ms,
        recommendation_count=len(results),
    )

    return RecommendationResponse(
        user_id=user_id,
        top_k=top_k,
        candidate_pool_size=service.candidate_pool_size,
        recommendations=[item.__dict__ for item in results],
        note=_model_note(),
    )


@app.post("/recommendations/sample", response_model=RecommendationResponse)
def sample_recommendations(request: SampleRecommendationRequest) -> RecommendationResponse:
    """Return top-k recommendations for a stateless external sample profile."""
    start = perf_counter()
    service = get_service()

    try:
        results, ignored_interests = service.get_recommendations_for_sample(
            user_id=request.user_id,
            age=request.age,
            city=request.city,
            country=request.country,
            interests=request.interests,
            top_k=request.top_k,
        )
    except ValueError as exc:
        duration_ms = round((perf_counter() - start) * 1000, 2)
        log_event(
            logger,
            "sample_recommendations_request",
            user_id=request.user_id,
            top_k=request.top_k,
            status_code=400,
            duration_ms=duration_ms,
            recommendation_count=0,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    duration_ms = round((perf_counter() - start) * 1000, 2)
    log_event(
        logger,
        "sample_recommendations_request",
        user_id=request.user_id,
        top_k=request.top_k,
        status_code=200,
        duration_ms=duration_ms,
        recommendation_count=len(results),
        ignored_interests=ignored_interests,
    )

    note = _model_note()
    if ignored_interests:
        note = (
            f"{note} Ignored interests not present in the model vocabulary: "
            f"{', '.join(ignored_interests)}."
        )

    return RecommendationResponse(
        user_id=request.user_id,
        top_k=request.top_k,
        candidate_pool_size=service.candidate_pool_size,
        recommendations=[item.__dict__ for item in results],
        note=note,
    )


def _model_note() -> str:
    return (
        "Scores are produced by a PyTorch ranker trained on proxy labels. "
        "The dataset has no real engagement labels, so this demonstrates the "
        "ranking pipeline rather than proven production recommendation accuracy."
    )
