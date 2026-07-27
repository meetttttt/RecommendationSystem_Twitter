"""High-level recommendation orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from floter_ai.config import DEFAULT_CANDIDATE_POOL_SIZE, DEFAULT_TOP_K, DATA_PATH, MODEL_PATH
from floter_ai.data_loader import load_processed_users
from floter_ai.features import build_pairwise_features
from floter_ai.ranker import RankingModel
from floter_ai.retrieval import CandidateRetriever


@dataclass(frozen=True)
class Recommendation:
    """Single ranked recommendation returned by the service."""

    user_id: str
    name: str
    city: str
    country: str
    interests: list[str]
    score: float
    ann_similarity: float
    interest_jaccard_similarity: float
    same_country: bool
    same_city_country_key: bool
    age_difference: float


class RecommendationService:
    """Two-stage recommendation service: retrieve candidates, then rerank them."""

    def __init__(
        self,
        data_path: Path = DATA_PATH,
        model_path: Path = MODEL_PATH,
        candidate_pool_size: int = DEFAULT_CANDIDATE_POOL_SIZE,
    ) -> None:
        self.users_df = load_processed_users(data_path)
        self.ranking_model = RankingModel(model_path)
        self.retriever = CandidateRetriever(self.users_df, self.ranking_model.interest_vocab)
        self.candidate_pool_size = candidate_pool_size
        self.interest_lookup = {
            interest.casefold(): interest for interest in self.ranking_model.interest_vocab
        }

    def get_recommendations(self, user_id: str, top_k: int = DEFAULT_TOP_K) -> list[Recommendation]:
        """Return top-k ranked recommendations for a target user."""
        self._validate_top_k(top_k)

        target_user = self._get_user(user_id)
        candidates = self.retriever.retrieve(user_id, top_k=self.candidate_pool_size)
        return self._rank_candidates(target_user=target_user, candidates=candidates, top_k=top_k)

    def get_recommendations_for_sample(
        self,
        user_id: str,
        age: int,
        city: str,
        country: str,
        interests: list[str],
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[list[Recommendation], list[str]]:
        """Return top-k recommendations for a stateless external user profile."""
        self._validate_top_k(top_k)
        known_interests, ignored_interests = self._normalize_sample_interests(interests)
        if not known_interests:
            raise ValueError("Sample must include at least one known interest from the model vocabulary")

        target_user = pd.Series(
            {
                "user_id": user_id,
                "name": user_id,
                "age": float(age),
                "city": city,
                "country": country,
                "city_country_key": f"{city}, {country}",
                "interests": known_interests,
            }
        )
        candidates = self.retriever.retrieve_for_interests(
            interests=known_interests,
            top_k=self.candidate_pool_size,
        )
        recommendations = self._rank_candidates(
            target_user=target_user,
            candidates=candidates,
            top_k=top_k,
        )

        return recommendations, ignored_interests

    def _rank_candidates(
        self,
        target_user: pd.Series,
        candidates: pd.DataFrame,
        top_k: int,
    ) -> list[Recommendation]:
        """Rank retrieved candidates for a target profile."""
        pair_features = build_pairwise_features(target_user, candidates)
        pair_features["predicted_relevance_score"] = self.ranking_model.predict(pair_features)

        ranked = pair_features.sort_values(
            by="predicted_relevance_score",
            ascending=False,
        ).head(top_k)

        return [self._row_to_recommendation(row) for _, row in ranked.iterrows()]

    def _validate_top_k(self, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if top_k > self.candidate_pool_size:
            raise ValueError(f"top_k cannot exceed candidate pool size {self.candidate_pool_size}")

    def _get_user(self, user_id: str) -> pd.Series:
        matches = self.users_df[self.users_df["user_id"] == user_id]
        if matches.empty:
            raise KeyError(f"Unknown user_id: {user_id}")
        return matches.iloc[0]

    def _normalize_sample_interests(self, interests: list[str]) -> tuple[list[str], list[str]]:
        known_interests = []
        ignored_interests = []
        seen_known = set()
        seen_ignored = set()

        for interest in interests:
            normalized_interest = " ".join(str(interest).strip().split())
            canonical_interest = self.interest_lookup.get(normalized_interest.casefold())

            if canonical_interest is None:
                if normalized_interest and normalized_interest.casefold() not in seen_ignored:
                    ignored_interests.append(normalized_interest)
                    seen_ignored.add(normalized_interest.casefold())
                continue

            if canonical_interest.casefold() not in seen_known:
                known_interests.append(canonical_interest)
                seen_known.add(canonical_interest.casefold())

        return known_interests, ignored_interests

    @staticmethod
    def _row_to_recommendation(row: pd.Series) -> Recommendation:
        return Recommendation(
            user_id=str(row["candidate_user_id"]),
            name=str(row["candidate_name"]),
            city=str(row["candidate_city"]),
            country=str(row["candidate_country"]),
            interests=list(row["candidate_interests"]),
            score=round(float(row["predicted_relevance_score"]), 6),
            ann_similarity=round(float(row["ann_similarity"]), 6),
            interest_jaccard_similarity=round(float(row["interest_jaccard_similarity"]), 6),
            same_country=bool(row["same_country"]),
            same_city_country_key=bool(row["same_city_country_key"]),
            age_difference=round(float(row["age_difference"]), 2),
        )
