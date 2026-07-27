"""Feature engineering for target-candidate ranking pairs."""

from __future__ import annotations

import pandas as pd


FEATURE_COLUMNS = [
    "ann_similarity",
    "interest_overlap_count",
    "interest_jaccard_similarity",
    "same_country",
    "same_city_country_key",
    "age_difference",
    "age_similarity",
    "target_interest_count",
    "candidate_interest_count",
]


def build_pairwise_features(target_user: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Build model-ready features for one target user and many candidate users."""
    target_interests = set(target_user["interests"])
    rows = []

    for _, candidate in candidates.iterrows():
        candidate_interests = set(candidate["interests"])
        overlap_count = len(target_interests & candidate_interests)
        union_count = len(target_interests | candidate_interests)
        jaccard = overlap_count / union_count if union_count else 0.0
        age_difference = abs(float(target_user["age"]) - float(candidate["age"]))
        age_similarity = max(0.0, 1.0 - age_difference / 60.0)

        rows.append(
            {
                "target_user_id": target_user["user_id"],
                "candidate_user_id": candidate["user_id"],
                "candidate_name": candidate["name"],
                "candidate_city": candidate["city"],
                "candidate_country": candidate["country"],
                "candidate_interests": candidate["interests"],
                "ann_similarity": float(candidate["ann_similarity"]),
                "interest_overlap_count": overlap_count,
                "interest_union_count": union_count,
                "interest_jaccard_similarity": jaccard,
                "same_country": int(target_user["country"] == candidate["country"]),
                "same_city_country_key": int(target_user["city_country_key"] == candidate["city_country_key"]),
                "age_difference": age_difference,
                "age_similarity": age_similarity,
                "target_interest_count": len(target_interests),
                "candidate_interest_count": len(candidate_interests),
            }
        )

    return pd.DataFrame(rows)

