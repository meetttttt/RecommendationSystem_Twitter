"""Nearest-neighbor candidate retrieval."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


class CandidateRetriever:
    """Retrieve candidate users using cosine nearest-neighbor search over interests."""

    def __init__(self, users_df: pd.DataFrame, interest_vocab: list[str]) -> None:
        self.users_df = users_df.reset_index(drop=True)
        self.interest_vocab = interest_vocab
        self.interest_to_idx = {interest: idx for idx, interest in enumerate(interest_vocab)}
        self.interest_matrix = self._build_interest_matrix()
        self.index = NearestNeighbors(
            n_neighbors=min(101, len(self.users_df)),
            metric="cosine",
            algorithm="brute",
        )
        self.index.fit(self.interest_matrix)

    def retrieve(self, target_user_id: str, top_k: int) -> pd.DataFrame:
        """Return top candidate users for a target user, excluding the target itself."""
        matches = self.users_df.index[self.users_df["user_id"] == target_user_id]
        if len(matches) == 0:
            raise KeyError(f"Unknown user_id: {target_user_id}")

        target_idx = int(matches[0])
        query_vector = self.interest_matrix[target_idx : target_idx + 1]
        return self.retrieve_for_vector(
            query_vector=query_vector,
            top_k=top_k,
            exclude_user_id=target_user_id,
        )

    def retrieve_for_interests(self, interests: list[str], top_k: int) -> pd.DataFrame:
        """Return top candidate users for an external interest profile."""
        query_vector = self._vectorize_interests(interests)
        if query_vector.sum() == 0:
            raise ValueError("At least one interest must match the model interest vocabulary")

        return self.retrieve_for_vector(query_vector=query_vector.reshape(1, -1), top_k=top_k)

    def retrieve_for_vector(
        self,
        query_vector: np.ndarray,
        top_k: int,
        exclude_user_id: str | None = None,
    ) -> pd.DataFrame:
        """Return top candidate users for a prepared interest vector."""
        neighbor_count = min(top_k + 1, len(self.users_df))
        distances, indices = self.index.kneighbors(
            query_vector,
            n_neighbors=neighbor_count,
        )

        candidate_rows = []
        for distance, idx in zip(distances[0], indices[0]):
            candidate = self.users_df.iloc[int(idx)].copy()
            if exclude_user_id is not None and candidate["user_id"] == exclude_user_id:
                continue

            candidate["ann_similarity"] = 1 - float(distance)
            candidate_rows.append(candidate)

            if len(candidate_rows) == top_k:
                break

        return pd.DataFrame(candidate_rows).reset_index(drop=True)

    def _build_interest_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.users_df), len(self.interest_vocab)), dtype=np.float32)
        for row_idx, interests in enumerate(self.users_df["interests"]):
            for interest in interests:
                if interest in self.interest_to_idx:
                    matrix[row_idx, self.interest_to_idx[interest]] = 1.0
        return matrix

    def _vectorize_interests(self, interests: list[str]) -> np.ndarray:
        vector = np.zeros(len(self.interest_vocab), dtype=np.float32)
        for interest in interests:
            if interest in self.interest_to_idx:
                vector[self.interest_to_idx[interest]] = 1.0
        return vector
