"""PyTorch neural ranking model and inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class NeuralRanker(nn.Module):
    """Small MLP ranker trained on target-candidate pair features."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RankingModel:
    """Load the saved PyTorch ranker and score candidate feature rows."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        # The checkpoint was created by this project notebook and contains numpy arrays
        # for scaler stats, so weights_only=False is required for this trusted local file.
        artifact = torch.load(model_path, map_location="cpu", weights_only=False)
        self.feature_columns = artifact["feature_columns"]
        self.scaler_mean = artifact["scaler_mean"]
        self.scaler_scale = artifact["scaler_scale"]
        self.interest_vocab = artifact["interest_vocab"]

        self.model = NeuralRanker(input_dim=len(self.feature_columns))
        self.model.load_state_dict(artifact["model_state_dict"])
        self.model.eval()

    def predict(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Predict relevance scores for candidate feature rows."""
        missing_columns = [column for column in self.feature_columns if column not in feature_df.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")

        features = feature_df[self.feature_columns].to_numpy(dtype=np.float32)
        scaled_features = (features - self.scaler_mean) / self.scaler_scale

        with torch.no_grad():
            predictions = self.model(torch.from_numpy(scaled_features.astype(np.float32)))

        return predictions.numpy().reshape(-1)

