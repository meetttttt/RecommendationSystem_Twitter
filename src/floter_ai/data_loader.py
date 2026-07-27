"""Load and normalize processed user profile data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_processed_users(data_path: Path) -> pd.DataFrame:
    """Load processed user data and parse JSON interest lists."""
    if not data_path.exists():
        raise FileNotFoundError(f"Processed data file not found: {data_path}")

    users_df = pd.read_csv(data_path, dtype={"user_id": "string"})
    users_df["interests"] = users_df["interests_deduped"].apply(json.loads)
    users_df["age"] = users_df["age"].astype(float)
    return users_df

