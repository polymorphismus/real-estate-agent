"""Minimal parquet loader with cached dataframe access."""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from config.constants import DATASET_PATH
from src.data.constants import EXPECTED_COLUMNS


def _validate_columns(dataframe: pd.DataFrame) -> None:
    """Validate required columns exist in dataframe."""
    missing = [column for column in EXPECTED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


@lru_cache(maxsize=4)
def _load_dataframe(dataset_path: str = DATASET_PATH) -> pd.DataFrame:
    """Load parquet once per dataset path and validate schema."""
    dataframe = pd.read_parquet(dataset_path)
    _validate_columns(dataframe)
    return dataframe


def get_dataframe(
    *, copy: bool = True, dataset_path: str = DATASET_PATH
) -> pd.DataFrame:
    """Return cached dataframe, optionally as a copy."""
    dataframe = _load_dataframe(dataset_path=dataset_path)
    return dataframe.copy() if copy else dataframe
