from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


TEXT_COLUMNS = (
    "text",
    "tweet",
    "full_text",
    "content",
    "body",
    "review",
    "message",
    "comment",
)


def _find_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    by_key = {column.strip().lower(): column for column in columns}
    for candidate in candidates:
        if candidate in by_key:
            return by_key[candidate]
    return None


def load_xquik_texts(frame: pd.DataFrame) -> list[str]:
    """Return non-empty text rows from a Xquik export-like frame."""
    text_column = _find_column(frame.columns, TEXT_COLUMNS)
    if text_column is None:
        return []
    return [
        value
        for value in frame[text_column].fillna("").astype(str).str.strip().tolist()
        if value
    ]
