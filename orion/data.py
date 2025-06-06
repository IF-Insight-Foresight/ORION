import re
import pandas as pd
from typing import Iterable, Sequence


def preprocess_text(text: str, stop_words: Iterable[str] | None = None) -> str:
    """Lowercase, remove punctuation and stop words, and collapse whitespace."""
    stop_words = set(stop_words or [])
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def custom_tokenizer(text: str) -> list[str]:
    """Basic tokenizer used for Count/Tfidf vectorizers."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return [tok for tok in tokens if len(tok) > 2]


def build_preprocessed_column(
    df: pd.DataFrame,
    stop_words: Iterable[str] | None = None,
    *,
    source_cols: Sequence[str] = ("Title", "Description", "Tags"),
    target_col: str = "PreprocessedText",
) -> pd.DataFrame:
    """Create ``target_col`` by concatenating and preprocessing ``source_cols``."""
    combined = (
        df[source_cols[0]].fillna("")
        + " "
        + df[source_cols[1]].fillna("")
        + " "
        + df[source_cols[2]].fillna("")
    )
    df[target_col] = combined.apply(lambda t: preprocess_text(t, stop_words))
    return df
