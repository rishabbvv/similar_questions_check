from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from text_utils import clean_text


REQUIRED_COLUMNS = {"question1", "question2", "is_duplicate"}


def load_question_pairs(path: str, sample_size: Optional[int] = None, random_state: int = 42) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[["question1", "question2", "is_duplicate"]].dropna()
    df["question1"] = df["question1"].map(clean_text)
    df["question2"] = df["question2"].map(clean_text)
    df["is_duplicate"] = df["is_duplicate"].astype(int)

    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=random_state)

    return df.reset_index(drop=True)


def split_pairs(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["is_duplicate"],
    )
