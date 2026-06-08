import re
from typing import Iterable, List


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(value: object) -> List[str]:
    return TOKEN_RE.findall(clean_text(value))


def tokenized_questions(questions: Iterable[object]) -> List[List[str]]:
    return [tokenize(question) for question in questions]
