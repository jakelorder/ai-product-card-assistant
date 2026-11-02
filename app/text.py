"""Small text utilities shared by verification, generation and formatting."""

from __future__ import annotations

import re
import unicodedata

from app.models import DESCRIPTION_MAX, DESCRIPTION_MIN, TITLE_MAX

_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212\u2043"
_NUMBERS = re.compile(r"\d+(?:[.,]\d+)?")
_WHITESPACE = re.compile(r"\s+")


def normalize(value: str) -> str:
    """Normalize a fact for comparison without changing its meaning.

    Handles case, ``ё``, dash variants, non-breaking spaces, decimal commas
    and duplicated whitespace. Punctuation at the edges is ignored.
    """
    text = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    for dash in _DASHES:
        text = text.replace(dash, "-")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = _WHITESPACE.sub(" ", text).strip()
    return text.strip(" .,;:")


def number_tokens(value: str) -> set[str]:
    """Extract comparable numeric tokens (``1,50`` and ``1.5`` are equal)."""
    tokens: set[str] = set()
    for token in _NUMBERS.findall(normalize(value)):
        token = token.replace(",", ".")
        if "." in token:
            token = token.rstrip("0").rstrip(".")
        else:
            token = token.lstrip("0") or "0"
        tokens.add(token)
    return tokens


def truncate_title(title: str, limit: int = TITLE_MAX) -> tuple[str, bool]:
    """Trim a title to ``limit`` characters at a word boundary."""
    text = _WHITESPACE.sub(" ", title).strip()
    if len(text) <= limit:
        return text, False

    cut = text[:limit]
    if " " in cut:
        cut = cut[: cut.rfind(" ")]
    cut = cut.rstrip(" ,;:.-–—«»\"'")
    if not cut:
        cut = text[:limit]
    return cut, True


def clamp_description(
    text: str,
    limit: int = DESCRIPTION_MAX,
    minimum: int = DESCRIPTION_MIN,
) -> tuple[str, bool]:
    """Cut a too-long description at a paragraph/sentence boundary.

    The result stays above ``minimum`` when a suitable boundary exists, so the
    required 800–1200 range is preserved. Too-short text is returned as is:
    padding it would mean inventing facts.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    if len(text) <= limit:
        return text, False

    threshold = max(minimum, int(limit * 0.75))
    window = text[:limit]
    for separator in ("\n\n", ". ", "! ", "? "):
        index = window.rfind(separator)
        if index >= threshold:
            return window[: index + len(separator)].strip(), True

    index = window.rfind(" ")
    if index > 0:
        return window[:index].rstrip(" ,;:.-–—"), True
    return window, True
