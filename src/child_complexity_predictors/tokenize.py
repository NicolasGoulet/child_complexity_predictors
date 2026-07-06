"""Small tokenizer for cleaned child-language utterances."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
VOWEL_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def tokenize_words(text: str, *, lowercase: bool = True) -> list[str]:
    if text is None:
        return []
    value = str(text)
    if lowercase:
        value = value.lower()
    return WORD_RE.findall(value)


def estimate_syllables(word: str) -> int:
    """Approximate syllables with vowel groups; deterministic and auditable."""

    letters = re.sub(r"[^A-Za-z]", "", word.lower())
    if not letters:
        return 0
    groups = VOWEL_RE.findall(letters)
    count = len(groups)
    if letters.endswith("e") and count > 1 and not letters.endswith(("le", "ye")):
        count -= 1
    return max(1, count)


def phoneme_proxy_count(word: str) -> int:
    """Simple spoken-effort proxy until a real phoneme mapper is selected."""

    return len(re.sub(r"[^A-Za-z]", "", word))
