"""Complexity predictor extraction."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from .manifest import ComplexityManifest
from .tokenize import estimate_syllables, phoneme_proxy_count, tokenize_words


def predictors_for_text(text: str, *, lowercase: bool = True) -> dict[str, Any]:
    words = tokenize_words(text, lowercase=lowercase)
    word_lengths = [len(word) for word in words]
    syllables = [estimate_syllables(word) for word in words]
    phoneme_proxy = [phoneme_proxy_count(word) for word in words]
    total_chars = sum(word_lengths)
    word_count = len(words)
    return {
        "orthographic_word_count": word_count,
        "orthographic_char_count": total_chars,
        "mean_word_length": total_chars / word_count if word_count else 0.0,
        "utterance_type_count": len(set(words)),
        "estimated_syllable_count": sum(syllables),
        "estimated_phoneme_proxy_count": sum(phoneme_proxy),
        "cleaned_is_empty_or_punctuation": int(word_count == 0),
    }


def extract_rows(manifest: ComplexityManifest):
    for row in iter_csv_dicts(manifest.input_csv):
        base: dict[str, Any] = {"complexity_run_id": manifest.run_id}
        for column in manifest.id_columns:
            base[column] = row.get(column, "")
        for column in manifest.carry_columns:
            base[column] = row.get(column, "")
        text = row.get(manifest.text_column, "")
        yield {
            **base,
            **predictors_for_text(text, lowercase=manifest.lowercase),
        }


def output_fieldnames(manifest: ComplexityManifest) -> list[str]:
    fields = [
        "complexity_run_id",
        *manifest.id_columns,
        *manifest.carry_columns,
        "orthographic_word_count",
        "orthographic_char_count",
        "mean_word_length",
        "utterance_type_count",
        "estimated_syllable_count",
        "estimated_phoneme_proxy_count",
        "cleaned_is_empty_or_punctuation",
    ]
    seen: set[str] = set()
    deduped: list[str] = []
    for field in fields:
        if field not in seen:
            seen.add(field)
            deduped.append(field)
    return deduped


def run_extract(manifest: ComplexityManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    row_count = write_csv_dicts(
        manifest.output_csv,
        extract_rows(manifest),
        fieldnames=output_fieldnames(manifest),
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.run_id,
        "row_count": row_count,
        "input_csv": str(manifest.input_csv),
        "output_csv": str(manifest.output_csv),
        "input_sha256": sha256_file(manifest.input_csv),
        "output_sha256": sha256_file(manifest.output_csv),
        "manifest": {
            key: str(value) if key.endswith("_csv") else value
            for key, value in asdict(manifest).items()
            if key != "raw"
        },
    }
    write_json(manifest.audit_json, audit)
    return audit
