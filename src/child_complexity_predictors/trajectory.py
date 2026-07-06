"""Lexical and MLU trajectory exports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import iter_csv_dicts, sha256_file, write_csv_dicts, write_json
from .manifest import ComplexityManifest
from .predictors import predictors_for_text
from .tokenize import tokenize_words


def _sort_key(row: dict[str, str], age_column: str, session_column: str | None, row_index: int):
    age_value = row.get(age_column, "")
    try:
        age = float(age_value)
    except ValueError:
        age = float("inf")
    return (row.get("child_id", ""), age, row.get(session_column or "", ""), row_index)


def trajectory_output_path(manifest: ComplexityManifest) -> Path:
    value = manifest.raw.get("trajectory_output_csv")
    if value:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return manifest.output_csv.parent / path
    if manifest.output_csv.suffix == ".gz":
        stem = manifest.output_csv.with_suffix("").with_suffix("")
    else:
        stem = manifest.output_csv.with_suffix("")
    return stem.with_name(stem.name + ".trajectory.csv.gz")


def aggregate_output_path(manifest: ComplexityManifest) -> Path:
    value = manifest.raw.get("aggregate_output_csv")
    if value:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return manifest.output_csv.parent / path
    if manifest.output_csv.suffix == ".gz":
        stem = manifest.output_csv.with_suffix("").with_suffix("")
    else:
        stem = manifest.output_csv.with_suffix("")
    return stem.with_name(stem.name + ".age_bin_summary.csv.gz")


def build_trajectory_rows(manifest: ComplexityManifest) -> list[dict[str, Any]]:
    child_column = str(manifest.raw.get("child_column", "child_id"))
    age_column = str(manifest.raw.get("age_column", "age_months"))
    age_bin_column = str(manifest.raw.get("age_bin_column", "age_bin"))
    session_column = manifest.raw.get("session_column", "session_id")

    indexed_rows = list(enumerate(iter_csv_dicts(manifest.input_csv)))
    indexed_rows.sort(key=lambda item: _sort_key(item[1], age_column, str(session_column), item[0]))

    cumulative_vocab: dict[str, set[str]] = defaultdict(set)
    cumulative_tokens: dict[str, int] = defaultdict(int)
    age_bin_vocab: dict[tuple[str, str], set[str]] = defaultdict(set)
    age_bin_tokens: dict[tuple[str, str], int] = defaultdict(int)

    outputs: list[dict[str, Any]] = []
    for _, row in indexed_rows:
        child = row.get(child_column, "")
        age_bin = row.get(age_bin_column, "")
        tokens = tokenize_words(row.get(manifest.text_column, ""), lowercase=manifest.lowercase)
        cumulative_vocab[child].update(tokens)
        cumulative_tokens[child] += len(tokens)
        age_key = (child, age_bin)
        age_bin_vocab[age_key].update(tokens)
        age_bin_tokens[age_key] += len(tokens)

        output: dict[str, Any] = {"complexity_run_id": manifest.run_id}
        for column in manifest.id_columns:
            output[column] = row.get(column, "")
        for column in manifest.carry_columns:
            output[column] = row.get(column, "")
        output.update(predictors_for_text(row.get(manifest.text_column, ""), lowercase=manifest.lowercase))
        cumulative_token_count = cumulative_tokens[child]
        output.update(
            {
                "lexical_cumulative_child_vocab_size": len(cumulative_vocab[child]),
                "lexical_cumulative_child_token_count": cumulative_token_count,
                "lexical_cumulative_child_ttr": (
                    len(cumulative_vocab[child]) / cumulative_token_count if cumulative_token_count else 0.0
                ),
                "lexical_age_bin_vocab_size_so_far": len(age_bin_vocab[age_key]),
                "lexical_age_bin_token_count_so_far": age_bin_tokens[age_key],
                "lexical_age_bin_ttr_so_far": (
                    len(age_bin_vocab[age_key]) / age_bin_tokens[age_key] if age_bin_tokens[age_key] else 0.0
                ),
            }
        )
        outputs.append(output)
    return outputs


def build_age_bin_summary_rows(manifest: ComplexityManifest, trajectory_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    child_column = str(manifest.raw.get("child_column", "child_id"))
    age_bin_column = str(manifest.raw.get("age_bin_column", "age_bin"))
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    token_sets: dict[tuple[str, str], set[str]] = defaultdict(set)

    source_rows_by_id = {
        tuple(row.get(column, "") for column in manifest.id_columns): row
        for row in iter_csv_dicts(manifest.input_csv)
    }
    for row in trajectory_rows:
        child = str(row.get(child_column, ""))
        age_bin = str(row.get(age_bin_column, ""))
        key = (child, age_bin)
        if key not in grouped:
            grouped[key] = {
                "complexity_run_id": manifest.run_id,
                child_column: child,
                age_bin_column: age_bin,
                "utterance_count": 0,
                "token_count": 0,
                "orthographic_word_count_sum": 0,
                "estimated_syllable_count_sum": 0,
                "estimated_phoneme_proxy_count_sum": 0,
                "empty_or_punctuation_count": 0,
            }
        group = grouped[key]
        group["utterance_count"] += 1
        group["token_count"] += int(row["orthographic_word_count"])
        group["orthographic_word_count_sum"] += int(row["orthographic_word_count"])
        group["estimated_syllable_count_sum"] += int(row["estimated_syllable_count"])
        group["estimated_phoneme_proxy_count_sum"] += int(row["estimated_phoneme_proxy_count"])
        group["empty_or_punctuation_count"] += int(row["cleaned_is_empty_or_punctuation"])
        source = source_rows_by_id.get(tuple(row.get(column, "") for column in manifest.id_columns), {})
        token_sets[key].update(tokenize_words(source.get(manifest.text_column, ""), lowercase=manifest.lowercase))

    summaries: list[dict[str, Any]] = []
    for key, group in grouped.items():
        token_count = int(group["token_count"])
        utterance_count = int(group["utterance_count"])
        vocab_size = len(token_sets[key])
        group["age_bin_vocab_size"] = vocab_size
        group["age_bin_ttr"] = vocab_size / token_count if token_count else 0.0
        group["mean_words_per_utterance"] = token_count / utterance_count if utterance_count else 0.0
        group["mean_syllables_per_utterance"] = (
            int(group["estimated_syllable_count_sum"]) / utterance_count if utterance_count else 0.0
        )
        summaries.append(group)
    summaries.sort(key=lambda row: (row.get(child_column, ""), row.get(age_bin_column, "")))
    return summaries


def trajectory_fieldnames(manifest: ComplexityManifest) -> list[str]:
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
        "lexical_cumulative_child_vocab_size",
        "lexical_cumulative_child_token_count",
        "lexical_cumulative_child_ttr",
        "lexical_age_bin_vocab_size_so_far",
        "lexical_age_bin_token_count_so_far",
        "lexical_age_bin_ttr_so_far",
    ]
    seen: set[str] = set()
    return [field for field in fields if not (field in seen or seen.add(field))]


def summary_fieldnames(manifest: ComplexityManifest) -> list[str]:
    child_column = str(manifest.raw.get("child_column", "child_id"))
    age_bin_column = str(manifest.raw.get("age_bin_column", "age_bin"))
    return [
        "complexity_run_id",
        child_column,
        age_bin_column,
        "utterance_count",
        "token_count",
        "age_bin_vocab_size",
        "age_bin_ttr",
        "mean_words_per_utterance",
        "mean_syllables_per_utterance",
        "orthographic_word_count_sum",
        "estimated_syllable_count_sum",
        "estimated_phoneme_proxy_count_sum",
        "empty_or_punctuation_count",
    ]


def run_trajectory(manifest: ComplexityManifest) -> dict[str, Any]:
    manifest.validate_existing_inputs()
    trajectory_rows = build_trajectory_rows(manifest)
    trajectory_csv = trajectory_output_path(manifest)
    summary_csv = aggregate_output_path(manifest)
    trajectory_count = write_csv_dicts(
        trajectory_csv,
        trajectory_rows,
        fieldnames=trajectory_fieldnames(manifest),
    )
    summary_rows = build_age_bin_summary_rows(manifest, trajectory_rows)
    summary_count = write_csv_dicts(
        summary_csv,
        summary_rows,
        fieldnames=summary_fieldnames(manifest),
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest.run_id,
        "input_csv": str(manifest.input_csv),
        "trajectory_output_csv": str(trajectory_csv),
        "aggregate_output_csv": str(summary_csv),
        "trajectory_row_count": trajectory_count,
        "aggregate_row_count": summary_count,
        "input_sha256": sha256_file(manifest.input_csv),
        "trajectory_output_sha256": sha256_file(trajectory_csv),
        "aggregate_output_sha256": sha256_file(summary_csv),
        "manifest": {
            key: str(value) if key.endswith("_csv") else value
            for key, value in asdict(manifest).items()
            if key != "raw"
        },
    }
    audit_path = trajectory_csv.with_name(trajectory_csv.name.replace(".csv.gz", ".audit.json"))
    write_json(audit_path, audit)
    return audit
