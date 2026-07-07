"""Prepare complexity manifests from the strict naturalistic bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .io import open_text, sha256_file, write_json

DEFAULT_PBM_DATASETS = ("Brown", "Manchester", "Providence")
DEFAULT_AGE_BINS = (
    {"label": "006-023", "start": 6, "end": 23},
    {"label": "024-029", "start": 24, "end": 29},
    {"label": "030-035", "start": 30, "end": 35},
    {"label": "036-041", "start": 36, "end": 41},
    {"label": "042-047", "start": 42, "end": 47},
    {"label": "048-053", "start": 48, "end": 53},
    {"label": "054-059", "start": 54, "end": 59},
    {"label": "060-065", "start": 60, "end": 65},
)
SOURCE_COLUMNS = {
    "real": "chi_utterance_clean",
    "random": "random_model_utterance_bin6",
    "unigram": "unigram_model_utterance_bin6",
    "bigram": "bigram_model_utterance_bin6",
    "trigram": "trigram_model_utterance_bin6",
}
REAL_FIELDNAMES = (
    "row_uid",
    "dataset",
    "child_id",
    "source_group",
    "session_id",
    "age_months",
    "age_bin",
    "file",
    "line_no",
    "utt_id",
    "context_id",
    "context_k3",
    "utterance_text",
)
CANDIDATE_FIELDNAMES = (
    "row_uid",
    "source_model",
    "candidate_uid",
    "target_variant",
    "dataset",
    "child_id",
    "source_group",
    "session_id",
    "age_months",
    "age_bin",
    "file",
    "line_no",
    "utt_id",
    "context_id",
    "context_k3",
    "utterance_text",
    "real_utterance",
)


def _stable_id(parts: list[str], *, length: int = 24) -> str:
    payload = "\x1f".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _context_id(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()[:24]


def _token_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text or ""))


def _read_bundle_manifest(bundle_root: Path) -> list[dict[str, str]]:
    manifest_csv = bundle_root / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing bundle manifest: {manifest_csv}")
    with manifest_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_age_bins(bundle_root: Path) -> list[dict[str, Any]]:
    age_bins_json = bundle_root / "age_ngram_dicts" / "merged_early_006_023" / "age_bins.json"
    if age_bins_json.exists():
        payload = json.loads(age_bins_json.read_text(encoding="utf-8"))
        return list(payload["bins"])
    return [dict(item) for item in DEFAULT_AGE_BINS]


def _age_bin_for(age_months: str, age_bins: list[dict[str, Any]]) -> str:
    try:
        month = int(float(age_months))
    except (TypeError, ValueError):
        return ""
    for age_bin in age_bins:
        if int(age_bin["start"]) <= month <= int(age_bin["end"]):
            return str(age_bin["label"])
    return ""


def _resolve_bundle_path(bundle_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = bundle_root / path
    if candidate.exists():
        return candidate
    parts = path.parts
    for anchor in ("preprocessed_data", "age_ngram_dicts"):
        if anchor in parts:
            return bundle_root / Path(*parts[parts.index(anchor) :])
    return candidate


def _iter_scoring_files(bundle_root: Path, datasets: set[str]):
    for manifest_row in _read_bundle_manifest(bundle_root):
        dataset = manifest_row.get("dataset", "")
        if dataset not in datasets:
            continue
        if manifest_row.get("child_scoring_ready") != "1":
            continue
        scoring_csv = _resolve_bundle_path(bundle_root, manifest_row.get("child_scoring_csv", ""))
        if not scoring_csv.exists():
            raise FileNotFoundError(f"Missing child scoring CSV: {scoring_csv}")
        yield scoring_csv


def _base_row(row: dict[str, str], *, age_bins: list[dict[str, Any]], context_column: str) -> dict[str, str] | None:
    real_utterance = row.get("chi_utterance_clean", "")
    if _token_count(real_utterance) == 0:
        return None
    age_bin = _age_bin_for(row.get("age_months", ""), age_bins)
    if not age_bin:
        return None
    context_text = row.get(context_column, "")
    row_uid = _stable_id(
        [
            row.get("dataset", ""),
            row.get("child_id", ""),
            row.get("session_id", ""),
            row.get("file", ""),
            row.get("line_no", ""),
            row.get("utt_id", ""),
        ]
    )
    return {
        "row_uid": row_uid,
        "dataset": row.get("dataset", ""),
        "child_id": row.get("child_id", ""),
        "source_group": row.get("source_group", ""),
        "session_id": row.get("session_id", ""),
        "age_months": row.get("age_months", ""),
        "age_bin": age_bin,
        "file": row.get("file", ""),
        "line_no": row.get("line_no", ""),
        "utt_id": row.get("utt_id", ""),
        "context_id": _context_id(context_text),
        "context_k3": context_text,
        "utterance_text": real_utterance,
    }


def _write_real_csv(
    path: Path,
    *,
    bundle_root: Path,
    datasets: set[str],
    age_bins: list[dict[str, Any]],
    context_column: str,
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REAL_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for scoring_csv in _iter_scoring_files(bundle_root, datasets):
            with scoring_csv.open(newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                required = {"chi_utterance_clean", "age_months", context_column}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{scoring_csv} is missing required columns: {sorted(missing)}")
                for row in reader:
                    base = _base_row(row, age_bins=age_bins, context_column=context_column)
                    if base is None:
                        continue
                    writer.writerow(base)
                    count += 1
    return count


def _write_candidate_csv(
    path: Path,
    *,
    bundle_root: Path,
    datasets: set[str],
    age_bins: list[dict[str, Any]],
    context_column: str,
    source_models: tuple[str, ...],
) -> tuple[int, dict[str, int]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {source_model: 0 for source_model in source_models}
    skipped = 0
    with open_text(path, "wt") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANDIDATE_FIELDNAMES), extrasaction="ignore")
        writer.writeheader()
        for scoring_csv in _iter_scoring_files(bundle_root, datasets):
            with scoring_csv.open(newline="", encoding="utf-8") as input_handle:
                reader = csv.DictReader(input_handle)
                required = {"chi_utterance_clean", "age_months", context_column}
                required.update(SOURCE_COLUMNS[source_model] for source_model in source_models)
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{scoring_csv} is missing required columns: {sorted(missing)}")
                for row in reader:
                    base = _base_row(row, age_bins=age_bins, context_column=context_column)
                    if base is None:
                        continue
                    real_utterance = base["utterance_text"]
                    for source_model in source_models:
                        candidate = row.get(SOURCE_COLUMNS[source_model], "")
                        if _token_count(candidate) == 0:
                            skipped += 1
                            continue
                        writer.writerow(
                            {
                                **base,
                                "source_model": source_model,
                                "candidate_uid": _stable_id([base["row_uid"], source_model]),
                                "target_variant": "real" if source_model == "real" else "generated_baseline",
                                "utterance_text": candidate,
                                "real_utterance": real_utterance,
                            }
                        )
                        counts[source_model] += 1
    return sum(counts.values()), {**counts, "skipped_empty_candidates": skipped}


def prepare_pbm_complexity_manifests(
    *,
    bundle_root: str | Path,
    output_root: str | Path,
    run_id: str = "pbm_complexity",
    datasets: set[str] | None = None,
    context_column: str = "context_k3",
    source_models: tuple[str, ...] = ("real", "random", "unigram", "bigram", "trigram"),
) -> dict[str, Any]:
    bundle_root = Path(bundle_root).resolve()
    output_root = Path(output_root).resolve()
    datasets = datasets or set(DEFAULT_PBM_DATASETS)
    age_bins = _load_age_bins(bundle_root)

    real_csv = output_root / "inputs" / "pbm_real_child_rows.csv.gz"
    candidate_csv = output_root / "inputs" / "pbm_candidate_rows.csv.gz"
    real_manifest_json = output_root / "manifests" / "pbm_real_complexity_manifest.json"
    candidate_manifest_json = output_root / "manifests" / "pbm_candidate_complexity_manifest.json"

    real_count = _write_real_csv(
        real_csv,
        bundle_root=bundle_root,
        datasets=datasets,
        age_bins=age_bins,
        context_column=context_column,
    )
    candidate_count, candidate_counts = _write_candidate_csv(
        candidate_csv,
        bundle_root=bundle_root,
        datasets=datasets,
        age_bins=age_bins,
        context_column=context_column,
        source_models=source_models,
    )
    if real_count == 0:
        raise ValueError("No real child rows were written for complexity predictors.")
    if candidate_count == 0:
        raise ValueError("No candidate rows were written for complexity predictors.")

    carry_columns = [
        "dataset",
        "child_id",
        "source_group",
        "session_id",
        "age_months",
        "age_bin",
        "file",
        "line_no",
        "utt_id",
        "context_id",
        "context_k3",
    ]
    real_manifest = {
        "run_id": f"{run_id}_real",
        "input_csv": str(real_csv),
        "output_csv": str(output_root / "predictors" / "pbm_real_complexity.csv.gz"),
        "trajectory_output_csv": "pbm_real_complexity_trajectory.csv.gz",
        "aggregate_output_csv": "pbm_real_complexity_age_bin_summary.csv.gz",
        "text_column": "utterance_text",
        "id_columns": ["row_uid"],
        "carry_columns": carry_columns,
        "child_column": "child_id",
        "age_column": "age_months",
        "age_bin_column": "age_bin",
        "session_column": "session_id",
    }
    candidate_manifest = {
        "run_id": f"{run_id}_candidates",
        "input_csv": str(candidate_csv),
        "output_csv": str(output_root / "predictors" / "pbm_candidate_complexity.csv.gz"),
        "text_column": "utterance_text",
        "id_columns": ["row_uid", "source_model"],
        "carry_columns": [
            "candidate_uid",
            "target_variant",
            *carry_columns,
            "real_utterance",
        ],
    }
    write_json(real_manifest_json, real_manifest)
    write_json(candidate_manifest_json, candidate_manifest)

    audit = {
        "status": "ok",
        "bundle_root": str(bundle_root),
        "output_root": str(output_root),
        "real_manifest_json": str(real_manifest_json),
        "candidate_manifest_json": str(candidate_manifest_json),
        "real_csv": str(real_csv),
        "candidate_csv": str(candidate_csv),
        "real_row_count": real_count,
        "candidate_row_count": candidate_count,
        "candidate_source_counts": candidate_counts,
        "real_sha256": sha256_file(real_csv),
        "candidate_sha256": sha256_file(candidate_csv),
        "datasets": sorted(datasets),
    }
    audit_json = output_root / "manifests" / "pbm_complexity_manifests.audit.json"
    write_json(audit_json, audit)
    audit["audit_json"] = str(audit_json)
    return audit
