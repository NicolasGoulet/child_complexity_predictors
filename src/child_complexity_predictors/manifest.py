"""Manifest parsing for complexity predictor extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ComplexityManifest:
    run_id: str
    input_csv: Path
    output_csv: Path
    text_column: str
    id_columns: tuple[str, ...]
    carry_columns: tuple[str, ...] = ()
    lowercase: bool = True
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: str | Path) -> "ComplexityManifest":
        manifest_path = Path(path).resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = manifest_path.parent

        def resolve(value: str) -> Path:
            candidate = Path(value)
            if candidate.is_absolute():
                return candidate
            return (base / candidate).resolve()

        required = ["run_id", "input_csv", "output_csv", "text_column", "id_columns"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"Manifest is missing required fields: {', '.join(missing)}")
        if not payload["id_columns"]:
            raise ValueError("id_columns must not be empty")

        return cls(
            run_id=str(payload["run_id"]),
            input_csv=resolve(payload["input_csv"]),
            output_csv=resolve(payload["output_csv"]),
            text_column=str(payload["text_column"]),
            id_columns=tuple(payload["id_columns"]),
            carry_columns=tuple(payload.get("carry_columns", ())),
            lowercase=bool(payload.get("lowercase", True)),
            raw=payload,
        )

    def validate_existing_inputs(self) -> None:
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Missing input file: {self.input_csv}")

    @property
    def audit_json(self) -> Path:
        if self.output_csv.suffix == ".gz":
            stem = self.output_csv.with_suffix("").with_suffix("")
        else:
            stem = self.output_csv.with_suffix("")
        return stem.with_name(stem.name + ".audit.json")
