from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from child_complexity_predictors.manifest import ComplexityManifest
from child_complexity_predictors.predictors import predictors_for_text, run_extract


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ComplexityPredictorTests(unittest.TestCase):
    def test_predictors_for_text(self) -> None:
        result = predictors_for_text("More milk please!", lowercase=True)
        self.assertEqual(result["orthographic_word_count"], 3)
        self.assertEqual(result["utterance_type_count"], 3)
        self.assertGreater(result["estimated_syllable_count"], 0)
        self.assertEqual(result["cleaned_is_empty_or_punctuation"], 0)

    def test_extract_writes_output_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "child.csv"
            output_csv = root / "complexity.csv.gz"
            manifest_path = root / "manifest.json"

            write_csv(
                input_csv,
                [
                    {
                        "utterance_id": "u1",
                        "dataset": "Brown",
                        "age_bin": "024-029",
                        "utterance_clean": "more milk",
                    },
                    {
                        "utterance_id": "u2",
                        "dataset": "Brown",
                        "age_bin": "024-029",
                        "utterance_clean": "...",
                    },
                ],
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "unit",
                        "input_csv": "child.csv",
                        "output_csv": "complexity.csv.gz",
                        "text_column": "utterance_clean",
                        "id_columns": ["utterance_id"],
                        "carry_columns": ["dataset", "age_bin"],
                    }
                ),
                encoding="utf-8",
            )

            audit = run_extract(ComplexityManifest.from_path(manifest_path))

            self.assertEqual(audit["row_count"], 2)
            self.assertTrue(output_csv.exists())

            with gzip.open(output_csv, "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(rows[0]["orthographic_word_count"], "2")
            self.assertEqual(rows[1]["cleaned_is_empty_or_punctuation"], "1")


if __name__ == "__main__":
    unittest.main()
