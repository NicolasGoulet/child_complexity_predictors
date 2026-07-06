from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from child_complexity_predictors.manifest import ComplexityManifest
from child_complexity_predictors.predictors import predictors_for_text, run_extract
from child_complexity_predictors.trajectory import run_trajectory


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

    def test_trajectory_outputs_cumulative_vocabulary_and_age_bin_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_csv = root / "child.csv"
            manifest_path = root / "manifest.json"

            write_csv(
                input_csv,
                [
                    {
                        "utterance_id": "u1",
                        "dataset": "Brown",
                        "child_id": "c1",
                        "session_id": "s1",
                        "age_months": "24",
                        "age_bin": "024-029",
                        "utterance_clean": "more milk",
                    },
                    {
                        "utterance_id": "u2",
                        "dataset": "Brown",
                        "child_id": "c1",
                        "session_id": "s2",
                        "age_months": "25",
                        "age_bin": "024-029",
                        "utterance_clean": "more cookie",
                    },
                    {
                        "utterance_id": "u3",
                        "dataset": "Brown",
                        "child_id": "c2",
                        "session_id": "s1",
                        "age_months": "24",
                        "age_bin": "024-029",
                        "utterance_clean": "go home",
                    },
                ],
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "run_id": "trajectory-unit",
                        "input_csv": "child.csv",
                        "output_csv": "complexity.csv.gz",
                        "trajectory_output_csv": "trajectory.csv.gz",
                        "aggregate_output_csv": "age_bin_summary.csv.gz",
                        "text_column": "utterance_clean",
                        "id_columns": ["utterance_id"],
                        "carry_columns": [
                            "dataset",
                            "child_id",
                            "session_id",
                            "age_months",
                            "age_bin",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            audit = run_trajectory(ComplexityManifest.from_path(manifest_path))

            self.assertEqual(audit["trajectory_row_count"], 3)
            self.assertEqual(audit["aggregate_row_count"], 2)
            with gzip.open(root / "trajectory.csv.gz", "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            c1_rows = [row for row in rows if row["child_id"] == "c1"]
            self.assertEqual(c1_rows[0]["lexical_cumulative_child_vocab_size"], "2")
            self.assertEqual(c1_rows[1]["lexical_cumulative_child_vocab_size"], "3")
            with gzip.open(root / "age_bin_summary.csv.gz", "rt", newline="", encoding="utf-8") as handle:
                summary = list(csv.DictReader(handle))
            self.assertEqual(len(summary), 2)


if __name__ == "__main__":
    unittest.main()
