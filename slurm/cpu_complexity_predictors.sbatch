#!/bin/bash
#SBATCH --job-name=complexity
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

MANIFEST="${1:-configs/complexity_example.json}"
PYTHON_BIN="${PYTHON:-python3}"

mkdir -p logs
"${PYTHON_BIN}" -m child_complexity_predictors validate-manifest --manifest "${MANIFEST}"
"${PYTHON_BIN}" -m child_complexity_predictors extract --manifest "${MANIFEST}"
"${PYTHON_BIN}" -m child_complexity_predictors trajectory --manifest "${MANIFEST}"
