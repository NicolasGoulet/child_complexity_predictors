#!/bin/bash
#SBATCH --job-name=pbm-complexity
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/src/child_complexity_predictors" ]]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch $0 /path/to/default_naturalistic_merged_006_023 [output_root]" >&2
  exit 2
fi

BUNDLE_ROOT="$1"
JOB_ID="${SLURM_JOB_ID:-manual}"
SCRATCH_ROOT="${SCRATCH:-${HOME}/scratch}"
OUTPUT_ROOT="${2:-${SCRATCH_ROOT}/communicative_efficiency_runs/pbm_complexity/${JOB_ID}}"
PYTHON_BIN="${PYTHON:-python3}"
RUN_ID="${RUN_ID:-pbm_complexity}"
DATASETS="${DATASETS:-Brown,Manchester,Providence}"
CONTEXT_COLUMN="${CONTEXT_COLUMN:-context_k3}"
SOURCE_MODELS="${SOURCE_MODELS:-real,random,unigram,bigram,trigram}"

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" -m child_complexity_predictors prepare-pbm-manifests \
  --bundle-root "${BUNDLE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --datasets "${DATASETS}" \
  --context-column "${CONTEXT_COLUMN}" \
  --source-models "${SOURCE_MODELS}"

REAL_MANIFEST="${OUTPUT_ROOT}/manifests/pbm_real_complexity_manifest.json"
CANDIDATE_MANIFEST="${OUTPUT_ROOT}/manifests/pbm_candidate_complexity_manifest.json"

"${PYTHON_BIN}" -m child_complexity_predictors validate-manifest --manifest "${REAL_MANIFEST}" --check-inputs
"${PYTHON_BIN}" -m child_complexity_predictors extract --manifest "${REAL_MANIFEST}"
"${PYTHON_BIN}" -m child_complexity_predictors trajectory --manifest "${REAL_MANIFEST}"

"${PYTHON_BIN}" -m child_complexity_predictors validate-manifest --manifest "${CANDIDATE_MANIFEST}" --check-inputs
"${PYTHON_BIN}" -m child_complexity_predictors extract --manifest "${CANDIDATE_MANIFEST}"

cat > "${OUTPUT_ROOT}/cleanup_after_rsync.sh" <<EOF
#!/bin/bash
set -euo pipefail
rm -rf -- '${OUTPUT_ROOT}'
EOF
chmod +x "${OUTPUT_ROOT}/cleanup_after_rsync.sh"

echo "Outputs written under ${OUTPUT_ROOT}"
echo "After rsyncing needed outputs, clean scratch with:"
echo "  bash '${OUTPUT_ROOT}/cleanup_after_rsync.sh'"
