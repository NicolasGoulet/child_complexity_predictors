# child_complexity_predictors

CPU-first child-language complexity predictors for communicative-efficiency
analyses.

This repo is for MLU-style grammatical and lexical complexity predictors. It
does not score Mistral surprisal and does not generate baseline utterances.

## Predictor Families

Implemented in the scaffold:

- orthographic word count
- orthographic character count
- mean word length
- utterance type count
- simple estimated syllable count
- simple phoneme proxy count

Planned:

- CHAT morpheme counts and missingness audits
- cumulative child vocabulary size
- age-bin vocabulary size
- type-token ratio and moving lexical diversity
- dependency-length feasibility audits
- phonotactic complexity measures

## Repo Boundary

- `communicative_efficiency`: reports, joined analysis tables, and project
  design notes.
- `compute_surprisal_mila`: neural surprisal scoring.
- `generate_baselines_mila`: baseline utterance generation.
- `bayes_efficiency_mila`: Bayes decomposition scoring.
- `child_complexity_predictors`: MLU, lexical, and structural predictors.

## Quick Start

Validate a manifest:

```bash
python3 -m child_complexity_predictors validate-manifest --manifest configs/complexity_example.json
```

Extract predictors:

```bash
python3 -m child_complexity_predictors extract --manifest configs/complexity_example.json
```

Run the PBM real-data predictor job on Mila after the strict-naturalistic
bundle has been extracted under scratch. This writes real-child MLU/lexical
trajectories and utterance-level complexity predictors for the PBM real +
n-gram candidate cloud.

```bash
cd "$HOME/communicative_efficiency_repos/child_complexity_predictors"
sbatch --output="$SCRATCH/pbm-complexity-%j.out" \
  slurm/pbm_complexity_predictors.sbatch \
  "$SCRATCH/communicative_efficiency_data/big_cleaned_dataset/default_naturalistic_merged_006_023"
```

Run tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Data Policy

Do not commit CHILDES data, derived full-corpus predictor tables, or large
exports. Commit only code, tests, configs, and small documentation.

On Mila, keep the permanent Git checkout in `$HOME` beside the other modular
repos. Put production manifests, temporary files, full cleaned inputs, and
derived predictor exports under `$SCRATCH`, then remove scratch job directories
after outputs have been rsynced back or are no longer needed.
