# TODO.md

Production checklist for `child_complexity_predictors`.

This repo adds MLU, grammatical, lexical, phonological, and structural
complexity predictors to the communicative-efficiency project. These predictors
are add-ons to existing information/effort analyses, not replacements.

## Core Contract

- [x] Keep this repo CPU-first and predictor-only.
- [x] Preserve row ids, child/session/age/corpus provenance, and empty-row
      diagnostics.
- [x] Provide a Slurm script that `cd` to repo root and sets `PYTHONPATH=src`.
- [ ] Add production manifests pointing to strict naturalistic child rows and
      any required CHAT morphology exports.
- [ ] Add a manifest audit command that checks required columns, duplicate row
      ids, age-bin coverage, and missing text.

## Utterance-Level Predictors

- [x] Extract orthographic word count, character count, mean word length,
      utterance type count, estimated syllables, phoneme proxy, and
      empty/punctuation flags.
- [ ] Add optional morpheme-count passthrough from an existing column when
      CHAT-derived morphology is available.
- [ ] Add missingness audits for morphemes by corpus, child, age bin, and
      session.
- [ ] Add documented phonological/phonotactic predictors beyond the current
      deterministic syllable and phoneme proxies.

## Lexical And Developmental Trajectories

- [x] Add per-child cumulative vocabulary size.
- [x] Add per-child/session and per-child/age-bin type-token ratio.
- [ ] Add moving-average lexical diversity.
- [ ] Add age-bin vocabulary size and lexical rarity predictors.
- [x] Export compact trajectory tables that can join back to Route 1/Route 2
      analysis rows.
- [x] Add unit tests for utterance-level predictors, cumulative vocabulary
      trajectories, and age-bin summaries.

## Dependency / Structural Complexity

- [ ] Add a parser feasibility audit on a small corpus/age-stratified sample.
- [ ] Do not promote dependency length to a primary predictor until parser
      failure modes on short child utterances are documented.
- [ ] If parser dependencies are added, keep parser model/version/config in
      the audit output.

## Verification Commands

```bash
PYTHONPYCACHEPREFIX=/tmp/child_complexity_predictors_pycache PYTHONPATH=src python3 -m unittest discover -s tests
bash -n slurm/*.sbatch
PYTHONPATH=src python3 -m child_complexity_predictors validate-manifest --manifest configs/complexity_example.json
```
