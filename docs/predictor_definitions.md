# Predictor Definitions

## Orthographic MLU

Orthographic MLU-style predictors are derived from cleaned utterance text:

- `orthographic_word_count`
- `orthographic_char_count`
- `mean_word_length`

These are CPU-only and should be available for all nonempty cleaned utterances.

## Grammatical Complexity

Planned grammatical complexity predictors:

- morpheme count from CHAT morphology where available
- mean morphemes per utterance
- parse-derived dependency length after feasibility audit

Do not treat parser-derived dependency length as primary until short
child-utterance parser failures have been audited by corpus and age bin.

## Phonological / Phonotactic Complexity

The scaffold includes a deterministic estimated syllable count and a simple
phoneme proxy. These are useful for a first pass, but production phonological
predictors need documentation of unmapped or uncertain forms.

## Lexical Complexity

Planned lexical predictors:

- cumulative child vocabulary size
- age-bin vocabulary size
- type-token ratio
- moving-average lexical diversity
- age-conditioned lexical rarity

Lexical complexity is conceptually distinct from utterance length: a short
utterance can still be lexically advanced.
