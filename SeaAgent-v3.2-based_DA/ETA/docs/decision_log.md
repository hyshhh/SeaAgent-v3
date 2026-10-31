# Experimental decision log

## 2026-10-30: use cached sparse observations rather than 5-second resampling

- Paper setting: construct candidate observations at 5-second intervals.
- Observed source constraint: the trajectory memory is already sparse, with at
  most three cached keyframes per track. Strict 5-second binning retains 257 of
  571 candidates.
- Pilot problem: allowing an empty KEP history for every later candidate made
  all but one AH sample positive, so the model could learn a constant classifier.
- The final strict-stride audit with non-empty later histories still produced
  262 positive and one negative AH sample, confirming that the issue was not
  caused by history construction alone.
- Protocol decision made before main training:
  1. retain all cached keyframes for the main run;
  2. pair an empty history only with a track's first candidate;
  3. pair every later candidate only with non-empty chronological KEP subsets;
  4. keep `configs/experiments/stride5.yaml` so the stricter paper sampling can
     be audited and compared.

The decision changes sample composition, not the paper's model equations. It is
reported as a deviation rather than presented as a faithful reconstruction of
the paper's human-labelled TEA dataset.

## 2026-10-30: fixed test protocol before structural ablations

Structural variants, capacities, seed, optimizer, class weighting, checkpoint
rule, metrics, and Stop threshold were fixed in configuration before launching
the complete comparison series. Intermediate test results were not used to
change any variant or hyperparameter.
