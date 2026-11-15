# Final Waterbirds results and figures

This directory contains the aggregated held-out test results from the final three-seed experiment (seeds 42, 123, and 2026; six runs total).

## Files
- `final_metrics_summary.csv`: overall, macro-group, and worst-group accuracy (mean and sample SD).
- `final_subgroup_accuracy.csv`: per-subgroup accuracy (mean and sample SD).
- `final_per_seed_worst_group.csv`: worst-group accuracy for each seed and method.
- `final_metrics_summary.json`: experiment metadata, headline results, and limitations.
- `../figures/final_overall_comparison.svg`: overall, macro-group, and worst-group comparison.
- `../figures/final_subgroup_comparison.svg`: subgroup comparison.

## Interpretation
GroupDRO improved mean worst-group accuracy by 17.89 percentage points (62.93% to 80.82%). Gains were concentrated in the two background-mismatched subgroups, while accuracy decreased in the two aligned subgroups. The results are benchmark-specific.

## Reporting cautions
Three seeds are reported; no statistical significance claim is made. The GroupDRO implementation is minibatch-based and is not presented as an exact canonical reproduction. Grad-CAM is qualitative inspection, not causal evidence. These files contain aggregate results only; raw per-example predictions and model checkpoints are not included.
