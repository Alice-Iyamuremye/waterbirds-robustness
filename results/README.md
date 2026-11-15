# Final Waterbirds results and figures

These files are based on the executed final experiment outputs embedded in the submitted notebook (`project15ml-5.ipynb`). The notebook records six held-out test evaluations: ERM and GroupDRO, each trained with seeds 42, 123, and 2026.

## Results files
- `final_metrics_summary.csv`: aggregate overall, macro-group, and worst-group accuracy (mean and sample SD).
- `final_subgroup_accuracy.csv`: aggregate accuracy for each of the four subgroups.
- `final_test_results_per_seed.csv`: per-run metrics transcribed from the executed notebook's consolidated results table (reported to four decimal places).
- `final_subgroup_error_analysis.csv`: pooled subgroup prediction counts, correct predictions, errors, accuracy and error rate from the notebook's subgroup error analysis.
- `final_per_seed_worst_group.csv`: worst-group accuracy by seed and method.
- `final_metrics_summary.json`: experiment configuration, headline metrics and reporting limitations.

## Figures
- `../figures/final_overall_comparison.svg`: mean overall, macro-group and worst-group accuracy with sample-SD error bars.
- `../figures/final_subgroup_comparison.svg`: mean accuracy for each subgroup.
- `../figures/final_seed_stability.svg`: worst-group accuracy for each seed and method.

## Headline result
Mean worst-group accuracy increased from 62.93% (ERM) to 80.82% (GroupDRO), a gain of 17.89 percentage points. GroupDRO improved both background-mismatched subgroups while reducing accuracy on the two background-aligned subgroups.

## Method and reporting cautions
The experiment uses an ImageNet-pretrained ResNet-18 and a minibatch GroupDRO implementation. The notebook's consolidated per-seed table is rounded to four decimal places; aggregate tables report the mean and sample standard deviation across three seeds. No statistical significance claim is made. Grad-CAM is qualitative model-attention inspection, not causal evidence that a model uses or ignores a feature. Results are specific to the WILDS Waterbirds benchmark and do not establish elimination of shortcut reliance or generalization to other datasets.

The notebook includes a Grad-CAM comparison output and confusion-matrix/error-analysis outputs. The per-example predictions and model checkpoint binaries are not included in this repository snapshot; the notebook documents their Kaggle working paths, which are session-specific.
