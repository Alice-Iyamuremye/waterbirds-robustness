# Experimental Report: Spurious Background Reliance in Bird Classification

## 1. Research objective

This experiment compares Empirical Risk Minimization (ERM) with Group Distributionally Robust Optimization (GroupDRO) on WILDS Waterbirds v1.0. The objective is to measure how the two training objectives perform across subgroups defined by bird class and background.

## 2. Dataset and subgroup definition

The dataset contains 11,788 images and uses the official WILDS train, validation and test splits: 4,795 / 1,199 / 5,794.

The binary label is landbird (0) or waterbird (1). Group IDs are: 0 = landbird on land; 1 = landbird on water; 2 = waterbird on land; 3 = waterbird on water.

The test split contains 2,255 landbird-on-land, 2,255 landbird-on-water, 642 waterbird-on-land and 642 waterbird-on-water examples.

## 3. Experimental configuration

Both methods use the same ImageNet-pretrained ResNet-18 architecture with a two-class classification head. Training uses 50 epochs, batch size 64, AdamW, learning rate 1e-4, weight decay 1e-4, and cosine annealing. Three seeds (42, 123, 2026) were run for each method, for six runs and 300 training epochs in total.

ERM minimizes the example-averaged cross-entropy objective. GroupDRO maintains group weights and updates them using group losses. The implementation is a minibatch GroupDRO variant; it should not be presented as an exact reproduction of every canonical GroupDRO implementation.

For each run, checkpoint selection is based on validation worst-group accuracy. The test set is used only for final evaluation after checkpoint selection.

## 4. Metrics

- **Overall accuracy:** fraction of all test examples classified correctly.
- **Macro-group accuracy:** unweighted mean of the four subgroup accuracies.
- **Worst-group accuracy:** minimum accuracy among the four subgroups.
- **Subgroup accuracy:** accuracy calculated separately for each bird/background combination.

Reported standard deviations are sample standard deviations (ddof=1) across the three seeds. Differences are absolute percentage-point differences.

## 5. Final held-out test results

### 5.1 Main comparison

| Metric | ERM mean ± SD | GroupDRO mean ± SD | Difference |
|---|---:|---:|---:|
| Overall accuracy | 82.44% ± 0.14% | 87.28% ± 0.36% | +4.84 pp |
| Macro-group accuracy | 81.33% ± 0.47% | 87.07% ± 0.21% | +5.74 pp |
| Worst-group accuracy | 62.93% ± 2.82% | 80.82% ± 2.00% | +17.89 pp |

### 5.2 Subgroup comparison

| Subgroup | ERM mean ± SD | GroupDRO mean ± SD | Difference |
|---|---:|---:|---:|
| Landbird on land | 98.51% ± 0.32% | 92.70% ± 1.80% | −5.81 pp |
| Landbird on water | 68.14% ± 0.80% | 82.20% ± 2.03% | +14.06 pp |
| Waterbird on land | 62.93% ± 2.82% | 81.57% ± 2.13% | +18.64 pp |
| Waterbird on water | 95.74% ± 0.24% | 91.80% ± 1.56% | −3.94 pp |

### 5.3 Per-seed worst-group test accuracy

| Seed | ERM | GroupDRO |
|---:|---:|---:|
| 42 | 65.58% | 83.02% |
| 123 | 63.24% | 80.31% |
| 2026 | 59.97% | 79.13% |

GroupDRO has higher worst-group test accuracy in all three paired seeds.

## 6. Error analysis

ERM's lowest-performing subgroup is waterbird on land for all three seeds. GroupDRO improves both background-mismatched subgroups (landbird on water and waterbird on land), while its accuracy is lower on the two background-aligned subgroups. This pattern is consistent with the method's objective of improving the lower-performing groups rather than maximizing accuracy only on the easiest groups.

Background-stratified confusion matrices show the same qualitative asymmetry: ERM has more difficulty classifying the bird class that is less typical for the observed background. These matrices aggregate predictions from three models and are descriptive summaries, not independent samples for inferential testing.

## 7. Interpretability analysis

Preliminary Grad-CAM maps were generated for selected test images using the final convolutional stage (layer4) of ResNet-18. Some examples show different prediction outcomes and highlighted regions between ERM and GroupDRO.

These maps are qualitative, coarse-resolution explanations. The inspected sample is small, and no validated ground-truth bird segmentation masks were available for quantitative inside-bird versus background activation measurement. Grad-CAM therefore does not prove causal background reliance or establish that GroupDRO has eliminated shortcut learning.

## 8. Interpretation

Across the three seeds, GroupDRO improves mean worst-group test accuracy by 17.89 percentage points and macro-group accuracy by 5.74 points. It also improves overall accuracy by 4.84 points in this experiment. The largest mean subgroup gain is on waterbird-on-land (+18.64 pp).

The reduction in accuracy on landbird-on-land and waterbird-on-water indicates a trade-off across subgroups. The results support the conclusion that this GroupDRO implementation improves worst-group performance on this benchmark under the reported configuration.

## 9. Limitations

1. The experiment uses one dataset and one model architecture.
2. Three seeds are a limited basis for estimating run-to-run variability.
3. No statistical significance claim is made.
4. GroupDRO relies on group annotations during training.
5. Grad-CAM is qualitative and cannot establish causality.
6. The segmentation-mask experiment was not completed and is not part of the reported quantitative evidence.
7. Further work should include additional robustness baselines, datasets and controlled background interventions.

## 10. Conclusion

Under the reported 50-epoch, three-seed setup, GroupDRO outperforms ERM on macro-group and worst-group test accuracy. The mean worst-group improvement is 17.89 percentage points. The result is consistent across all three seeds, while subgroup-level results reveal a trade-off between background-aligned and background-mismatched groups.

The findings are benchmark-specific and should be treated as evidence of improved subgroup robustness in this experimental setting, not as proof of causal shortcut elimination or generalization to all image-classification settings.
