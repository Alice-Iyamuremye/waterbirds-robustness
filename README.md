# Spurious Background Reliance in Bird Classification

**A reproducible comparison of Empirical Risk Minimization (ERM) and Group Distributionally Robust Optimization (GroupDRO) on the WILDS Waterbirds benchmark.**

![Python](https://img.shields.io/badge/Python-3.12-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c) ![Dataset](https://img.shields.io/badge/Benchmark-WILDS%20Waterbirds-green)

## Research question

How does group-robust optimization affect subgroup performance when bird-class labels are spuriously correlated with image backgrounds?

Waterbirds contains landbirds and waterbirds photographed against land and water backgrounds. The four combinations expose models to background–class correlations that can make average accuracy conceal failures on less common subgroups.

## Experimental setup

- **Dataset:** WILDS Waterbirds v1.0 (11,788 images).
- **Official splits:** 4,795 train / 1,199 validation / 5,794 test.
- **Architecture:** ImageNet-pretrained ResNet-18 with a two-class output layer.
- **Methods:** ERM and a minibatch GroupDRO implementation.
- **Training:** 50 epochs per run; seeds 42, 123 and 2026 for each method (six runs total).
- **Selection:** Best checkpoint selected by validation worst-group accuracy.
- **Final evaluation:** Held-out test set, evaluated after checkpoint selection.
- **Primary metric:** Worst-group accuracy; overall, macro-group and four subgroup accuracies are also reported.

Group IDs: 0 = landbird on land; 1 = landbird on water; 2 = waterbird on land; 3 = waterbird on water.

## Final held-out test results

Mean and sample standard deviation across three independently trained seeds:

| Metric | ERM | GroupDRO | Difference |
|---|---:|---:|---:|
| Overall accuracy | 82.44% ± 0.14% | **87.28% ± 0.36%** | +4.84 pp |
| Macro-group accuracy | 81.33% ± 0.47% | **87.07% ± 0.21%** | +5.74 pp |
| Worst-group accuracy | 62.93% ± 2.82% | **80.82% ± 2.00%** | **+17.89 pp** |

### Per-subgroup test accuracy

| Test subgroup | ERM | GroupDRO | Difference |
|---|---:|---:|---:|
| Landbird on land | 98.51% ± 0.32% | 92.70% ± 1.80% | −5.81 pp |
| Landbird on water | 68.14% ± 0.80% | 82.20% ± 2.03% | +14.06 pp |
| Waterbird on land | 62.93% ± 2.82% | 81.57% ± 2.13% | +18.64 pp |
| Waterbird on water | 95.74% ± 0.24% | 91.80% ± 1.56% | −3.94 pp |

**Interpretation.** GroupDRO improves accuracy on both background–class-mismatched subgroups, while reducing accuracy on the two aligned subgroups. This is consistent with a robustness–average subgroup trade-off. The worst-group improvement is consistent across all three seeds.

These are descriptive results from three seeds. They do not establish statistical significance or prove that GroupDRO has eliminated background reliance.

## Analysis completed

- Per-seed held-out test evaluation.
- Mean and sample standard deviation across seeds.
- Per-subgroup accuracy and error-rate analysis.
- Confusion matrices stratified by background.
- Preliminary Grad-CAM comparisons on selected examples.

Grad-CAM is qualitative evidence about class-discriminative regions, not a causal test of feature reliance. The current visual inspection covers a small number of images and should not be generalized to the full test set.

## Reproducibility and artifacts

The experiment code and notebook are provided in this repository. The final Kaggle run used six 50-epoch checkpoints. The dataset is not redistributed here; obtain WILDS Waterbirds through the [WILDS project](https://wilds.stanford.edu/datasets/).

The final run produced per-seed JSON test reports and CSV summaries, along with three performance figures and background-stratified confusion matrices. Model checkpoints are large and are not included in this repository.

## Limitations

- One benchmark dataset and one backbone (ResNet-18).
- Three random seeds provide limited information about run-to-run variability.
- GroupDRO requires group labels during training.
- Confusion matrices pooled across seeds are descriptive; pooled predictions are not independent observations.
- Grad-CAM is approximate and does not establish causal reliance on backgrounds.
- No validated ground-truth bird segmentation masks were available for quantitative attention localization.
- Results should be replicated on additional datasets and under controlled background interventions before making broader claims.

## References

1. Sagawa, S. et al. (2020). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization*. ICLR. https://arxiv.org/abs/1911.08731
2. Koh, P. W. et al. (2021). *WILDS: A Benchmark of In-the-Wild Distribution Shifts*. ICML. https://arxiv.org/abs/2012.07421
3. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV.
4. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR.

---
This repository documents an experimental research project. The reported findings are benchmark results, not evidence of deployment readiness.
