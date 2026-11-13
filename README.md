# Robust Image Classification and Spurious Background Reliance on the WILDS Waterbirds Benchmark

**An experimental study of Empirical Risk Minimization (ERM) and Group Distributionally Robust Optimization (Group DRO) under spurious background correlations.**

## Overview

Image classifiers can exploit correlations that are predictive in the training data but unreliable outside it. In the WILDS Waterbirds benchmark, bird class is correlated with background type. This creates a setting in which a model may rely on the background rather than on features of the bird.

This project investigates that problem by personally training and evaluating image-classification models and comparing standard ERM training with a subgroup-aware Group DRO approach.

## Research questions

1. How does standard ERM perform across the Waterbirds subgroups when class and background are spuriously correlated?
2. Does Group DRO change worst-group performance relative to ERM?
3. What do subgroup metrics and Grad-CAM visualisations reveal—and not reveal—about model behaviour?

## Dataset

The project uses the **Waterbirds** benchmark from the WILDS benchmark suite.

The task distinguishes landbirds from waterbirds across four groups defined by bird class and background:

- Landbird on land
- Landbird on water
- Waterbird on land
- Waterbird on water

The minority combinations are particularly important because average accuracy can conceal poor performance on these groups.

Dataset access and licensing are governed by the original dataset and WILDS distribution. Follow the official WILDS instructions rather than committing dataset images to this repository.

## Methods

### Empirical Risk Minimization (ERM)

ERM is used as the standard empirical-risk training baseline.

### Group Distributionally Robust Optimization (Group DRO)

Group DRO is used to emphasise higher-loss groups during training, with the aim of improving performance on the most challenging subgroup rather than optimising only aggregate accuracy.

### Model and inspection

The main experiment documented in this repository uses an ImageNet-pretrained **ResNet-18** backbone. Grad-CAM is used as a qualitative visualisation method to inspect regions associated with model predictions.

Grad-CAM maps are exploratory evidence about model activations. They do not, by themselves, prove that a model uses a particular causal feature or explain its complete decision process.

## Evaluation

The evaluation focuses on:

- Overall test accuracy.
- Accuracy for each of the four subgroups.
- Worst-group accuracy.
- The difference between average and worst-group performance.
- Qualitative Grad-CAM inspection.

**Numerical results are intentionally not presented in this README while the experiment records are being reconciled.** Earlier versions of the related project repositories contain inconsistent metrics and model descriptions. A headline result should be restored only after the exact run, backbone, checkpoint, evaluation split, and metric calculation have been verified against the source code and saved outputs.

## Repository contents

The notebook and supporting files in this repository contain the experiment workflow. Review the actual notebook cells and configuration before assuming a particular training command, runtime, or dependency version.

## Reproducibility checklist

Before treating a result as reproducible, record:

1. Dataset version, split, and group definitions.
2. Model architecture and pretrained-weight source.
3. Training configuration, optimiser, learning rate, batch size, epochs, and checkpoint selection.
4. Group DRO update rule and hyperparameters.
5. Random seeds and hardware/software environment.
6. Exact evaluation script and per-group sample counts.
7. The checkpoint used to produce each reported metric.
8. Grad-CAM target layer and visualisation procedure.

Do not compare results across runs unless these settings are made explicit.

## Limitations

- This is an experimental benchmark study, not evidence of robustness across all image domains.
- Results on Waterbirds do not establish performance under arbitrary distribution shifts.
- Worst-group accuracy depends on the benchmark's group definitions and test-set sample sizes.
- Grad-CAM is a qualitative inspection tool and should not be treated as causal proof.
- A single backbone and benchmark are insufficient to establish general claims about robust machine learning.

## References

- Sagawa, S. et al. (2020). *Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization.* [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- Koh, P. W. et al. (2021). *WILDS: A Benchmark of in-the-Wild Distribution Shifts.* [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV.

## Author

**Alice Iyamuremye**  
Research interests: robust machine learning, subgroup performance, explainable AI, and trustworthy AI.
