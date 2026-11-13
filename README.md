# Robust Image Classification and Spurious Background Reliance on the WILDS Waterbirds Benchmark

**An experimental comparison of Empirical Risk Minimization (ERM) and Group Distributionally Robust Optimization (Group DRO) under spurious background correlations.**

## Research question

When bird class is spuriously correlated with background in the WILDS Waterbirds benchmark, how does standard ERM perform across the four bird/background groups, and how does Group DRO affect worst-group performance?

## Dataset and task

This project uses the Waterbirds benchmark distributed through WILDS. The task classifies images as landbirds or waterbirds. Each example also belongs to one of four groups formed by crossing bird class with background type:

- Landbird on land
- Landbird on water
- Waterbird on land
- Waterbird on water

The notebook reports 11,788 examples overall, with 4,795 training, 1,199 validation, and 5,794 test examples. Group membership is used by the Group DRO training procedure and for subgroup evaluation.

## Methods

- **ERM:** standard empirical risk minimization baseline.
- **Group DRO:** group-aware robust optimization that updates group weights to place greater emphasis on higher-loss groups.
- **Backbone:** ImageNet-pretrained ResNet-18 for both methods.
- **Visual inspection:** Grad-CAM saliency maps generated for selected examples from each group.

The notebook trains each method for up to 50 epochs and selects checkpoints using validation worst-group accuracy. The saved notebook output reports a best validation worst-group accuracy of 0.5940 for ERM and 0.7293 for Group DRO.

## Test-set results

The following values are taken from the saved test evaluation outputs in the accompanying Kaggle notebook (test set n = 5,794).

| Method | Average accuracy | Worst-group accuracy | Average–worst gap |
|---|---:|---:|---:|
| ERM | 83.07% | 62.62% | 20.45 pp |
| Group DRO | 86.36% | 74.77% | 11.59 pp |

Group DRO improves worst-group accuracy by **12.15 percentage points** and average accuracy by **3.29 percentage points** relative to ERM.

### Per-group test accuracy

| Group | ERM | Group DRO | Change |
|---|---:|---:|---:|
| Landbird on land | 98.63% | 94.28% | −4.35 pp |
| Landbird on water | 75.88% | 85.41% | +9.53 pp |
| Waterbird on land | 62.62% | 74.77% | +12.15 pp |
| Waterbird on water | 95.17% | 90.97% | −4.20 pp |

The results show a trade-off: Group DRO improves both minority-group accuracies and the worst-group metric, while reducing accuracy on the two majority groups. Overall accuracy alone would not communicate this behaviour.

## Grad-CAM: qualitative inspection

The notebook generates Grad-CAM visualisations for both trained checkpoints, with two examples per group. These maps can help inspect image regions associated with model predictions. They are qualitative diagnostic artefacts; they do not prove that a model relies on a particular causal feature or establish that shortcut learning has been eliminated.

## Reproducing the experiment

The Kaggle notebook contains the setup/bootstrap workflow, training calls, evaluation, saliency generation, and comparison code:

- [Kaggle notebook: project15ml](https://www.kaggle.com/code/aliceiyamuremye/project15ml)

The saved notebook output records the following workflow:

1. Set up the WILDS Waterbirds data and project training code in the Kaggle working environment.
2. Train the ERM model for 50 epochs.
3. Train the Group DRO model for 50 epochs.
4. Evaluate both selected checkpoints on the 5,794-example test set.
5. Generate Grad-CAM maps and compare subgroup metrics.

The notebook output reports approximately 40 minutes per training run on the configured Kaggle GPU environment. Runtime is hardware- and environment-dependent. The notebook's checkpoint check skips training when the expected checkpoint files already exist; therefore, a fresh reproduction should confirm that both checkpoints were newly generated for the run being reported.

For a fully reproducible release, the repository should also preserve the exact generated training source files, dependency versions, data preparation details, random seeds, and checkpoint-selection configuration. Do not commit the Waterbirds image dataset or large model checkpoints unless licensing and repository storage choices have been reviewed.

## Limitations

- This is one benchmark and one backbone; results do not establish robustness across arbitrary distribution shifts.
- Group DRO depends on group labels being available during training.
- The test results show gains on the worst-performing groups but losses on the majority groups.
- The Grad-CAM maps are qualitative and cannot establish causal feature use.
- The notebook reports a single training run; variation across independent random seeds has not been established here.
- Additional methods, architectures, and datasets would be needed for broader conclusions.

## References

- Sagawa, S. et al. (2020). *Distributionally Robust Neural Networks for Group Shifts*. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
- Koh, P. W. et al. (2021). *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization*. ICCV.

## Author

**Alice Iyamuremye**  
Research interests: robust machine learning, subgroup performance, explainable AI, and trustworthy AI.
