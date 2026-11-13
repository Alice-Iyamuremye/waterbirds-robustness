# Spurious Background Reliance in Bird Classification

**Evaluating ERM and Group DRO on the WILDS Waterbirds benchmark**

This project investigates whether empirical risk minimization (ERM) performs unevenly across bird–background subgroups and evaluates Group Distributionally Robust Optimization (Group DRO) as a mitigation approach.

## Recorded results

The table below reproduces the saved test-evaluation results in `project15ml.ipynb`. These are the recorded outputs of this project run, not results from a multi-seed study.

| Method | Average accuracy | Worst-group accuracy | Average–worst gap |
|---|---:|---:|---:|
| ERM | 83.07% | 62.62% | 20.45 pp |
| Group DRO | 86.36% | 74.77% | 11.59 pp |

Group DRO's recorded worst-group accuracy is **12.15 percentage points higher** than ERM's; average accuracy is **3.29 points higher**.

### Test accuracy by subgroup

| Subgroup | ERM | Group DRO | Difference |
|---|---:|---:|---:|
| Landbird on land | 98.63% | 94.28% | −4.35 pp |
| Landbird on water | 75.88% | 85.41% | +9.53 pp |
| Waterbird on land | 62.62% | 74.77% | +12.15 pp |
| Waterbird on water | 95.17% | 90.97% | −4.20 pp |

The largest improvement is on waterbird-on-land, the worst-performing subgroup under ERM. Group DRO does not improve every subgroup: accuracy is lower on the two majority subgroups in these recorded results.

## Dataset and approach

- **Dataset:** WILDS Waterbirds, 11,788 images.
- **Official split sizes:** 4,795 train / 1,199 validation / 5,794 test.
- **Groups:** four combinations of bird type (landbird/waterbird) and background (land/water).
- **Backbone:** ImageNet-pretrained ResNet-18 with a two-class classification head (11,177,538 parameters in the notebook's model-build check).
- **Methods:** ERM and Group DRO, with subgroup labels used by Group DRO during training.
- **Checkpoint selection:** validation worst-group accuracy, as described in the notebook.
- **Analysis:** average and worst-group accuracy, subgroup-level accuracy, training curves, and Grad-CAM visualizations.

## Interpretation and limitations

The recorded subgroup results are consistent with a robustness benefit from Group DRO on the minority groups in this run. They do not, by themselves, establish that the models causally rely on background features or that Group DRO will generalize to other datasets.

Grad-CAM is used here as a **qualitative visualization aid**. Saliency maps can suggest which image regions contribute to a prediction, but they do not prove causal feature reliance or establish that a model has stopped using background information.

Important limitations:
- Results represent a single recorded experiment; no multi-seed mean, standard deviation, or confidence interval is reported.
- Only one benchmark dataset and one backbone (ResNet-18) are evaluated.
- Group DRO depends on group labels during training.
- The four subgroup results show trade-offs: Group DRO improves the two minority-group accuracies while reducing accuracy on the two majority groups in this run.
- The notebook's environment setup is Kaggle-specific and expects the required dataset/source files and artifacts.

## Repository contents

- `project15ml.ipynb` — experiment workflow, saved outputs, evaluation, and visualizations.
- `report.md` — written project report.
- `erm_best_saliency-2.png`, `groupdro_best_saliency.png` — saved saliency visualizations.

## Running the notebook

The notebook was developed for a Kaggle environment with GPU support. Review the setup cell before execution: it installs WILDS, retrieves the Waterbirds dataset when needed, and expects the project's ERM source files to be available through the Kaggle input attachment described in the notebook. Training can take substantially longer than simply reviewing the saved outputs.

## References

1. Sagawa et al. (2020), *Distributionally Robust Neural Networks for Group Shifts*, ICLR. https://arxiv.org/abs/1911.08731
2. Koh et al. (2021), *WILDS: A Benchmark of in-the-Wild Distribution Shifts*, ICML. https://arxiv.org/abs/2012.07421
3. Selvaraju et al. (2017), *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV.
4. He et al. (2016), *Deep Residual Learning for Image Recognition*, CVPR.
5. Wah et al. (2011), *The Caltech-UCSD Birds-200-2011 Dataset*, California Institute of Technology.

---
**Scope note:** This is an educational research project on benchmark robustness, not evidence of deployment-ready performance or a comprehensive fairness audit.
