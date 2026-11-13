# Spurious Background Reliance in Bird Classification

**Detecting and Mitigating Background Shortcuts with ERM and Group DRO on the Waterbirds Benchmark**

**Course:** Machine Learning / Robustness  
**Project:** Project 15 — Spurious Background Reliance in Bird Classification  
**Date:** 2026

---

## 1. Problem Statement

Image classifiers often learn **spurious correlations** instead of the actual class-defining features. The most common form is **background reliance**: a model may classify birds based on the background (water vs land) rather than the bird itself.

### Research Questions

1. Does standard training (ERM) exploit this background shortcut?
2. How large is the gap between average and worst-subgroup accuracy?
3. Can a robustness method (Group DRO) close this gap?
4. Is the shortcut visible in the model's attention (saliency maps)?

**Why this matters:** A model that relies on background shortcuts will fail systematically when the "wrong" background appears (e.g., a waterbird photographed on land). This is a real-world deployment hazard.

---

## 2. Dataset — Waterbirds (WILDS)

**Source:** Stanford WILDS benchmark (Sagawa et al., 2019; Koh et al., 2021)  
**URL:** [https://wilds.stanford.edu/datasets/](https://wilds.stanford.edu/datasets/)

| Property | Value |
|----------|-------|
| Total images | 11,788 |
| Train / Val / Test | 4,795 / 1,199 / 5,794 |
| Classes | 2 (landbird, waterbird) |
| Subgroups | 4 (bird type × background) |

### The Four Subgroups (Training Distribution)

| Group ID | Subgroup | Train % | Role |
|----------|----------|---------|------|
| 0 | Landbird-on-land | 73.0% | **Majority** |
| 1 | Landbird-on-water | 3.8% | **Minority (HARD)** |
| 2 | Waterbird-on-land | 1.2% | **Minority (HARDEST)** |
| 3 | Waterbird-on-water | 22.0% | **Majority** |

**Key property:** The training set is **deliberately biased** — majority subgroups make up ~95% of training data, while minority subgroups together make up only ~5%. This creates a strong spurious correlation between background and bird type.

---

## 3. Methodology

### Full Workflow

1. **Data loading:** Use WILDS package to load Waterbirds with image transforms (RandomResizedCrop + flip for train, Resize for eval).
2. **Split design:** Use WILDS official splits (train/val/test). Train uses heavily biased distribution; test has more balanced subgroups to make worst-group accuracy a meaningful metric.
3. **Feature extraction:** ImageNet-pretrained ResNet-18 backbone (last FC layer replaced with 2-class output).
4. **Modeling:**
   - **ERM baseline:** minimize average cross-entropy loss across all training examples (equal weight per example).
   - **Group DRO mitigation:** maintain per-group weights and upweight groups with higher loss via dual ascent.
5. **Validation:** Every epoch, compute both average and worst-group validation accuracy. **Select best checkpoint by worst-group val accuracy** (not average — selecting on average rewards shortcut learning).
6. **Analysis:** Per-subgroup test accuracy, gap analysis, and Grad-CAM saliency visualization.

### Selected Models

**Baseline: ERM (Empirical Risk Minimization)**
- Objective: minimize average loss
- Standard training procedure used in virtually all deep learning.
- On biased data, ERM minimizes average loss by exploiting the shortcut.

**Main model: Group DRO (Distributionally Robust Optimization)**
- Maintains weights for each of the 4 subgroups.
- Updates weights via **dual ascent**, then re-normalizes.
- Forces the optimizer to focus on the worst-performing subgroup.

### Architecture

Both models use the same backbone: **ResNet-18** pretrained on ImageNet (11.2M parameters with our 2-class head). The only difference is the loss function.

---

## 4. Evaluation Criterion

### Primary Metric: Worst-Group Accuracy

```
worst_group_accuracy = min(accuracy_on_subgroup_0, ..., accuracy_on_subgroup_3)
```

Following the WILDS protocol, the headline metric is the **accuracy on the worst-performing subgroup**.

**Why worst-group accuracy?**
- Average accuracy hides shortcut learning (a model can get 90% average while failing on 30% of test examples).
- Worst-group accuracy forces the model to perform well on ALL subgroups.
- Real-world relevance: deployment failure on a minority subgroup is exactly the kind of bias we want to detect.

### Secondary Metrics

- Per-subgroup accuracy (breakdown across all 4 subgroups)
- Average accuracy (standard metric for context)
- Gap (average − worst-group, the size of the disparity)
- Grad-CAM saliency (qualitative visual evidence of where the model "looks")

---

## 5. Results

### 5.1 Headline Comparison

| Method | Macro-average subgroup accuracy | Worst-Group | Gap |
|--------|---------|-------------|-----|
| **ERM** | **83.07%** | **62.62%** | **20.45 pp** |
| **Group DRO** | **86.36%** | **74.77%** | **11.59 pp** |

**Key result:** Group DRO improves worst-group accuracy by **+12.15 percentage points** (from 62.62% to 74.77%) while also improving macro-average subgroup accuracy by **+3.29 points**.

### 5.2 Per-Subgroup Breakdown

| Subgroup | ERM | Group DRO | Delta |
|----------|-----|-----------|-------|
| Landbird-on-land (majority) | 98.63% | 94.28% | −4.35 pp |
| Landbird-on-water (minority) | 75.88% | 85.41% | **+9.53 pp** |
| **Waterbird-on-land (minority)** | **62.62%** | **74.77%** | **+12.15 pp** |
| Waterbird-on-water (majority) | 95.17% | 90.97% | −4.20 pp |

The biggest improvement is on **waterbird-on-land** (+12.15 pp), the subgroup where ERM failed most. This subgroup has the **fewest training examples** (~56 images), making it the hardest case for ERM but exactly what Group DRO targets.

The second biggest improvement is **landbird-on-water** (+9.53 pp), showing Group DRO helps both minority groups significantly.

### 5.3 Visual Evidence: Grad-CAM Saliency

The saved Grad-CAM heatmaps are used as qualitative visualizations of regions contributing to model predictions. They may help formulate hypotheses about bird-versus-background attention, but saliency maps do not establish causal feature reliance and cannot confirm that a model has stopped using background cues. The visualizations should therefore be interpreted alongside the quantitative subgroup results, not as proof of shortcut learning or its removal.

### 5.4 Training Dynamics

- **ERM:** Quickly drives train loss to ~0.0001 (severe overfitting) but val worst-group accuracy plateaus around 0.50–0.70.
- **Group DRO:** Has higher train loss but achieves much better worst-group validation accuracy (best: 0.7293 vs ERM's 0.5940), demonstrating the trade-off works as intended.

---

## 6. Error Analysis and Limitations

### Where Group DRO Still Fails

After Group DRO, waterbird-on-land improves to 74.77%, but **25.23% of those examples are still misclassified** (roughly 162 out of 642 test images). Possible reasons:

- **Insufficient training examples:** Waterbird-on-land has only ~56 training examples — even with upweighting, the model sees very few examples.
- **Strong visual similarity:** Some waterbird species (e.g., cormorants on rocks) genuinely look similar to landbirds when on land backgrounds.
- **Background still partially used:** Even Group DRO may not fully eliminate background reliance when training data is so heavily biased.

### Limitations of This Work

| Limitation | Explanation |
|------------|-------------|
| Single backbone | Only ResNet-18 tested. Vision Transformers or other ConvNets may behave differently. |
| Single dataset | Waterbirds is one benchmark. Conclusions should be validated on CelebA, CivilComments, etc. |
| Single mitigation | Only Group DRO compared. Other methods (IRM, CORAL, reweighting) could provide additional baselines. |
| Group labels required | Group DRO needs group labels during training, which may not be available in real-world settings. |
| Saliency interpretation | Grad-CAM is qualitative and does not prove causal reliance on image regions; the reported 7×7 layer4 maps are also spatially coarse. |

### What Could Be Improved

- **Longer training:** 50 epochs may be insufficient with appropriate LR scheduling.
- **Data augmentation:** More aggressive augmentation (RandAugment, CutMix) might improve generalization on minority subgroups.
- **Ensemble methods:** Combining multiple models trained with different seeds could reduce variance.
- **Test-time intervention:** Methods like TENT or batchnorm adaptation could help at inference time.

---

## 7. Conclusion

This project demonstrated three key findings:

1. **ERM exploits background shortcuts** in the Waterbirds dataset, producing a 20.45-percentage-point gap between macro-average subgroup accuracy (83.07%) and worst-group (62.62%) accuracy.
2. **Grad-CAM provides qualitative visualizations** that can be inspected for possible bird/background attention patterns; it does not prove causal shortcut reliance.
3. **Group DRO closes this gap dramatically**, improving worst-group accuracy to 74.77% (a **+12.15 pp gain**) while also improving macro-average subgroup accuracy to 86.36%.

### Answer to the Original Research Questions

**Yes, ERM exploits background shortcuts, and yes, Group DRO effectively mitigates this failure mode.**

The recorded results are consistent with the motivation for group-robust optimization on a deliberately imbalanced benchmark. Because this is a single recorded experiment on one dataset and backbone, it should not be described as an independent reproduction or general proof of mitigation.

### Practical Implications

- Models deployed in real-world settings should be evaluated on **worst-group accuracy**, not just average accuracy.
- When training data is biased, consider using **group-robust optimization** methods like Group DRO.
- **Visualization** (including Grad-CAM) can support qualitative error analysis, but should be combined with quantitative tests and should not be treated as causal evidence on its own.

---

## 8. References

1. **Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P.** (2020). *Distributionally Robust Neural Networks for Group Shifts*. ICLR 2020. [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
2. **Koh, P. W., Sagawa, S., et al.** (2021). *WILDS: A Benchmark of in-the-Wild Distribution Shifts*. ICML 2021. [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)
3. **Selvaraju, R. R., et al.** (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017.
4. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
5. **Wah, C., Branson, S., Welinder, P., Perona, P., & Belongie, S.** (2011). *The Caltech-UCSD Birds-200-2011 Dataset*. California Institute of Technology.

---

## Appendix: Reproducibility

The historical notebook records the following pipeline; the revised source code is version-controlled, but a clean rerun is still required before claiming reproducibility:

- **Setup cell:** Downloads data and writes source files (~5 min)
- **Training:** ERM + Group DRO take ~40 min each on a single GPU (Tesla T4)
- **Evaluation and saliency:** ~2 min
- **Artifacts saved to:** `artifacts/models/`, `artifacts/results/`, `artifacts/saliency/`

The notebook uses a self-healing mechanism: if checkpoint files already exist, training is skipped automatically, allowing quick re-execution of the evaluation and plotting cells.

## Metric clarification

The historical notebook's “Average” column is the unweighted mean of the four subgroup accuracies. It is therefore macro-average subgroup accuracy, not sample-weighted accuracy over all test images. The revised evaluator reports these separately. The historical values above are retained as recorded and should not be treated as results from the revised Group DRO implementation until it has been rerun.
