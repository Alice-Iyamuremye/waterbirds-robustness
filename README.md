# 🐦 Spurious Background Reliance in Bird Classification

**Detecting and Mitigating Background Shortcuts with ERM and Group DRO on the Waterbirds Benchmark**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c)](https://pytorch.org)
[![WILDS](https://img.shields.io/badge/WILDS-Waterbirds-green)](https://wilds.stanford.edu)

---

## 📌 Overview

Image classifiers often take shortcuts. Instead of learning what a bird **looks like**, they learn what **background** the bird usually sits on. A duck on water? → "Waterbird." That same duck on grass? → "Landbird" (wrong).

This project demonstrates the problem and a fix:

1. **ERM** (standard training) — exploits the background shortcut, producing a **20.45 percentage point gap** between average and worst-group accuracy.
2. **Group DRO** (robust training) — forces the model to care about rare, hard cases, cutting the gap in **half**.

---

## 📊 Headline Results

| Method       | Average Accuracy | Worst-Group Accuracy | Gap      |
|--------------|-----------------|---------------------|----------|
| **ERM**      | 83.07%          | 62.62%              | 20.45 pp |
| **Group DRO**| **86.36%**      | **74.77%**          | **11.59 pp** |

✅ **Worst-group improvement:** +12.15 percentage points  
✅ **Average improvement:** +3.29 percentage points

---

## 📁 Dataset — Waterbirds (WILDS)

- **11,788** images of birds
- **2 classes:** landbird, waterbird
- **4 subgroups:** bird type × background (land/water)
- **Deliberately biased:** ~95% of training data has the "expected" background → strong spurious correlation

| Subgroup | Train % | Role |
|----------|---------|------|
| Landbird on land | 73.0% | Majority (easy) |
| Landbird on water | 3.8% | Minority (hard) |
| Waterbird on land | 1.2% | **Minority (hardest)** |
| Waterbird on water | 22.0% | Majority (easy) |

---

## 🧠 Methods

### ERM (Empirical Risk Minimization)
- Standard training — every image gets equal weight
- On biased data, ERM minimizes average loss by exploiting the shortcut
- **Baseline** — represents "do nothing special"

### Group DRO (Distributionally Robust Optimization)
- Maintains per-group weights updated via dual ascent
- Automatically upweights groups with higher loss (the rare, hard cases)
- Forces the model to perform well on **all** subgroups, not just the majority

**Architecture:** Both models use ResNet-18 pretrained on ImageNet (11.2M parameters).

---

## 📈 Per-Subgroup Breakdown

| Subgroup | ERM | Group DRO | Delta |
|----------|-----|-----------|-------|
| Landbird-on-land | 98.63% | 94.28% | **−4.35 pp** |
| Landbird-on-water | 75.88% | 85.41% | **+9.53 pp** |
| **Waterbird-on-land** ⚠️ | **62.62%** | **74.77%** | **+12.15 pp** |
| Waterbird-on-water | 95.17% | 90.97% | **−4.20 pp** |

The biggest win is **waterbird-on-land** (+12.15 pp) — the subgroup with only **56 training examples**. Group DRO also dramatically improved **landbird-on-water** (+9.53 pp).

---

## 🔍 Visual Evidence (Grad-CAM)

Grad-CAM heatmaps show **where the model looks** when classifying:

- **ERM** — attention is biased toward the **background** for minority subgroups (especially waterbird-on-land), confirming shortcut learning.
- **Group DRO** — attention is consistently centered on the **bird itself**, proving it learned actual features.

---

## 🛠️ How to Run

```bash
# 1. Clone the repo
git clone https://github.com/your-username/waterbirds-robustness.git
cd waterbirds-robustness

# 2. Install dependencies
pip install torch torchvision wilds matplotlib

# 3. Run the notebook
jupyter notebook waterbirds_project.ipynb
```

> ⏱ Training takes ~40 min per model on a single GPU (Tesla T4).  
> If checkpoints exist, training is skipped automatically.

---

## 📚 References

1. Sagawa et al. (2020) — *Distributionally Robust Neural Networks for Group Shifts* — [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)
2. Koh et al. (2021) — *WILDS: A Benchmark of in-the-Wild Distribution Shifts* — [arXiv:2012.07421](https://arxiv.org/abs/2012.07421)
3. Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks* — ICCV 2017

---

## 📝 License

This project is for educational and research purposes.