# Implementation and Reproducibility Audit

**Scope:** Review of the committed `project15ml.ipynb`, its saved cell outputs, `README.md`, and `report.md`.

**Status:** Partial audit. The notebook's orchestration and saved evaluation outputs are visible, but the actual training and evaluation implementation modules are not committed in this repository.

## 1. What the notebook evidence supports

- The notebook loads the WILDS Waterbirds dataset and records 11,788 examples with official split sizes of 4,795 train, 1,199 validation, and 5,794 test.
- The saved metadata inspection reports a strongly imbalanced training split:
  - landbird-on-land: 3,498 (73.0%)
  - landbird-on-water: 184 (3.8%)
  - waterbird-on-land: 56 (1.2%)
  - waterbird-on-water: 1,057 (22.0%)
- The validation and test splits are balanced across the two bird/background combinations within each class as recorded in the notebook outputs: 2,255 / 2,255 / 642 / 642 for test.
- Both models use the same ImageNet-pretrained ResNet-18 architecture. The notebook's model-build output reports 11,177,538 parameters.
- The notebook invokes 50-epoch ERM and Group DRO training functions. Saved logs report best validation worst-group accuracy of 0.5940 for ERM and 0.7293 for Group DRO.
- The saved test-evaluation output reports:
  - ERM: 83.07% average accuracy, 62.62% worst-group accuracy.
  - Group DRO: 86.36% average accuracy, 74.77% worst-group accuracy.
- The saved outputs are internally consistent with the notebook's displayed per-group table to rounding precision.

These observations establish what the notebook records. They do not independently verify the hidden training implementation or prove that the run can be reproduced from this GitHub repository.

## 2. Main reproducibility blocker: implementation is external

The bootstrap cell searches under `/kaggle/input` for a Kaggle attachment containing an `erm/` directory, then copies its Python files into `/kaggle/working/erm`. The saved output shows that this external attachment supplied:

- `run_erm.py`
- `data.py`
- `evaluate.py`
- `model.py`
- `train_erm.py`
- `config.py`
- `train_groupdro.py`
- `saliency.py`
- `inspect_data.py`

None of these source files is present in the GitHub repository tree. Consequently, the exact Group DRO loss calculation, group-loss aggregation, update rule, step size, numerical stabilization, optimizer, learning-rate schedule, random-seed setup, checkpoint-selection implementation, and test metric aggregation cannot be audited from the repository as it currently stands.

**Required action:** add the exact source files used for the recorded run to this repository (or a versioned, accessible source archive), and record their commit/version in the notebook. Do not reconstruct or replace the original algorithm silently.

## 3. Group DRO checks still outstanding

The notebook's prose describes the standard multiplicative update

`q_g \leftarrow q_g \exp(\eta L_g)`

followed by normalization. The training logs show four normalized-looking group weights evolving from approximately equal weights to `[0.270, 0.240, 0.200, 0.290]` by epoch 50.

This is useful diagnostic evidence, but it is not enough to validate the implementation. Once `train_groupdro.py` is available, verify:

1. The loss for each group is computed from the correct examples and group IDs.
2. The group weights are updated using the intended per-group loss, with a documented step size and stable normalization.
3. The weighted objective actually uses the updated group weights when computing the model gradient.
4. Empty groups or batches with no examples from a group are handled safely.
5. The group-ID mapping is identical in the dataset loader, training loop, evaluation, and printed subgroup names.
6. The optimizer, scheduler, checkpoint rule, and seed configuration are stated and consistent between ERM and Group DRO where intended.

The saved logs alone cannot answer these questions. In particular, the weight trajectory should not be interpreted as proof that the highest-loss group was correctly upweighted without examining the implementation and the group-loss values.

## 4. Evaluation pipeline checks

The notebook calls an external `evaluate_on_test` function and prints subgroup accuracy, average accuracy, and worst-group accuracy. The printed outputs are consistent with the published table, but the evaluator source is missing.

When `evaluate.py` is available, verify:

- Test split only is used for final reporting; validation is used only for checkpoint selection.
- Each test example is counted exactly once and labels/group IDs are aligned.
- Per-group accuracy is computed as correct predictions divided by the number of examples in that group.
- Overall accuracy is computed over all test examples, not as an unweighted mean of subgroup accuracies unless explicitly intended.
- Worst-group accuracy is the minimum of the four subgroup accuracies.
- Both models are evaluated on the same split, transforms, class mapping, and metric definitions.
- The saved JSON reports include subgroup counts as well as accuracies.

## 5. Issues visible in the notebook itself

### 5.1 External Kaggle attachment dependency

A clean clone cannot run the bootstrap cell without the Kaggle input attachment containing the `erm/` source directory. This dependency is not represented as a versioned Git submodule, package, or committed source folder.

### 5.2 Misleading subgroup label in the bootstrap output

The notebook rewrites `inspect_data.py` with a diagnostic label rule that calls a group a majority only when its share is greater than 30%. As a result, the 22.0% waterbird-on-water group is printed as `MINORITY`, even though the report treats it as one of the two majority/correlated groups. This is a labeling-threshold issue in the inspection display; it does not establish that the training group IDs are wrong. The diagnostic should use explicit group definitions rather than a generic 30% cutoff.

### 5.3 Checkpoint reuse is not provenance-safe

The notebook skips training if both checkpoint filenames already exist. It does not visibly verify that those checkpoints were produced from the current source revision, configuration, dataset version, or seed. A stale checkpoint could therefore be evaluated alongside a changed notebook or source attachment.

### 5.4 Incomplete run provenance

The notebook output records PyTorch 2.10.0+cu128, CUDA availability, and a Tesla T4 GPU. It does not visibly record a complete environment lockfile, source hashes, seed values, checkpoint hashes, or the exact Kaggle attachment version. These should be added before claiming end-to-end reproducibility.

### 5.5 Notebook setup instructions need correction

The appendix says to enable a T4 x2 GPU, while the saved run output reports a single Tesla T4. The setup also describes source files as if the notebook writes them itself, but they are copied from an external Kaggle input attachment.

## 6. Recommended next steps

1. Add the exact `erm/` source files used in this run to the review branch.
2. Audit `train_groupdro.py`, `train_erm.py`, `data.py`, `model.py`, `config.py`, and `evaluate.py` line by line.
3. Add explicit seeds and environment/dependency versions, then make checkpoint provenance checks configuration-aware.
4. Fix the subgroup diagnostic labels and add group counts to evaluation JSON.
5. Run a clean end-to-end reproduction from the repository, ideally across at least three seeds, and report mean and standard deviation. Keep the current saved single-run results clearly labeled as historical results until that is completed.

## 7. Current audit conclusion

The recorded outputs support a **single-run benchmark comparison** in which Group DRO has higher average and worst-group accuracy than ERM on Waterbirds. The GitHub repository is **not yet self-contained or independently reproducible**, and the Group DRO implementation and evaluator cannot yet be fully validated because their source files are external to the repository.

Do not describe this as a verified reproduction of the Group DRO algorithm until the source implementation and a clean rerun have been checked.
