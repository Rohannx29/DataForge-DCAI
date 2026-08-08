# Data-Centric Active Learning for Manufacturing Defect Detection

_Final report skeleton — matches the structure required by project instructions.
Sections marked [TODO] require content once experiments are complete._

## Title
Data-Centric Active Learning for Manufacturing Defect Detection: A Controlled
Comparison of Data Quality Interventions

## Abstract
[TODO — write last, after results are final. ~200 words summarizing problem,
method, key result.]

## Introduction
[TODO]

## Literature Survey
See `reports/literature_survey.md`.

## Problem Statement
[TODO]

## Objectives
1. Quantify the individual contribution of data cleaning, label-noise
   correction, and data-centric interventions (targeted augmentation, active
   learning) on defect classification performance, while holding model
   architecture constant.
2. Demonstrate that active-learning-guided label acquisition reaches target
   performance with fewer labeled samples than random sampling.
3. Produce a reproducible, well-documented pipeline suitable for extension
   into a research paper.

## Dataset Description
See `docs/dataset_description.md`.

## Methodology
See `docs/architecture.md` for the full controlled-experiment rationale.

## System Architecture
See `docs/architecture.md` — includes data flow diagram.

## Algorithms Used
- ResNet18 / EfficientNet-B0 (transfer learning) — `src/models/architectures.py`
- Confident Learning (cleanlab) — `src/labels/noise_detection.py`
- Uncertainty Sampling (entropy-based active learning) — `src/active_learning/sampling.py`
- Isolation Forest (outlier detection) — `src/dcai/outlier_detection.py`

## Implementation
See repository README.md for structure; `docs/setup.md` for run instructions.

## Experimental Results
[TODO — populate from `scripts/compare_experiments.py` output:
`reports/comparison_table.csv` and `reports/figures/condition_comparison.png`]

## Discussion
[TODO]

## Future Scope
- Extend to defect segmentation using MVTec AD's pixel-level masks
- Hybrid uncertainty + diversity active learning sampling
- Synthetic defect generation via GAN-based augmentation
- Cross-category generalization testing (train on one MVTec category, test on another)

## Conclusion
[TODO]

## References
See `reports/literature_survey.md`. Do not include unverified citations.

---
## Progress Log
_(Informal tracking — remove before final submission)_

- [x] Repository structure established
- [ ] Casting dataset downloaded and validated
- [ ] Baseline training pipeline validated end-to-end
- [ ] Label noise detection wired up (cleanlab cross-validation)
- [ ] Active learning loop wired up (model inference in loop.py)
- [ ] MVTec AD experiments run
- [ ] Final comparison table + statistical significance tests
- [ ] Report written
