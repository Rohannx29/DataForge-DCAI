# Notebooks

Exploratory analysis only. Nothing in `src/` imports from this directory —
any reusable logic developed here should be refactored into `src/` before
being relied upon by scripts.

Suggested notebooks:
- `01_eda.ipynb` — initial dataset exploration (mirrors `scripts/run_eda.py`)
- `02_label_audit_review.ipynb` — manual review of samples flagged by `cleanlab`
- `03_results_analysis.ipynb` — exploratory analysis of comparison results before finalizing report figures
