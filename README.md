# Data Quality-Aware Knowledge-Based AI

**Research prototype — synthetic business data; not real-world entrepreneurial evidence.**

Companion software for *Data Quality-Aware Knowledge-Based AI for Digital Transformation: An Explainable Decision-Support Framework for Innovation and Entrepreneurship* (preliminary short communication).

## Reproduction
```bash
python -m pip install -r requirements.txt
python experiment.py
```
Python 3.11+ recommended. No external data, APIs, credentials, or proprietary information needed.

The experiment generates 3,200 synthetic firms per seed, across five fixed seeds and four missingness settings (0%, 10%, 25%, 40%). It uses a stratified 70/30 split and training-only median imputation, comparing (1) standard gradient-boosted ML, (2) quality-feature-augmented ML, and (3) quality-aware ML with an auditable liquidity/leverage adjustment.

The executable script **generates two PNG figures and two CSV tables**, plus the complete `results.csv`. Corresponding SVG figure previews are already committed:
- `figure_1_architecture.png`: experimental workflow
- `figure_2_performance.png`: ROC-AUC and Brier under missingness
- `table_1_design.csv`: experimental protocol
- `table_2_results.csv`: results mean and standard deviation
- `results.csv`: per-seed observations

The repository includes lightweight CSV summary tables and SVG versions of the two figures for immediate review. Running `python experiment.py` regenerates the CSV tables and outputs publication-resolution PNG figures, along with the detailed per-seed `results.csv`.

## Scientific limits
The knowledge constraint is simple rule-based post-processing, not a validated novel algorithm. The label generator deliberately incorporates a similar liquidity/leverage relationship; the experiment is therefore favorable to that constraint. Performance differences are modest and not uniformly positive. The findings do not demonstrate empirical impacts on digital transformation, entrepreneurship or innovation. External validation on actual business datasets, stronger competing methods and statistical uncertainty estimates are required before drawing generalizable conclusions.

No funding, IRB approval, affiliations, or real-world validation are claimed.
