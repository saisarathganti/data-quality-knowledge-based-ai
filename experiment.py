"""Synthetic-data pilot for data-quality-aware, rule-guided classification.
Not evidence of actual innovation or entrepreneurial outcomes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, balanced_accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FEATURES = ("liquidity", "growth", "digital_maturity", "market_demand", "execution", "leverage")
SEEDS = (7, 11, 19, 23, 31)
LEVELS = (0.0, 0.1, 0.25, 0.4)

def generate(seed, n=3200):
    """Generate synthetic firm attributes and a deliberately rule-sensitive label."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 6))
    x = 1 / (1 + np.exp(-z))
    x[:, 1] = np.clip(.55 * x[:, 1] + .45 * x[:, 3] + rng.normal(0, .1, n), 0, 1)
    logit = (-1.9 + 1.2*x[:, 0] + 1.3*x[:, 1] + 1.1*x[:, 2]
             + 1.5*x[:, 3] + .8*x[:, 4] - 1.1*x[:, 5]
             - 2.8*((x[:, 0] < .25) & (x[:, 5] > .65)))
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(x, columns=FEATURES), y

def corrupt(frame, rate, seed):
    """Introduce label-independent missingness and measurement noise."""
    rng = np.random.default_rng(seed)
    x = frame.to_numpy(copy=True)
    noise = rng.random(x.shape) < rate*.45
    x[noise] = np.clip(x[noise] + rng.normal(0, .34, noise.sum()), 0, 1)
    missing = rng.random(x.shape) < rate
    x[missing] = np.nan
    completeness = 1 - np.isnan(x).mean(axis=1)
    anomaly = (np.where(np.isnan(x), 0, np.abs(x-.5)) > .44).mean(axis=1)
    return pd.DataFrame(x, columns=FEATURES), completeness, anomaly

def classifier():
    return HistGradientBoostingClassifier(
        max_iter=110, max_leaf_nodes=12, l2_regularization=3,
        learning_rate=.055, random_state=42
    )

def domain_rule(observed):
    """Only fire when both required raw measurements are present."""
    return ((observed["liquidity"].to_numpy() < .25)
            & (observed["leverage"].to_numpy() > .65))

def run():
    rows = []
    for seed in SEEDS:
        clean, target = generate(seed)
        train, test = train_test_split(
            np.arange(len(target)), test_size=.3,
            stratify=target, random_state=seed
        )
        for level in LEVELS:
            raw, completeness, anomaly = corrupt(clean, level, seed + 1099)
            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(raw.iloc[train])
            x_test = imputer.transform(raw.iloc[test])
            q_train = np.column_stack((
                x_train, completeness[train], anomaly[train],
                raw.iloc[train].isna().to_numpy(dtype=int)
            ))
            q_test = np.column_stack((
                x_test, completeness[test], anomaly[test],
                raw.iloc[test].isna().to_numpy(dtype=int)
            ))
            baseline = classifier().fit(x_train, target[train]).predict_proba(x_test)[:, 1]
            aware = classifier().fit(q_train, target[train]).predict_proba(q_test)[:, 1]
            # Fixed rule penalty; deliberately not tuned against test labels.
            rule = domain_rule(raw.iloc[test])
            guided = np.clip(aware - .16*rule, 0, 1)
            for method, prediction in (
                ("ML baseline", baseline),
                ("Quality-aware ML", aware),
                ("Knowledge-guided AI", guided)
            ):
                rows.append({
                    "seed": seed, "corruption": level, "method": method,
                    "roc_auc": roc_auc_score(target[test], prediction),
                    "brier": brier_score_loss(target[test], prediction),
                    "balanced_accuracy": balanced_accuracy_score(
                        target[test], prediction >= .5
                    ),
                    "rule_coverage": float(rule.mean())
                })
    results = pd.DataFrame(rows)
    results.to_csv(ROOT/"results.csv", index=False, float_format="%.6f")
    summary = results.groupby(["corruption", "method"])[
        ["roc_auc", "brier", "balanced_accuracy"]
    ].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(
        ROOT/"table_2_results.csv", index=False, float_format="%.4f"
    )
    design = pd.DataFrame([
        ("Observations per seed", 3200, "Simulated firms; binary outcome"),
        ("Seeds", len(SEEDS), str(SEEDS)),
        ("Split", "70% / 30%", "Stratified per seed"),
        ("Missingness levels", str(LEVELS), "MCAR per cell"),
        ("Measurement noise", "0.45 × corruption rate", "Gaussian SD 0.34"),
        ("Knowledge rule", "liquidity < .25 AND leverage > .65",
         "Subtract .16 when both attributes are observed"),
        ("Target", "Simulated innovation outcome", "Not real entrepreneurial outcomes")
    ], columns=["design_component", "setting", "interpretation"])
    design.to_csv(ROOT/"table_1_design.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    colors = {"ML baseline": "#64748b", "Quality-aware ML": "#0284c7",
              "Knowledge-guided AI": "#ca8a04"}
    for name, subset in results.groupby("method"):
        means = subset.groupby("corruption").mean(numeric_only=True)
        std = subset.groupby("corruption").std(numeric_only=True)
        for axis, metric in zip(ax, ("roc_auc", "brier")):
            axis.errorbar(means.index, means[metric], yerr=std[metric],
                          marker="o", capsize=3, label=name, color=colors[name])
    for axis, title in zip(
        ax, ("ROC-AUC (higher is better)", "Brier score (lower is better)")
    ):
        axis.set(xlabel="Missingness probability", ylabel=title)
        axis.grid(alpha=.18)
    ax[0].legend(fontsize=8)
    fig.savefig(ROOT/"figure_2_performance.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(11, 2.8), layout="constrained")
    axis.axis("off")
    labels = ["Synthetic firms", "MCAR + noise", "Train-only imputation",
              "Quality features", "Predictive ML", "Rule constraint",
              "Decision score + audit"]
    xs = np.linspace(.075, .925, len(labels))
    for i, (x, label) in enumerate(zip(xs, labels)):
        axis.text(x, .5, label, ha="center", va="center", fontsize=8,
                  transform=axis.transAxes,
                  bbox=dict(boxstyle="round,pad=.5",
                            facecolor="#eef3f9", edgecolor="#456b9d"))
        if i < len(labels)-1:
            axis.annotate("", xy=(xs[i+1]-.062, .5), xytext=(x+.062, .5),
                          xycoords=axis.transAxes,
                          arrowprops=dict(arrowstyle="->", color="#334155"))
    fig.savefig(ROOT/"figure_1_architecture.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(summary.round(4).to_string())

if __name__ == "__main__":
    run()
