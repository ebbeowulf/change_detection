#!/usr/bin/env python3
"""Analysis steps 1-2: summarize recall error per (concept, N) and compare it with the binomial.

Reads every results_dir/<feature_model>/recall_error_shard*.csv (from recall_error.py).

Per (concept, N), over the draws:
  true_recall  mean test-set recall
  bias         mean error (val_recall - test_recall)
  spread       standard deviation of the error
  binom_sd     spread expected from binomial sampling of the validation positives alone:
               sqrt(mean over draws of R(1-R) / n_val), R = that draw's test recall
  inflation    spread / binom_sd   (1 = error fully explained by N)
  infl_p       two-sided chi-square test of inflation = 1
  coverage     fraction of draws whose test recall falls inside the 90% Wilson
               interval around the validation recall (nominal 0.90)

Then, per N, tests whether inflation varies across concepts more than the draws alone
would cause: compares the observed spread of log(inflation) across concepts with the
spread expected from estimating a standard deviation from K draws.

Writes results_dir/<feature_model>/error_summary.csv.
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")  # write files only; no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

Z90 = 1.645  # two-sided 90% normal quantile


def wilson(k, n, z=Z90):
    """Wilson score interval for k successes out of n."""
    p = k / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half

REFERENCE = {"inflation": 1.0, "bias": 0.0, "coverage": 0.90}  # value meaning "matches the binomial"


def inflation_interval(infl, k):
    """90% interval for an inflation factor estimated from k draws (chi-square on the variance)."""
    lo = infl * np.sqrt((k - 1) / chi2.ppf(0.95, k - 1))
    hi = infl * np.sqrt((k - 1) / chi2.ppf(0.05, k - 1))
    return infl - lo, hi - infl  # distances below and above, as matplotlib expects


def plot_summary(s, title, variables, sizes, plot_dir):
    """One figure per N, one panel per variable, one bar per category, sorted by the first variable."""
    os.makedirs(plot_dir, exist_ok=True)
    for n in sizes:
        g = s[s.n_pos == n].sort_values(variables[0])
        if g.empty:
            print(f"N={n}: no rows, no plot")
            continue
        y = np.arange(len(g))
        fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 0.2 * len(g) + 1.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, var in zip(axes, variables):
            err = None
            if var == "inflation":
                err = np.array(inflation_interval(g.inflation.to_numpy(), g.draws.to_numpy()))
            ax.barh(y, g[var], xerr=err, color="tab:blue", ecolor="gray", capsize=1.5, height=0.7)
            if var in REFERENCE:
                ax.axvline(REFERENCE[var], color="red", linewidth=1)
            ax.set_title(var)
            ax.grid(axis="x", alpha=0.3)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(g.concept, fontsize=7)
        fig.suptitle(f"{title}, N = {n} labeled positives ({int(g.n_val_pos.iloc[0])} validation)")
        fig.tight_layout()
        out_file = plot_dir + f"/error_summary_N{n}.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_file}")

def summarize(g, min_binom_sd):
    """One (concept, N) group of draws -> one summary row."""
    n_val = g.n_val_pos.iloc[0]
    k = len(g)
    spread = g.error.std(ddof=1)
    # Validation and test recall are independent binomial measurements of the same head's recall,
    # so their variances add; the test term matters only for rare categories with few test positives.
    binom_sd = np.sqrt((g.test_recall * (1 - g.test_recall)).mean() * (1 / n_val + 1 / g.n_test_pos.iloc[0]))

    # Inflation is undefined when recall is pinned at 0 or 1 (binomial spread ~ 0).
    if binom_sd < min_binom_sd:
        inflation, infl_p = np.nan, np.nan
    else:
        inflation = spread / binom_sd
        stat = (k - 1) * inflation ** 2  # chi-square with k-1 degrees of freedom if inflation = 1
        infl_p = 2 * min(chi2.cdf(stat, k - 1), chi2.sf(stat, k - 1))

    lo, hi = wilson(np.round(g.val_recall * n_val), n_val)
    coverage = ((g.test_recall >= lo) & (g.test_recall <= hi)).mean()
    recall_sd = g.test_recall.std(ddof=1)  # how much the classifier you end up with depends on which N images were labeled

    return pd.Series({"draws": k, "n_val_pos": n_val, "true_recall": g.test_recall.mean(), "recall_sd": recall_sd,
                      "val_recall": g.val_recall.mean(), "bias": g.error.mean(), "spread": spread,
                      "binom_sd": binom_sd, "inflation": inflation, "infl_p": infl_p,
                      "coverage": coverage, "test_spec": g.test_spec.mean()})


def main():
    p = argparse.ArgumentParser(description="Summarize recall error per concept and N; compare with binomial.")
    p.add_argument("--results_dir", type=str, required=True, help="Results folder")
    p.add_argument("--feature_model", type=str, required=True, help="Feature model whose recall_error files to read")
    p.add_argument("--pattern", type=str, default="recall_error_shard*.csv", help="File pattern in the feature model folder")
    p.add_argument("--min_binom_sd", type=float, default=0.02, help="Skip inflation when the binomial spread is below this")
    p.add_argument("--plot_sizes", type=str, default="20,50,200", help="Comma-separated values of N to plot, one figure each; empty = no plots")
    p.add_argument("--plot_variables", type=str, default="inflation,bias,spread,recall_sd", help="Comma-separated columns to plot; the first sets the sort order")
    args = p.parse_args()

    model_dir = args.results_dir + "/" + args.feature_model
    files = sorted(glob.glob(model_dir + "/" + args.pattern))
    assert files, f"no files matching {args.pattern} in {model_dir}"
    d = pd.concat([pd.read_csv(f) for f in files])
    print(f"read {len(files)} files, {len(d)} rows, {d.concept.nunique()} concepts")

    # ---- Step 1: one summary row per (concept, N) ----
    s = d.groupby(["concept", "n_pos"]).apply(summarize, args.min_binom_sd).reset_index()

    # ---- Step 2: per N, does inflation vary across concepts beyond sampling noise? ----
    # With K draws, log(estimated sd) has sampling std of about 1/sqrt(2(K-1)).
    print("\nPer N, across concepts:")
    print(f"{'N':>5} {'concepts':>8} {'bias':>7} {'infl_med':>8} {'coverage':>8} "
          f"{'sd_log_infl':>11} {'expected':>8} {'ratio':>6}")
    for n, g in s.groupby("n_pos"):
        g = g.dropna(subset=["inflation"])
        if len(g) < 3:
            continue
        observed = np.log(g.inflation).std(ddof=1)
        expected = np.sqrt(1 / (2 * (g.draws.mean() - 1)))
        print(f"{n:>5} {len(g):>8} {g.bias.mean():>+7.3f} {g.inflation.median():>8.2f} "
              f"{g.coverage.mean():>8.2f} {observed:>11.3f} {expected:>8.3f} {observed / expected:>6.2f}")
    print("ratio near 1: concept-to-concept differences in inflation are what K draws alone would produce.")
    print("ratio well above 1: inflation genuinely varies by concept.")

    out_file = model_dir + "/error_summary.csv"
    s.round(4).to_csv(out_file, index=False)
    print(f"\nWrote {out_file}")

    if args.plot_sizes:
        plot_summary(s, args.feature_model, args.plot_variables.split(","),
                     [int(x) for x in args.plot_sizes.split(",")], model_dir + "/plots")

if __name__ == "__main__":
    main()