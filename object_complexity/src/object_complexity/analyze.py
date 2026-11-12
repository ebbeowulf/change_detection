#!/usr/bin/env python3
"""Stage 5: relate query-profile statistics to the labeled-data cost N* from the learning curves.

N* = smallest number of training positives at which the mean (over seeds) of --metric reaches
--target, interpolated linearly in log N between the two bracketing sizes. Concepts that never
reach the target are reported as not reached (N* = inf).

Profile features per concept (from the query matching the concept name):
  target_auc, target_dprime, target_pos_std  use the target's own labels: upper-bound reference only
  loo_max_auc        highest AUC among the other categories (strongest associate / confuser)
  loo_n_active       number of other categories with AUC >= --active_auc
  loo_mean_auc       mean AUC over the other categories

Prints the per-concept table and Spearman correlations with log N*, and writes the table to
results_dir/<feature_model>/analysis_<query_model>[_tag].csv.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def n_star(curve, target):
    """curve: Series of mean metric indexed by n_pos (sorted). Log-linear interpolation of the crossing."""
    ns, vals = curve.index.to_numpy(dtype=float), curve.to_numpy()
    hit = np.where(vals >= target)[0]
    if len(hit) == 0:
        return np.inf
    i = hit[0]
    if i == 0:
        return ns[0]
    x0, x1, y0, y1 = np.log(ns[i - 1]), np.log(ns[i]), vals[i - 1], vals[i]
    return float(np.exp(x0 + (target - y0) / (y1 - y0) * (x1 - x0)))


def main():
    p = argparse.ArgumentParser(description="Relate profile statistics to learning-curve cost.")
    p.add_argument("--results_dir", type=str, required=True, help="Results folder")
    p.add_argument("--query_model", type=str, required=True, help="Model whose profile file supplies the statistics")
    p.add_argument("--feature_model", type=str, required=True, help="Model whose curves file supplies N*")
    p.add_argument("--profile_file", type=str, default="profile_categories.csv", help="Profile file name in the query model folder")
    p.add_argument("--curves_file", type=str, default="curves.csv", help="Curves file name in the feature model folder")
    p.add_argument("--metric", type=str, default="p_at_r0.8", help="Curves column used to define N*")
    p.add_argument("--target", type=float, default=0.9, help="Value of --metric that defines N*")
    p.add_argument("--active_auc", type=float, default=0.7, help="AUC at or above which a non-target category counts as active")
    p.add_argument("--out_tag", type=str, default="", help="Optional tag appended to the output file name")
    args = p.parse_args()

    prof = pd.read_csv(args.results_dir + "/" + args.query_model + "/" + args.profile_file)
    curves = pd.read_csv(args.results_dir + "/" + args.feature_model + "/" + args.curves_file)

    mean_curve = curves.groupby(["concept", "n_pos"])[args.metric].mean().unstack("concept")
    print(f"Mean {args.metric} by number of positives:")
    print(mean_curve.round(3).to_string(), "\n")

    rows = []
    for concept in mean_curve.columns:
        q = prof[prof["query"] == concept]
        tgt, other = q[q.is_target == 1], q[q.is_target == 0]
        if tgt.empty:
            continue
        rows.append({
            "concept": concept,
            "n_star": n_star(mean_curve[concept].dropna(), args.target),
            "target_auc": tgt.auc.iloc[0],
            "target_dprime": tgt.d_prime.iloc[0],
            "target_pos_std": tgt.pos_std.iloc[0],
            "loo_max_auc": other.auc.max(),
            "loo_n_active": int((other.auc >= args.active_auc).sum()),
            "loo_mean_auc": other.auc.mean(),
        })
    table = pd.DataFrame(rows).sort_values("n_star")
    print(f"N* = positives needed for mean {args.metric} >= {args.target}")
    print(table.round(3).to_string(index=False), "\n")

    reached = table[np.isfinite(table.n_star)]
    print(f"Spearman correlation with log N* ({len(reached)} concepts reached the target):")
    if len(reached) < 3:
        print("  too few concepts to correlate")
    else:
        for col in table.columns[2:]:
            rho, pval = spearmanr(reached[col], np.log(reached.n_star))
            print(f"  {col:16s} rho = {rho:+.2f}  (p = {pval:.2f})")

    out_file = (args.results_dir + "/" + args.feature_model + "/analysis_" + args.query_model +
                ("_" + args.out_tag if args.out_tag else "") + ".csv")
    table.to_csv(out_file, index=False)
    print(f"\nWrote {out_file}")


if __name__ == "__main__":
    main()