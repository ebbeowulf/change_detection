"""RecallBound: everything the recall lower-bound methods share.

For each COCO category with synthetic images, and each specificity level, the operating-points table holds:
  threshold    score that a given share (the specificity level) of real training-pool images WITHOUT the
               category fall below
  true_recall  share of the category's real test-set images above the threshold
  k, n, r_hat  synthetic images above the threshold (k), synthetic images (n), estimated recall k / n
  log_size     log of the category's estimated size (object_sizes.json; no labels of the category used)
  prof_d1..K   query profile: d' of the K reference categories the query separates best (own category removed)
  prof_s1..K   pos_std of those same K categories
  acc1..K      the K highest acceptance rates of other categories at this threshold (share of their
               training-pool images above it), sorted high to low
The table is computed once and cached in <out_dir>/<model>/operating_points.csv; it is rebuilt only when
the settings it depends on change (or with --recompute 1).

run() holds out one category at a time (all its specificity levels together), calls fit() on the other
categories and lower_bound() on the held-out one, then saves the results and a summary.

A method subclass overrides lower_bound(), and fit() if it learns from the other categories.
"""
import csv
import datetime
import json
import os

import numpy as np
import pandas as pd

import models

TABLE_VERSION = 2  # bump when the columns of the operating-points table change, so old caches are rebuilt


class RecallBound:
    name = "base"  # each subclass sets its own; used in file names and the --method choice

    def __init__(self, args):
        self.args = args
        self.coverage = args.coverage
        self.out_dir = args.out_dir + "/" + args.model
        os.makedirs(self.out_dir, exist_ok=True)

    # ---------------------------------------------------------------- arguments
    @classmethod
    def add_arguments(cls, p):
        """Arguments every method shares."""
        text_models = [m for m, s in models.MODELS.items() if s["family"] != "dino"]
        p.add_argument("--results_dir", type=str, required=True, help="Real COCO results folder (index.csv, labels.csv, dataset.json, <model>/)")
        p.add_argument("--synth_dir", type=str, required=True, help="Synthetic results folder from synthetic_index.py and embed.py")
        p.add_argument("--model", type=str, required=True, choices=text_models, help="Open-vocabulary model providing the scores")
        p.add_argument("--out_dir", type=str, required=True, help="Output folder (e.g. results_eval); written under <out_dir>/<model>/")
        p.add_argument("--coverage", type=float, default=0.9, help="Target share of cases whose true recall is at or above the bound")
        p.add_argument("--specs", type=str, default="0.8,0.9,0.95,0.99", help="Comma-separated specificity levels")
        p.add_argument("--template", type=str, default="{}", help="Prompt template applied to each category name")
        p.add_argument("--profile_file", type=str, default="profile_categories.csv", help="Query profiles from stats.py, in results_dir/<model>")
        p.add_argument("--sizes_file", type=str, default="object_sizes.json", help="Object size lookup table in results_dir")
        p.add_argument("--size_key", type=str, default="image_fraction", help="Which size value to use: image_fraction or surface_area_m2")
        p.add_argument("--top_k", type=int, default=10, help="Number of reference categories in the profile and acceptance columns")
        p.add_argument("--test_split", type=str, default="val2017", help="Split always placed in the test set")
        p.add_argument("--extra_test", type=int, default=20000, help="Additional images moved from other splits into the test set")
        p.add_argument("--split_seed", type=int, default=0, help="Seed for the test split")
        p.add_argument("--recompute", type=int, default=0, help="1 = rebuild the cached operating points even if settings match")

    @classmethod
    def add_method_arguments(cls, p):
        """Arguments specific to one method; subclasses add theirs here."""

    def method_params(self):
        """Settings specific to this method, recorded with its results; subclasses add theirs here."""
        return {}

    # ---------------------------------------------------------------- what a method overrides
    def fit(self, calib):
        """Learn from the other categories' rows (all specificity levels). Default: nothing to learn."""

    def lower_bound(self, rows):
        """Return the recall lower bound Y for each row of the held-out category."""
        raise NotImplementedError

    # ---------------------------------------------------------------- shared data: operating points
    def table(self):
        """Operating points for every (category, specificity level); computed once, then read from cache."""
        a = self.args
        settings = {"table_version": TABLE_VERSION, "results_dir": a.results_dir, "synth_dir": a.synth_dir,
                    "model": a.model, "specs": a.specs, "template": a.template, "profile_file": a.profile_file,
                    "sizes_file": a.sizes_file, "size_key": a.size_key, "top_k": a.top_k,
                    "test_split": a.test_split, "extra_test": a.extra_test, "split_seed": a.split_seed}
        cache = self.out_dir + "/operating_points.csv"
        if not a.recompute and os.path.exists(cache) and os.path.exists(cache + ".json"):
            with open(cache + ".json") as f:
                if json.load(f) == settings:
                    return pd.read_csv(cache)
        print("Computing operating points (one time for these settings)...")
        t = self._compute_table()
        t.to_csv(cache, index=False)
        with open(cache + ".json", "w") as f:
            json.dump(settings, f, indent=2)
        return t

    def _compute_table(self):
        a = self.args
        K = a.top_k
        # Real COCO: labels, embeddings, test split
        with open(a.results_dir + "/dataset.json") as f:
            categories = json.load(f)["categories"]
        with open(a.results_dir + "/index.csv") as f:
            index = list(csv.DictReader(f))
        emb = models.load_embeddings(a.results_dir + "/" + a.model)
        assert emb.shape[0] == len(index), "COCO embeddings incomplete: finish embed.py first"
        col = {c: j for j, c in enumerate(categories)}
        Y = np.zeros((len(index), len(categories)), dtype=np.int8)
        with open(a.results_dir + "/labels.csv") as f:
            for r in csv.DictReader(f):
                Y[int(r["row"]), col[r["category"]]] = 1
        test = _test_mask(index, a.test_split, a.extra_test, a.split_seed)
        pool = ~test
        pos_pool = [np.where(pool & (Y[:, k] == 1))[0] for k in range(len(categories))]  # each category's pool images

        # Synthetic images: which category each one shows, and their embeddings
        with open(a.synth_dir + "/dataset.json") as f:
            synth_categories = json.load(f)["categories"]
        with open(a.synth_dir + "/labels.csv") as f:
            synth_cat = {int(r["row"]): r["category"] for r in csv.DictReader(f)}
        synth_emb = models.load_embeddings(a.synth_dir + "/" + a.model)
        assert synth_emb.shape[0] == len(synth_cat), "synthetic embeddings incomplete: run embed.py on synth_dir"

        # Side information that uses no labels of the category itself
        with open(a.results_dir + "/" + a.sizes_file) as f:
            sizes = json.load(f)
        prof = pd.read_csv(a.results_dir + "/" + a.model + "/" + a.profile_file)
        prof = prof[prof.is_target == 0]

        # Query vectors (GPU chosen by CUDA_VISIBLE_DEVICES)
        model, processor = models.load(a.model, "cuda")
        text = models.encode_texts(a.model, model, processor,
                                   [a.template.format(c) for c in categories], "cuda").cpu().numpy()

        specs = [float(s) for s in a.specs.split(",")]
        rows = []
        for name in synth_categories:
            if name not in col:
                continue
            c = col[name]
            s = emb @ text[c]
            neg_pool, pos_test, neg_test = s[pool & (Y[:, c] == 0)], s[test & (Y[:, c] == 1)], s[test & (Y[:, c] == 0)]
            synth = synth_emb[[i for i, cat in synth_cat.items() if cat == name]] @ text[c]

            # Query profile: the K reference categories with the highest d' (own category already removed)
            p = prof[prof["query"] == name].sort_values("d_prime", ascending=False).head(K)
            prof_cols = {**{f"prof_d{i + 1}": v for i, v in enumerate(p.d_prime)},
                         **{f"prof_s{i + 1}": v for i, v in enumerate(p.pos_std)}}

            for spec in specs:
                thr = float(np.quantile(neg_pool, spec))
                k = int((synth > thr).sum())
                acc = np.sort([(s[pos_pool[j]] > thr).mean() for j in range(len(categories)) if j != c])[::-1][:K]
                rows.append({"category": name, "spec": spec, "threshold": thr,
                             "true_recall": float((pos_test > thr).mean()),
                             "true_spec": float((neg_test <= thr).mean()), "n_test_pos": len(pos_test),
                             "k": k, "n": len(synth), "r_hat": k / len(synth),
                             "log_size": float(np.log(sizes[name][a.size_key])), **prof_cols,
                             **{f"acc{i + 1}": v for i, v in enumerate(acc)}})
        return pd.DataFrame(rows)

    # ---------------------------------------------------------------- the evaluation
    def run(self):
        t = self.table()
        out = []
        for c in t.category.unique():
            held, calib = t[t.category == c], t[t.category != c]  # all specificity levels held out together
            self.fit(calib)
            out.append(held.assign(Y=np.clip(self.lower_bound(held), 0, 1)))
        res = pd.concat(out, ignore_index=True)
        res["held"] = res.true_recall >= res.Y  # did the promise hold?

        res_file = self.out_dir + "/bounds_" + self.name + ".csv"
        res.round(5).to_csv(res_file, index=False)
        info = {"method": self.name, "coverage": self.coverage, **self.method_params(),
                "arguments": vars(self.args), "date": datetime.datetime.now().isoformat(timespec="seconds")}
        with open(res_file + ".json", "w") as f:
            json.dump(info, f, indent=2)
        print(f"Wrote {res_file}")
        self.summary(res)

    def summary(self, res):
        """Same tab-separated table for every method, ready to paste into a spreadsheet, plus one plain sentence.

        held_pct       share of cases whose true recall was at or above the bound (target: coverage)
        shortfall      when the promise broke: average of (Y - true recall)
        cushion        when the promise held: average of (true recall - Y)
        mean_Y         average bound (recall assured); mean_margin = average (r_hat - Y)
        pinball_loss   single score for ranking methods; lower = better
        """
        tau = 1 - self.coverage
        gap = res.true_recall - res.Y
        res = res.assign(gap=gap, pinball=np.where(gap >= 0, tau * gap, -(1 - tau) * gap))
        groups = [("all", res)] + [(str(s), g) for s, g in res.groupby("spec")]
        header = ["method", "coverage", "spec", "categories", "held_pct", "shortfall", "cushion",
                  "mean_Y", "mean_margin", "mean_true_recall", "mean_r_hat", "pinball_loss"]
        lines = ["\t".join(header)]
        for label, g in groups:
            broke, kept = g[~g.held], g[g.held]
            shortfall = -broke.gap.mean() if len(broke) else 0.0
            cushion = kept.gap.mean() if len(kept) else 0.0
            lines.append("\t".join([self.name, f"{self.coverage:.2f}", label, str(g.category.nunique()),
                                    f"{100 * g.held.mean():.1f}", f"{shortfall:.3f}", f"{cushion:.3f}",
                                    f"{g.Y.mean():.3f}", f"{(g.r_hat - g.Y).mean():.3f}",
                                    f"{g.true_recall.mean():.3f}", f"{g.r_hat.mean():.3f}",
                                    f"{g.pinball.mean():.4f}"]))
            if label == "all":
                sentence = (f"{self.name}: the promised recall held in {100 * g.held.mean():.0f}% of cases "
                            f"(target {100 * self.coverage:.0f}%). When it failed, true recall fell {shortfall:.2f} "
                            f"short on average; when it held, true recall was {cushion:.2f} above the promise.")
        text = "\n".join(lines)
        print("\n" + sentence + "\n\n" + text)
        with open(self.out_dir + "/summary_" + self.name + ".tsv", "w") as f:
            f.write(sentence + "\n\n" + text + "\n")


def _test_mask(index, test_split, extra_test, seed):
    """True = test image: all of test_split plus about extra_test images (whole groups) from the other splits."""
    split = np.array([r["split"] for r in index])
    group = np.array([r["group"] for r in index])
    test = split == test_split
    if extra_test <= 0:
        return test
    groups, counts = np.unique(group[~test], return_counts=True)
    order = np.random.default_rng(seed).permutation(len(groups))
    n_needed = np.searchsorted(np.cumsum(counts[order]), extra_test) + 1
    test |= np.isin(group, groups[order[:n_needed]])
    return test