# bound_methods

Methods that put a lower bound Y on the recall of an open-vocabulary query, so that true recall is at or
above Y for a stated share of queries (default 90%). Run them through `run_bound.py` in the parent folder.

## Run

```bash
python run_bound.py --method binomial      --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval
python run_bound.py --method global_margin --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval
python run_bound.py --method scaled_margin --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval --features rhat,profile,size
```

Needs, all for the same `--model`:
- `results/`: real COCO (`coco_index.py` + `embed.py`), `results/object_sizes.json`, and
  `results/<model>/profile_categories.csv` (from `stats.py`)
- `results_synth/`: synthetic images (`synthetic_index.py` + `embed.py`)

## After adding synthetic images

New images change the synthetic index, but two caches would otherwise keep the old data: `embed.py` skips
embedding shards that already exist, and `run_bound.py` reuses its operating-points table when the settings
match. Rerun in this order:

```bash
# 1. Re-index the synthetic folder
python synthetic_index.py --synthetic_root /data3/datasets/coco/<synthetic_folder> --coco_results_dir results --results_dir results_synth --min_images 5

# 2. Delete the old synthetic embeddings (row order has changed), then re-embed
rm -r results_synth/siglip2-so400m
python embed.py --results_dir results_synth --model siglip2-so400m

# 3. Rebuild the operating points once with --recompute 1, then run the other methods normally
python run_bound.py --method binomial --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval --recompute 1
python run_bound.py --method global_margin --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval
python run_bound.py --method scaled_margin --results_dir results --synth_dir results_synth --model siglip2-so400m --out_dir results_eval --features rhat,profile,size
```

Keep `--min_images` the same between runs so results stay comparable. Each method run overwrites its
`bounds_<method>.csv` and `summary_<method>.tsv`; copy them first to keep earlier results.

## What every run does (base.py, class RecallBound)

1. **Operating points**, per category with synthetic images and per specificity level (0.8, 0.9, 0.95, 0.99):

| Column | Meaning |
|---|---|
| `threshold` | Score that the given share of real training-pool images without the category fall below |
| `true_recall` | Share of the category's real test-set images above the threshold |
| `k`, `n`, `r_hat` | Synthetic images above the threshold, synthetic images, estimated recall k / n |
| `log_size` | Log of the category's estimated size (object_sizes.json) |
| `prof_d1..10`, `prof_s1..10` | Query profile: d' and pos_std of the 10 reference categories the query separates best (own category removed) |
| `acc1..10` | The 10 highest acceptance rates of other categories at this threshold, sorted high to low |

   Computed once and cached in `<out_dir>/<model>/operating_points.csv`. Rebuilt when a setting it depends on
   changes, when the table's columns change (`TABLE_VERSION` in base.py), or with `--recompute 1`. It is not
   rebuilt when only the synthetic images change; use `--recompute 1` then (see above).
2. **Leave one category out**: each category is held out in turn (all its specificity levels together); the
   method learns from the other categories (`fit`) and returns Y for the held-out one (`lower_bound`).
3. **Outputs** in `<out_dir>/<model>/`:

| File | Contents |
|---|---|
| `bounds_<method>.csv` | One row per category and level: the table above plus Y and held (true_recall >= Y) |
| `bounds_<method>.csv.json` | Settings used and date |
| `summary_<method>.tsv` | A one-sentence result, then a tab-separated table with the same columns for every method (paste into a spreadsheet) |

## Reading the summary

The sentence says how well the method worked, for example (COCO, global_margin):

> global_margin: the promised recall held in 90% of cases (target 90%). When it failed, true recall fell 0.14
> short on average; when it held, true recall was 0.27 above the promise.

| Column | Meaning |
|---|---|
| `spec` | Specificity level, or `all` |
| `categories` | Categories evaluated |
| `held_pct` | Share of cases where true recall was at or above Y. Target: the coverage (90%) |
| `shortfall` | When the promise broke: how far true recall fell below Y, on average |
| `cushion` | When the promise held: how far true recall sat above Y, on average (the cost of being safe) |
| `mean_Y` | Average bound: the recall assured |
| `mean_margin` | Average r_hat - Y: how far below the estimate the bound sits |
| `mean_true_recall`, `mean_r_hat` | Average true and estimated recall, for scale |
| `pinball_loss` | Single score for ranking methods only (lower = better); not for explaining results |

A method works if `held_pct` is near the target at every level. Among methods that work, a smaller cushion
(higher `mean_Y`) is better, and a smaller shortfall means its failures matter less.

## Methods

| Method | File | Margin comes from | Notes |
|---|---|---|---|
| `binomial` | `binomial.py` | Clopper-Pearson bound on k of n synthetic examples | Uses no other categories; assumes synthetic examples are a random sample of the real target, so it ignores the synthetic-to-real gap |
| `global_margin` | `global_margin.py` | Errors (r_hat - true_recall) of the other categories: the ceil(0.9 x (K + 1))-th smallest of K | Corrects for the gap on average; same margin for every query. One margin per level by default; `--pool_specs 1` pools all levels |
| `scaled_margin` | `scaled_margin.py` | Y = r_hat - mu - k x s. Random forests predict the overshoot mu and its spread s from the inputs; k is set on the other categories (each scored by forests trained without it) | Margin adapts to each query. `--features` picks inputs: `rhat` (r_hat, n), `profile`, `size`, `acc`. Specificity is not an input. `--k_per_spec 1` (default) sets k separately for each level. Options: `--trees`, `--min_leaf`, `--min_spread`. Takes several minutes |

## Adding a method

1. New file with a subclass of `RecallBound`: set `name`, override `lower_bound(rows)`, and `fit(calib)` if it
   learns from other categories. Optional: `add_method_arguments` (prefix help text with the method name),
   `method_params`.
2. Add the class to `METHODS` in `__init__.py`.
3. Add a row to the Methods table above.

Methods needing a trained model can also read a precomputed column; that column must come from a model
that never saw the category it predicts.