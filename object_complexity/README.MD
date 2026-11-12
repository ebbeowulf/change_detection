# Object Complexity

Estimates how far an open-vocabulary detector (CLIP / SigLIP image-text scores) can be trusted for a new
query: its recall at a chosen specificity, and how much to trust that estimate. Two tracks:

- **Low-shot:** how far a recall measured on N labeled images is from the true recall, and whether the
  query's profile against known categories predicts that error.
- **Zero-shot:** predicting recall with no real labels (query profile, object size, synthetic images), and
  putting a lower bound on it that holds for a stated share of queries (`bound_methods/`).

Each script is one stage. Scripts read only datasets or earlier results and write only to results folders.

## Results folders

| Folder | Contents |
|---|---|
| `results/` | COCO 2017: index, labels, embeddings per model, query profiles, `object_sizes.json`, low-shot outputs |
| `results_synth/` | Synthetic COCO images (inpainted into COCO backgrounds): index and embeddings |
| `results_ovad/` | OVAD attribute + category concepts (2,000 COCO val images) |
| `results_oi/` | Open Images V7 validation (pool) + test, for the classes in `oi_classes.txt` |
| `results_eval/` | Recall-bound evaluation (`run_bound.py`): cached operating points, bounds, summaries |

Each holds `index.csv` (one row per image), `labels.csv` (image, category, status), `dataset.json`
(categories, counts, `exhaustive`), and one subfolder per model with embedding shards.

## Stage 1: Index datasets

| Script | Dataset | Command |
|---|---|---|
| `coco_index.py` | COCO 2017 train + val | `python coco_index.py --coco_root /data3/datasets/coco/coco/ --results_dir results` |
| `synthetic_index.py` | Synthetic images, `<root>/<category>/images/<img_id>_syn.jpg` | `python synthetic_index.py --synthetic_root /data3/datasets/coco/<synthetic_folder> --coco_results_dir results --results_dir results_synth --min_images 5` |
| `ovad_index.py` | OVAD attribute + category concepts | `python ovad_index.py --ovad_root /data3/datasets/coco/ovad --coco_root /data3/datasets/coco/coco --results_dir results_ovad` |
| `openimages_index.py` | Open Images V7 val + test, listed classes | `python openimages_index.py --oi_root /data3/datasets/openimages --class_list /data3/datasets/openimages/oi_classes.txt --results_dir results_oi` |

Splits: COCO test = val2017 + 20,000 train images (the rest is the pool). Open Images test = its test split;
its validation split is the pool (thresholds, synthetic backgrounds); use `--test_split test --extra_test 0`
downstream. `openimages_index.py` expands broader classes through the class hierarchy and writes
`class_report.csv` (counts, kept or dropped, reason).

## Stage 2: Embed images (`embed.py`, `models.py`)

```bash
python embed.py --results_dir results --model siglip2-so400m
```

Models: `clip-l14`, `siglip-so400m`, `siglip2-so400m`, `dinov2-l14` (DINOv2: classifier features only, no
text). Resumable: existing shards are skipped, so **delete `<results_dir>/<model>/` before re-embedding a
dataset whose index changed** (e.g. after adding synthetic images). `models.py` is the only shared module.

## Stage 3: Query profiles (`stats.py`)

```bash
python stats.py --ref_dir results --model siglip2-so400m                             # COCO category names
python stats.py --ref_dir results --model siglip2-so400m --query_file <queries.txt>  # any query list
```

Writes `results/<model>/profile_<query_set>.csv`: one row per (query, reference category) with pos_mean,
pos_std, neg_mean, neg_std, d_prime, auc, and is_target. The reference is always COCO.

Object sizes: `results/object_sizes.json` (LLM-estimated typical image fraction and surface area per category;
no labels used). New datasets need entries for their classes.

## Low-shot track

| Script | Job | Command |
|---|---|---|
| `recall_error.py` | Per concept, N, and random draw: train a small head on 80% of N positives, threshold on held-out negatives (specificity 0.9), measured recall on the other 20% vs. test-set recall | Two windows, one per GPU: `python recall_error.py --results_dir results --feature_model siglip2-so400m --shard 0 --num_shards 2 --out_tag shard0` (and `--shard 1 ... --out_tag shard1`) |
| `error_summary.py` | Bias, spread, binomial comparison, coverage per concept and N; optional bar charts | `python error_summary.py --results_dir results --feature_model siglip2-so400m` |
| `sorted_vector/predict_spread.py` | Predict spread (how much measured recall varies) from measured recall + query profile; separate model per N or combined | `python predict_spread.py --results_dir ../results --query_model siglip2-so400m --feature_model siglip2-so400m --per_n 1 --profile_cols d_prime,pos_std --out_tag perN` |

OVAD runs of `recall_error.py` add `--test_split test --extra_test 0 --sizes 10,20,50 --n_neg 500`.

## Zero-shot track

| Script | Job | Command |
|---|---|---|
| `sorted_vector/zeroshot_recall.py` | Recall at a negatives-based threshold from a sorted query vector (first version, superseded by basket_recall.py) | `python zeroshot_recall.py --results_dir ../results --model siglip2-so400m` |
| `basket_recall.py` | Recall from a fixed basket of 40 reference categories and/or estimated object size, one model per specificity level, over 5 random baskets | `python basket_recall.py --results_dir results --model siglip2-so400m --out_tag size` |
| `synth_recall.py` | Raw synthetic recall vs. the predictors above, same categories and thresholds | `python synth_recall.py --results_dir results --synth_dir results_synth --model siglip2-so400m` |
| `run_bound.py` + `bound_methods/` | Lower bound Y on recall (binomial, global margin, scaled margin), leave one category out; plain-language summary | See `bound_methods/README.md` |

## Status

| Item | Status |
|---|---|
| COCO index, embeddings (CLIP, SigLIP 2), query profiles | done |
| Low-shot recall error and spread prediction (CLIP, SigLIP 2) | done |
| Zero-shot basket, size, and synthetic comparisons (COCO) | done |
| Recall bounds on COCO: binomial, global_margin, scaled_margin | done; rerun after adding synthetic images |
| Open Images: download, index, embeddings, synthetic images | in progress |
| Bounds trained on COCO, evaluated on Open Images | planned: needs an evaluation-dataset option in `bound_methods/base.py` |

## Obsolete files (safe to delete)

`curves.py`, `analyze.py`, `run_recall_error.sh`, `common.py`, `operating_points.py`, `plot_error_summary.py`
(each replaced by a script above).