import argparse
import json
import os
import random
from make_splits import list_runs, count_run, totals, pos_rate, balanced_pick


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Leave-one-environment-out split: train/val on the other environments, '
                                                 'test on one environment restricted to selected label sources')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('split_file', type=str, help='name of the split file, saved as <dataset_root>/splits/<split_file> (e.g. holdout_s120.json)')
    parser.add_argument('--test_env', type=str, required=True, help='environment held out for testing (e.g. s120)')
    parser.add_argument('--test_include', type=str, nargs='+', required=True, help='test only on label-source files whose name contains one of these (e.g. save_results_unfiltered__sam3_0.1_openVocab)')
    parser.add_argument('--test_exclude', type=str, nargs='*', default=[], help='skip test label-source files whose name contains one of these (e.g. tight)')
    parser.add_argument('--train_envs', type=str, nargs='*', default=[], help='training environments (default = every other environment in dataset_root)')
    parser.add_argument('--val_runs_per_env', type=int, default=1, help='runs per training environment held out for validation (default=1)')
    parser.add_argument('--min_pos', type=int, default=1, help='minimum positive samples in train and val per environment when picking val runs (default=1)')
    parser.add_argument('--min_neg', type=int, default=1, help='minimum negative samples in train and val per environment when picking val runs (default=1)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help='train/val label-source filter used for counting (match what you pass to train_resnet.py)')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='train/val label-source exclusions used for counting')
    parser.add_argument('--delta_model', type=str, default='', help='if set, count only samples whose delta_change/<model> file exists (default = count all labels)')
    parser.add_argument('--seed', type=int, default=0, help='seed for balanced tie-breaking (default=0)')
    args = parser.parse_args()

    delta_model = args.delta_model if args.delta_model else None
    rng = random.Random(args.seed)
    train_envs = args.train_envs if args.train_envs else sorted(
        e for e in os.listdir(args.dataset_root)
        if e not in (args.test_env, 'splits', 'runs') and os.path.isdir(args.dataset_root + "/" + e))
    if args.test_env in train_envs:
        raise ValueError(f"{args.test_env} is both the test environment and a training environment")

    split = {'train': [], 'val': [], 'test': []}
    counts = {}

    # training environments: all label sources, val runs picked to match each environment's class balance
    for env in train_envs:
        names = []
        for name in [env + "/" + r for r in list_runs(args.dataset_root, env)]:
            counts[name] = count_run(args.dataset_root, name, delta_model, args.include, args.exclude)
            if counts[name]['pos'] + counts[name]['neg'] > 0:
                names.append(name)
        if not names:
            print(f"WARNING: {env} has no runs with samples - skipped")
            continue
        if args.val_runs_per_env >= len(names):
            raise ValueError(f"{env} has {len(names)} runs - not enough to hold out {args.val_runs_per_env} for val")
        if args.val_runs_per_env == 0:
            test_unused, val, train = [], [], names
        else:
            # reuse the balanced picker with no test runs
            test_unused, val, train = balanced_pick(names, counts, 0, args.val_runs_per_env, args.min_pos, args.min_neg, rng)
        split['train'] += sorted(train)
        split['val'] += sorted(val)
        t, v = totals(train, counts), totals(val, counts)
        print(f"{env}: train {len(train)} runs ({t['pos']} pos / {t['neg']} neg), val {[n.split('/')[1] for n in val]} ({v['pos']} pos / {v['neg']} neg)")

    # test environment: every run, but only the selected label sources
    for name in [args.test_env + "/" + r for r in list_runs(args.dataset_root, args.test_env)]:
        counts[name] = count_run(args.dataset_root, name, delta_model, args.test_include, args.test_exclude)
        c = counts[name]
        if c['pos'] + c['neg'] == 0:
            print(f"  {name}: no samples from {args.test_include} - skipped")
            continue
        split['test'].append(name)
        print(f"  {name}: {c['pos']} pos / {c['neg']} neg samples ({c['cluster_pos']} pos / {c['cluster_neg']} neg clusters)")
    if not split['test']:
        raise ValueError(f"no {args.test_env} runs contain label sources matching {args.test_include}")

    split['filters'] = {'test': {'include': args.test_include, 'exclude': args.test_exclude}}
    split['counts'] = {}
    for name in ('train', 'val', 'test'):
        t = totals(split[name], counts)
        split['counts'][name] = t
        print(f"{name:5s}: {len(split[name])} runs, {t['pos']} pos / {t['neg']} neg samples "
              f"({100 * pos_rate(t):.1f}% pos), {t['cluster_pos']} pos / {t['cluster_neg']} neg clusters")

    split['args'] = vars(args)
    os.makedirs(args.dataset_root + "/splits", exist_ok=True)
    out_path = args.dataset_root + "/splits/" + args.split_file
    with open(out_path, 'w') as f:
        json.dump(split, f, indent=2)
    print(f"Saved {out_path}")