import argparse
import itertools
import json
import os
import random
from delta_dataset import load_samples


def list_runs(dataset_root, env):
    env_dir = dataset_root + "/" + env
    return sorted(r for r in os.listdir(env_dir) if os.path.isfile(env_dir + "/" + r + "/meta.json"))


def count_run(dataset_root, run, delta_model, include, exclude):
    """Positive/negative counts for one run, as frame samples and as unique labeled clusters."""
    samples, _ = load_samples(dataset_root, [run], delta_model, include, exclude)
    pos = sum(s['label'] for s in samples)
    clusters = {(s['source_file'], s['cluster_id']): s['label'] for s in samples}
    cpos = sum(clusters.values())
    return {'pos': pos, 'neg': len(samples) - pos, 'cluster_pos': cpos, 'cluster_neg': len(clusters) - cpos}


def totals(runs, counts):
    t = {'pos': 0, 'neg': 0, 'cluster_pos': 0, 'cluster_neg': 0}
    for r in runs:
        for k in t:
            t[k] += counts[r][k]
    return t


def pos_rate(t):
    n = t['pos'] + t['neg']
    return t['pos'] / n if n else 0.0


def balanced_pick(runs, counts, n_test, n_val, min_pos, min_neg, rng):
    """Try every (test, val) choice of runs and keep the one whose test and val sets best match the
    environment's overall positive rate and sample share. Shortfalls below min_pos/min_neg in any split
    are penalized heavily (train > test > val), so they are only accepted when no assignment avoids them.
    Ties are broken in seeded random order."""
    overall = totals(runs, counts)
    n_all = max(overall['pos'] + overall['neg'], 1)
    rate = pos_rate(overall)
    target_test, target_val = n_test / len(runs), n_val / len(runs)

    candidates = []
    for test in itertools.combinations(runs, n_test):
        rest = [r for r in runs if r not in test]
        for val in itertools.combinations(rest, n_val):
            candidates.append((list(test), list(val), [r for r in rest if r not in val]))
    rng.shuffle(candidates)

    best, best_score = None, None
    for test, val, train in candidates:
        t = {name: totals(rs, counts) for name, rs in (('test', test), ('val', val), ('train', train))}
        # when shortfalls are unavoidable, protect train first, then test, then val;
        # a split with no runs requested (e.g. --test_runs_per_env 0) is not checked
        weight = {'train': 3000.0, 'test': 2000.0 if n_test else 0.0, 'val': 1000.0 if n_val else 0.0}
        score = sum(weight[n] * (max(0, min_pos - t[n]['pos']) + max(0, min_neg - t[n]['neg'])) for n in t)
        for name, target in (('test', target_test), ('val', target_val)):
            if target == 0:
                continue
            n = t[name]['pos'] + t[name]['neg']
            score += abs(pos_rate(t[name]) - rate) + abs(n / n_all - target)
        if best_score is None or score < best_score:
            best, best_score = (test, val, train), score
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build a reusable train/val/test split of whole runs from the change dataset')
    parser.add_argument('dataset_root', type=str, help='root of the change dataset (e.g. /data2/datasets/change_dataset)')
    parser.add_argument('split_file', type=str, help='name of the split file, saved as <dataset_root>/splits/<split_file> (e.g. phone_runs.json)')
    parser.add_argument('--environments', type=str, nargs='+', required=True, help='environments to include (e.g. living_room south_haven)')
    parser.add_argument('--test_envs', type=str, nargs='*', default=[], help='environments held out entirely for testing (new-environment evaluation)')
    parser.add_argument('--test_runs_per_env', type=int, default=1, help='runs per remaining environment held out for testing (default=1)')
    parser.add_argument('--val_runs_per_env', type=int, default=1, help='runs per remaining environment held out for validation (default=1)')
    parser.add_argument('--pick', type=str, default='balanced', help="how held-out runs are chosen: 'balanced' (match class balance), 'random' (seeded), or 'last' (last runs in name order) (default=balanced)")
    parser.add_argument('--min_pos', type=int, default=1, help='balanced: minimum positive samples required in each split per environment (default=1)')
    parser.add_argument('--min_neg', type=int, default=1, help='balanced: minimum negative samples required in each split per environment (default=1)')
    parser.add_argument('--delta_model', type=str, default='', help='if set, count only samples whose delta_change/<model> file exists (default = count all labels)')
    parser.add_argument('--include', type=str, nargs='*', default=[], help='count only label-source files whose name contains one of these (match your training filters)')
    parser.add_argument('--exclude', type=str, nargs='*', default=[], help='skip label-source files whose name contains one of these')
    parser.add_argument('--seed', type=int, default=0, help='random seed for random picks and balanced tie-breaking (default=0)')
    args = parser.parse_args()

    split = {'train': [], 'val': [], 'test': []}
    counts = {}
    rng = random.Random(args.seed)
    delta_model = args.delta_model if args.delta_model else None
    for env in args.environments:
        runs = list_runs(args.dataset_root, env)
        if not runs:
            print(f"WARNING: no runs with meta.json under {args.dataset_root}/{env}")
            continue
        names = []
        for name in [env + "/" + r for r in runs]:
            counts[name] = count_run(args.dataset_root, name, delta_model, args.include, args.exclude)
            c = counts[name]
            if c['pos'] + c['neg'] == 0:
                print(f"  {name}: no samples - skipped")
                continue
            names.append(name)
            print(f"  {name}: {c['pos']} pos / {c['neg']} neg samples  ({c['cluster_pos']} pos / {c['cluster_neg']} neg clusters)")
        if not names:
            print(f"WARNING: {env} has no runs with samples - skipped")
            continue

        if env in args.test_envs:
            split['test'] += names
            print(f"{env}: all {len(names)} runs -> test")
            continue

        n_test, n_val = args.test_runs_per_env, args.val_runs_per_env
        if n_test + n_val >= len(names):
            raise ValueError(f"{env} has {len(names)} runs - not enough to hold out {n_test} test and {n_val} val and still train")
        if args.pick == 'balanced':
            test, val, train = balanced_pick(names, counts, n_test, n_val, args.min_pos, args.min_neg, rng)
            for name, rs, n_req in (('train', train, 1), ('val', val, n_val), ('test', test, n_test)):
                if n_req == 0:
                    continue
                t = totals(rs, counts)
                if t['pos'] < args.min_pos or t['neg'] < args.min_neg:
                    print(f"  WARNING: {env} {name} has only {t['pos']} pos / {t['neg']} neg samples - "
                          f"no assignment of runs meets --min_pos {args.min_pos} / --min_neg {args.min_neg}")
        else:
            order = list(names)
            if args.pick == 'random':
                rng.shuffle(order)
            elif args.pick == 'last':
                order = order[::-1]
            else:
                raise ValueError(f"unknown --pick {args.pick}")
            test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
        split['test'] += sorted(test)
        split['val'] += sorted(val)
        split['train'] += sorted(train)

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