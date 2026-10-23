"""
Run a sequence of filters over the clusters written by build_change_candidates.py.

  python run_filters.py <config_dir> --stages siamese --siamese_model <training output folder>

  config_dir   a step-3 configuration folder, e.g. <run>/change2.0_results/sam3_0.1_openVocab
  --stages     filters to run, in order (see filters/__init__.py); each filter's own options are
               prefixed with its name, e.g. --siamese_model
  --name       output folder name (default: the stage names joined by '+')

Each filter only sees clusters still active after the previous ones. Output, in <config_dir>/filters/<name>/:
  results.json    every cluster's record (see records.py) plus the command line and every filter's settings
  accepted.json   clusters still active after the last stage - what the ROS server reads
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # sibling modules of cluster_filters
from records import load_records, accepted_list, write_json
from filters import FILTERS


def parse_args():
    parser = argparse.ArgumentParser(description='Run a sequence of filters over step-3 change candidates')
    parser.add_argument('config_dir', type=str, help='step-3 configuration folder, e.g. <run>/change2.0_results/sam3_0.1_openVocab')
    parser.add_argument('--stages', type=str, nargs='+', required=True, help=f'filters to run, in order: {" ".join(FILTERS)}')
    parser.add_argument('--name', type=str, default='', help="output folder name under filters/ (default = stages joined by '+')")
    for f in FILTERS.values():
        f.add_args(parser)
    args = parser.parse_args()
    unknown = [s for s in args.stages if s not in FILTERS]
    if unknown:
        parser.error(f"unknown stage(s) {unknown}; available: {list(FILTERS)}")
    return args

if __name__ == '__main__':
    args = parse_args()
    config_dir = args.config_dir.rstrip('/')
    with open(config_dir + "/run_settings.json") as f:
        run_settings = json.load(f)
    out_dir = config_dir + "/filters/" + (args.name if args.name else "+".join(args.stages))
    os.makedirs(out_dir, exist_ok=True)

    records = load_records(config_dir)
    print(f"{config_dir}: {len(records)} clusters, stages: {' -> '.join(args.stages)}")

    # run the stages in order; each one only sees clusters still active
    stage_settings = {}
    for stage_name in args.stages:
        stage = FILTERS[stage_name](args, run_settings, config_dir)
        stage_settings[stage_name] = stage.settings
        active = [r for r in records.values() if r['status'] == 'active']
        for record in active:
            if not stage.apply(record):
                record['status'] = 'rejected'
                record['rejected_by'] = stage_name
        stage.finish()
        kept = sum(1 for r in records.values() if r['status'] == 'active')
        print(f"  {stage_name}: {len(active)} clusters in, {kept} kept")

    run_info = {'command': " ".join(sys.argv), 'config_dir': os.path.abspath(config_dir),
                'stages': args.stages, 'stage_settings': stage_settings}
    write_json(out_dir + "/results.json", {'run': run_info, 'clusters': records})
    accepted = accepted_list(records)
    write_json(out_dir + "/accepted.json", accepted)
    print(f"{len(accepted)} of {len(records)} clusters accepted -> {out_dir}/accepted.json")