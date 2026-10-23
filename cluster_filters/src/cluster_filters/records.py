"""
The per-cluster record shared by every filter, and the two output files.

Record (one per cluster, keyed by its permanent ID from step 3):
    uid, prompt, box3d, centroid, points_file, n_views
    status        'active' or 'rejected'
    rejected_by   the stage that rejected it (None while active)
    views         step-3 views, each with 'active' (view filters may set False) and, if a view filter
                  replaced its box, 'original_bbox'
    stages        {stage name: that stage's scores, decision and details}

Outputs, in <config_dir>/filters/<name>/:
    results.json    {"run": command line + settings of every stage, "clusters": {uid: record}}
    accepted.json   clusters still active after the last stage (what the ROS server reads)
"""
import glob
import json
import os


def load_records(config_dir):
    """Starting record for every step-3 cluster: active, with all of its views active."""
    records = {}
    for path in sorted(glob.glob(config_dir + "/clusters/*_[0-9][0-9][0-9][0-9].json")):
        with open(path) as f:
            c = json.load(f)
        records[c['uid']] = {
            'uid': c['uid'], 'prompt': c['prompt'], 'box3d': c['box3d'], 'centroid': c['centroid'],
            'points_file': "clusters/" + os.path.basename(path)[:-5] + ".npz",
            'n_views': c['n_views'],
            'status': 'active', 'rejected_by': None,
            'views': [dict(v, active=True) for v in c['views']],
            'stages': {}}
    return records


def active_views(record):
    return [v for v in record['views'] if v['active']]


def accepted_list(records):
    """Clusters still active, with every stage's cluster score and the three highest-scoring views of the
    last stage that scored views."""
    out = []
    for r in records.values():
        if r['status'] != 'active':
            continue
        scores = {name: st.get('cluster_score') for name, st in r['stages'].items()}
        view_scores = {}
        for st in r['stages'].values():
            view_scores = st.get('view_scores', view_scores)
        best = sorted(view_scores, key=lambda k: -view_scores[k])[:3]
        out.append({'uid': r['uid'], 'prompt': r['prompt'], 'box3d': r['box3d'], 'centroid': r['centroid'],
                    'points_file': r['points_file'], 'n_views': r['n_views'], 'scores': scores, 'best_views': best})
    return out


def write_json(path, data):
    """Write via a temporary file, so an interrupted run never leaves a half-written file."""
    tmp = path + ".tmp"
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)