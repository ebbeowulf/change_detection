"""
NaiveBayesFusionFilter: combine the siamese and VLM change filters with naive Bayes.

  python run_filters.py <config_dir> --stages nb_fusion \
      --siamese_model <siamese training output folder> \
      --vlm_change_model qwen3.5:35b+nothink+json \
      --nb_fusion_train_root /data3/datasets/smart_change/j234

Components: a SiameseFilter and a VLMChangeFilter, built from their usual options (--siamese_*, --vlm_change_*).
Each scores every cluster this stage sees (their caches are used as usual) but neither rejects anything;
both of their entries are kept in the cluster's record.

Fusion: posterior log-odds = log prior odds + LLR_siamese(score) + LLR_vlm(score), where each LLR is
log P(score bin | changed) - log P(score bin | not changed). The cluster passes if the posterior probability
>= --nb_fusion_threshold. The posterior is stored as this stage's cluster_score.

Likelihoods are learned at startup from existing standalone results: every configuration under
--nb_fusion_train_root (same configuration name as the one being filtered) that has labels and both
filters/<--nb_fusion_siamese_results>/results.json and filters/<--nb_fusion_vlm_results>/results.json.
Each filter's cluster score is put in one of --nb_fusion_bins equal bins on [0, 1]; counts per bin and class
get add-one smoothing. A cluster a filter could not score (no views answered) contributes LLR 0 for it.
The learned tables are written into results.json with the other settings.
For an in-sample fit use the folder being filtered as the train root; for leave-one-out use the others.
"""
import glob
import json
import math
import os
from .base import ClusterFilter
from . import siamese as siamese_module
from . import vlm_change as vlm_module


def score_bin(score, n_bins):
    return min(int(score * n_bins), n_bins - 1)


class NaiveBayesFusionFilter(ClusterFilter):
    name = 'nb_fusion'

    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--nb_fusion_train_root', type=str, nargs='+', default=[], help='folder(s) searched for labeled standalone results to learn the likelihoods from')
        parser.add_argument('--nb_fusion_siamese_results', type=str, default='siamese', help='standalone siamese results folder name under filters/ (default=siamese)')
        parser.add_argument('--nb_fusion_vlm_results', type=str, default='', help='standalone VLM results folder name under filters/ (default = vlm_change__<VLM model name>)')
        parser.add_argument('--nb_fusion_bins', type=int, default=10, help='score bins per filter (default=10)')
        parser.add_argument('--nb_fusion_threshold', type=float, default=0.5, help='posterior probability of change a cluster needs (default=0.5)')

    def __init__(self, args, run_settings, config_dir):
        super().__init__(args, run_settings, config_dir)
        self.siamese = siamese_module.SiameseFilter(args, run_settings, config_dir)
        self.vlm = vlm_module.VLMChangeFilter(args, run_settings, config_dir)
        self.n_bins = args.nb_fusion_bins
        self.threshold = args.nb_fusion_threshold
        self.config_name = os.path.basename(os.path.abspath(config_dir))
        self.siamese_results = args.nb_fusion_siamese_results
        self.vlm_results = args.nb_fusion_vlm_results or "vlm_change__" + self.vlm.model_name.replace(':', '_').replace('/', '_')
        if not args.nb_fusion_train_root:
            raise SystemExit("ERROR: nb_fusion needs --nb_fusion_train_root (folder(s) with labeled standalone results)")
        self.fit(args.nb_fusion_train_root)
        self.settings = {'siamese': self.siamese.settings, 'vlm_change': self.vlm.settings,
                         'threshold': self.threshold, 'bins': self.n_bins,
                         'train_roots': [os.path.abspath(r) for r in args.nb_fusion_train_root],
                         'siamese_results': self.siamese_results, 'vlm_results': self.vlm_results,
                         'train_configs': self.train_configs, 'train_clusters': self.n_changed + self.n_unchanged,
                         'train_changed': self.n_changed, 'train_unchanged': self.n_unchanged,
                         'prior_log_odds': self.prior_log_odds, 'llr_siamese': self.llr['siamese'],
                         'llr_vlm': self.llr['vlm']}

    # ---------------------------------------------------------- learning the likelihoods

    @staticmethod
    def stage_score(results, uid, stage):
        """A filter's cluster score from a results file, or None if it scored no views of that cluster."""
        st = results.get(uid, {}).get('stages', {}).get(stage)
        if not st or st.get('views_scored', 0) == 0:
            return None
        return st['cluster_score']

    def fit(self, train_roots):
        counts = {m: {True: [0] * self.n_bins, False: [0] * self.n_bins} for m in ('siamese', 'vlm')}
        n = {True: 0, False: 0}
        self.train_configs = 0
        for root in train_roots:
            pattern = os.path.abspath(root) + "/**/change2.0_results/" + self.config_name + "/filters/" + self.siamese_results + "/results.json"
            for s_path in sorted(glob.glob(pattern, recursive=True)):
                config_dir = os.path.dirname(os.path.dirname(os.path.dirname(s_path)))
                v_path = config_dir + "/filters/" + self.vlm_results + "/results.json"
                labels = {}
                for lp in glob.glob(config_dir + "/*.labels.json"):
                    with open(lp) as f:
                        labels.update(json.load(f))
                if not os.path.exists(v_path) or not labels:
                    continue
                with open(s_path) as f:
                    s_res = json.load(f)['clusters']
                with open(v_path) as f:
                    v_res = json.load(f)['clusters']
                self.train_configs += 1
                for uid, lab in labels.items():
                    if uid not in s_res or uid not in v_res:
                        continue
                    changed = bool(lab.get('changed'))
                    n[changed] += 1
                    for m, res, stage in (('siamese', s_res, 'siamese'), ('vlm', v_res, 'vlm_change')):
                        score = self.stage_score(res, uid, stage)
                        if score is not None:
                            counts[m][changed][score_bin(score, self.n_bins)] += 1
        if n[True] == 0 or n[False] == 0:
            raise SystemExit(f"ERROR: nb_fusion found {n[True]} changed and {n[False]} unchanged labeled clusters with both "
                             f"'{self.siamese_results}' and '{self.vlm_results}' results under {train_roots} - need both classes")

        self.n_changed, self.n_unchanged = n[True], n[False]
        self.prior_log_odds = math.log((n[True] + 1) / (n[False] + 1))
        self.llr = {}
        for m in ('siamese', 'vlm'):
            c1, c0 = counts[m][True], counts[m][False]
            t1, t0 = sum(c1) + self.n_bins, sum(c0) + self.n_bins          # add-one smoothing
            self.llr[m] = [math.log((c1[b] + 1) / t1) - math.log((c0[b] + 1) / t0) for b in range(self.n_bins)]
        print(f"  nb_fusion: likelihoods from {self.train_configs} configurations, "
              f"{n[True]} changed / {n[False]} unchanged clusters")

    # ---------------------------------------------------------- filtering

    def apply(self, record):
        self.siamese.apply(record)                 # both components score the cluster; their decisions are ignored
        self.vlm.apply(record)
        s = self.stage_score({record['uid']: record}, record['uid'], 'siamese')
        v = self.stage_score({record['uid']: record}, record['uid'], 'vlm_change')
        llr_s = self.llr['siamese'][score_bin(s, self.n_bins)] if s is not None else 0.0
        llr_v = self.llr['vlm'][score_bin(v, self.n_bins)] if v is not None else 0.0
        log_odds = self.prior_log_odds + llr_s + llr_v
        posterior = 1.0 / (1.0 + math.exp(-log_odds))
        decision = posterior >= self.threshold
        self.store(record, {'siamese_score': s, 'vlm_score': v, 'llr_siamese': llr_s, 'llr_vlm': llr_v,
                            'log_odds': log_odds, 'cluster_score': posterior, 'decision': decision})
        return decision

    def finish(self):
        self.siamese.finish()
        self.vlm.finish()