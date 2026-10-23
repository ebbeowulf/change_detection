import csv, sys, numpy as np
from collections import defaultdict
r = [x for x in csv.DictReader(open(sys.argv[1])) if x['label'] == '1']
g = defaultdict(list)
for x in r:
    g[(x['run'], x['source_file'], x['cluster_id'])].append(float(x['prob']))
rows = sorted((np.median(v), len(v), k) for k, v in g.items())
for med, n, (run, src, cid) in rows:
    print(f"{med:.4f}  n={n:3d}  {run}  {src}  cluster {cid}")