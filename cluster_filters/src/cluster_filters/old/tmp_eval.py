import csv, sys, numpy as np
from sklearn.metrics import roc_curve
r = list(csv.DictReader(open(sys.argv[1])))           # .../runs/<run>/test_predictions.csv
p = np.array([float(x['prob']) for x in r]); y = np.array([int(x['label']) for x in r])
fpr, tpr, thr = roc_curve(y, p)
hm = 2 * tpr * (1 - fpr) / np.maximum(tpr + 1 - fpr, 1e-9)
i = hm.argmax()
print(f"best image HM {hm[i]:.3f} at threshold {thr[i]:.3f} (TPR {tpr[i]:.3f}, TNR {1 - fpr[i]:.3f})")

for lab in ('1', '0'):
    p = np.array([float(x['prob']) for x in r if x['label'] == lab])
    print(f"label {lab}: median {np.median(p):.4f}  25-75% {np.percentile(p, 25):.4f}-{np.percentile(p, 75):.4f}")