import json
import pdb
import numpy as np

TARGET_FILE="backpack.json.old"
with open(TARGET_FILE,'r') as fin:
    results=json.load(fin)

res_hash=dict()
for key in results:
    for stat,match in zip(results[key]['stats'],results[key]['matches']):
        K1=int(stat[0]*100)
        K2=int(stat[1])
        if K1 not in res_hash:
            res_hash[K1]=dict()
        if K2 not in res_hash[K1]:
            res_hash[K1][K2]=[]
        res_hash[K1][K2].append(match)

all_stats=[]
for thresh in res_hash:
    for minC in res_hash[thresh]:
        cum_sum=np.array(res_hash[thresh][minC]).sum(0)
        if (cum_sum[0]+cum_sum[2])==0:
            precision=0.0
        else:
            precision=cum_sum[0]/(cum_sum[0]+cum_sum[2])
        recall=cum_sum[1]/(cum_sum[1]+cum_sum[3])
        all_stats.append([thresh,minC,precision,recall])

all_stats=np.array(all_stats)
import matplotlib.pyplot as plt
plt.plot(all_stats[:,3],all_stats[:,2],'o')
for thresh in res_hash:
    pdb.set_trace()
    F=np.where(all_stats[0]==thresh)
    plt.plot(all_stats[F][:,3],all_stats[F][:,2])
plt.show()
pdb.set_trace()