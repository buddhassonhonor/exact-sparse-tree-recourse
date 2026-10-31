from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('.');d=pd.read_csv('exp/followup_20261101/scaling_summary.csv')
plt.rcParams.update({'font.size':16})
fig,axes=plt.subplots(1,2,figsize=(12,6))
g=d[(d.dataset=='noise')&(d.method=='leaves_-1')].sort_values('positive_leaves')
axes[0].plot(g.positive_leaves,g.median_ms,'o-');axes[0].set(xlabel='Positive leaves',ylabel='Exact median latency (ms)')
g=d[(d.dataset=='noise')&(d.n_records==d.n_records.max())]
for _,z in g.iterrows():
 label={'leaves_-1':'Exact','leaves_1':'Leaf 1','leaves_5':'Leaf 5','leaves_20':'Leaf 20','anchors_1':'Anchor 1','anchors_5':'Anchor 5','anchors_20':'Anchor 20'}[z.method]
 axes[1].scatter(z.median_ms,z.coverage)
 axes[1].annotate(label,(z.median_ms,z.coverage),xytext=(2,4),textcoords='offset points',fontsize=16)
axes[1].set(xlabel='Median latency (ms)',ylabel='Coverage')
fig.tight_layout();fig.savefig('figures/scaling_tradeoff.png',dpi=100);fig.savefig('figures/scaling_tradeoff.pdf')

