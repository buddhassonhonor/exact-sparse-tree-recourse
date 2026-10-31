"""Aggregate equal-weight units, paired costs, exact cluster tests and scientific figures."""
from followup_study import *
from scipy.stats import t
from itertools import product
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18})
DECISION=['credit_default','adult_income','german_credit','bank_marketing','compas']
keys=['dataset','seed','depth','protocol','query']
def unit_summary(df,groups):
 return df.groupby(groups+['dataset','seed'],as_index=False).agg(coverage=('success','mean'),cost=('cost','mean'),changed=('changed','mean'),N=('query','size'))
def aggregate(df,groups,name):
 u=unit_summary(df,groups);u.to_csv(OUT/f'{name}_units.csv',index=False)
 v=u.groupby(groups,as_index=False).agg(coverage=('coverage','mean'),cost=('cost','mean'),changed=('changed','mean'),units=('N','size'),N=('N','sum'))
 v.to_csv(OUT/f'{name}_summary.csv',index=False);return v
def paired(df):
 out=[]
 pivot=df.set_index(keys+['method'])[['success','cost','changed']].unstack('method')
 for baseline in ['lf_prr1','lf_prr20']:
  for (depth,proto),pp in pivot.groupby(['depth','protocol']):
   joint=pp[pp[('success','exact')].eq(1)&pp[('success',baseline)].eq(1)].copy()
   diff=joint[('cost',baseline)]-joint[('cost','exact')]
   assert diff.min()>-1e-6
   vals=pd.DataFrame({'exact_cost':joint[('cost','exact')],'baseline_cost':joint[('cost',baseline)],
    'exact_changed':joint[('changed','exact')],'baseline_changed':joint[('changed',baseline)],'savings':diff})
   unit=vals.groupby(['dataset','seed']).mean()
   out.append(dict(depth=depth,protocol=proto,baseline=baseline,N=len(joint),strict=int((diff>1e-8).sum()),
    **unit.mean().to_dict(),pooled_savings=diff.mean()))
 pd.DataFrame(out).to_csv(OUT/'paired_summary.csv',index=False)
def cluster_tests(df):
 u=unit_summary(df,['depth','protocol','method'])
 e=u[u.method=='exact'].set_index(['depth','protocol','dataset','seed']).coverage
 rows=[];bydataset=[]
 for method in ['lf_prr1','lf_prr20','leaves_5','leaves_10']:
  m=u[u.method==method].set_index(['depth','protocol','dataset','seed']).coverage
  d=(e-m).reset_index(name='gap')
  dd=d.groupby(['depth','protocol','dataset'],as_index=False).gap.mean()
  dd['method']=method;bydataset.extend(dd.to_dict('records'))
  for subset,names in [('all8',list(b.DATASET_NAMES)),('decision5',DECISION)]:
   for (depth,proto),g in dd[dd.dataset.isin(names)].groupby(['depth','protocol']):
    v=g.gap.to_numpy();n=len(v);se=v.std(ddof=1)/np.sqrt(n);half=t.ppf(.975,n-1)*se
    draws=np.array(list(product([-1,1],repeat=n)))@v/n
    p=float(np.mean(np.abs(draws)>=abs(v.mean())-1e-12))
    rows.append(dict(method=method,subset=subset,depth=depth,protocol=proto,gap=v.mean(),low=v.mean()-half,high=v.mean()+half,signflip_p=p,datasets=n))
 pd.DataFrame(rows).to_csv(OUT/'cluster_tests.csv',index=False)
 ds=pd.DataFrame(bydataset);ds.to_csv(OUT/'dataset_gaps.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,6))
 g=ds[(ds.method=='lf_prr20')&(ds.depth==5)&(ds.protocol!='free_l2')]
 for proto,color,offset in [('immutable_l2','#2166ac',-.13),('immutable_top3_l1','#b2182b',.13)]:
  v=g[g.protocol==proto].set_index('dataset').reindex(b.DATASET_NAMES).gap
  ax.scatter(v,np.arange(8)+offset,c=color,label='Immutable' if offset<0 else 'Top-3')
 ax.axvline(0,color='gray',ls='--');ax.set_yticks(np.arange(8),b.DATASET_NAMES);ax.set_xlabel('Exact minus 20-anchor coverage');ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/dataset_coverage_gaps.png',dpi=100);plt.close(fig)
def main():
 q=pd.read_csv(OUT/'query_results.csv');summary=aggregate(q,['depth','protocol','method'],'main');paired(q);cluster_tests(q)
 funnel=pd.read_csv(OUT/'funnel.csv')
 f=funnel.groupby(['depth','protocol','dataset','seed','stage']).size().unstack('stage',fill_value=0)
 f=f.div(f.sum(axis=1),axis=0).groupby(['depth','protocol']).mean().reset_index()
 f.to_csv(OUT/'funnel_summary.csv',index=False)
 ranks=funnel.groupby(['depth','protocol']).first_compatible_rank.agg(['median',lambda x:x.quantile(.9)]).reset_index()
 ranks.columns=['depth','protocol','median_rank','p90_rank'];ranks.to_csv(OUT/'rank_summary.csv',index=False)
 a=pd.read_csv(OUT/'anchor.csv')
 subset=a[a.method=='mechanism'][keys].drop_duplicates()
 same=q[q.method.isin(['exact','lf_prr1','lf_prr20'])].merge(subset,on=keys)
 aggregate(pd.concat([a[~a.method.isin(['segment','mechanism'])],same],ignore_index=True),['depth','protocol','method'],'anchor')
 seg=a[a.method=='segment'].copy()
 seg['multi']=seg.transitions>1
 seg.groupby(['depth','protocol'],as_index=False).agg(N=('query','size'),multi_fraction=('multi','mean'),mean_excess=('bisection_excess','mean'),p90_excess=('bisection_excess',lambda x:x.quantile(.9)),max_excess=('bisection_excess','max')).to_csv(OUT/'segment_summary.csv',index=False)
 mechanism=a[a.method=='mechanism'].copy()
 dist=mechanism.groupby(['depth','protocol','dataset','seed','reason']).size().unstack('reason',fill_value=0)
 dist.div(dist.sum(axis=1),axis=0).groupby(['depth','protocol']).mean().to_csv(OUT/'mechanism_summary.csv')
 mechanism.groupby(['depth','protocol'],as_index=False).agg(scaled_median_rank=('scaled_first_valid_rank','median'),support_median_rank=('support_first_valid_rank','median'),scaled_p90_rank=('scaled_first_valid_rank',lambda x:x.quantile(.9)),support_p90_rank=('support_first_valid_rank',lambda x:x.quantile(.9)),valid_fraction=('valid_anchor_fraction','mean'),multi_fraction=('multi_transition_fraction','mean')).to_csv(OUT/'anchor_rank_summary.csv',index=False)
 su=pd.read_csv(OUT/'support.csv');aggregate(su,['depth','protocol','rule','method'],'support')
 practical=pd.read_csv(OUT/'practical.csv')
 # Rounded candidates are conditional on the existence of the unrounded solution.
 aggregate(practical,['depth','protocol','margin','method'],'practical')
 dose=pd.read_csv(OUT/'dose.csv');dose['miss_given_feasible']=1-dose.repair_success
 du=dose.groupby(['dataset','seed','depth','immutable_fraction','k'],as_index=False).agg(exact_coverage=('exact_success','mean'),repair_coverage=('repair_success','mean'))
 du.groupby(['depth','immutable_fraction','k'],as_index=False)[['exact_coverage','repair_coverage']].mean().to_csv(OUT/'dose_summary.csv',index=False)
 feasible=dose[dose.exact_success==1].copy()
 feasible['bin']=pd.cut(feasible.compatible_fraction,[0,.01,.05,.1,.25,.5,1],include_lowest=True)
 db=feasible.groupby(['dataset','seed','bin'],observed=True).miss_given_feasible.mean().reset_index()
 db.groupby('bin',observed=True).agg(failure=('miss_given_feasible','mean'),units=('seed','size')).to_csv(OUT/'dose_bins.csv')
 # Dataset-cluster t intervals in bins.
 dc=feasible.groupby(['dataset','bin'],observed=True).miss_given_feasible.mean().reset_index()
 vals=[]
 for bin,g in dc.groupby('bin',observed=True):
  n=len(g);v=g.miss_given_feasible;half=t.ppf(.975,n-1)*v.std(ddof=1)/np.sqrt(n) if n>1 else np.nan
  vals.append(dict(bin=str(bin),failure=v.mean(),low=v.mean()-half,high=v.mean()+half,datasets=n))
 pd.DataFrame(vals).to_csv(OUT/'dose_cluster_intervals.csv',index=False)
 fig,ax=plt.subplots(figsize=(10,6))
 for h,color in [(5,'#2166ac'),(11,'#b2182b')]:
  g=summary[(summary.depth==h)&(summary.protocol=='immutable_top3_l1')&summary.method.str.startswith('leaves_')].copy()
  g['K']=g.method.str.split('_').str[1].astype(int);g=g[g.K>0].sort_values('K')
  ax.plot(g.K,g.coverage,'o-',label=f'Depth {h}',color=color)
  ex=summary[(summary.depth==h)&(summary.protocol=='immutable_top3_l1')&(summary.method=='exact')].coverage.iloc[0]
  ax.axhline(ex,color=color,ls='--')
 ax.set(xlabel='Positive-leaf budget K',ylabel='Top-3 coverage',xticks=[1,2,3,5,10]);ax.legend();fig.tight_layout();fig.savefig(ROOT/'figures/leaf_budget_curve.png',dpi=100);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,6))
 v=pd.DataFrame(vals);ax.errorbar(np.arange(len(v)),v.failure,yerr=[v.failure-v.low,v.high-v.failure],fmt='o',capsize=4)
 ax.set_xticks(np.arange(len(v)),v.bin,rotation=25);ax.set(xlabel='Fraction of compatible positive leaves',ylabel='Repair failure given feasible Exact');fig.tight_layout();fig.savefig(ROOT/'figures/compatibility_failure.png',dpi=100);plt.close(fig)
 if (OUT/'external.csv').exists():
  aggregate(pd.read_csv(OUT/'external.csv'),['protocol','method'],'external')
 if (OUT/'scaling.csv').exists():
  sc=pd.read_csv(OUT/'scaling.csv')
  sc.groupby(['dataset','n_records','positive_leaves','total_leaves','depth','method'],as_index=False).agg(median_ms=('ms','median'),q25_ms=('ms',lambda x:x.quantile(.25)),q75_ms=('ms',lambda x:x.quantile(.75)),coverage=('success','mean')).to_csv(OUT/'scaling_summary.csv',index=False)
 print(summary.to_string(index=False))
if __name__=='__main__':main()

