"""Support policies, paired engine polishing and paired environment contrasts."""
from mechanism_assets import mean,units,interval,P,PN,R,NAMES
import pandas as pd,numpy as np,json
O=R/'exp/polish_20261101';OLD=R/'exp/mechanism_20261101';E=R/'exp/evidence_20261101'
KEYS=['dataset','seed','depth','protocol','query']
def main():
 breadth=pd.read_csv(OLD/'breadth.csv');breadth=breadth[breadth.depth==5]
 summary=[]
 envrows=[]
 for p in P:
  for env in ['iid','pc1_low','pc1_high','mahalanobis_tail']:
   g=breadth[(breadth.protocol==p)&(breadth.dataset!='synthetic_overlap')&(breadth[env]==1)]
   m=g[g.method=='lf_prr20'][KEYS+['success']].merge(g[g.method=='exact'][KEYS+['success']],on=KEYS,suffixes=('_b','_e'))
   m['difference']=m.success_e-m.success_b;u=units(m,'difference');lo,hi=interval(u)
   envrows.append(dict(protocol=p,environment=env,datasets=len(u),queries=len(m),gap=u.mean(),low=lo,high=hi))
 pd.DataFrame(envrows).to_csv(O/'environment_pairs.csv',index=False)
 base=breadth[(breadth.protocol==P[1])&breadth.method.isin(['lf_prr20','exact','native'])]
 cv=base.groupby(['dataset','seed','method']).success.mean().groupby('method').mean()
 ratio=(cv['native']-cv['exact'])/(cv['exact']-cv['lf_prr20'])
 facts=dict(primary_queries_per_protocol=int(len(base[base.method=='exact'])),primary_LF20=float(cv['lf_prr20']),primary_fixed=float(cv['exact']),primary_joint=float(cv['native']),nearest_specific_ratio=float(ratio))
 timing=pd.read_csv(E/'polish_timing.csv').groupby(['dataset','seed','protocol','query','method']).ms.median().reset_index().groupby(['dataset','protocol','method']).ms.median().unstack('method')
 timing.to_csv(O/'timing_by_dataset.csv');facts['timing_exact_faster_counts']={p:int((g.exact<g.lf_prr20).sum()) for p,g in timing.groupby('protocol')}
 if (O/'support.csv').exists():
  d=pd.read_csv(O/'support.csv');assert d.dataset.nunique()==24
  native=d[d.method=='native']
  for method in ['nearest_positive','native']:
   old=base[base.method==('exact' if method=='nearest_positive' else 'native')];m=d[d.method==method].merge(old,on=KEYS,suffixes=('_new','_old'))
   assert len(m)==len(old) and (m.success_new==m.success_old).all()
   assert np.allclose(m.cost_new,m.cost_old,equal_nan=True,atol=1e-7)
  for method,g in d.groupby('method'):
   u=units(g);lo,hi=interval(u);j=g.merge(native,on=KEYS,suffixes=('_b','_e'));assert not ((j.success_b==1)&(j.success_e==0)).any();j=j[(j.success_b==1)&(j.success_e==1)]
   summary.append(dict(method=method,datasets=len(u),queries=len(g)//(5 if method=='random' else 1),draws=5 if method=='random' else 1,coverage=u.mean(),low=lo,high=hi,pooled=g.success.mean(),own_cost=mean(g[g.success==1],'cost'),joint_observations=len(j),paired_method_cost=mean(j,'cost_b'),paired_native_cost=mean(j,'cost_e')))
  pd.DataFrame(summary).to_csv(O/'support_summary.csv',index=False)
  heur=[v['coverage'] for v in summary if v['method']!='native'];facts.update(support_min=min(heur),support_max=max(heur),support_native=float(next(v['coverage'] for v in summary if v['method']=='native')))
 if (O/'engines.csv').exists():
  d=pd.read_csv(O/'engines.csv');assert d.dataset.nunique()==24 and (d.success==d.polished_success).all();rows=[]
  old=pd.read_csv(E/'external.csv');matches=[]
  for (p,method),g in d.groupby(['protocol','method']):
   j=g[g.success==1].copy();assert (j.cost_after<=j.cost_before+1e-7).all();assert (j.cost_after>=j.exact_cost-1e-7).all()
   u=j.groupby(['dataset','seed'])[['cost_before','cost_after']].mean().groupby('dataset').mean();ratio=1-u.cost_after.mean()/u.cost_before.mean()
   rng=np.random.default_rng(965);v=u.to_numpy();samples=v[rng.integers(0,len(u),(5000,len(u)))].mean(axis=1);rr=1-samples[:,1]/samples[:,0];lo,hi=np.quantile(rr,[.025,.975])
   m=g.merge(old[(old.method==method)&(old.protocol==p)],on=KEYS+['method'],suffixes=('_new','_old'))
   matches.append(dict(protocol=p,method=method,outcomes_identical=bool((m.success_new==m.success_old).all()),successful_costs_equal=bool(np.allclose(m.cost_before,m.cost,equal_nan=True,atol=1e-7))))
   rows.append(dict(protocol=p,method=method,datasets=g.dataset.nunique(),successful_datasets=j.dataset.nunique(),queries=len(g),pairs=len(j),coverage=mean(g),pooled=g.success.mean(),before=mean(j,'cost_before'),after=mean(j,'cost_after'),exact_paired=mean(j,'exact_cost'),reduction=ratio,reduction_low=lo,reduction_high=hi,changed_before=mean(j,'changed_before'),changed_after=mean(j,'changed_after'),strict_improvements=int((j.cost_after<j.cost_before-1e-7).sum())))
  pd.DataFrame(rows).to_csv(O/'engine_summary.csv',index=False);(O/'regeneration_check.json').write_text(json.dumps(matches,indent=2))
  # The upstream CARLA kernel uses set iteration to order sampled coordinates.
  # Current comparisons share the fresh fixed-Python-hash-seed returns, rather
  # than pairing a new cost with the old realization's success population.
  fresh=d[d.method=='carla_gs_kernel'].rename(columns={'cost_before':'cost','changed_before':'changed'})
  ext=pd.concat([old[old.method!='carla_gs_kernel'],fresh[[*KEYS,'method','success','cost','changed','status']]],ignore_index=True)
  ext.to_csv(O/'external_refreshed.csv',index=False)
  es=[]
  for (p,m),g in ext.groupby(['protocol','method']):
   j=g.merge(ext[(ext.protocol==p)&(ext.method=='bounded_exact')],on=KEYS,suffixes=('_b','_e'));j=j[(j.success_b==1)&(j.success_e==1)]
   es.append(dict(protocol=p,method=m,coverage=mean(g),cost=mean(g[g.success==1],'cost'),joint=len(j),paired_cost=mean(j,'cost_b'),paired_exact=mean(j,'cost_e'),timeouts=int(g.status.fillna('').str.contains('TimeoutError').sum())))
  pd.DataFrame(es).to_csv(O/'external_summary.csv',index=False)
  facts['engine_reduction_min']=min(x['reduction'] for x in rows);facts['engine_reduction_max']=max(x['reduction'] for x in rows)
 (O/'facts.json').write_text(json.dumps(facts,indent=2));print(json.dumps(facts,indent=2));print(pd.DataFrame(summary).to_string(index=False))
if __name__=='__main__':main()
