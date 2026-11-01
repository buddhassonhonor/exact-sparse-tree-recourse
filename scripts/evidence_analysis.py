"""Saved-result summaries, identity audit and publication-size figures."""
from mechanism_assets import *
O=R/'exp/evidence_20261101';OLD=R/'exp/mechanism_20261101'
NICE={n:({'credit_default':'Credit Default','adult_income':'Adult','german_credit':'German','bank_marketing':'Bank','breast_cancer':'Breast Cancer','wine_binary':'Wine','synthetic_overlap':'Synthetic','compas':'COMPAS'}.get(n,n)) for n in NAMES}
for n in NAMES:
 if n.startswith('openml_'):NICE[n]=str(CAT.loc[int(n.split('_')[1]),'name']).replace('blood-transfusion-service-center','Blood transfusion').replace('banknote-authentication','Banknote').replace('credit-approval','Credit approval')
def ratio_ci(g):
 u=g.groupby(['dataset','seed'])[['bracket','within_leaf','leaf_choice','total']].mean().groupby('dataset').mean()
 rng=np.random.default_rng(897);z=u.to_numpy()[rng.integers(0,len(u),(5000,len(u)))].mean(axis=1);fr=z[:,1]/z[:,3]
 return u.mean(),np.quantile(fr,[.025,.975])
def main():
 a=pd.read_csv(OLD/'breadth.csv');q=a[a.depth==5].copy();extra=pd.read_csv(O/'main_extra.csv');q=pd.concat([q,extra],ignore_index=True)
 summary=[]
 for (p,m),g in q.groupby(['protocol','method']):
  if m.startswith('fraction_'):continue
  u=units(g);l,h=interval(u);summary.append(dict(protocol=p,method=m,unit_coverage=u.mean(),low=l,high=h,pooled_coverage=g.success.mean(),queries=len(g),cost=mean(g[g.success==1],'cost'),changed=mean(g[g.success==1],'changed')))
 pd.DataFrame(summary).to_csv(O/'main_summary.csv',index=False)
 me=pd.read_csv(O/'mechanism.csv');summ=[]
 for (h,p,m),g in me.groupby(['depth','protocol','method']):
  joint=paired(me[(me.depth==h)&(me.protocol==p)],m)
  summ.append(dict(depth=h,protocol=p,method=m,unit_coverage=mean(g),pooled_coverage=g.success.mean(),total_queries=len(g),pairs=len(joint),cost_b=mean(joint,'cost_b'),cost_e=mean(joint,'cost_e'),changed_b=mean(joint,'changed_b'),changed_e=mean(joint,'changed_e')))
 pd.DataFrame(summ).to_csv(O/'paired_summary.csv',index=False)
 dec=pd.read_csv(O/'decomposition.csv');ds=[]
 for (h,p,m),g in dec.groupby(['depth','protocol','baseline']):
  avg,ci_=ratio_ci(g);ds.append(dict(depth=h,protocol=p,baseline=m,within_fraction=avg.within_leaf/avg.total,within_low=ci_[0],within_high=ci_[1],**avg.to_dict()))
 pd.DataFrame(ds).to_csv(O/'decomposition_summary.csv',index=False)
 checks=paired(me,'all_output_polish');bad=(checks.cost_b>checks.cost_e+1e-7)
 ranks=pd.read_csv(OLD/'ranks.csv');feas=ranks[ranks.exact_success==1];assert (feas.first_rank==feas.cost99_rank).all() and (feas.first_rank==feas.cost95_rank).all()
 boundary=pd.read_csv(R/'exp/boundary_audit/20260917-193249/split_checks.csv')
 audit=dict(rank_feasible_queries=len(feas),first_equals_cost99=int((feas.first_rank==feas.cost99_rank).sum()),all_output_pairs=len(checks),all_output_excess=int(bad.sum()),old_mechanism_depth5_queries=int(pd.read_csv(OLD/'mechanism.csv').query("method=='exact' and depth==5 and protocol=='immutable_l2'").shape[0]),old_internal_split_checks=len(boundary))
 (O/'identity_audit.json').write_text(json.dumps(audit,indent=2))
 # Observed-subpopulation table, same cluster inference as the primary quality study.
 env=[]
 for p in P:
  for e in ['iid','pc1_low','pc1_high','mahalanobis_tail']:
   for m in ['lf_prr20','leaves10','exact']:
    g=q[(q.protocol==p)&(q.method==m)&(q.dataset!='synthetic_overlap')&(q[e]==1)];u=units(g);l,h=interval(u);env.append(dict(protocol=p,environment=e,method=m,coverage=u.mean(),low=l,high=h,queries=len(g)))
 pd.DataFrame(env).to_csv(O/'environments.csv',index=False)
 # Dataset forest. Within-row uncertainty describes sensitivity to five splits.
 fig0,ax=plt.subplots(figsize=(5.7,6.4));forest=[]
 for offset,(p,color) in enumerate(zip(P,['#1764a0','#cc6b24'])):
  wide=q[q.protocol==p].groupby(['dataset','seed','method']).success.mean().unstack('method')
  for i,name in enumerate(NAMES):
   v=wide.loc[name,'exact']-wide.loc[name,'lf_prr20'];l,h=interval(v);y=i+(offset-.5)*.25
   ax.errorbar(v.mean(),y,xerr=[[v.mean()-l],[h-v.mean()]],fmt='o',ms=3,color=color,capsize=2,label=PN[p] if i==0 else None)
   forest.append(dict(dataset=name,protocol=p,gap=v.mean(),split_low=l,split_high=h))
 ax.axvline(0,color='grey',lw=.7);ax.set_yticks(range(len(NAMES)),[NICE[n] for n in NAMES],fontsize=8);ax.invert_yaxis();ax.set(xlabel='Exact minus LF-PRR20 coverage');ax.legend(fontsize=8,loc='lower right');fig0.tight_layout();savefig(fig0,'coverage_gap_forest.png');pd.DataFrame(forest).to_csv(O/'forest.csv',index=False)
 # Pooled leaf-budget ranks, with the redundant cost column deliberately absent.
 nr=[]
 for h,g in ranks[ranks.protocol==P[1]].groupby('depth'):
  v=g[g.exact_success==1];nr.append(dict(depth=h,Nplus=g.positive_leaves.median(),fraction10=np.minimum(10/g.positive_leaves,1).median(),K95=np.quantile(v.first_rank,.95,method='higher'),K99=np.quantile(v.first_rank,.99,method='higher')))
 pd.DataFrame(nr).to_csv(O/'normalized_budgets.csv',index=False)
 # Pool-level leaf coverage and query-specific compatible-leaf omissions.
 pool=pd.read_csv(O/'pools.csv');pr=[]
 for (h,p,name),g in pool.groupby(['depth','protocol','pool']):
  pr.append(dict(depth=h,protocol=p,pool=name,coverage=mean(g),positive_leaf_coverage=mean(g,'positive_leaf_coverage'),uncovered_compatible_fraction=mean(g,'uncovered_compatible_fraction')))
 pd.DataFrame(pr).to_csv(O/'pool_summary.csv',index=False)
 pe=me[me.method=='exact'][['dataset','seed','depth','protocol','query','success']].rename(columns={'success':'exact_success'})
 cp=pool.merge(pe,on=['dataset','seed','depth','protocol','query']);cp=cp[cp.exact_success==1].copy();cp['loss']=1-cp.success;cp['bin']=pd.cut(cp.uncovered_compatible_fraction,[-1e-9,.001,.1,.25,.5,.75,1.0001])
 fig0,ax=plt.subplots(figsize=(5.4,3.1));ps=[]
 for p,color in zip(P,['#1764a0','#cc6b24']):
  xs=[];ys=[];los=[];his=[]
  for bb,g in cp[cp.protocol==p].groupby('bin',observed=True):
   u=g.groupby('dataset').loss.mean()
   if len(u)<4:continue
   l,h=interval(u);xs.append(max(0,bb.mid));ys.append(u.mean());los.append(l);his.append(h);ps.append(dict(protocol=p,bin=str(bb),clusters=len(u),loss=u.mean(),low=l,high=h))
  ax.errorbar(xs,ys,yerr=[np.array(ys)-los,np.array(his)-ys],marker='o',ms=3,capsize=2,label=PN[p],color=color)
 ax.set(xlabel='Uncovered fraction of compatible leaves',ylabel='Coverage loss | Exact feasible',ylim=(-.02,1.02));ax.legend(fontsize=8);fig0.tight_layout();savefig(fig0,'anchor_coverage_loss.png');pd.DataFrame(ps).to_csv(O/'pool_bins.csv',index=False)
 dose=pd.read_csv(O/'dose.csv');dose=dose[dose.exact_success==1].copy();dose['failure']=1-dose.repair_success
 statsrows=[];fig0,axs=plt.subplots(2,2,figsize=(5.65,5.3))
 for ax,key,sel,title in [(axs[0,0],'immutable_fraction',(dose.k==3)&(dose.depth==5),'Frozen fraction'),(axs[0,1],'k',(dose.immutable_fraction==.5)&(dose.depth==5),'Cardinality k'),(axs[1,0],'depth',(dose.immutable_fraction==.5)&(dose.k==3),'Maximum depth')]:
  xs=[];ys=[];lo=[];hi=[]
  for val,g in dose[sel].groupby(key):
   u=g.groupby('dataset').failure.mean();l,h=interval(u);xs.append(val);ys.append(u.mean());lo.append(l);hi.append(h)
   statsrows.append(dict(axis=key,value=val,failure=u.mean(),low=l,high=h,clusters=len(u),activation=mean(g.assign(seed=g.seed),'direct')))
  ax.errorbar(xs,ys,yerr=[np.array(ys)-lo,np.array(hi)-ys],marker='o',ms=3,capsize=2);ax.set(xlabel=title,ylabel='Conditional failure',ylim=(0,.55));ax.tick_params(labelsize=8)
 dose['ratio_bin']=pd.cut(dose.cost_ratio,[0,.1,.25,.5,.75,.9,1.00001],include_lowest=True);xs=[];ys=[];lo=[];hi=[]
 for bb,g in dose.groupby('ratio_bin',observed=True):
  u=g.groupby('dataset').failure.mean()
  if len(u)<4:continue
  l,h=interval(u);xs.append(bb.mid);ys.append(u.mean());lo.append(l);hi.append(h);statsrows.append(dict(axis='cost_ratio',value=bb.mid,failure=u.mean(),low=l,high=h,clusters=len(u)))
 axs[1,1].errorbar(xs,ys,yerr=[np.array(ys)-lo,np.array(hi)-ys],marker='o',ms=3,capsize=2);axs[1,1].set(xlabel='Unrestricted / compatible cost',ylabel='Conditional failure',ylim=(0,.55));axs[1,1].tick_params(labelsize=8)
 fig0.tight_layout(pad=.7);savefig(fig0,'constraint_dose_results.png');pd.DataFrame(statsrows).to_csv(O/'dose_summary.csv',index=False)
 # Refreshed inferential table omits the deterministic equality entirely.
 infer=pd.read_csv(OLD/'inference.csv');infer=infer[infer.method!='filter_first'].copy();infer['subset']=infer['subset'].replace({'context14':'subset14'})
 minima=[]
 for _,v in infer.iterrows():
  names=NAMES if v['subset']=='all24' else DECISION
  w=q[(q.protocol==v.protocol)&(q.dataset.isin(names))].groupby(['dataset','seed','method']).success.mean().unstack('method').groupby('dataset').mean();gap=w.exact-w[v.method]
  radius=stats.t.ppf(.95,len(gap)-1)*gap.std(ddof=1)/np.sqrt(len(gap));minima.append(max(abs(gap.mean()-radius),abs(gap.mean()+radius)))
 infer['minimum_margin90']=minima;infer.to_csv(O/'inference.csv',index=False)
 # Timing strata remain grouped by tree configuration, with no definitional coverage columns.
 lat=pd.read_csv(OLD/'latency.csv');strata=[]
 for (d,n,alpha,m,st),g in lat.groupby(['dataset','n_records','alpha','method','stratum']):
  qq=g.groupby(['seed','query']).agg(ms=('ms','median'),success=('success','first'));strata.append(dict(dataset=d,n_records=n,alpha=alpha,method=m,stratum=st,queries=len(qq),median_ms=qq.ms.median(),q25=qq.ms.quantile(.25),q75=qq.ms.quantile(.75),coverage=qq.success.mean()))
 pd.DataFrame(strata).to_csv(O/'timing_by_tree_stratum.csv',index=False)
 # External summaries and explicit stochastic misses.
 ext=pd.read_csv(O/'external.csv');es=[]
 for (p,m),g in ext.groupby(['protocol','method']):
  j=paired(ext[ext.protocol==p],m,'bounded_exact');es.append(dict(protocol=p,method=m,coverage=mean(g),pooled=g.success.mean(),queries=len(g),cost=mean(g[g.success==1],'cost'),joint=len(j),paired_cost=mean(j,'cost_b'),paired_exact=mean(j,'cost_e'),timeouts=int(g.status.fillna('').str.contains('TimeoutError').sum())))
 pd.DataFrame(es).to_csv(O/'external_summary.csv',index=False)
 gs=pd.read_csv(O/'gs_repeats.csv');gsstats=[]
 for (p,rep),g in gs.groupby(['protocol','repeat']):
  gsstats.append(dict(protocol=p,repeat=rep,coverage=mean(g),cost=mean(g[g.success==1],'cost'),misses=int(((g.exact_success==1)&(g.success==0)).sum()),exact_feasible=int(g.exact_success.sum())))
 pd.DataFrame(gsstats).to_csv(O/'gs_summary.csv',index=False)
 print('Audit',audit);print('Prepared summaries and figures from saved observations.')
if __name__=='__main__':main()
