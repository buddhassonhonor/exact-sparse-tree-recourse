"""Build manuscript tables/plots only from saved observations; no model reruns."""
from pathlib import Path
import json,re,sys
import numpy as np,pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];O=R/'exp/mechanism_20261101';F=R/'figures'
sys.path.insert(0,str(R/'script'))
from mechanism_core import NAMES,DECISION,CAT,b
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
P=['immutable_l2','immutable_top3_l1'];PN={'immutable_l2':'Immutable','immutable_top3_l1':'Top-3','free_l2':'Free'}
M=['lf_prr1','lf_prr20','leaves10','filter_first','exact'];MN={'lf_prr1':'LF-PRR1','lf_prr20':'LF-PRR20','leaves10':'First 10 leaves','filter_first':'Filter-first','exact':'Exact'}
def esc(s):return str(s).replace('adult_income','Adult').replace('credit_default','Credit').replace('bank_marketing','Bank').replace('breast_cancer','Breast').replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')
def num(x):return f'{x:.3f}' if np.isfinite(x) else '--'
def interval(x,boots=5000):
 x=np.asarray(x,float);x=x[np.isfinite(x)]
 if not len(x):return (np.nan,np.nan)
 rng=np.random.default_rng(871);means=x[rng.integers(0,len(x),(boots,len(x)))].mean(axis=1)
 return np.quantile(means,[.025,.975])
def units(df,value='success'):
 return df.groupby(['dataset','seed'])[value].mean().groupby('dataset').mean()
def mean(df,value='success'):return units(df,value).mean()
def ci(x):a,z=interval(x);return f'[{a:.3f}, {z:.3f}]'
def table(label,caption,headers,rows,widths=None):
 spec='@{}' + ('l'+'c'*(len(headers)-1) if widths is None else ''.join('p{'+w+r'\textwidth}' for w in widths))+'@{}'
 sep={'tab:inference':'2pt','tab:scaling':'1.5pt','tab:strata':'2pt'}.get(label,'3pt')
 out=r'\begin{table}[t!]'+ '\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\centering\scriptsize\setlength{\tabcolsep}{'+sep+'}\n'+r'\begin{tabular}{'+spec+'}'+r'\toprule'+'\n'
 out+=' & '.join(headers)+r' \\\midrule'+'\n'
 for row in rows:out+=' & '.join(str(x) for x in row)+r' \\'+'\n'
 return out+r'\bottomrule\end{tabular}'+'\n'+r'\end{table}'+'\n'
def fig(file,label,caption,width='.95'):
 return r'\begin{figure}[t!]\centering\includegraphics[width='+width+r'\textwidth]{'+file+'}\n'+r'\caption{'+caption+'}\n'+r'\label{'+label+'}\n'+r'\end{figure}'+'\n'
def savefig(fig,name):fig.savefig(F/name,bbox_inches='tight');plt.close(fig)
def paired(df,method,reference='exact'):
 keys=['dataset','seed','depth','protocol','query'];a=df[df.method==method];e=df[df.method==reference]
 return a.merge(e,on=keys,suffixes=('_b','_e')).query('success_b==1 and success_e==1')

def main():
 a=pd.read_csv(O/'breadth.csv');q=a[a.depth==5];me=pd.read_csv(O/'mechanism.csv');de=pd.read_csv(O/'decomposition.csv');po=pd.read_csv(O/'pools.csv');ra=pd.read_csv(O/'ranks.csv')
 ext=pd.read_csv(O/'external.csv');sol=pd.read_csv(O/'solver.csv');pr=pd.read_csv(O/'practical.csv');lat=pd.read_csv(O/'latency.csv')
 out=r'''\subsection{Coverage, Metrics and Support}\label{sec:main}
The broadened benchmark separates search configuration from the feasible set. Table~\ref{tab:breadth} reports unit-weighted and pooled coverage at depth five. Free-support metric-matched methods coincide with Exact. The filter-first and Exact coverage columns coincide by Theorem~\ref{thm:anchors}, and were also asserted equal query by query in the implementation.
'''
 rows=[];summary=[]
 for p in P:
  for m in M+['native']:
   z=q[(q.protocol==p)&(q.method==m)];v=units(z)
   rows.append([PN[p],MN.get(m,'Native support'),num(v.mean()),ci(v),num(z.success.mean()),num(mean(z[z.success==1],'cost'))])
   summary.append(dict(protocol=p,method=m,coverage=v.mean(),pooled=z.success.mean(),conditional_cost=mean(z[z.success==1],'cost')))
 out+=table('tab:breadth','Depth-five coverage and conditional cost on 24 datasets, five seeds and the same randomly sampled queries. Unit means average within each tree and then equally across datasets. Pooled means weight every query equally. Costs use each method\'s own successes and do not establish paired cost superiority.',['Protocol','Method','Unit coverage','95\% cluster CI','Pooled','Cost'],rows)
 summary=pd.DataFrame(summary);summary.to_csv(O/'breadth_summary.csv',index=False)
 im=summary.query("protocol=='immutable_l2'").set_index('method');sp=summary.query("protocol=='immutable_top3_l1'").set_index('method')
 headline=f"Across 24 datasets and five splits, LF-PRR20 coverage is {im.loc['lf_prr20','coverage']:.3f}/{sp.loc['lf_prr20','coverage']:.3f} under Immutable/top-3, versus {im.loc['exact','coverage']:.3f}/{sp.loc['exact','coverage']:.3f} for Exact."
 out+=r'\revadd{'+headline+r'} Native support optimization changes the feasible set; its gain is not a search-within-support effect.'+'\n'
 out+=r'''The original eight-dataset, ten-seed benchmark retains the objective-mismatch ablation in Table~\ref{tab:waterfall}. Euclidean LF-PRR1's 0.742-to-0.954 comparison mixes metric choice with search configuration. Metric matching, one-anchor repair, 20-anchor repair, complete fixed-support search, and native support selection are displayed separately. Adjacent coverage differences are descriptive increments under these stated configurations, not independent causal effects.
'''
 out+=table('tab:waterfall','Configuration waterfall on the original eight datasets and ten seeds (depth five). Matching the objective and adding repair are separate steps.',['Configuration','Immutable','Top-3'],[
 ['Euclidean projection','0.659','0.301'],['Euclidean LF-PRR1','0.742','0.420'],['Metric-matched projection','0.793','0.474'],['Metric-matched LF-PRR1','0.860','0.544'],['Metric-matched LF-PRR20','0.943','0.641'],['Fixed-support Exact','0.954','0.685'],['Native support Exact','0.954','0.954']])
 out+=r'\subsection{Visited-Leaf Cost Decomposition}\label{sec:mechanism-results}'+'\n'
 n=me.query("method=='exact' and depth==5 and protocol=='immutable_l2'").shape[0]
 out+=f'The paired mechanism study uses eight datasets, five seeds, depths 5/11 and up to 40 randomly ordered negative queries per tree ({n:,} per protocol at depth five). Every method in each row uses the identical queries and supports. '+'\n'
 rows=[];pairrows=[]
 for h in [5,11]:
  for p in P:
   d=me[(me.depth==h)&(me.protocol==p)]
   for m in ['lf_prr20','lf_prr20_output_polish','lf_prr20_oracle','lf_prr20_polish','filter_first','filter_first_output_polish','all_repair','all_repair_output_polish','all_output_polish','all_endpoint_polish']:
    z=paired(d,m);cv=mean(d[d.method==m]);cb=mean(z,'cost_b');ce=mean(z,'cost_e')
    if h==5:rows.append([PN[p],esc(m.replace('_',' ')),num(cv),str(len(z)),num(cb),num(ce),num(mean(z,'changed_b')),num(mean(z,'changed_e'))])
    pairrows.append(dict(depth=h,protocol=p,method=m,coverage=cv,n=len(z),cost_b=cb,cost_e=ce,changed_b=mean(z,'changed_b'),changed_e=mean(z,'changed_e')))
 out+=table('tab:polish','Depth-five paired cost and changed-coordinate counts. B and E columns use each row\'s jointly successful queries, averaged equally over nonempty dataset--seed units. Oracle means the validated interval reference including the bisection candidate; polish projects into its reached leaf. Different rows can have different joint sets.',['Protocol','Configuration','Cov.','Pairs','B cost','E cost','B dims','E dims'],rows)
 rows=[]
 for (h,p,m),g in de.groupby(['depth','protocol','baseline']):
  rows.append([str(h),PN[p],esc(m.replace('_',' '))]+[num(mean(g,v)) for v in ['bracket','within_leaf','leaf_choice','total']])
 out+=table('tab:decomposition',r'Same-anchor telescoping cost decomposition, Equation~\ref{eq:cost-decomposition}, on jointly successful queries. Means use identical query pairs for all four entries within a row. The interval reference includes the observed bisection output.',['Depth','Protocol','Baseline','Segment','Within leaf','Leaf choice','Total'],rows)
 dfpairs=pd.DataFrame(pairrows);dfpairs.to_csv(O/'paired_summary.csv',index=False)
 check=paired(me,'all_output_polish');strict=(check.cost_b>check.cost_e+1e-7)
 out+=r"\revadd{All restored-endpoint polishing matched Exact's candidate exactly for every feasible query.}"+' '
 out+=f'Polishing all bisection outputs retained excess cost in {int(strict.sum()):,} of {len(check):,} joint successes in this benchmark. '+r'However, the explicit construction in Appendix~\ref{sec:polish-counterexample} gives cost 40.013 after repair-output polishing versus 2.000 for Exact and endpoint polishing. Visiting every endpoint leaf and visiting every repair-output leaf are different operations.'+'\n'
 out+=r'The decomposition attributes most measured excess to within-leaf points rather than segment refinement. Because all differences in a row use the same chosen anchor and queries, the components sum exactly. The corresponding changed-coordinate identity is descriptive; an exact distance optimum need not minimize cardinality across leaves.'+'\n'
 out+=r'\subsection{When Anchor Completeness Is Removed}\label{sec:pool}'+'\n'
 rows=[]
 for name,g in po.groupby('pool'):
  vals=[num(mean(g[(g.depth==h)&(g.protocol==p)])) for h in [5,11] for p in P]
  rows.append([esc(name.replace('_',' '))]+vals)
 out+=table('tab:pools','Anchor-pool stress test on the same random paired queries. Directly valid projections are retained in every condition. Subsampling uses a seeded random pool shared across queries; prototypes use at most 32 k-means centers, filtered by the fitted model. Margins screen anchors by scaled distance from all finite leaf faces.',['Pool','D5 Immutable','D5 Top-3','D11 Immutable','D11 Top-3'],rows)
 out+=r'''\revadd{Full training-pool equality is the predicted consequence of leaf coverage. Subsampled, prototype and margin-screened pools can miss compatible leaves and lose coverage.} Ground-truth-and-model-positive and held-out-model-positive pools need not cover every leaf, even where their aggregate result is close. Held-out positive records come from the same test split but are disjoint from the evaluated model-negative queries; test labels are not consulted for that pool. They are not an independently recruited validation population. Pool construction is fixed before inspecting query success.
\subsection{Normalized Leaf Budgets and Depth}\label{sec:depth}
'''
 rows=[]
 for h in [3,5,7,9,11,20]:
  g=ra[(ra.depth==h)&(ra.protocol==P[1])];v=g[g.exact_success==1];rat=np.minimum(10/g.positive_leaves,1)
  # Required K is the empirical quantile of the first-compatible rank among feasible queries.
  rows.append([str(h),f'{g.positive_leaves.median():.0f}',f'{rat.median():.2f}',str(int(np.quantile(v.first_rank,.95,method='higher'))),str(int(np.quantile(v.first_rank,.99,method='higher'))),str(int(np.quantile(v.cost99_rank,.95,method='higher')))])
 out+=table('tab:normalized','Top-3 positive-leaf budgets on 24 datasets. K95/K99 are pooled first-compatible-rank quantiles, conditional on Exact feasibility; the last column is the 95th percentile of the per-query budget attaining cost at most Exact/0.99. These pooled rank summaries are distinct from unit-weighted coverage.',['Max depth','Median $N_+$','Median $\min(10/N_+,1)$','K95','K99','Cost99 K95'],rows)
 large=pd.read_csv(O/'large_ranks.csv');rows=[]
 for name,g in large.groupby('dataset'):
  v=g[g.exact_success==1];rows.append([esc(name),f'{g.positive_leaves.median():.0f}',f'{(10/g.positive_leaves).median():.4f}',str(int(np.quantile(v.first_rank,.95,method='higher'))),str(int(np.quantile(v.first_rank,.99,method='higher'))),f'{v.fraction.quantile(.95):.3f}'])
 out+=table('tab:large-budgets','The same normalized diagnostic on full unpruned real-data trees and 120,000-record noise trees (five seeds, 100 random queries each). K quantiles condition on Exact feasibility. The final column is the pooled 95th percentile of first-compatible rank/$N_+$.',['Tree','Median $N_+$','Median $10/N_+$','K95','K99','Fraction95'],rows)
 fig0,axs=plt.subplots(1,2,figsize=(10,3.25))
 for ax,p in zip(axs,P):
  for m in M:
   xs=[];ys=[];lo=[];hi=[]
   for h in [3,5,7,9,11,20]:
    v=units(a[(a.depth==h)&(a.protocol==p)&(a.method==m)]);l,u=interval(v);xs.append(h);ys.append(v.mean());lo.append(l);hi.append(u)
   ax.plot(xs,ys,marker='o',ms=3,label=MN[m],ls='--' if m=='filter_first' else '-');ax.fill_between(xs,lo,hi,alpha=.055)
  ax.set(xlabel='Maximum depth',ylabel='Coverage',title=PN[p],ylim=(0,1.02))
 axs[1].legend(fontsize=7,loc='lower right');fig0.tight_layout();savefig(fig0,'mechanism_depth.png')
 out+=fig('mechanism_depth.png','fig:depth','Matched methods at depths 3--20 on 24 datasets. Bands are 95\% dataset-cluster bootstrap intervals. Filter-first and Exact overlap by theorem. Five splits and at most 100 random negative queries per tree are used.')
 fig0,axs=plt.subplots(1,2,figsize=(10,3.25))
 for ax,p in zip(axs,P):
  for h in [5,11,20]:
   xx=[];yy=[];lo=[];hi=[]
   for frac in [.01,.05,.1,.25,.5,1.]:
    v=units(a[(a.depth==h)&(a.protocol==p)&(a.method=='fraction_'+str(frac))]);l,u=interval(v);xx.append(frac);yy.append(v.mean());lo.append(l);hi.append(u)
   ax.plot(xx,yy,marker='o',ms=3,label=f'Depth {h}');ax.fill_between(xx,lo,hi,alpha=.08)
  ax.set(xlabel='Requested fraction of positive leaves',ylabel='Coverage',title=PN[p],ylim=(0,1.02));ax.legend(fontsize=8)
 fig0.tight_layout();savefig(fig0,'mechanism_leaf_fraction.png')
 out+=fig('mechanism_leaf_fraction.png','fig:normalized',r'Coverage for $K=\max(1,\lceil\rho N_+\rceil)$; the horizontal axis is requested $\rho$, while actual K/$N_+$ is discretized upward on small trees. Bands use the same dataset-cluster bootstrap. A fixed ten-leaf budget is not interpreted as size independent.')
 out+=r'''At depth five ten leaves often inspect most or all positive leaves. Their near-complete coverage is therefore not a universal small-budget result. Lemma~\ref{lem:budget} establishes the worst case, and the normalized curves expose the budget relative to tree size. Full sorting still evaluates and ranks every leaf, so reducing the output budget alone does not reduce all projection work.
\subsection{Dataset-Cluster Inference and Observed Subpopulations}\label{sec:inference}
'''
 infer=[];rows=[]
 for subset,names in [('all24',NAMES),('context14',DECISION)]:
  for p in P:
   u=q[(q.dataset.isin(names))&(q.protocol==p)].groupby(['dataset','seed','method']).success.mean().unstack('method').groupby('dataset').mean()
   for m in ['lf_prr1','lf_prr20','leaves10','filter_first']:
    diff=u.exact-u[m];n=len(diff);se=diff.std(ddof=1)/np.sqrt(n);delta=[]
    for margin in [.02,.03]:
     if se==0:pv=0. if abs(diff.mean())<margin else 1.
     else:pv=max(stats.t.sf((diff.mean()+margin)/se,n-1),stats.t.cdf((diff.mean()-margin)/se,n-1))
     delta.append(pv)
    rng=np.random.default_rng(711);draw=rng.choice([-1,1],(100000,n));pv=(1+np.count_nonzero(np.abs(draw@diff.to_numpy()/n)>=abs(diff.mean())-1e-12))/100001
    infer.append(dict(subset=subset,protocol=p,method=m,D=n,gap=diff.mean(),ci_low=interval(diff)[0],ci_high=interval(diff)[1],signflip=pv,tost02=delta[0],tost03=delta[1]))
    rows.append([subset,PN[p],MN[m],num(diff.mean()),ci(diff),num(pv),num(delta[0]),num(delta[1])])
 pd.DataFrame(infer).to_csv(O/'inference.csv',index=False)
 out+=table('tab:inference','Exact minus baseline coverage at depth five. Cluster CIs resample datasets; sign-flip p-values use 100,000 Monte Carlo draws with add-one correction. TOST reports the larger one-sided t-test p-value for symmetric equivalence margins 0.02/0.03 (alpha 0.05). Context14 is the prespecified finance/clinical/social-context subset, not a collection of validated intervention tasks.',['Subset','Protocol','Baseline','Gap','95\% CI','Sign-flip p','TOST .02','TOST .03'],rows)
 out+=r'\revadd{A nonsignificant superiority test is not evidence of equivalence. TOST evaluates the stated practical margins separately; the all-pool filter-first equality is proved rather than inferred from these tests.} Comparisons are exploratory and have no multiplicity adjustment.'+'\n'
 envnames=['iid','pc1_low','pc1_high','mahalanobis_tail'];ev=[];rows=[];fig0,axs=plt.subplots(1,2,figsize=(10,3.2))
 for ax,p in zip(axs,P):
  for m in M:
   yy=[];lo=[];hi=[]
   for env in envnames:
    g=q[(q.protocol==p)&(q.method==m)&(q.dataset!='synthetic_overlap')&(q[env]==1)];v=units(g);l,u=interval(v);yy.append(v.mean());lo.append(l);hi.append(u)
    ev.append(dict(protocol=p,method=m,environment=env,coverage=v.mean(),low=l,high=u,D=len(v),queries=len(g)))
   ax.errorbar(np.arange(4),yy,yerr=[np.array(yy)-lo,np.array(hi)-yy],marker='o',ms=3,capsize=2,label=MN[m],ls='--' if m=='filter_first' else '-')
  ax.set_xticks(np.arange(4),['IID','PC1 low','PC1 high','Mahalanobis']);ax.tick_params(axis='x',labelsize=8);ax.set(title=PN[p],ylabel='Coverage',ylim=(0,1.03))
 axs[1].legend(fontsize=7,loc='lower right');fig0.tight_layout();savefig(fig0,'mechanism_subpopulations.png');pd.DataFrame(ev).to_csv(O/'environments.csv',index=False)
 out+=fig('mechanism_subpopulations.png','fig:reweighting','Coverage on unchanged test records from 23 real datasets. Training-standardized principal-component and covariance transforms are fitted only on training data; tails are quartiles of each sampled negative-query set. Intervals resample datasets, as in the depth study. Filter-first and Exact coincide.')
 out+=r'''The subpopulation analysis uses the same random query pool before selecting lower/upper first-principal-component (PC1) quartiles and the upper Ledoit--Wolf Mahalanobis-distance quartile. It changes observed composition without generating new records. Small subgroups have fewer queries; no temporal or geographic distribution-shift guarantee is inferred.
\subsection{Controlled Constraint Dose}\label{sec:support-dose}
'''
 dose=pd.read_csv(R/'exp/followup_20261101/dose.csv');fig0,axs=plt.subplots(2,2,figsize=(9,6));dose_stats=[]
 for ax,var,sel,title in [(axs[0,0],'immutable_fraction',(dose.k==3)&(dose.depth==5),'Frozen fraction (k=3, depth=5)'),(axs[0,1],'k',(dose.immutable_fraction==.5)&(dose.depth==5),'Cardinality (frozen fraction=0.5, depth=5)'),(axs[1,0],'depth',(dose.immutable_fraction==.5)&(dose.k==3),'Depth (frozen fraction=0.5, k=3)')]:
  xx=[];yy=[];ls=[];us=[]
  for val,g in dose[sel& (dose.exact_success==1)].groupby(var):
   v=(1-g.groupby('dataset').repair_success.mean());
   if len(v)<4:continue
   l,u=interval(v);xx.append(val);yy.append(v.mean());ls.append(l);us.append(u);dose_stats.append(dict(axis=var,value=val,D=len(v),failure=v.mean(),low=l,high=u))
  ax.errorbar(xx,yy,yerr=[np.array(yy)-ls,np.array(us)-yy],marker='o',capsize=3);ax.set(title=title,ylabel='P(LF-PRR20 fails | Exact feasible)',ylim=(0,1))
 d=dose[dose.exact_success==1].copy();d['bin']=pd.cut(d.cost_ratio,[0,.1,.25,.5,.75,1.000001],include_lowest=True)
 xx=[];yy=[];ls=[];us=[]
 for bb,g in d.groupby('bin',observed=True):
  v=1-g.groupby('dataset').repair_success.mean()
  if len(v)<4:continue
  l,u=interval(v);xx.append(bb.mid);yy.append(v.mean());ls.append(l);us.append(u);dose_stats.append(dict(axis='cost_ratio',value=bb.mid,D=len(v),failure=v.mean(),low=l,high=u))
 axs[1,1].errorbar(xx,yy,yerr=[np.array(yy)-ls,np.array(us)-yy],marker='o',capsize=3);axs[1,1].set(title='Unrestricted / compatible optimal cost',xlabel='Cost ratio',ylabel='Conditional failure',ylim=(0,1),xlim=(0,1))
 fig0.tight_layout();savefig(fig0,'mechanism_doses.png');pd.DataFrame(dose_stats).to_csv(O/'dose_summary.csv',index=False)
 out+=fig('mechanism_doses.png','fig:dose','Controlled dose comparisons use eight datasets and five seeds, with the other displayed factors held fixed. The final panel bins the unrestricted/compatible optimum cost ratio across the sweep. Points require at least four dataset clusters; bounded 95\% percentile bootstrap intervals resample datasets. The cost ratio is a derived geometric quantity, not a controlled treatment.')
 out+=r'''Random frozen-coordinate permutations are seeded per query and nested across fractions; k changes the nearest-positive support within the remaining coordinates. The existing sweep spans fractions 0/0.25/0.5/0.75, k=1--5 and depths 3/5/11/20. Compatible-leaf fraction alone is not presented as the controlled cause of failure.
\subsection{External Methods with Aligned Objectives}\label{sec:external}
'''
 rows=[]
 for m,g in ext.groupby('method'):
  cells=[]
  for p in P:
   z=g[g.protocol==p];cells.append(num(mean(z))+' / '+num(mean(z[z.success==1],'cost')))
  rows.append([esc(m.replace('_',' '))]+cells+[str(int(g.status.fillna('').str.contains('TimeoutError').sum())),f'{g.seconds.median()*1000:.2f}'])
 out+=table('tab:external','Numeric-subspace external comparison on eight datasets, three seeds and at most 20 random queries per tree. Cells show unit-weighted coverage / cost conditional on each method\'s successes. Times are median milliseconds from concurrent quality runs, not the isolated latency study. Timeout cases count as failures.',['Method','Immutable','Top-3','Timeouts','Median ms'],rows)
 timeout=int(ext.status.fillna('').str.contains('TimeoutError').sum())
 out+=r'''DiCE (Diverse Counterfactual Explanations) random search uses 2,000 samples and five alternatives. Genetic search uses five alternatives, at most 100 iterations and a two-second prediction-call deadline. The default row retains DiCE's loss; the aligned adaptation replaces its proximity function with exactly $c_1$/$c_2$, sets proximity weight one and diversity/sparsity weights zero, and retains its target-class loss. Internal proximity is evaluated by each configured DiCE loss on its returned candidate and reported in Appendix~\ref{sec:external-details}. '''+f'{timeout} deadline cases are retained as failures.'+'\n'
 out+=r'''All methods share fixed supports and training-range action bounds. Constrained Growing Spheres samples 512 random directions per shell, with 14 radii from 0.01 to 64 in SD units and radii uniform over each [r/2,r] interval, clips to range, then tries restoring each changed coordinate once in increasing scaled-change order while preserving positivity. It is a constrained adaptation, not an anchor method. The Tolomei-style rows enumerate positive paths and move finite faces inward by epsilon=0.001/0.01 training SD before support restoration; their treatment of constraints and epsilon scaling is also an adaptation \cite{tolomei2017feature}. No universal external-method ranking follows from these budgets.
\subsection{Independent Optimization and Executable Precision}\label{sec:practical}
'''
 rows=[]
 for p,g in sol.groupby('protocol'):
  opt=g[g.status=='optimal'];rows.append([PN[p],str(len(g)),str(len(opt)),str(int((g.status=='infeasible').sum())),f'{opt.objective_difference.abs().max():.2e}',str(int((opt.raw_valid==0).sum())),f'{opt.reconstructed_difference.abs().max():.1e}'])
 out+=table('tab:solver','Independent node-flow encoding from raw tree arrays: HiGHS MILP for weighted L1; ECOS\_BB MISOCP for scaled L2. Raw invalid counts identify numerical solver outputs failing inference/support before path reconstruction. Reconstructed candidates use the solver-selected raw path, not the shared cached leaf boxes.',['Protocol','Checks','Optimal','Infeasible','Max obj. error','Raw invalid','Rebuilt error'],rows)
 out+=r'''\revadd{The independent solver encoding never reads the cached leaf-box list.} Binary node flows enforce root activation, child-flow conservation and zero flow to negative leaves; each active edge imposes its raw split inequality using finite range-derived big-M bounds. Frozen coordinates have zero standardized displacement. HiGHS uses a ten-second limit and relative gap $10^{-9}$; ECOS\_BB uses 10,000 integer iterations and $10^{-9}$ continuous feasibility/absolute/relative tolerances, with integer absolute/relative gaps $10^{-8}$. Model construction is outside the recorded solve time. Near one-ulp faces, raw solver coordinates can route incorrectly despite the correct objective; reconstruction clips along the solver-selected raw path and revalidates inference. These checks verify selected small trees, not all possible trees.
'''
 rows=[]
 for m,g in pr.groupby('method'):
  rows.append([esc(m.replace('_',' '))]+[num(mean(g[g.protocol==p])) for p in P])
 out+=table('tab:practical','Precision and feature-direction sensitivity on eight datasets, five seeds and up to 40 random queries per tree. Semantic rows use the four listed policy datasets and a new policy support, so they are not paired search improvements over the generic support.',['Method','Immutable','Top-3'],rows)
 case=json.loads((O/'credit_margin_case.json').read_text());cv=case[2];val=cv['changes']['pay_amt3'][1]
 out+=r'''Directed rounding changes only coordinates that differ from the query: increases round upward and decreases downward to the training-inferred decimal grid. Nearest rounding and directed rounding are revalidated, since either can cross another leaf face. Grid optimization instead searches every positive leaf on the declared grid, retaining an observed off-grid value when no change is required. Neither rounding rule is asserted to preserve validity without revalidation.

Feature-specific sensitivity policies allow Credit Default's limit balance and payment amounts to increase, Adult's education number/capital gain/hours to increase, German Credit's requested duration/amount/installment rate to decrease, and Bank Marketing's balance to increase. Other variables are frozen; each change is bounded by one training SD and the training range. Bank call duration and contact history are frozen. Credit payment amounts are retrospective measurements: the directions are hypothetical planning analogies, not an ability to alter recorded history or a validated causal intervention.

The credit example retains the same negative query, support and full tree path. Without margin, payment amount 3 changes from 5,000 to 8,688.0009765625 at cost 0.210, just above the strict split 8,688. '''+r'\revadd{At a 0.01-SD inward margin it changes to '+f'{val:,.7f}'+r' at cost '+f'{cv["cost"]:.3f}'+r', still changing one coordinate and passing model validation.}'+r''' Margins 0.001/0.05 yield 8,705.556640625/9,565.78125 and costs 0.211/0.260. The full positive path is payment status 0 $>1.5$, bill amount 2 $>1016$, payment amount 5 $\le3370.5$, payment amount 3 $>8688$. Margin feasibility is a declared numerical policy, not robustness to unknown execution errors.
\subsection{Indexed Anchors and Constraint-Pruned Exact}\label{sec:scaling}
'''
 rows=[];lat_summary=[]
 for (d,n,alpha),g in lat.groupby(['dataset','n_records','alpha']):
  counts=g.drop_duplicates(['seed','query']);leaf=g.drop_duplicates('seed');cells=[]
  for m in ['exact','pruned','leaves10_partition','lf20_sort','lf20_index','filter_first','all_repair']:
   v=g[g.method==m];cells.append(f'{v.ms.median():.3f}');lat_summary.append(dict(dataset=d,n=n,alpha=alpha,method=m,median_ms=v.ms.median(),coverage=v.drop_duplicates(['seed','query']).success.mean(),queries=len(counts)))
  rows.append([esc(d)+(f' {n//1000}k' if d=='noise' else '')+(' pruned' if alpha else ''),f'{leaf.total_leaves.median():.0f}/{leaf.positive_leaves.median():.0f}',str(len(counts))]+cells)
 out+=table('tab:scaling','Isolated single-process timing (median ms over five fitted trees, up to 100 random queries each and three repeats). Total/positive leaf counts are medians over trees. Partial leaf ranking uses argpartition; indexed anchors use a cKDTree in full scaled feature space. Real-data pruned rows use ccp\_alpha=0.0001.',['Tree','Leaves total/+','n','Exact','DFS','K10','LF20 sort','LF20 index','Filter','All repair'],rows)
 rows=[]
 for (d,n,alpha),g in lat.groupby(['dataset','n_records','alpha']):
  cells=[]
  for m in ['exact','leaves10_partition','lf20_index','filter_first','all_repair']:
   v=g[g.method==m].drop_duplicates(['seed','query']);cells.append(num(v.groupby('seed').success.mean().mean()))
  rows.append([esc(d)+(f' {n//1000}k' if d=='noise' else '')+(' pruned' if alpha else '')]+cells)
 out+=table('tab:scaling-quality','Coverage on the same scaling queries, equally weighted over the five trees in each row. Timing repeats do not add queries. DFS is identical to Exact by per-query assertion.',['Tree','Exact/DFS','K10','LF20 index','Filter','All repair'],rows)
 pd.DataFrame(lat_summary).to_csv(O/'latency_summary.csv',index=False)
 strata=lat.groupby(['method','stratum']).agg(ms=('ms','median'),coverage=('success','mean'),n=('query','size')).reset_index();strata.to_csv(O/'latency_strata.csv',index=False)
 out+=r'''\revadd{Latency conclusions depend on implementation and query stratum; the experiment does not claim that Exact is uniformly faster.} The same support is precomputed for every method and its KD-tree nearest-anchor lookup is recorded separately. Query timing includes each method's leaf projection/ranking, restoration, model calls and repair, but excludes fitting, cache construction, indexing and Numba compilation. Each method receives one warm-up per query; three measured repeats run in seeded random order. DFS candidates are asserted bit-identical to cached enumeration. Index construction time, coordinate-storage bytes, support time, node counts, stratum-specific IQRs and coverage are reported in Appendix~\ref{sec:timing}. KD-tree worst-case query time remains linear in anchors in high dimension.

Global interval search is timed only when $A N_+\le10,000$; this explicit workspace gate avoids materializing large anchor-by-leaf arrays. Excluded configurations are unmeasured, not failures or claimed speedups. Restore/filter and all-anchor repair inspect up to A anchors, often far more than $N_+$. Total leaves and positive leaves are kept separate. Independent-label Gaussian trees isolate computation; their label-flipping actions have no application interpretation.
'''
 (R/'mechanism_results.tex').write_text(out,encoding='utf8')
 # Appendix: dataset identities, per-dataset coverage, full timing and iteration settings.
 ap=r'\section{Dataset Identities and Frozen Targets}\label{sec:datasets}'+'\n'
 ap+=r'''The additional OpenML tasks use provider-native binary labels and numeric columns only, excluding id/index/row\_id fields. Numeric missing values are imputed with training medians. Their diagnostic immutable set is the first max(1,floor(d/5)) numeric columns; it is an index-based stress policy, not a domain actionability judgment. The artifact freezes feature lists, class encodings and source IDs. All named real tasks are distinct benchmark records; finance/clinical/social context is a descriptive grouping. In particular, a model target is not automatically a favorable clinical or lending outcome.
'''
 rows=[]
 for name in NAMES:
  if name.startswith('openml_'):
   v=CAT.loc[int(name.split('_')[1])];rows.append([str(v.name),esc(v['name']),str(int(v.n)),str(int(v.numeric)),('yes' if name in DECISION else 'no')])
 ap+=table('tab:datasets','Sixteen added binary OpenML datasets. Sizes are source counts before any seed-specific stratified cap of 6,000. Numeric-column counts exclude the target. The original eight datasets and actionability policies are described in Appendix B.',['OpenML ID','Name','Source n','Numeric d','Context subset'],rows)
 ap+=r'\section{A Counterexample for Repair-Output Polishing}\label{sec:polish-counterexample}'+'\n'
 ap+=r'''Let $x=(0,0,0)$ with adjustable coordinates $(u,v)$ and frozen $w=0$, and unit-scaled L2 cost. A constructed tree has positive regions $w>1/4,u\le1/4,v\le1/4$ and, for $w\le1/4$, the two boxes $1<u\le1.5,40<v\le60$ and $2<u\le3,-0.001<v\le100$. Axis-aligned splits realize these disjoint regions. The pool $\{(0,0,1/2),(1.25,50,0),(2.5,100,0)\}$ covers every positive leaf. The nearest unrestricted leaf restores to $x$. On the segment to $(2.5,100,0)$ the first midpoint $(1.25,50,0)$ is positive in the expensive box; eight-step bisection thereafter retains that box, as does the segment to its own anchor. The upper anchor restores negative. Polishing all retained outputs gives $(\operatorname{succ}_{32}(1),\operatorname{succ}_{32}(40),0)$, cost 40.012502. Polishing all valid restored endpoints visits the other compatible leaf and returns $(\operatorname{succ}_{32}(2),0,0)$, cost 2.000000238, exactly Exact. A direct sklearn-tree construction verifies the leaf memberships and both outputs. This separates the empirical equality in the benchmark from a general guarantee.
'''
 ap+=r'\section{External-Method Joint Costs}\label{sec:external-details}'+'\n'
 rows=[]
 for p in P:
  for method in sorted(set(ext.method)-{'bounded_exact'}):
   z=paired(ext[ext.protocol==p],method,'bounded_exact');g=ext[(ext.protocol==p)&(ext.method==method)&(ext.success==1)]
   iv=g.internal_proximity.mean() if 'internal_proximity' in g else np.nan
   rows.append([PN[p],esc(method.replace('_',' ')),str(len(z)),num(mean(z,'cost_b')),num(mean(z,'cost_e')),f'{iv:.5g}' if np.isfinite(iv) else '--'])
 ap+=table('tab:external-paired','Joint-success costs against range-bounded Exact on identical random queries. Internal proximity is DiCE\'s configured loss on returned candidates (default normalized/MAD-weighted loss or the adapted shared cost), averaged over candidates with a valid return. Random search has no internal proximity objective.',['Protocol','Method','Pairs','Method cost','Exact cost','Internal proximity'],rows)
 ap+=r'\section{Full Timing and Stratification}\label{sec:timing}'+'\n'
 rows=[]
 for (m,st),g in lat.groupby(['method','stratum']):
  uq=g.drop_duplicates(['dataset','n_records','seed','alpha','query']);tree=uq.groupby(['dataset','n_records','alpha','seed']).success.mean()
  # Dependence is at fitted-tree level here, not repeat level.
  l,u=interval(tree)
  rows.append([esc(m.replace('_',' ')),esc(st.replace('_',' ')),str(len(uq)),f'{g.ms.median():.3f}',f'{g.ms.quantile(.25):.3f}--{g.ms.quantile(.75):.3f}',num(tree.mean()),f'[{l:.3f},{u:.3f}]'])
 ap+=table('tab:strata','Timing strata across all scaling configurations. Query counts are unique per tree and exclude repeats. Coverage intervals resample fitted trees within this timing collection; these are separate from the dataset-cluster quality inference. Direct means the metric-selected projection is already valid; repair feasible means it is invalid but Exact is feasible.',['Method','Stratum','Queries','Median ms','IQR ms','Coverage','95\% tree CI'],rows)
 rows=[]
 for (d,n,alpha),g in lat.groupby(['dataset','n_records','alpha']):
  source_size=n if d=='noise' else 0
  metas=[json.loads((O/'latency_units'/f'{d}-{source_size}-{seed}-{alpha}.json').read_text()) for seed in sorted(g.seed.unique())]
  rows.append([esc(d)+(f' {n//1000}k' if d=='noise' else '')+(' pruned' if alpha else ''),f'{np.median([z["cache_ms"] for z in metas]):.2f}',f'{np.median([z["index_ms"]+z["raw_index_ms"] for z in metas]):.2f}',f'{np.median([z["cache_bytes"] for z in metas])/1e6:.3f}',f'{np.median([z["index_data_bytes"] for z in metas])/1e6:.3f}',f'{np.mean([np.mean([v["visited"] for v in z["queries"]]) for z in metas]):.1f}'])
 ap+=table('tab:cache-index','One-time cache and two KD-tree construction costs (median over five trees), coordinate-array storage and mean nodes visited. Index MB counts coordinate arrays only and excludes KD-tree nodes/allocator overhead; it is not peak RSS.',['Tree','Cache ms','Index ms','Boxes MB','Index data MB','DFS nodes'],rows)
 # Retain only the six-setting iteration table from prior results, with explicit historical design.
 old=(R/'followup_appendix.tex').read_text(encoding='utf8') if (R/'followup_appendix.tex').exists() else ''
 match=re.search(r'\\begin\{table\}.*?\\label\{tab:iteration\}.*?\\end\{table\}',old,re.S)
 if match:ap+=r'\section{Iteration Sensitivity on the Original Benchmark}\label{sec:iteration}'+'\n'+r'The original eight-dataset, ten-seed full-query study retains all six iteration settings. Coverage for positive iteration budgets is invariant by endpoint retention; more iterations refine cost.'+'\n'+match.group(0)+'\n'
 ap+=r'\section{Notation}\label{sec:notation}'+'\n'
 ap+=table('tab:notation','Symbols kept distinct throughout.',['Symbol','Meaning'],[[r'$\ell$','Leaf index; $(u,v)$ denote coordinates in the constructed examples'],[r'$N_+$','Number of positive leaves'],[r'$\mathcal A,\mathcal I,S$','Actionable, immutable and fixed support coordinate sets'],[r'$\mathcal Q,A=|\mathcal Q|$','Anchor pool and anchor count'],[r'$B_a,T,R$','Repair attempts, bisection steps and far-anchor coordinate'],[r'$\mathcal N,\mathcal C$','Numeric coordinate set and categorical groups'],[r'$B,O,P,E$','Bisection point, interval reference, leaf-polished point, global optimum']])
 (R/'mechanism_appendix.tex').write_text(ap,encoding='utf8')
 (O/'headline.json').write_text(json.dumps(dict(headline=headline,summary=summary.to_dict('records'),mechanism_queries=int(me.query("method=='exact' and depth==5 and protocol=='immutable_l2'").shape[0])),indent=2))

if __name__=='__main__':main()
