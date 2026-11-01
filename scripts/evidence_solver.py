"""Independent raw-tree node-flow MILP / MISOCP validation (no leaf-box input)."""
from mechanism_core import *
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)
import cvxpy as cp,warnings
from joblib import Parallel,delayed,parallel_config

def solve_raw(f,x,S,scale,lower,upper,metric):
 build_start=time.perf_counter()
 t=f.tree_;d=len(x);n=t.node_count;F=np.setdiff1d(np.arange(d),S)
 low=(lower.astype(float)-x)/scale;high=(upper.astype(float)-x)/scale
 low[F]=0.;high[F]=0.
 z=cp.Variable(d);flow=cp.Variable(n,boolean=True)
 con=[z>=low,z<=high,flow[0]==1];parent={};edges={}
 for node in range(n):
  l,r=int(t.children_left[node]),int(t.children_right[node])
  if l==r:
   if np.argmax(t.value[node,0])!=1:con.append(flow[node]==0)
   continue
  con.append(flow[l]+flow[r]==flow[node]);j=int(t.feature[node]);threshold=t.threshold[node]
  a=np.float32(threshold)
  pred=np.nextafter(a,np.float32(-np.inf)) if float(a)>threshold else a
  succ=np.nextafter(a,np.float32(np.inf)) if float(a)<=threshold else a
  a=(float(pred)-float(x[j]))/scale[j];bnd=(float(succ)-float(x[j]))/scale[j]
  con.extend([z[j]<=a+max(0.,high[j]-a)*(1-flow[l]),z[j]>=bnd-max(0.,bnd-low[j])*(1-flow[r])])
  if low[j]>a:con.append(flow[l]==0)
  if high[j]<bnd:con.append(flow[r]==0)
  parent[l]=parent[r]=node;edges[l]=(j,'upper',pred);edges[r]=(j,'lower',succ)
 objective=cp.norm1(z) if metric=='weighted_l1' else cp.norm2(z)
 problem=cp.Problem(cp.Minimize(objective),con)
 build_seconds=time.perf_counter()-build_start
 st=time.perf_counter()
 try:
  with warnings.catch_warnings():
   warnings.simplefilter('ignore')
   if metric=='weighted_l1':problem.solve(solver='HIGHS',highs_options={'time_limit':10.,'mip_rel_gap':1e-9,'mip_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9,'threads':1})
   else:problem.solve(solver='ECOS_BB',mi_max_iters=1000,abstol=1e-9,reltol=1e-9,feastol=1e-9,mi_abs_eps=1e-8,mi_rel_eps=1e-8)
 except Exception as err:return dict(status=type(err).__name__,objective=np.nan,seconds=time.perf_counter()-st,build_seconds=build_seconds,raw_valid=0,candidate=None)
 result=dict(status=problem.status,objective=problem.value,seconds=time.perf_counter()-st,build_seconds=build_seconds,raw_valid=0,candidate=None)
 if z.value is None or flow.value is None:return result
 candidate=(x.astype(float)+np.asarray(z.value)*scale).astype('float32')
 result['raw_valid']=int(f.predict(candidate[None])[0]==1 and np.array_equal(candidate[F],x[F]))
 # Numerically reconstruct the solver-selected path using the raw split arrays.
 # This corrects solver tolerance at one-ulp boundaries without using shared boxes.
 leaves=[q for q in range(n) if t.children_left[q]==t.children_right[q] and np.argmax(t.value[q,0])==1]
 leaf=max(leaves,key=lambda q:float(flow.value[q]));lo=lower.copy();hi=upper.copy();node=leaf
 while node:
  j,kind,value=edges[node]
  if kind=='upper':hi[j]=min(hi[j],value)
  else:lo[j]=max(lo[j],value)
  node=parent[node]
 lo[F]=np.maximum(lo[F],x[F]);hi[F]=np.minimum(hi[F],x[F])
 if np.all(lo<=hi):
  candidate=np.minimum(np.maximum(x,lo),hi)
  if f.predict(candidate[None])[0]==1:result['candidate']=candidate
 return result


def task(name,seed,depth):
 folder=OUT/'solver_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}-{depth}.csv'
 if dest.exists() and 'comparison_timing_full' in pd.read_csv(dest,nrows=1).columns:return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,depth,4);pos=tr[f.predict(tr)==1];rows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,_=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S)
   lo=tr.min(axis=0);hi=tr.max(axis=0);lo[F]=x[F];hi[F]=x[F]
   ll=np.maximum(lb,lo);uu=np.minimum(ub,hi);zz=projections(x,ll,uu);good=np.all(ll<=uu,axis=1)&(f.predict(zz)==1)
   metrics=['weighted_l1']+(['scaled_l2'] if seed==7 and depth in [3,5] else [])
   for metric in metrics:
    st=time.perf_counter();ll=np.maximum(lb,lo);uu=np.minimum(ub,hi);zz=projections(x,ll,uu);good=np.all(ll<=uu,axis=1)&(f.predict(zz)==1);e=pick(x,zz,good,scale,metric);exact_seconds=time.perf_counter()-st
    result=solve_raw(f,x,S,scale,lo,hi,metric);cand=result.pop('candidate')
    ce=float(costs(x,e,scale,metric)) if e is not None else np.nan
    cc=float(costs(x,cand,scale,metric)) if cand is not None else np.nan
    obj=result['objective'] if result['objective'] is not None else np.nan
    rows.append(dict(dataset=name,seed=seed,depth=depth,protocol=protocol.name,metric=metric,query=int(qi),positive_leaves=len(leaves),
     exact_success=int(e is not None),exact_cost=ce,reconstructed_cost=cc,objective_difference=obj-ce,reconstructed_difference=cc-ce,exact_seconds=exact_seconds,comparison_timing_full=True,**result))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,depth,flush=True)

if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=2)(delayed(task)(d,s,h) for h in [3,5,11] for d in NAMES for s in b.DEFAULT_SEEDS[:2])
 pd.concat([pd.read_csv(OUT/'solver_units'/f'{d}-{s}-{h}.csv') for h in [3,5,11] for d in NAMES for s in b.DEFAULT_SEEDS[:2]],ignore_index=True).to_csv(OUT/'solver.csv',index=False)
