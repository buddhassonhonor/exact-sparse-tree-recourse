"""External recourse comparisons and an independent HiGHS mixed-integer check."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,contextlib,io,inspect,warnings
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.optimize import OptimizeResult
from joblib import Parallel,delayed,parallel_config
import dice_ml
from followup_study import *
def growing(f,x,S,scale,rng,lower,upper,metric):
 best=None
 for radius in np.geomspace(.01,64,14):
  directions=rng.normal(size=(512,len(S))); directions/=np.maximum(np.linalg.norm(directions,axis=1,keepdims=True),1e-12)
  rr=rng.uniform(radius/2,radius,size=(512,1))**(1/max(len(S),1))
  # Uniform directions with radii in the current shell.
  rr=rng.uniform(radius/2,radius,size=(512,1))
  z=np.broadcast_to(x,(512,len(x))).copy()
  z[:,S]=np.clip(x[S]+directions*rr*scale[S],lower[S],upper[S])
  yes=f.predict(z)==1
  if yes.any(): best=pick(x,z,yes,scale,metric);break
 if best is not None:
  for j in S[np.argsort(np.abs((best[S]-x[S])/scale[S]))]:
   if x[j]<lower[j] or x[j]>upper[j]: continue
   z=best.copy();z[j]=x[j]
   if f.predict(z[None])[0]==1: best=z
 return best
def independent_milp(x,S,scale,lb,ub,lower,upper):
 feasible=np.all(np.maximum(lb,lower)<=np.minimum(ub,upper),axis=1)
 lb,ub=lb[feasible],ub[feasible]
 if len(lb)==0:return OptimizeResult(status=2,success=False,fun=None)
 d=len(x);L=len(lb);n=2*d+L
 low=(lower.astype(float)-x)/scale;high=(upper.astype(float)-x)/scale
 frozen=np.setdiff1d(np.arange(d),S);low[frozen]=0;high[frozen]=0
 objective=np.r_[np.zeros(d),np.ones(d),np.zeros(L)]
 vl=np.r_[low,np.zeros(d),np.zeros(L)];vu=np.r_[high,np.full(d,np.inf),np.ones(L)]
 A=[];rhs=[]
 for j in range(d):
  row=np.zeros(n);row[j]=1;row[d+j]=-1;A.append(row);rhs.append(0.)
  row=np.zeros(n);row[j]=-1;row[d+j]=-1;A.append(row);rhs.append(0.)
 for ell in range(L):
  for j in range(d):
   if np.isfinite(lb[ell,j]):
    bound=(float(lb[ell,j])-float(x[j]))/scale[j];M=max(0.,bound-low[j])
    row=np.zeros(n);row[j]=-1;row[2*d+ell]=M;A.append(row);rhs.append(M-bound)
   if np.isfinite(ub[ell,j]):
    bound=(float(ub[ell,j])-float(x[j]))/scale[j];M=max(0.,high[j]-bound)
    row=np.zeros(n);row[j]=1;row[2*d+ell]=M;A.append(row);rhs.append(M+bound)
 row=np.zeros(n);row[2*d:]=1
 integrality=np.r_[np.zeros(2*d),np.ones(L)]
 result=milp(objective,integrality=integrality,bounds=Bounds(vl,vu),
  constraints=[LinearConstraint(np.asarray(A),-np.inf,np.asarray(rhs)),LinearConstraint(row,1,1)],
  options={'time_limit':10.,'mip_rel_gap':1e-9,'mip_feasibility_tolerance':1e-10,'primal_feasibility_tolerance':1e-10})
 return result
def external_task(dataset,seed):
 out=OUT/'external_units_v2';out.mkdir(parents=True,exist_ok=True)
 dest=out/f'{dataset}-{seed}.csv'
 if dest.exists():return
 ds,train,test,scale,f,lb,ub,pos,ids=context(dataset,seed,5,20)
 names=[f'x{j}' for j in range(train.shape[1])]
 df=pd.DataFrame(train.astype(float),columns=names);df['outcome']=ds.y[train_test_split(np.arange(len(ds.y)),test_size=.35,stratify=ds.y,random_state=seed)[0]]
 # Labels passed to DiCE are the supervised training labels; anchors use model predictions.
 dd=dice_ml.Data(dataframe=df,continuous_features=names,outcome_name='outcome',continuous_features_precision={n:8 for n in names})
 model=dice_ml.Model(model=f,backend='sklearn')
 explainers={k:dice_ml.Dice(dd,model,method=k) for k in ['random','genetic']}
 rows=[];checks=[]
 for protocol in b.PROTOCOLS[1:]:
  action=c.actionable_indices(train.shape[1],b.immutable_indices(ds,protocol))
  for qi in ids:
   x=test[qi];nn=c.nearest_positive_train(x,pos);S=c.choose_support(x,nn,action,protocol.max_changes)
   lower=train.min(axis=0);upper=train.max(axis=0)
   lower[np.setdiff1d(np.arange(len(x)),S)]=x[np.setdiff1d(np.arange(len(x)),S)]
   upper[np.setdiff1d(np.arange(len(x)),S)]=x[np.setdiff1d(np.arange(len(x)),S)]
   ll=np.maximum(lb,lower);uu=np.minimum(ub,upper)
   zz=projections(x,ll,uu);zz[:,np.setdiff1d(np.arange(len(x)),S)]=x[np.setdiff1d(np.arange(len(x)),S)]
   yes=np.all(ll<=uu,axis=1)&np.all(zz>=ll,axis=1)&np.all(zz<=uu,axis=1)&(f.predict(zz)==1)
   exact=pick(x,zz,yes,scale,protocol.metric)
   base=dict(dataset=dataset,seed=seed,depth=5,protocol=protocol.name,query=int(qi))
   rows.append(record(base,'bounded_exact',x,exact,f,S,scale,protocol.metric))
   z=growing(f,x,S,scale,np.random.default_rng(seed*1009+int(qi)),lower,upper,protocol.metric)
   rows.append(record(base,'constrained_growing_spheres',x,z,f,S,scale,protocol.metric,exact))
   for method,exp in explainers.items():
    status='ok';candidate=None;start=time.perf_counter()
    params=dict(query_instance=pd.DataFrame([x.astype(float)],columns=names),total_CFs=5,desired_class=1,
     features_to_vary=[names[j] for j in S],permitted_range={names[j]:[float(lower[j]),float(upper[j])] for j in S},
     stopping_threshold=.5,posthoc_sparsity_param=0.,verbose=False)
    if method=='random':params.update(sample_size=2000,random_seed=seed+int(qi))
    else:params.update(maxiterations=100,initialization='random');np.random.seed(seed+int(qi))
    original_predict=exp.predict_fn_scores if method=='genetic' else None
    def bounded_predict(*args,**kwargs):
     if time.perf_counter()-start>2.: raise TimeoutError('two-second per-query budget')
     return original_predict(*args,**kwargs)
    if method=='genetic': exp.predict_fn_scores=bounded_predict
    try:
     if exact is None: raise ValueError('certified_infeasible_support_and_range')
     with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),warnings.catch_warnings():
      warnings.simplefilter('ignore')
      params['query_instances']=params.pop('query_instance')
      res=exp.generate_counterfactuals(**params).cf_examples_list[0]
     if res.final_cfs_df is not None:
      z=res.final_cfs_df[names].to_numpy(dtype='float32')
      frozen=np.setdiff1d(np.arange(len(x)),S)
      valid=(f.predict(z)==1)&np.all(z[:,frozen]==x[frozen],axis=1)&np.all(z[:,S]>=lower[S],axis=1)&np.all(z[:,S]<=upper[S],axis=1)
      candidate=pick(x,z,valid,scale,protocol.metric)
      if candidate is None:status='no_valid_candidate'
     else:status='no_candidate'
    except Exception as err:status=type(err).__name__+':'+str(err)[:120]
    finally:
     if method=='genetic': exp.predict_fn_scores=original_predict
    rows.append(record(base,'dice_'+method,x,candidate,f,S,scale,protocol.metric,exact,status=status,seconds=time.perf_counter()-start))
   # Independent L1 objective for both supports; do not claim an L2 MILP check.
   exactl1=pick(x,zz,yes,scale,'weighted_l1')
   result=independent_milp(x,S,scale,lb,ub,lower,upper)
   diff=float(result.fun-costs(x,exactl1,scale,'weighted_l1')) if result.fun is not None and exactl1 is not None else np.nan
   checks.append(dict(**base,solver_status=result.status,exact_success=int(exactl1 is not None),
    milp_success=int(result.success),objective=result.fun,exact_cost=float(costs(x,exactl1,scale,'weighted_l1')) if exactl1 is not None else np.nan,difference=diff))
   if result.success and exactl1 is not None:assert abs(diff)<=1e-6,(base,diff)
   if exactl1 is None:assert result.status==2,(base,result.status)
 pd.DataFrame(checks).to_csv(out/f'{dataset}-{seed}-milp.csv',index=False)
 pd.DataFrame(rows).to_csv(dest,index=False);print(f'external {dataset} {seed}',flush=True)
def main():
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=4)(delayed(external_task)(d,s) for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:3])
 for kind in ['external','milp']:
  files=[OUT/'external_units_v2'/f'{d}-{s}{"" if kind=="external" else "-milp"}.csv' for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:3]]
  pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(OUT/f'{kind}.csv',index=False)
if __name__=='__main__':main()

