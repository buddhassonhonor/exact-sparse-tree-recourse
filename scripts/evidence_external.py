"""24-dataset constrained, post-restoration, live-region and upstream CARLA comparisons."""
from mechanism_core import *
from followup_external import growing
from mechanism_external import aligned_proximity
from mechanism_practical import inward
from vendor.carla_growing_spheres import growing_spheres_search
import dice_ml,types,contextlib,io,warnings
from joblib import Parallel,delayed,parallel_config
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)

def bounded_exact(x,S,lb,ub,lower,upper,scale,metric):
 ll=np.maximum(lb,lower);uu=np.minimum(ub,upper);zz=projections(x,ll,uu)
 return pick(x,zz,np.all(ll<=uu,axis=1),scale,metric)

class CarlaModel:
 def __init__(self,f,scale,lower,upper,S):self.f=f;self.scale=scale;self.lower=lower;self.upper=upper;self.S=S
 def predict_proba(self,z):
  raw=(np.asarray(z,float)*self.scale).astype('float32');p=self.f.predict_proba(raw)
  good=np.all(raw[:,self.S]>=self.lower[self.S],axis=1)&np.all(raw[:,self.S]<=self.upper[self.S],axis=1)
  p[~good]=[1.,0.];return p

def dice_run(exp,method,x,S,names,lower,upper,scale,metric,seed,budget=2.,iterations=100,post=False):
 exp._raw_query=x.astype(float);exp._scale=scale;exp._metric=metric
 status='ok';candidate=None;internal=np.nan;start=time.perf_counter();random=method.startswith('random')
 vary=np.arange(len(x)) if post else S
 params=dict(query_instances=pd.DataFrame([x.astype(float)],columns=names),total_CFs=5,desired_class=1,features_to_vary=[names[j] for j in vary],
  permitted_range={names[j]:[float(lower[j]),float(upper[j])] for j in vary},stopping_threshold=.5,posthoc_sparsity_param=0.,verbose=False)
 if random:params.update(sample_size=2000,random_seed=seed)
 else:
  params.update(maxiterations=iterations,initialization='random');np.random.seed(seed)
  if method=='genetic_aligned':params.update(proximity_weight=1.,sparsity_weight=0.,diversity_weight=0.,feature_weights={n:1. for n in names})
 original=exp.predict_fn_scores if not random else None
 def deadline(*args,**kwargs):
  if time.perf_counter()-start>budget:raise TimeoutError('prediction-call deadline')
  return original(*args,**kwargs)
 if original is not None:exp.predict_fn_scores=deadline
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),warnings.catch_warnings():
   warnings.simplefilter('ignore');res=exp.generate_counterfactuals(**params).cf_examples_list[0]
  if res.final_cfs_df is not None:
   z=res.final_cfs_df[names].to_numpy(dtype='float32');F=np.setdiff1d(np.arange(len(x)),S)
   if post:z[:,F]=x[F]
   good=(exp.model.model.predict(z)==1)&np.all(z[:,F]==x[F],axis=1)&np.all(z[:,S]>=lower[S],axis=1)&np.all(z[:,S]<=upper[S],axis=1)
   candidate=pick(x,z,good,scale,metric)
   if candidate is None:status='no_valid_candidate'
   elif not random:internal=float(exp.compute_proximity_loss(candidate.astype(float)[None],exp.data_interface.normalize_data(x.astype(float)[None]))[0])
  else:status='no_candidate'
 except Exception as err:status=type(err).__name__+':'+str(err)[:80]
 finally:
  if original is not None:exp.predict_fn_scores=original
 return candidate,status,time.perf_counter()-start,internal

def task(name,seed):
 folder=OUT/'external_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,20);pos=tr[f.predict(tr)==1];names=[f'x{j}' for j in range(tr.shape[1])]
 df=pd.DataFrame(tr.astype(float),columns=names);df['outcome']=yt
 dd=dice_ml.Data(dataframe=df,continuous_features=names,outcome_name='outcome',continuous_features_precision={n:8 for n in names})
 model=dice_ml.Model(model=f,backend='sklearn');exps={m:dice_ml.Dice(dd,model,method='random' if m.startswith('random') else 'genetic') for m in ['random','random_postrestore','genetic_default','genetic_aligned']}
 exps['genetic_aligned'].compute_proximity_loss=types.MethodType(aligned_proximity,exps['genetic_aligned'])
 live=set(f.apply(tr));live_rows=np.asarray([j for j,l in enumerate(leaves) if l in live]);assert len(live_rows)==len(leaves)
 rows=[];repeatrows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,act=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S);lo=tr.min(axis=0);hi=tr.max(axis=0)
   lower=lo.copy();upper=hi.copy();lower[F]=x[F];upper[F]=x[F]
   base=dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi))
   start=time.perf_counter();E=bounded_exact(x,S,lb,ub,lower,upper,scale,protocol.metric)
   rows.append(rec(base,'bounded_exact',x,E,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='ok'))
   start=time.perf_counter();L=bounded_exact(x,S,lb[live_rows],ub[live_rows],lower,upper,scale,protocol.metric)
   assert (L is None)==(E is None)
   if E is not None:assert np.array_equal(L,E)
   rows.append(rec(base,'lire_single',x,L,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='identity_expected'))
   nearest=pos[np.argmin(np.linalg.norm((pos.astype(float)-x)/scale,axis=1))].copy();nearest[F]=x[F]
   rows.append(rec(base,'nearest_point_postrestore',x,nearest,f,scale,protocol.metric,seconds=np.nan,status='wrapper'))
   # Both path variants use the same inward-rounded boxes. Postrestore first
   # selects an unrestricted path; compatibility-first enumerates feasible paths.
   for eps in [.001,.01]:
    l=lb.astype(float).copy();u=ub.astype(float).copy();l[np.isfinite(l)]+=np.broadcast_to(eps*scale,l.shape)[np.isfinite(l)];u[np.isfinite(u)]-=np.broadcast_to(eps*scale,u.shape)[np.isfinite(u)]
    l,u=inward(l,u)
    start=time.perf_counter();z=bounded_exact(x,S,l,u,lower,upper,scale,protocol.metric)
    rows.append(rec(base,'path_compatible_'+str(eps),x,z,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='adaptation'))
    start=time.perf_counter();ll=np.maximum(l,lo);uu=np.minimum(u,hi);z=projections(x,ll,uu);good=np.all(ll<=uu,axis=1)
    z=pick(x,z,good,scale,protocol.metric)
    if z is not None:z[F]=x[F]
    rows.append(rec(base,'path_postrestore_'+str(eps),x,z,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='wrapper'))
   for repeat in range(5):
    rngseed=seed*1009+int(qi)+repeat*100003;start=time.perf_counter();z=growing(f,x,S,scale,np.random.default_rng(rngseed),lower,upper,protocol.metric)
    record=rec(dict(**base,repeat=repeat),'growing_constrained',x,z,f,scale,protocol.metric,seconds=time.perf_counter()-start,exact_success=int(E is not None))
    repeatrows.append(record)
    if repeat==0:rows.append({k:v for k,v in record.items() if k not in ['repeat','exact_success']})
   start=time.perf_counter();z=growing(f,x,np.arange(len(x)),scale,np.random.default_rng(seed*1009+int(qi)),lo,hi,protocol.metric)
   if z is not None:z[F]=x[F]
   rows.append(rec(base,'growing_postrestore',x,z,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='wrapper'))
   start=time.perf_counter();np.random.seed(seed*1009+int(qi));adapter=CarlaModel(f,scale,lower,upper,S)
   z=growing_spheres_search(pd.Series(x.astype(float)/scale,index=names),[names[j] for j in S],[names[j] for j in F],names,[],names,adapter,n_search_samples=512,p_norm=1 if protocol.metric=='weighted_l1' else 2,step=.2,max_iter=50)
   z=(z*scale).astype('float32') if np.isfinite(z).all() else None
   rows.append(rec(base,'carla_gs_kernel',x,z,f,scale,protocol.metric,seconds=time.perf_counter()-start,status='upstream_kernel_with_adapter'))
   for method,exp in exps.items():
    if E is None:z,status,seconds,internal=None,'certified_infeasible_prescreen',0.,np.nan
    else:
     l,h=(lo,hi) if method.endswith('postrestore') else (lower,upper)
     z,status,seconds,internal=dice_run(exp,method,x,S,names,l,h,scale,protocol.metric,seed+int(qi),post=method.endswith('postrestore'))
    rows.append(rec(base,'dice_'+method,x,z,f,scale,protocol.metric,seconds=seconds,status=status,internal_proximity=internal))
 pd.DataFrame(repeatrows).to_csv(folder/f'{name}-{seed}-gs.csv',index=False);pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=4)(delayed(task)(d,s) for d in NAMES for s in b.DEFAULT_SEEDS[:3])
 for suffix,out in [('', 'external'),('-gs','gs_repeats')]:
  pd.concat([pd.read_csv(OUT/'external_units'/f'{d}-{s}{suffix}.csv') for d in NAMES for s in b.DEFAULT_SEEDS[:3]],ignore_index=True).to_csv(OUT/f'{out}.csv',index=False)
