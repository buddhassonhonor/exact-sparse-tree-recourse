"""Objective-aligned DiCE adaptation and epsilon path-tweaking comparison."""
from mechanism_core import *
from followup_external import growing
import dice_ml,types,contextlib,io,warnings
from joblib import Parallel,delayed,parallel_config

def aligned_proximity(self,candidates,query_normalized):
 # DiCE passes raw candidates and min--max-normalized query. The closure stores
 # the raw query and training SD, avoiding MAD/min--max objective mismatch.
 d=(np.asarray(candidates,float)-self._raw_query)/self._scale
 return np.abs(d).sum(axis=1) if self._metric=='weighted_l1' else np.linalg.norm(d,axis=1)

def task(name,seed):
 folder=OUT/'external_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists() and 'internal_proximity' in pd.read_csv(dest,nrows=1).columns:return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,20)
 pos=tr[f.predict(tr)==1];names=[f'x{j}' for j in range(tr.shape[1])]
 df=pd.DataFrame(tr.astype(float),columns=names);df['outcome']=yt
 dd=dice_ml.Data(dataframe=df,continuous_features=names,outcome_name='outcome',continuous_features_precision={n:8 for n in names})
 model=dice_ml.Model(model=f,backend='sklearn')
 exps={m:dice_ml.Dice(dd,model,method=('random' if m=='random' else 'genetic')) for m in ['random','genetic_default','genetic_aligned']}
 exps['genetic_aligned'].compute_proximity_loss=types.MethodType(aligned_proximity,exps['genetic_aligned'])
 rows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,act=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S)
   lower=tr.min(axis=0);upper=tr.max(axis=0);lower[F]=x[F];upper[F]=x[F]
   ll=np.maximum(lb,lower);uu=np.minimum(ub,upper);zz=projections(x,ll,uu);zz[:,F]=x[F]
   good=np.all(ll<=uu,axis=1)&np.all(zz>=ll,axis=1)&np.all(zz<=uu,axis=1)&(f.predict(zz)==1)
   e=pick(x,zz,good,scale,protocol.metric)
   base=dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi))
   st=time.perf_counter();z=pick(x,zz,good,scale,protocol.metric)
   rows.append(rec(base,'bounded_exact',x,z,f,scale,protocol.metric,seconds=time.perf_counter()-st,status='ok'))
   st=time.perf_counter();z=growing(f,x,S,scale,np.random.default_rng(seed*1009+int(qi)),lower,upper,protocol.metric)
   rows.append(rec(base,'growing',x,z,f,scale,protocol.metric,seconds=time.perf_counter()-st,status='ok'))
   # Tolomei-style path tweaking, explicitly an axis-aligned constrained adaptation.
   # A scaled epsilon is moved inward on each split face before restoring support.
   for epsilon in [.001,.01]:
    st=time.perf_counter();l=lb.astype(float).copy();u=ub.astype(float).copy()
    l[np.isfinite(l)]+=np.broadcast_to(epsilon*scale,l.shape)[np.isfinite(l)]
    u[np.isfinite(u)]-=np.broadcast_to(epsilon*scale,u.shape)[np.isfinite(u)]
    l=np.maximum(l,lower);u=np.minimum(u,upper);z=projections(x,l,u).astype('float32');z[:,F]=x[F]
    valid=np.all(l<=u,axis=1)&np.all(z>=l,axis=1)&np.all(z<=u,axis=1)&(f.predict(z)==1)
    cand=pick(x,z,valid,scale,protocol.metric)
    rows.append(rec(base,'path_epsilon_'+str(epsilon),x,cand,f,scale,protocol.metric,seconds=time.perf_counter()-st,status='ok'))
   for method,exp in exps.items():
    exp._raw_query=x.astype(float);exp._scale=scale;exp._metric=protocol.metric
    status='ok';candidate=None;internal=np.nan;st=time.perf_counter()
    params=dict(query_instances=pd.DataFrame([x.astype(float)],columns=names),total_CFs=5,desired_class=1,
     features_to_vary=[names[j] for j in S],permitted_range={names[j]:[float(lower[j]),float(upper[j])] for j in S},
     stopping_threshold=.5,posthoc_sparsity_param=0.,verbose=False)
    if method=='random':params.update(sample_size=2000,random_seed=seed+int(qi))
    else:
     params.update(maxiterations=100,initialization='random');np.random.seed(seed+int(qi))
     if method=='genetic_aligned':params.update(proximity_weight=1.,sparsity_weight=0.,diversity_weight=0.,feature_weights={n:1. for n in names})
    original=exp.predict_fn_scores if method!='random' else None
    def deadline(*args,**kwargs):
     if time.perf_counter()-st>2.:raise TimeoutError('2-second prediction-call deadline')
     return original(*args,**kwargs)
    if original is not None:exp.predict_fn_scores=deadline
    try:
     if e is None:raise ValueError('certified_infeasible')
     with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()),warnings.catch_warnings():
      warnings.simplefilter('ignore');res=exp.generate_counterfactuals(**params).cf_examples_list[0]
     if res.final_cfs_df is not None:
      z=res.final_cfs_df[names].to_numpy(dtype='float32')
      good=(f.predict(z)==1)&np.all(z[:,F]==x[F],axis=1)&np.all(z[:,S]>=lower[S],axis=1)&np.all(z[:,S]<=upper[S],axis=1)
      candidate=pick(x,z,good,scale,protocol.metric)
      if candidate is None:status='no_valid_candidate'
      else:
       if method!='random':
        internal=float(exp.compute_proximity_loss(candidate.astype(float)[None],dd.normalize_data(x.astype(float)[None]))[0])
     else:status='no_candidate'
    except Exception as err:status=type(err).__name__+':'+str(err)[:80]
    finally:
     if original is not None:exp.predict_fn_scores=original
    rows.append(rec(base,'dice_'+method,x,candidate,f,scale,protocol.metric,seconds=time.perf_counter()-st,status=status,internal_proximity=internal,candidate=json.dumps(candidate.tolist()) if candidate is not None else ''))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,flush=True)

if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=4)(delayed(task)(d,s) for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:3])
 pd.concat([pd.read_csv(OUT/'external_units'/f'{d}-{s}.csv') for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:3]],ignore_index=True).to_csv(OUT/'external.csv',index=False)
