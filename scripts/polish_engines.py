"""Within-returned-leaf polishing of constrained published-engine outputs."""
from evidence_external import *
OUT=ROOT/'exp/polish_20261101';OUT.mkdir(exist_ok=True)
# Upstream CARLA forms feature lists through Python sets. Fix both its process
# hash seed (inherited by loky workers) and its explicit per-query NumPy seed.
os.environ['PYTHONHASHSEED']='0'
def valid(z,x,S,f,lower,upper):
 if z is None:return False
 F=np.setdiff1d(np.arange(len(x)),S)
 return bool(f.predict(z[None])[0]==1 and np.array_equal(z[F],x[F]) and np.all(z>=lower) and np.all(z<=upper))
def task(name,seed):
 folder=OUT/'engine_units_hash0';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,20);pos=tr[f.predict(tr)==1];names=[f'x{j}' for j in range(tr.shape[1])]
 df=pd.DataFrame(tr.astype(float),columns=names);df['outcome']=yt
 dd=dice_ml.Data(dataframe=df,continuous_features=names,outcome_name='outcome',continuous_features_precision={n:8 for n in names});model=dice_ml.Model(model=f,backend='sklearn');exp=dice_ml.Dice(dd,model,method='random');rows=[]
 leafrows={int(l):i for i,l in enumerate(leaves)}
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,act=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S);lower=tr.min(axis=0);upper=tr.max(axis=0);lower[F]=upper[F]=x[F]
   E=bounded_exact(x,S,lb,ub,lower,upper,scale,protocol.metric)
   for method in ['growing_constrained','dice_random','carla_gs_kernel']:
    status='ok'
    if method=='growing_constrained':z=growing(f,x,S,scale,np.random.default_rng(seed*1009+int(qi)),lower,upper,protocol.metric)
    elif method=='dice_random':
     if E is None:z,status=None,'certified_infeasible_prescreen'
     else:z,status,seconds,internal=dice_run(exp,'random',x,S,names,lower,upper,scale,protocol.metric,seed+int(qi))
    else:
     np.random.seed(seed*1009+int(qi));adapter=CarlaModel(f,scale,lower,upper,S)
     z=growing_spheres_search(pd.Series(x.astype(float)/scale,index=names),[names[j] for j in S],[names[j] for j in F],names,[],names,adapter,n_search_samples=512,p_norm=1 if protocol.metric=='weighted_l1' else 2,step=.2,max_iter=50)
     z=(z*scale).astype('float32') if np.isfinite(z).all() else None
    before=valid(z,x,S,f,lower,upper);P=None;cb=cp=ce=np.nan;changed_b=changed_p=np.nan
    if before:
     i=leafrows[int(f.apply(z[None])[0])];lo=np.maximum(lb[i],lower);hi=np.minimum(ub[i],upper);P=np.minimum(np.maximum(x,lo),hi).astype('float32');P[F]=x[F]
     assert valid(P,x,S,f,lower,upper) and int(f.apply(P[None])[0])==int(f.apply(z[None])[0])
     cb=float(costs(x,z,scale,protocol.metric));cp=float(costs(x,P,scale,protocol.metric));ce=float(costs(x,E,scale,protocol.metric))
     assert cp<=cb+1e-7 and ce<=cp+1e-7
     changed_b=len(c.changed_feature_indices(x,z));changed_p=len(c.changed_feature_indices(x,P))
    after=valid(P,x,S,f,lower,upper);assert before==after
    rows.append(dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi),method=method,success=int(before),polished_success=int(after),cost_before=cb,cost_after=cp,exact_cost=ce,changed_before=changed_b,changed_after=changed_p,status=status))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):Parallel(n_jobs=3)(delayed(task)(d,s) for d in NAMES for s in b.DEFAULT_SEEDS[:3])
 pd.concat([pd.read_csv(OUT/'engine_units_hash0'/f'{d}-{s}.csv') for d in NAMES for s in b.DEFAULT_SEEDS[:3]],ignore_index=True).to_csv(OUT/'engines.csv',index=False)
