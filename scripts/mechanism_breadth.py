"""Breadth, depth, normalized leaf budgets, and observed-subpopulation evaluation."""
from mechanism_core import *
from joblib import Parallel,delayed,parallel_config
from empirical_shift_analysis import environment_indices
def task(name,seed,depth):
 folder=OUT/'breadth_units';folder.mkdir(exist_ok=True)
 dest=folder/f'{name}-{seed}-{depth}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,depth,100)
 pos=tr[f.predict(tr)==1];rows=[];ranks=[]
 env=environment_indices(tr,te[ids]) if depth==5 and len(ids)>0 else {'iid':np.arange(len(ids))}
 for protocol in b.PROTOCOLS:
  for nq,qi in enumerate(ids):
   x=te[qi];S,action=support(ds,x,pos,protocol);metric=protocol.metric
   e,z,valid=exact(x,S,lb,ub,scale,metric)
   if len(lb)==0:continue
   order=np.argsort(costs(x,projections(x,lb,ub),scale,metric),kind='stable')
   raw=projections(x,lb,ub);rr=z[order[0]];direct=valid[order[0]]
   arank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')[:20]
   targets=np.broadcast_to(x,(len(pos),len(x))).copy();targets[:,S]=pos[:,S]
   yes=f.predict(targets)==1
   rawrank=np.argmin(np.linalg.norm(pos.astype(float)-x,axis=1))
   one,onevalid=bisect(f,x,rr,targets[rawrank:rawrank+1],S)
   many,mvalid=bisect(f,x,rr,targets[arank],S)
   one=rr if direct else pick(x,one,onevalid,scale,metric)
   many=rr if direct else pick(x,many,mvalid,scale,metric)
   ix=np.flatnonzero(yes)
   filtered=None
   if direct:filtered=rr
   elif len(ix):
    best=ix[np.argmin(np.linalg.norm((pos[ix].astype(float)-x)/scale,axis=1))]
    zz,vv=bisect(f,x,rr,targets[best:best+1],S);filtered=zz[0]
   p,visited=pruned(x,S,scale,metric,f)
   assert (e is None)==(filtered is None)==(p is None)
   if e is not None:
    assert np.array_equal(e,p),(name,seed,depth,metric,float(costs(x,e,scale,metric)),float(costs(x,p,scale,metric)))
   selected=order[:10];k10=pick(x,z[selected],valid[selected],scale,metric)
   native=b.project_exact_cached(x,f,lb,ub,action,scale,metric,max_changes=protocol.max_changes)
   uraw=raw[np.argmin(np.linalg.norm(raw.astype(float)-x,axis=1))]
   up=c.apply_support(x,uraw,S)
   up=up if f.predict(up[None])[0]==1 else None
   mp=rr if direct else None
   base=dict(dataset=name,seed=seed,depth=depth,protocol=protocol.name,query=int(qi),positive_leaves=len(lb),total_leaves=f.get_n_leaves(),
    tree_depth=f.get_depth(),pruned_nodes=visited,decision=int(name in DECISION),
    **{key:int(nq in set(values)) for key,values in env.items()})
   for m,candidate in [('raw_projection',up),('metric_projection',mp),('lf_prr1',one),('lf_prr20',many),('leaves10',k10),('filter_first',filtered),('exact',e),('native',native)]:
    rows.append(rec(base,m,x,candidate,f,scale,metric))
   goodrank=np.flatnonzero(valid[order])
   if len(goodrank):
    costs_sofar=np.minimum.accumulate(np.where(valid[order],costs(x,z[order],scale,metric),np.inf))
    exact_cost=float(costs(x,e,scale,metric))
    k95=int(goodrank[0]+1);k99=k95 # deterministic per-query coverage indicator: first feasible leaf.
    cost95=int(np.flatnonzero(costs_sofar<=exact_cost/0.95+1e-12)[0]+1)
    cost99=int(np.flatnonzero(costs_sofar<=exact_cost/0.99+1e-12)[0]+1)
   else:k95=k99=cost95=cost99=np.nan
   ranks.append(dict(**base,first_rank=k95,cost95_rank=cost95,cost99_rank=cost99,exact_success=int(e is not None)))
   for frac in [.01,.05,.1,.25,.5,1.]:
    K=max(1,int(np.ceil(frac*len(lb))));ss=order[:K]
    candidate=pick(x,z[ss],valid[ss],scale,metric)
    rows.append(rec(base,'fraction_'+str(frac),x,candidate,f,scale,metric,K=K,budget_fraction=K/len(lb)))
 pd.DataFrame(ranks).to_csv(folder/f'{name}-{seed}-{depth}-ranks.csv',index=False)
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,depth,len(lb),len(ids),flush=True)
if __name__=='__main__':
 st=time.time()
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=4)(delayed(task)(d,s,h) for h in [3,5,7,9,11,20] for d in NAMES for s in b.DEFAULT_SEEDS[:5])
 for suffix in ['', '-ranks']:
  files=[OUT/'breadth_units'/f'{d}-{s}-{h}{suffix}.csv' for h in [3,5,7,9,11,20] for d in NAMES for s in b.DEFAULT_SEEDS[:5]]
  pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(OUT/('breadth.csv' if not suffix else 'ranks.csv'),index=False)
 (OUT/'breadth_metadata.json').write_text(json.dumps(dict(datasets=NAMES,decision=DECISION,seeds=b.DEFAULT_SEEDS[:5],depths=[3,5,7,9,11,20],max_queries=100,seconds=time.time()-st,sklearn=__import__('sklearn').__version__,cpu=platform.processor()),indent=2))

