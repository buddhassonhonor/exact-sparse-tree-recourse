"""24-dataset controlled dose outcomes, with geometric and activation diagnostics."""
from mechanism_core import *
from joblib import Parallel,delayed,parallel_config
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)
def task(name,seed,depth):
 folder=OUT/'dose_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}-{depth}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,depth,25);pos=tr[f.predict(tr)==1];rows=[]
 for qi in ids:
  x=te[qi];nn=c.nearest_positive_train(x,pos);raw=projections(x,lb,ub);cc=costs(x,raw,scale,'weighted_l1');order=np.argsort(cc,kind='stable')
  rank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')[:20]
  perm=np.random.default_rng(seed*1009+int(qi)).permutation(len(x))
  for frac in [0.,.25,.5,.75]:
   action=np.sort(perm[int(np.floor(frac*len(x))):])
   for k in [1,2,3,4,5]:
    S=c.choose_support(x,nn,action,k);e,z,valid=exact(x,S,lb,ub,scale,'weighted_l1');r=z[order[0]]
    targets=np.broadcast_to(x,(len(rank),len(x))).copy();targets[:,S]=pos[rank][:,S]
    bb,yes=bisect(f,x,r,targets,S);cand=r if valid[order[0]] else pick(x,bb,yes,scale,'weighted_l1')
    rows.append(dict(dataset=name,seed=seed,depth=depth,query=int(qi),immutable_fraction=frac,k=k,d=len(x),positive_leaves=len(leaves),
     compatible_fraction=valid.mean(),exact_success=int(e is not None),direct=int(valid[order[0]]),repair_success=int(cand is not None),
     cost_ratio=float(cc[order[0]]/costs(x,e,scale,'weighted_l1')) if e is not None else np.nan))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,depth,len(ids),flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=2)(delayed(task)(d,s,h) for h in [3,5,11,20] for d in NAMES for s in b.DEFAULT_SEEDS[:5])
 pd.concat([pd.read_csv(OUT/'dose_units'/f'{d}-{s}-{h}.csv') for h in [3,5,11,20] for d in NAMES for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'dose.csv',index=False)
