"""Common 24-dataset main-table additions: post-hoc output polishing and raw-metric repair."""
from mechanism_core import *
from joblib import Parallel,delayed,parallel_config
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)
def task(name,seed):
 folder=OUT/'main_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,100);pos=tr[f.predict(tr)==1];rows=[]
 for protocol in b.PROTOCOLS:
  for qi in ids:
   x=te[qi];S,_=support(ds,x,pos,protocol);e,z,valid=exact(x,S,lb,ub,scale,protocol.metric);raw=projections(x,lb,ub)
   order=np.argsort(costs(x,raw,scale,protocol.metric),kind='stable');r=z[order[0]]
   iraw=np.argmin(np.linalg.norm(pos.astype(float)-x,axis=1));many=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')[:20]
   targets=np.broadcast_to(x,(len(many),len(x))).copy();targets[:,S]=pos[many][:,S]
   bb,yes=bisect(f,x,r,targets,S);B=r if valid[order[0]] else pick(x,bb,yes,scale,protocol.metric)
   P=polish(x,[B],f,leaves,lb,ub,S,scale,protocol.metric) if B is not None else None
   ru=z[np.argmin(np.linalg.norm(raw.astype(float)-x,axis=1))]
   target=x.copy();target[S]=pos[iraw,S];one,ov=bisect(f,x,ru,target[None],S)
   U=ru if f.predict(ru[None])[0]==1 else pick(x,one,ov,scale,protocol.metric)
   base=dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi))
   for m,candidate in [('lf_prr20_output_polish',P),('euclidean_repair',U)]:rows.append(rec(base,m,x,candidate,f,scale,protocol.metric))
   if P is not None:assert costs(x,P,scale,protocol.metric)<=costs(x,B,scale,protocol.metric)+1e-7
 pd.DataFrame(rows).to_csv(dest,index=False)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=2)(delayed(task)(d,s) for d in NAMES for s in b.DEFAULT_SEEDS[:5])
 pd.concat([pd.read_csv(OUT/'main_units'/f'{d}-{s}.csv') for d in NAMES for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'main_extra.csv',index=False)
