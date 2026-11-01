"""Isolated 24-dataset timing of LF-PRR20, output polishing and cached Exact."""
from mechanism_core import *
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)
def task(name,seed):
 folder=OUT/'polish_timing_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,100);pos=tr[f.predict(tr)==1];rows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,_=support(ds,x,pos,protocol);E,_,_=exact(x,S,lb,ub,scale,protocol.metric)
   def run(method):
    if method=='exact':return exact(x,S,lb,ub,scale,protocol.metric)[0]
    raw=projections(x,lb,ub);i=np.argmin(costs(x,raw,scale,protocol.metric));r=c.apply_support(x,raw[i],S)
    if f.predict(r[None])[0]==1:B=r
    else:
     rank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')[:20]
     targets=np.broadcast_to(x,(len(rank),len(x))).copy();targets[:,S]=pos[rank][:,S]
     bb,yes=bisect(f,x,r,targets,S);B=pick(x,bb,yes,scale,protocol.metric)
    if method.endswith('polish') and B is not None:return polish(x,[B],f,leaves,lb,ub,S,scale,protocol.metric)
    return B
   methods=['lf_prr20','lf_prr20_output_polish','exact']
   for m in methods:run(m)
   for repeat in range(3):
    for m in np.random.default_rng(seed+repeat).permutation(methods):
     start=time.perf_counter_ns();z=run(m);ms=(time.perf_counter_ns()-start)/1e6
     rows.append(dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi),repeat=repeat,method=m,ms=ms,success=int(z is not None)))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,flush=True)
if __name__=='__main__':
 for d in NAMES:
  for s in b.DEFAULT_SEEDS[:5]:task(d,s)
 pd.concat([pd.read_csv(OUT/'polish_timing_units'/f'{d}-{s}.csv') for d in NAMES for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'polish_timing.csv',index=False)
