"""Recompute interval-oracle diagnostics with adjacent-parameter endpoint handling."""
from followup_study import *
def task(d,s,h):
 ds,train,test,scale,f,lb,ub,pos,ids=context(d,s,h,200);rows=[]
 for protocol in b.PROTOCOLS[1:]:
  action=c.actionable_indices(train.shape[1],b.immutable_indices(ds,protocol))
  for qi in ids[:25]:
   x=test[qi];S=c.choose_support(x,c.nearest_positive_train(x,pos),action,protocol.max_changes)
   raw=projections(x,lb,ub);order=np.argsort(costs(x,raw,scale,protocol.metric),kind='stable')
   zz,vv=valid_candidates(f,x,raw,S);ex=pick(x,zz,vv,scale,protocol.metric);r=zz[order[0]]
   targets=np.broadcast_to(x,pos.shape).copy();targets[:,S]=pos[:,S]
   rank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')
   first,opt,trans=line_oracle(f,x,r,targets,S,lb,ub,scale,protocol.metric)
   allbr,valid=bisect(f,x,r,targets,S)
   zlist=[z for z in opt if z is not None]
   oracle=r if vv[order[0]] else (pick(x,np.asarray(zlist),np.ones(len(zlist),bool),scale,protocol.metric) if zlist else None)
   base=dict(dataset=d,seed=s,depth=h,protocol=protocol.name,query=int(qi))
   rows.append(record(base,'oracle_all',x,oracle,f,S,scale,protocol.metric,ex))
   for ix in rank[:20]:
    if opt[ix] is not None and valid[ix] and not vv[order[0]]:
     rows.append(dict(**base,method='segment',success=1,cost=float(costs(x,opt[ix],scale,protocol.metric)),changed=np.nan,
      transitions=int(trans[ix]),bisection_excess=float(costs(x,allbr[ix],scale,protocol.metric)-costs(x,opt[ix],scale,protocol.metric)),
      first_crossing_cost=float(costs(x,first[ix],scale,protocol.metric)) if first[ix] is not None else np.nan))
 return rows
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  parts=Parallel(n_jobs=4)(delayed(task)(d,s,h) for h in [5,11] for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS)
 df=pd.read_csv(OUT/'anchor.csv');new=pd.DataFrame([r for part in parts for r in part])
 pd.concat([df[~df.method.isin(['oracle_all','segment'])],new],ignore_index=True).to_csv(OUT/'anchor.csv',index=False)
 print('Refreshed interval diagnostics',len(new),flush=True)

