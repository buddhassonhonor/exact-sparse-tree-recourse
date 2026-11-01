"""Random paired mechanism decomposition and anchor-pool assumption stress tests."""
from mechanism_core import *
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(parents=True,exist_ok=True)
from joblib import Parallel,delayed,parallel_config
from sklearn.cluster import MiniBatchKMeans
def task(name,seed,depth):
 folder=OUT/'mechanism_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}-{depth}.csv'
 if dest.exists():return
 ds,tr,te,ytr,yte,scale,f,leaves,lb,ub,ids=context(name,seed,depth,40)
 pred=f.predict(tr);pos=tr[pred==1];ypool=tr[(pred==1)&(ytr==1)]
 labels=f.apply(pos);leafrow={l:i for i,l in enumerate(leaves)}
 # Pools are independent of each individual query; subsampling never observes query outcomes.
 rng=np.random.default_rng(seed+377)
 pools={'full_model_positive':pos,'true_and_model_positive':ypool,'heldout_model_positive':te[f.predict(te)==1]}
 for frac in [.0001,.001,.01,.1]:
  n=max(1,int(np.ceil(len(pos)*frac)));pools['subsample_'+str(frac)]=pos[rng.choice(len(pos),n,replace=False)]
 from sklearn.decomposition import PCA
 pc=PCA(n_components=1,random_state=seed).fit(tr/scale).transform(pos/scale).ravel()
 pools['shift_pc1_low']=pos[pc<=np.quantile(pc,.1)]
 pools['shift_pc1_high']=pos[pc>=np.quantile(pc,.9)]
 k=min(32,len(pos));km=MiniBatchKMeans(n_clusters=k,random_state=seed,n_init=3,batch_size=256).fit(pos/scale)
 prototypes=(km.cluster_centers_*scale).astype('float32');pools['kmeans32']=prototypes[f.predict(prototypes)==1]
 for margin in [.01,.1]:
  li=np.array([leafrow[l] for l in labels]);good=np.all(pos>=lb[li]+margin*scale,axis=1)&np.all(pos<=ub[li]-margin*scale,axis=1)
  pools['margin_'+str(margin)]=pos[good]
 rows=[];decomp=[];poolrows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,act=support(ds,x,pos,protocol);e,z,valid=exact(x,S,lb,ub,scale,protocol.metric)
   raw=projections(x,lb,ub);order=np.argsort(costs(x,raw,scale,protocol.metric),kind='stable');r=z[order[0]];direct=valid[order[0]]
   targets=np.broadcast_to(x,pos.shape).copy();targets[:,S]=pos[:,S]
   tv=f.predict(targets)==1;arank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')
   repaired,rv=bisect(f,x,r,targets,S);allids=np.flatnonzero(rv)
   methods={}
   for mode,ii in [('lf_prr20',arank[:20]),('all_repair',allids),('filter_first',arank[tv[arank]][:1])]:
    candidates=ii[rv[ii]]
    chosen=int(candidates[np.argmin(costs(x,repaired[candidates],scale,protocol.metric))]) if len(candidates) else None
    B=r if direct else (repaired[chosen] if chosen is not None else None)
    if B is not None and not direct:
     _,opt,_=line_oracle(f,x,r,targets[chosen:chosen+1],S,lb,ub,scale,protocol.metric)
     # Keep the observed feasible bisection output in the interval reference.
     # Analytic real-valued optima can round onto a neighbouring binary32 cell.
     pool=[B]+([opt[0]] if opt[0] is not None else [])
     O=pick(x,np.asarray(pool),np.ones(len(pool),bool),scale,protocol.metric)
     P=polish(x,[O],f,leaves,lb,ub,S,scale,protocol.metric)
    elif B is not None:O=B;P=B
    else:O=P=None
    methods[mode]=B;methods[mode+'_oracle']=O;methods[mode+'_polish']=P
    methods[mode+'_output_polish']=polish(x,[B],f,leaves,lb,ub,S,scale,protocol.metric) if B is not None else None
    if B is not None:
     cb,co,cp,ce=[float(costs(x,q,scale,protocol.metric)) for q in [B,O,P,e]]
     assert cb+1e-6>=co>=cp-1e-6 and cp+1e-6>=ce,(name,seed,depth,protocol.name,int(qi),mode,cb,co,cp,ce)
     nb,no,np_,ne=[len(c.changed_feature_indices(x,q)) for q in [B,O,P,e]]
     decomp.append(dict(dataset=name,seed=seed,depth=depth,protocol=protocol.name,query=int(qi),baseline=mode,
      bracket=cb-co,within_leaf=co-cp,leaf_choice=cp-ce,total=cb-ce,
      changed_bracket=nb-no,changed_within=no-np_,changed_leaf=np_-ne,changed_total=nb-ne))
   # Endpoint polishing covers every compatible leaf; polishing only repair outputs need not.
   EP=polish(x,targets[tv],f,leaves,lb,ub,S,scale,protocol.metric)
   RP=r if direct else polish(x,repaired[rv],f,leaves,lb,ub,S,scale,protocol.metric)
   assert (EP is None)==(e is None)
   if e is not None:assert np.array_equal(EP,e)
   methods['all_endpoint_polish']=EP;methods['all_output_polish']=RP;methods['exact']=e
   base=dict(dataset=name,seed=seed,depth=depth,protocol=protocol.name,query=int(qi),positive_leaves=len(leaves),anchors=len(pos),decision=int(name in DECISION))
   for method,cand in methods.items():rows.append(rec(base,method,x,cand,f,scale,protocol.metric))
   for poolname,pool in pools.items():
    rr=np.broadcast_to(x,pool.shape).copy();rr[:,S]=pool[:,S]
    good=f.predict(rr)==1 if len(rr) else np.zeros(0,bool)
    covered=set(f.apply(pool).tolist()) if len(pool) else set()
    covered_mask=np.array([leaf in covered for leaf in leaves])
    leaf_fraction=covered_mask.mean()
    missing_compatible=float((valid & ~covered_mask).sum()/max(1,valid.sum()))
    if direct:cand=r
    else:cand=pick(x,rr,good,scale,protocol.metric)
    poolrows.append(rec(dict(**base,pool=poolname,pool_count=len(pool),positive_leaf_coverage=leaf_fraction,uncovered_compatible_fraction=missing_compatible),'anchor_pool',x,cand,f,scale,protocol.metric))
 pd.DataFrame(poolrows).to_csv(folder/f'{name}-{seed}-{depth}-pools.csv',index=False)
 pd.DataFrame(decomp).to_csv(folder/f'{name}-{seed}-{depth}-decomposition.csv',index=False)
 pd.DataFrame(rows).to_csv(dest,index=False);print('mechanism',name,seed,depth,len(ids),flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=2)(delayed(task)(d,s,h) for h in [5,11] for d in NAMES for s in b.DEFAULT_SEEDS[:5])
 for suffix,name in [('', 'mechanism'),('-pools','pools'),('-decomposition','decomposition')]:
  files=[OUT/'mechanism_units'/f'{d}-{s}-{h}{suffix}.csv' for h in [5,11] for d in NAMES for s in b.DEFAULT_SEEDS[:5]]
  pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(OUT/f'{name}.csv',index=False)

