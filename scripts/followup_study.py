"""Controlled follow-up experiments: leaf budget, anchor mechanisms, supports and doses."""
from pathlib import Path
import os,sys,time,json,argparse,hashlib,platform
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import numpy as np
import pandas as pd
import scipy,sklearn,psutil
from joblib import Parallel,delayed,parallel_config
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/('script/mve' if (ROOT/'script/mve').exists() else 'scripts/mve')))
import run_review_benchmark as b
import recourse_core as c
OUT=ROOT/'exp/followup_20261101'
def context(dataset,seed,depth=5,limit=200,minleaf=5,subsample=True):
 ds=b.load_dataset(dataset,seed)
 tr,te=train_test_split(np.arange(len(ds.y)),test_size=.35,stratify=ds.y,random_state=seed)
 train,test=ds.X[tr].astype('float32'),ds.X[te].astype('float32')
 scale=np.std(train.astype(float),axis=0); scale=np.where(scale>b.EPS,scale,1.)
 f=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=minleaf,random_state=seed).fit(train,ds.y[tr])
 boxes=c.get_leaf_boxes(f); leaves=[n for n in boxes if np.argmax(f.tree_.value[n][0])==1]
 lb,ub=b.implementable_bound_matrices(boxes,leaves,train.shape[1])
 ids=np.flatnonzero(f.predict(test)==0)
 if len(ids)>limit: ids=np.sort(np.random.default_rng(seed+1009).choice(ids,limit,replace=False))
 return ds,train,test,scale,f,lb,ub,train[f.predict(train)==1],ids
def costs(x,z,scale,metric):
 d=(z.astype(float)-x.astype(float))/scale
 return np.abs(d).sum(axis=-1) if metric=='weighted_l1' else np.linalg.norm(d,axis=-1)
def projections(x,lb,ub):
 return np.minimum(np.maximum(x,lb),ub)
def valid_candidates(f,x,z,S):
 a=np.broadcast_to(x,z.shape).copy(); a[:,S]=z[:,S]
 return a,f.predict(a)==1
def pick(x,z,valid,scale,metric):
 if not np.any(valid): return None
 ix=np.flatnonzero(valid)
 return z[ix[np.argmin(costs(x,z[ix],scale,metric))]].copy()
def bisect(f,x,r,targets,S,steps=8):
 if len(targets)==0: return targets.copy(),np.zeros(0,dtype=bool)
 valid=f.predict(targets)==1
 lo=np.zeros(len(targets)); hi=np.ones(len(targets)); best=targets.copy()
 for _ in range(steps):
  mid=(lo+hi)/2
  z=np.broadcast_to(r,targets.shape).copy()
  z[:,S]=((1-mid[:,None])*r[S]+mid[:,None]*targets[:,S]).astype('float32')
  yes=f.predict(z)==1; best[yes]=z[yes]; hi[yes]=mid[yes]; lo[~yes]=mid[~yes]
 assert np.array_equal(best[:,np.setdiff1d(np.arange(len(x)),S)],np.broadcast_to(x,targets.shape)[:,np.setdiff1d(np.arange(len(x)),S)])
 return best,valid
def line_oracle(f,x,r,targets,S,lb,ub,scale,metric):
 """Intersect segments with preimages of binary32 positive-leaf boxes.
 Half-ulp cells account for inference rounding; validate every selected endpoint.
 Returns first crossing and minimum cost over all positive intervals, transitions.
 """
 A=len(targets); L=len(lb)
 low=np.zeros((A,L)); high=np.ones((A,L))
 for j in range(len(x)):
  lower=lb[:,j].astype(float); upper=ub[:,j].astype(float)
  lower=(lower+np.nextafter(lb[:,j],np.float32(-np.inf)).astype(float))/2
  upper=(upper+np.nextafter(ub[:,j],np.float32(np.inf)).astype(float))/2
  delta=targets[:,j].astype(float)-float(r[j])
  moving=delta!=0; fixed=~moving
  low[fixed,:]=np.where((float(r[j])>=lower)&(float(r[j])<=upper),low[fixed,:],np.inf)
  if np.any(moving):
   v=(lower[None,:]-float(r[j]))/delta[moving,None]
   w=(upper[None,:]-float(r[j]))/delta[moving,None]
   low[moving,:]=np.maximum(low[moving,:],np.minimum(v,w))
   high[moving,:]=np.minimum(high[moving,:],np.maximum(v,w))
 good=low<=high
 firsts=[]; optima=[]; transitions=[]
 for ai in range(A):
  ids=np.flatnonzero(good[ai])
  intervals=sorted((max(0.,low[ai,i]),min(1.,high[ai,i])) for i in ids)
  merged=[]
  for lo,hi in intervals:
   if merged and lo<=np.nextafter(merged[-1][1],np.inf): merged[-1][1]=max(hi,merged[-1][1])
   else: merged.append([lo,hi])
  transitions.append(sum(int(lo>0)+int(hi<1) for lo,hi in merged))
  direction=targets[ai].astype(float)-r.astype(float)
  lambdas=[]
  for lo,hi in intervals:
   left=np.nextafter(lo,hi); right=np.nextafter(hi,lo)
   candidates=[lo,left,min(hi,lo+8*abs(np.spacing(lo))),min(hi,lo+64*abs(np.spacing(lo))),right,hi,(lo+hi)/2]
   d=direction/scale; base=(r.astype(float)-x.astype(float))/scale
   if metric=='scaled_l2':
    if np.dot(d,d)>0: candidates.append(np.clip(-np.dot(base,d)/np.dot(d,d),left,right))
   else:
    candidates.extend(np.clip(-base[d!=0]/d[d!=0],left,right).tolist())
   lambdas.extend(candidates)
  if not lambdas:
   firsts.append(None);optima.append(None);continue
  ll=np.asarray(sorted(set(lambdas)))
  z=np.broadcast_to(r,(len(ll),len(r))).copy()
  z[:,S]=(r[S].astype(float)+ll[:,None]*direction[S]).astype('float32')
  yes=f.predict(z)==1
  firsts.append(z[np.flatnonzero(yes)[0]].copy() if yes.any() else None)
  optima.append(pick(x,z,yes,scale,metric))
 return firsts,optima,np.asarray(transitions)
def record(base,name,x,z,f,S,scale,metric,exact=None,**kw):
 success=z is not None and f.predict(z[None])[0]==1
 if success:
  frozen=np.setdiff1d(np.arange(len(x)),S); assert np.array_equal(z[frozen],x[frozen])
  cost=float(costs(x,z,scale,metric))
  if exact is not None: assert costs(x,exact,scale,metric)<=cost+1e-6
 return dict(**base,method=name,success=int(success),cost=cost if success else np.nan,
  changed=int(np.count_nonzero(np.abs(z.astype(float)-x.astype(float))>c.ACTION_EPS)) if success else np.nan,**kw)
def study_task(dataset,seed,depth,limit):
 folder=OUT/'units'; folder.mkdir(parents=True,exist_ok=True)
 path=folder/f'{dataset}-{seed}-{depth}.csv'
 if path.exists(): return
 ds,train,test,scale,f,lb,ub,pos,ids=context(dataset,seed,depth,limit)
 rows=[]; funnel=[]; anchor=[]; supports=[]; practical=[]
 for protocol in b.PROTOCOLS:
  if protocol.name=='free_l2': policies=['all']
  else: policies=['nearest']
  action=c.actionable_indices(train.shape[1],b.immutable_indices(ds,protocol))
  for nq,qi in enumerate(ids):
   x=test[qi]; nearest=c.nearest_positive_train(x,pos)
   S=c.choose_support(x,nearest,action,protocol.max_changes)
   raw=projections(x,lb,ub); order=np.argsort(costs(x,raw,scale,protocol.metric),kind='stable')
   restored,valid=valid_candidates(f,x,raw,S)
   exact=pick(x,restored,valid,scale,protocol.metric)
   r=restored[order[0]]
   frozen=np.setdiff1d(np.arange(len(x)),S)
   compatible=np.all((x[frozen]>=lb[:,frozen])&(x[frozen]<=ub[:,frozen]),axis=1)
   first_rank=int(np.flatnonzero(compatible[order])[0]+1) if compatible.any() else np.nan
   targets=np.broadcast_to(x,pos.shape).copy(); targets[:,S]=pos[:,S]
   arank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1),kind='stable')
   rawrank=np.argsort(np.linalg.norm(pos.astype(float)-x,axis=1),kind='stable')
   srank=np.argsort(np.linalg.norm(((pos.astype(float)-x)/scale)[:,S],axis=1),kind='stable')
   br,bv=bisect(f,x,r,targets[arank[:20]],S)
   one,ov=bisect(f,x,r,targets[rawrank[:1]],S)
   prr1=r if valid[order[0]] else pick(x,one,ov,scale,protocol.metric)
   prr20=r if valid[order[0]] else pick(x,br,bv,scale,protocol.metric)
   base=dict(dataset=dataset,seed=seed,depth=depth,protocol=protocol.name,query=int(qi),
     positive_leaves=len(lb),compatible_fraction=compatible.mean(),first_compatible_rank=first_rank)
   for name,z in [('exact',exact),('lf_prr1',prr1),('lf_prr20',prr20)]:
    rows.append(record(base,name,x,z,f,S,scale,protocol.metric,exact))
   for K in [1,2,3,5,10,-1]:
    selected=order if K<0 else order[:K]
    z=pick(x,restored,valid & np.isin(np.arange(len(lb)),selected),scale,protocol.metric)
    rows.append(record(base,f'leaves_{K}',x,z,f,S,scale,protocol.metric,exact))
   category='first_compatible' if compatible[order[0]] else ('restored_other_positive' if valid[order[0]] else ('anchor_repaired' if prr1 is not None else ('missed_feasible' if exact is not None else 'infeasible')))
   funnel.append(dict(**base,stage=category))
   if protocol.name!='free_l2' and nq<25:
    targetvalid=f.predict(targets)==1
    filtered=arank[targetvalid[arank]]
    allbr,allbv=bisect(f,x,r,targets,S)
    for name,idx in [('filter_1',filtered[:1]),('filter_20',filtered[:20]),('anchors_all',np.arange(len(pos)))]:
     z=r if valid[order[0]] else pick(x,allbr[idx],allbv[idx],scale,protocol.metric)
     anchor.append(record(base,name,x,z,f,S,scale,protocol.metric,exact))
    # Oracle over ALL training anchors, not only endpoint-valid ones.
    firsts,opts,trans=line_oracle(f,x,r,targets,S,lb,ub,scale,protocol.metric)
    zlist=[z for z in opts if z is not None]
    oracle=r if valid[order[0]] else (pick(x,np.asarray(zlist),np.ones(len(zlist),bool),scale,protocol.metric) if zlist else None)
    anchor.append(record(base,'oracle_all',x,oracle,f,S,scale,protocol.metric,exact))
    for ix in arank[:20]:
     if opts[ix] is not None and allbv[ix] and not valid[order[0]]:
      anchor.append(dict(**base,method='segment',success=1,cost=float(costs(x,opts[ix],scale,protocol.metric)),
       changed=np.nan,transitions=int(trans[ix]),bisection_excess=float(costs(x,allbr[ix],scale,protocol.metric)-costs(x,opts[ix],scale,protocol.metric)),
       first_crossing_cost=float(costs(x,firsts[ix],scale,protocol.metric)) if firsts[ix] is not None else np.nan))
    # Mechanistic decomposition; coverage deficits are not called independent evidence.
    missing='exact_infeasible' if exact is None else ('recovered' if prr20 is not None else (
      'no_valid_restored_training_anchor' if not targetvalid.any() else
      'valid_anchor_not_in_top20' if not bv.any() else 'other'))
    ranks={}
    for name,rr in [('scaled',arank),('support_scaled',srank)]:
     vv=targetvalid[rr]; ranks[name]=int(np.flatnonzero(vv)[0]+1) if vv.any() else np.nan
    anchor.append(dict(**base,method='mechanism',success=0,cost=np.nan,changed=np.nan,reason=missing,
      positive_anchors=len(pos),valid_anchor_fraction=targetvalid.mean(),scaled_first_valid_rank=ranks['scaled'],
      support_first_valid_rank=ranks['support_scaled'],multi_transition_fraction=float(np.mean(trans>1)),
      oracle_success=int(oracle is not None)))
   if protocol.name=='immutable_top3_l1' and nq<50:
    native=b.project_exact_cached(x,f,lb,ub,action,scale,protocol.metric,max_changes=3)
    rng=np.random.default_rng(seed*1009+int(qi))
    rules={'nearest':S,'random':np.sort(rng.choice(action,min(3,len(action)),replace=False)),
     'importance':action[np.argsort(-f.feature_importances_[action],kind='stable')[:3]],
     'native':c.changed_feature_indices(x,native) if native is not None else action[:0]}
    for rule,SS in rules.items():
     zz,vv=valid_candidates(f,x,raw,SS); ex=pick(x,zz,vv,scale,protocol.metric); rr=zz[order[0]]
     tt=np.broadcast_to(x,pos.shape).copy(); tt[:,SS]=pos[:,SS]
     bb,v=bisect(f,x,rr,tt[arank[:20]],SS)
     repaired=rr if vv[order[0]] else pick(x,bb,v,scale,protocol.metric)
     for name,z in [('exact',ex),('lf_prr20',repaired)]:
      supports.append(record(dict(**base,rule=rule),name,x,z,f,SS,scale,protocol.metric,ex))
   if protocol.name!='free_l2' and nq<50:
    # Action bounds: nondecreasing, at most one training SD, and train range.
    tl=train.min(axis=0); tu=train.max(axis=0)
    for delta in [0.,1e-4,1e-3,.01,.05]:
     ml=lb.astype(float).copy(); mu=ub.astype(float).copy()
     ml[np.isfinite(ml)]+=np.broadcast_to(delta*scale,ml.shape)[np.isfinite(ml)]
     mu[np.isfinite(mu)]-=np.broadcast_to(delta*scale,mu.shape)[np.isfinite(mu)]
     # Round bounds inward.
     ll=ml.astype('float32'); uu=mu.astype('float32')
     ll=np.where(ll.astype(float)<ml,np.nextafter(ll,np.float32(np.inf)),ll)
     uu=np.where(uu.astype(float)>mu,np.nextafter(uu,np.float32(-np.inf)),uu)
     zz=projections(x,ll,uu); zz[:,frozen]=x[frozen]
     yes=(np.all(ll<=uu,axis=1)&np.all(zz>=ll,axis=1)&np.all(zz<=uu,axis=1)&(f.predict(zz)==1))
     z=pick(x,zz,yes,scale,protocol.metric)
     practical.append(record(dict(**base,margin=delta), 'margin_exact',x,z,f,S,scale,protocol.metric))
    boundedlb=np.maximum(lb,np.maximum(tl,x)); boundedub=np.minimum(ub,np.minimum(tu,x+scale))
    zz=projections(x,boundedlb,boundedub); zz[:,frozen]=x[frozen]
    yes=np.all(boundedlb<=boundedub,axis=1)&np.all(zz>=boundedlb,axis=1)&np.all(zz<=boundedub,axis=1)&(f.predict(zz)==1)
    z=pick(x,zz,yes,scale,protocol.metric)
    practical.append(record(dict(**base,margin=0.),'direction_bounded',x,z,f,S,scale,protocol.metric))
    if exact is not None:
     # Precision grid inferred only from training data, per feature.
     quantum=[]
     for j in range(len(x)):
      u=np.unique(train[:,j]); precision=next((p for p in range(9) if np.array_equal(np.round(u.astype(float),p).astype('float32'),u)),8)
      quantum.append(10.**(-precision))
     changed=c.changed_feature_indices(x,exact)
     rounded=x.copy(); rounded[changed]=(np.round(exact[changed]/np.asarray(quantum)[changed])*np.asarray(quantum)[changed]).astype('float32')
     practical.append(record(dict(**base,margin=0.),'rounded_exact',x,rounded,f,S,scale,protocol.metric))
 for name,data in [('funnel',funnel),('anchor',anchor),('support',supports),('practical',practical)]:
  pd.DataFrame(data).to_csv(folder/f'{dataset}-{seed}-{depth}-{name}.csv',index=False)
 pd.DataFrame(rows).to_csv(path,index=False)
 print(f'study {dataset} {seed} {depth} {len(ids)}',flush=True)
def dose_task(dataset,seed,depth):
 ds,tr,te,scale,f,lb,ub,pos,ids=context(dataset,seed,depth,25)
 rows=[]
 for qi in ids:
  x=te[qi]; nn=c.nearest_positive_train(x,pos); raw=projections(x,lb,ub)
  order=np.argsort(costs(x,raw,scale,'weighted_l1')); rng=np.random.default_rng(seed*1009+int(qi))
  perm=rng.permutation(len(x))
  for frac in [0.,.25,.5,.75]:
   action=np.sort(perm[int(np.floor(frac*len(x))):])
   for k in [1,2,3,4,5]:
    S=c.choose_support(x,nn,action,k)
    zz,vv=valid_candidates(f,x,raw,S); exact=pick(x,zz,vv,scale,'weighted_l1'); r=zz[order[0]]
    rr=np.broadcast_to(x,pos.shape).copy();rr[:,S]=pos[:,S]
    rank=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1))[:20]
    bb,valid=bisect(f,x,r,rr[rank],S)
    z=r if vv[order[0]] else pick(x,bb,valid,scale,'weighted_l1')
    frozen=np.setdiff1d(np.arange(len(x)),S)
    compatible=np.all((x[frozen]>=lb[:,frozen])&(x[frozen]<=ub[:,frozen]),axis=1)
    closest=costs(x,raw,scale,'weighted_l1')[order[0]]
    rows.append(dict(dataset=dataset,seed=seed,depth=depth,query=int(qi),immutable_fraction=frac,k=k,
     positive_leaves=len(lb),compatible_fraction=compatible.mean(),exact_success=int(exact is not None),
     repair_success=int(z is not None),cost_ratio=closest/float(costs(x,exact,scale,'weighted_l1')) if exact is not None else np.nan))
 return rows
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=['study','dose'],default='study')
 p.add_argument('--workers',type=int,default=4);p.add_argument('--max-queries',type=int,default=200)
 a=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True);start=time.time()
 with parallel_config(backend='loky',inner_max_num_threads=1):
  if a.mode=='study':
   Parallel(n_jobs=a.workers)(delayed(study_task)(d,s,h,a.max_queries) for h in [5,11] for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS)
   for name in ['', '-funnel','-anchor','-support','-practical']:
    files=[OUT/'units'/f'{d}-{s}-{h}{name}.csv' for h in [5,11] for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS]
    frames=[pd.read_csv(f) for f in files if f.stat().st_size>2]
    pd.concat(frames,ignore_index=True).to_csv(OUT/f'{name.lstrip("-") or "query_results"}.csv',index=False)
  else:
   parts=Parallel(n_jobs=a.workers)(delayed(dose_task)(d,s,h) for h in [3,5,11,20] for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:5])
   pd.DataFrame([r for part in parts for r in part]).to_csv(OUT/'dose.csv',index=False)
 meta=dict(args=vars(a),seconds=time.time()-start,python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
  sklearn=sklearn.__version__,cpu=platform.processor(),memory_gb=psutil.virtual_memory().total/1e9,
  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),data_root=str(b.DATA_ROOT))
 (OUT/f'metadata_{a.mode}.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
if __name__=='__main__':main()

