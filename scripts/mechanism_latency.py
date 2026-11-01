"""Isolated single-process latency, exact pruning, KD-tree anchors and partial ranks."""
from mechanism_core import *
from scipy.spatial import cKDTree

def task(name,size,seed,alpha):
 folder=OUT/'latency_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{size}-{seed}-{alpha}.csv'
 if dest.exists():return
 st=time.perf_counter()
 if name=='noise':
  rng=np.random.default_rng(seed);X=rng.normal(size=(size,20)).astype('float32');y=rng.integers(0,2,size);immutable=np.arange(4)
 else:
  sub=b._subsample
  try:
   b._subsample=lambda X,y,seed,limit=6000:(X,y)
   ds=b.load_dataset(name,seed);X=ds.X.astype('float32');y=ds.y;immutable=b.immutable_indices(ds,b.PROTOCOLS[2])
  finally:b._subsample=sub
 tr,te=train_test_split(np.arange(len(y)),test_size=.35,stratify=y,random_state=seed)
 train=X[tr];test=X[te];f=DecisionTreeClassifier(min_samples_leaf=1,ccp_alpha=alpha,random_state=seed).fit(train,y[tr])
 if name=='breast_cancer':f=DecisionTreeClassifier(max_depth=5,min_samples_leaf=5,random_state=seed).fit(train,y[tr])
 fitsec=time.perf_counter()-st;st=time.perf_counter()
 boxes=c.get_leaf_boxes(f);leaves=np.asarray([n for n in boxes if np.argmax(f.tree_.value[n][0])==1]);lb,ub=b.implementable_bound_matrices(boxes,leaves,X.shape[1])
 pos=train[f.predict(train)==1];scale=train.astype(float).std(axis=0);scale=np.where(scale>b.EPS,scale,1.)
 cache_ms=1000*(time.perf_counter()-st);st=time.perf_counter();index=cKDTree(pos.astype(float)/scale);index_ms=1000*(time.perf_counter()-st)
 st=time.perf_counter();rawindex=cKDTree(pos.astype(float));rawindex_ms=1000*(time.perf_counter()-st)
 action=c.actionable_indices(X.shape[1],immutable);ids=np.flatnonzero(f.predict(test)==0);ids=np.random.default_rng(seed+1009).choice(ids,min(100,len(ids)),replace=False)
 rows=[];meta=[]
 for qi in ids:
  x=test[qi];st=time.perf_counter();_,near=rawindex.query(x);S=c.choose_support(x,pos[int(near)],action,3);support_ms=1000*(time.perf_counter()-st)
  e,zz,vv=exact(x,S,lb,ub,scale,'weighted_l1');raw=projections(x,lb,ub);r=c.apply_support(x,raw[np.argmin(costs(x,raw,scale,'weighted_l1'))],S)
  direct=f.predict(r[None])[0]==1;stratum='direct' if direct else ('repair_feasible' if e is not None else 'infeasible')
  def run(m):
   if m=='exact':return exact(x,S,lb,ub,scale,'weighted_l1')[0]
   if m=='pruned':return pruned(x,S,scale,'weighted_l1',f)[0]
   raw=projections(x,lb,ub);cc=costs(x,raw,scale,'weighted_l1')
   if m.startswith('leaves10'):
    n=min(10,len(lb));ii=np.argsort(cc,kind='stable')[:n] if m.endswith('sort') else np.argpartition(cc,n-1)[:n]
    z=raw[ii].copy();F=np.setdiff1d(np.arange(len(x)),S);z[:,F]=x[F]
    return pick(x,z,f.predict(z)==1,scale,'weighted_l1')
   r=c.apply_support(x,raw[np.argmin(cc)],S)
   if f.predict(r[None])[0]==1:return r
   if m=='lf20_index':ii=np.atleast_1d(index.query(x.astype(float)/scale,k=min(20,len(pos)))[1]).astype(int)
   elif m=='lf20_sort':ii=np.argsort(np.linalg.norm(pos.astype(float)/scale-x/scale,axis=1),kind='stable')[:20]
   else:ii=np.arange(len(pos))
   targets=np.broadcast_to(x,(len(ii),len(x))).copy();targets[:,S]=pos[ii][:,S]
   if m=='filter_first':
    yes=f.predict(targets)==1;ii=np.flatnonzero(yes)
    if not len(ii):return None
    a=ii[np.argmin(np.linalg.norm((pos[ii].astype(float)-x)/scale,axis=1))];targets=targets[a:a+1]
   if m=='global_interval':
    _,opts,_=line_oracle(f,x,r,targets,S,lb,ub,scale,'weighted_l1');pool=[z for z in opts if z is not None]
    return pick(x,np.asarray(pool),np.ones(len(pool),bool),scale,'weighted_l1') if pool else None
   z,valid=bisect(f,x,r,targets,S);return pick(x,z,valid,scale,'weighted_l1')
  methods=['exact','pruned','leaves10_sort','leaves10_partition','lf20_sort','lf20_index','filter_first','all_repair']
  # Global interval enumeration scales as A*N+; retain an explicit workspace gate.
  if len(pos)*len(lb)<=10000:methods.append('global_interval')
  for m in methods:run(m)
  for rep in range(3):
   for m in np.random.default_rng(seed+rep).permutation(methods):
    st=time.perf_counter_ns();z=run(m);ms=(time.perf_counter_ns()-st)/1e6
    if m in ['exact','pruned']:
     assert (z is None)==(e is None)
     if e is not None:assert np.array_equal(z,e)
    rows.append(dict(dataset=name,n_records=len(X),seed=seed,alpha=alpha,query=int(qi),repeat=rep,method=m,ms=ms,success=int(z is not None),
     stratum=stratum,positive_leaves=len(lb),total_leaves=f.get_n_leaves(),depth=f.get_depth(),anchors=len(pos),
     cost=float(costs(x,z,scale,'weighted_l1')) if z is not None else np.nan))
  meta.append(dict(query=int(qi),support_ms=support_ms,visited=pruned(x,S,scale,'weighted_l1',f)[1],stratum=stratum))
 pd.DataFrame(rows).to_csv(dest,index=False)
 (folder/f'{dest.stem}.json').write_text(json.dumps(dict(fit_seconds=fitsec,cache_ms=cache_ms,index_ms=index_ms,raw_index_ms=rawindex_ms,
  cache_bytes=lb.nbytes+ub.nbytes,index_data_bytes=index.data.nbytes+rawindex.data.nbytes,global_gate=10000,queries=meta),indent=2))
 print(name,size,seed,alpha,len(lb),len(ids),flush=True)

if __name__=='__main__':
 specs=[('breast_cancer',0,0.),('credit_default',0,0.),('adult_income',0,0.),('bank_marketing',0,0.)]+[('noise',n,0.) for n in [6000,20000,60000,120000]]+[(d,0,.0001) for d in ['credit_default','adult_income','bank_marketing']]
 for name,size,alpha in specs:
  for seed in b.DEFAULT_SEEDS[:5]:task(name,size,seed,alpha)
 pd.concat([pd.read_csv(OUT/'latency_units'/f'{d}-{n}-{s}-{a}.csv') for d,n,a in specs for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'latency.csv',index=False)
