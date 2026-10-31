"""Isolated repeated latency and large-tree scaling."""
from followup_study import *
def scaling():
 rows=pd.read_csv(OUT/'scaling.csv').to_dict('records') if (OUT/'scaling.csv').exists() else []
 for dataset,size in [('credit_default',0),('adult_income',0),('bank_marketing',0),('noise',6000),('noise',20000),('noise',60000),('noise',120000)]:
  if any(r['dataset']==dataset and (size==0 or r['n_records']==size) for r in rows):continue
  if dataset=='noise':
   rng=np.random.default_rng(711);X=rng.normal(size=(size,20)).astype('float32');y=rng.integers(0,2,size)
   immutable=np.arange(4)
  else:
   subs=b._subsample
   try:
    b._subsample=lambda X,y,seed,limit=6000:(X,y)
    ds=b.load_dataset(dataset,7);X=ds.X.astype('float32');y=ds.y
    immutable=b.immutable_indices(ds,b.PROTOCOLS[2])
   finally:b._subsample=subs
  tr,te=train_test_split(np.arange(len(y)),test_size=.35,stratify=y,random_state=7)
  train=X[tr];test=X[te];f=DecisionTreeClassifier(min_samples_leaf=1,random_state=7).fit(train,y[tr])
  boxes=c.get_leaf_boxes(f);leaves=[n for n in boxes if np.argmax(f.tree_.value[n][0])==1]
  lb,ub=b.implementable_bound_matrices(boxes,leaves,X.shape[1]);pos=train[f.predict(train)==1]
  scale=train.astype(float).std(axis=0);scale=np.where(scale>b.EPS,scale,1.)
  action=c.actionable_indices(X.shape[1],immutable);ids=np.flatnonzero(f.predict(test)==0)[:20]
  def run(x,method):
   nn=c.nearest_positive_train(x,pos);S=c.choose_support(x,nn,action,3)
   raw=projections(x,lb,ub); cc=costs(x,raw,scale,'weighted_l1')
   if method.startswith('leaves_'):
    K=int(method.split('_')[1]);ix=np.arange(len(lb)) if K<0 else np.argsort(cc)[:K]
    zz,vv=valid_candidates(f,x,raw[ix],S)
    return pick(x,zz,vv,scale,'weighted_l1')
   r=c.apply_support(x,raw[np.argmin(cc)],S)
   if f.predict(r[None])[0]==1:return r
   M=int(method.split('_')[1])
   ix=np.argsort(np.linalg.norm((pos.astype(float)-x)/scale,axis=1))[:M]
   tt=np.broadcast_to(x,(len(ix),len(x))).copy();tt[:,S]=pos[ix][:,S]
   bb,vv=bisect(f,x,r,tt,S)
   return pick(x,bb,vv,scale,'weighted_l1')
  methods=['leaves_1','leaves_5','leaves_20','leaves_-1','anchors_1','anchors_5','anchors_20']
  for qi in ids:
   x=test[qi]
   for m in methods:run(x,m)
   for rep in range(9):
    for m in np.random.default_rng(7+rep).permutation(methods):
     st=time.perf_counter_ns();z=run(x,m);ns=time.perf_counter_ns()-st
     rows.append(dict(dataset=dataset,n_records=len(X),positive_leaves=len(lb),total_leaves=f.get_n_leaves(),depth=f.get_depth(),query=int(qi),repeat=rep,method=m,
       ms=ns/1e6,success=int(z is not None),cost=float(costs(x,z,scale,'weighted_l1')) if z is not None else np.nan))
  print(dataset,len(X),len(lb),f.get_depth(),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'scaling.csv',index=False)
if __name__=='__main__':scaling()

