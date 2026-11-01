"""Normalized leaf-budget ranks for the same large-tree timing queries."""
from mechanism_latency import *

def main():
 rows=[]
 for name,size in [('credit_default',0),('adult_income',0),('bank_marketing',0),('noise',120000)]:
  for seed in b.DEFAULT_SEEDS[:5]:
   if name=='noise':
    rng=np.random.default_rng(seed);X=rng.normal(size=(size,20)).astype('float32');y=rng.integers(0,2,size);immutable=np.arange(4)
   else:
    sub=b._subsample
    try:
     b._subsample=lambda X,y,seed,limit=6000:(X,y)
     ds=b.load_dataset(name,seed);X=ds.X.astype('float32');y=ds.y;immutable=b.immutable_indices(ds,b.PROTOCOLS[2])
    finally:b._subsample=sub
   tr,te=train_test_split(np.arange(len(y)),test_size=.35,stratify=y,random_state=seed);train=X[tr];test=X[te]
   f=DecisionTreeClassifier(min_samples_leaf=1,random_state=seed).fit(train,y[tr]);boxes=c.get_leaf_boxes(f);leaves=[n for n in boxes if np.argmax(f.tree_.value[n,0])==1]
   lb,ub=b.implementable_bound_matrices(boxes,leaves,X.shape[1]);pos=train[f.predict(train)==1];scale=train.astype(float).std(axis=0);scale=np.where(scale>b.EPS,scale,1.)
   action=c.actionable_indices(X.shape[1],immutable);index=cKDTree(pos.astype(float))
   meta=json.loads((OUT/'latency_units'/f'{name}-{size}-{seed}-0.0.json').read_text())
   for v in meta['queries']:
    qi=v['query'];x=test[qi];_,near=index.query(x);S=c.choose_support(x,pos[int(near)],action,3)
    e,z,valid=exact(x,S,lb,ub,scale,'weighted_l1');raw=projections(x,lb,ub);order=np.argsort(costs(x,raw,scale,'weighted_l1'),kind='stable')
    feasible=np.flatnonzero(valid[order]);first=int(feasible[0]+1) if len(feasible) else np.nan
    costrank=np.nan
    if e is not None:
     best=np.minimum.accumulate(np.where(valid[order],costs(x,z[order],scale,'weighted_l1'),np.inf));costrank=int(np.flatnonzero(best<=float(costs(x,e,scale,'weighted_l1'))/.99+1e-12)[0]+1)
    rows.append(dict(dataset=name,seed=seed,query=qi,positive_leaves=len(leaves),first_rank=first,cost99_rank=costrank,exact_success=int(e is not None),fraction=first/len(leaves)))
   print(name,seed,len(leaves),flush=True)
 pd.DataFrame(rows).to_csv(OUT/'large_ranks.csv',index=False)

if __name__=='__main__':main()
