"""Re-evaluate rounding on changed coordinates, leaving unchanged values untouched."""
from followup_study import *
def task(d,s,h):
 ds,train,test,scale,f,lb,ub,pos,ids=context(d,s,h,200)
 precision=[next((p for p in range(9) if np.array_equal(np.round(np.unique(train[:,j]).astype(float),p).astype('float32'),np.unique(train[:,j]))),8) for j in range(train.shape[1])]
 quantum=10.**(-np.array(precision,dtype=float)); rows=[]
 for protocol in b.PROTOCOLS[1:]:
  action=c.actionable_indices(train.shape[1],b.immutable_indices(ds,protocol))
  for qi in ids[:50]:
   x=test[qi];S=c.choose_support(x,c.nearest_positive_train(x,pos),action,protocol.max_changes)
   raw=projections(x,lb,ub);zz,valid=valid_candidates(f,x,raw,S);exact=pick(x,zz,valid,scale,protocol.metric)
   base=dict(dataset=d,seed=s,depth=h,protocol=protocol.name,query=int(qi),margin=0.)
   if exact is not None:
    ch=c.changed_feature_indices(x,exact);z=x.copy();z[ch]=(np.round(exact[ch]/quantum[ch])*quantum[ch]).astype('float32')
    rows.append(record(base,'rounded_exact',x,z,f,S,scale,protocol.metric))
   # Grid-constrained reoptimization; leave frozen coordinates at the query.
   ll=np.ceil(lb.astype(float)/quantum)*quantum;uu=np.floor(ub.astype(float)/quantum)*quantum
   zz=projections(x,ll,uu).astype('float32');frozen=np.setdiff1d(np.arange(len(x)),S);zz[:,frozen]=x[frozen]
   yes=np.all(ll<=uu,axis=1)&np.all(zz>=lb,axis=1)&np.all(zz<=ub,axis=1)&(f.predict(zz)==1)
   z=pick(x,zz,yes,scale,protocol.metric)
   rows.append(record(base,'precision_grid_exact',x,z,f,S,scale,protocol.metric))
 return rows
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  parts=Parallel(n_jobs=4)(delayed(task)(d,s,h) for h in [5,11] for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS)
 new=pd.DataFrame([r for part in parts for r in part]);df=pd.read_csv(OUT/'practical.csv')
 pd.concat([df[~df.method.isin(['rounded_exact','precision_grid_exact'])],new],ignore_index=True).to_csv(OUT/'practical.csv',index=False)

