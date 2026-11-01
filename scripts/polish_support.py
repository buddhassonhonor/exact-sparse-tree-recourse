"""24-dataset support-policy sensitivity on the unchanged primary queries."""
from mechanism_core import *
from joblib import Parallel,delayed,parallel_config
OUT=ROOT/'exp/polish_20261101';OUT.mkdir(exist_ok=True)
def task(name,seed):
 folder=OUT/'support_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,100);pos=tr[f.predict(tr)==1];rows=[];protocol=b.PROTOCOLS[2]
 for qi in ids:
  x=te[qi];S,action=support(ds,x,pos,protocol)
  choices=[('nearest_positive',0,S),('cart_importance',0,action[np.argsort(-f.feature_importances_[action],kind='stable')[:3]])]
  for draw in range(5):
   rng=np.random.default_rng(seed*1000003+int(qi)*101+draw)
   choices.append(('random',draw,np.sort(rng.choice(action,size=min(3,len(action)),replace=False))))
  for policy,draw,chosen in choices:
   z,_,_=exact(x,chosen,lb,ub,scale,'weighted_l1')
   if z is not None:
    F=np.setdiff1d(np.arange(len(x)),chosen);assert f.predict(z[None])[0]==1 and np.array_equal(z[F],x[F]) and len(c.changed_feature_indices(x,z))<=3
   rows.append(rec(dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi),draw=draw),policy,x,z,f,scale,'weighted_l1'))
  z=b.project_exact_cached(x,f,lb,ub,action,scale,'weighted_l1',max_changes=3)
  rows.append(rec(dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi),draw=0),'native',x,z,f,scale,'weighted_l1'))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,seed,flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):Parallel(n_jobs=3)(delayed(task)(d,s) for d in NAMES for s in b.DEFAULT_SEEDS[:5])
 pd.concat([pd.read_csv(OUT/'support_units'/f'{d}-{s}.csv') for d in NAMES for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'support.csv',index=False)
