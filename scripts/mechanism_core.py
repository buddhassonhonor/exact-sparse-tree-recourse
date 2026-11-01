"""Domain-aligned recourse mechanism experiments and constraint-pruned exact search."""
from pathlib import Path
import os,sys,json,platform,hashlib,time
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from numba import njit
from followup_study import costs,projections,pick,bisect,line_oracle,c,b
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'exp/mechanism_20261101';OUT.mkdir(parents=True,exist_ok=True)
EXTRA_IDS=[29,37,38,40,44,49,59,1462,1464,1480,1489,1494,40701,40983,44089,45023]
CAT=pd.read_csv(ROOT/'config/breadth_sources.csv').set_index('id')
NAMES=list(b.DATASET_NAMES)+[f'openml_{i}' for i in EXTRA_IDS]
DECISION=list(['credit_default','adult_income','german_credit','bank_marketing','compas'])+[f'openml_{i}' for i in [29,37,38,49,1464,1480,40701,44089,45023]]
def load(name,seed):
 if not name.startswith('openml_'):return b.load_dataset(name,seed)
 item=CAT.loc[int(name.split('_')[1])]
 df=pd.read_parquet(b.DATA_ROOT/'openml/org/openml/www/datasets'/str(item.name)/f'dataset_{item.name}.pq')
 names=[x for x in df.select_dtypes('number').columns if x!=item.target and x.lower() not in ['id','index','row_id']]
 keep=df[item.target].notna();df=df.loc[keep];labels=sorted(df[item.target].astype(str).unique())
 desired={29:'+ ',37:'tested_negative',38:'negative',49:'0',1464:'1',1480:'2',40701:'0',44089:'0',45023:'Good'}.get(int(item.name))
 desired=desired.strip() if desired else labels[-1]
 if desired not in labels:desired=labels[-1]
 X=df[names].to_numpy(float);y=(df[item.target].astype(str)==desired).to_numpy(int)
 if len(y)>6000:
  ix,_=train_test_split(np.arange(len(y)),train_size=6000,stratify=y,random_state=seed);ix=np.sort(ix);X,y=X[ix],y[ix]
 return b.Dataset(name,X,y,tuple(names),tuple(names[:max(1,len(names)//5)]),'numeric-subspace benchmark')
def context(name,seed,depth=5,limit=100,minleaf=5,alpha=0.,random_queries=True):
 ds=load(name,seed);tr,te=train_test_split(np.arange(len(ds.y)),test_size=.35,stratify=ds.y,random_state=seed)
 imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(ds.X[tr])
 train=imp.transform(ds.X[tr]).astype('float32');test=imp.transform(ds.X[te]).astype('float32')
 scale=train.astype(float).std(axis=0);scale=np.where(scale>b.EPS,scale,1.)
 f=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=minleaf,ccp_alpha=alpha,random_state=seed).fit(train,ds.y[tr])
 boxes=c.get_leaf_boxes(f);leaves=[n for n in boxes if np.argmax(f.tree_.value[n][0])==1]
 lb,ub=b.implementable_bound_matrices(boxes,leaves,train.shape[1])
 ids=np.flatnonzero(f.predict(test)==0);rng=np.random.default_rng(seed+1009)
 if len(ids)>limit:ids=rng.choice(ids,limit,replace=False)
 elif random_queries:ids=rng.permutation(ids)
 return ds,train,test,ds.y[tr],ds.y[te],scale,f,np.asarray(leaves),lb,ub,ids
def support(ds,x,pos,protocol):
 immutable=b.immutable_indices(ds,protocol);action=c.actionable_indices(len(x),immutable)
 nn=c.nearest_positive_train(x,pos);return c.choose_support(x,nn,action,protocol.max_changes),action
def exact(x,S,lb,ub,scale,metric):
 frozen=np.setdiff1d(np.arange(len(x)),S);valid=np.all((x[frozen]>=lb[:,frozen])&(x[frozen]<=ub[:,frozen]),axis=1)
 z=projections(x,lb,ub);z[:,frozen]=x[frozen]
 return pick(x,z,valid,scale,metric),z,valid
def polish(x,pool,f,leaves,lb,ub,S,scale,metric):
 if len(pool)==0:return None
 ids=set(f.apply(np.asarray(pool,dtype='float32')).tolist());rows=np.array([i for i,l in enumerate(leaves) if l in ids],dtype=int)
 if len(rows)==0:return None
 z=projections(x,lb[rows],ub[rows]);frozen=np.setdiff1d(np.arange(len(x)),S);z[:,frozen]=x[frozen]
 return pick(x,z,f.predict(z)==1,scale,metric)
@njit(cache=True)
def pruned_kernel(x,allowed,scale,metric_l1,left,right,feature,threshold,positive,capacity):
 d=len(x);n=capacity;stack_nodes=np.empty(n,np.int64)
 # DFS stores only deferred path boxes; number of pending nodes is <=depth+1.
 stack_lo=np.empty((n,d),np.float32);stack_hi=np.empty((n,d),np.float32)
 stack_nodes[0]=0;stack_lo[0,:]=-np.inf;stack_hi[0,:]=np.inf;count=1
 best=np.empty(d,np.float32);best_cost=np.inf;best_leaf=-1;visited=0
 while count:
  count-=1;node=stack_nodes[count];lo=stack_lo[count].copy();hi=stack_hi[count].copy();visited+=1
  z=x.copy();bad=False
  for j in range(d):
   if not allowed[j] and (x[j]<lo[j] or x[j]>hi[j]):bad=True;break
   if allowed[j]:z[j]=min(max(x[j],lo[j]),hi[j])
  if bad:continue
  cost=0.
  for j in range(d):
   v=(float(z[j])-float(x[j]))/scale[j]
   cost+=abs(v) if metric_l1 else v*v
  if cost>best_cost:continue
  if left[node]==right[node]:
   if positive[node] and (cost<best_cost or (cost==best_cost and node>best_leaf)):
    best_cost=cost;best=z.copy();best_leaf=node
   continue
  j=feature[node];t=threshold[node];cast=np.float32(t)
  pred=np.nextafter(cast,np.float32(-np.inf)) if float(cast)>t else cast
  succ=np.nextafter(cast,np.float32(np.inf)) if float(cast)<=t else cast
  # Visit the query's own side first to obtain a small incumbent.
  own_left=x[j]<=t
  for branch in range(2):
   go_left=(branch==1) if own_left else (branch==0)
   ll=lo.copy();hh=hi.copy()
   if go_left:hh[j]=min(hh[j],pred);child=left[node]
   else:ll[j]=max(ll[j],succ);child=right[node]
   if ll[j]<=hh[j]:
    stack_nodes[count]=child;stack_lo[count]=ll;stack_hi[count]=hh;count+=1
 return best,best_leaf,visited
def pruned(x,S,scale,metric,f):
 allowed=np.zeros(len(x),dtype=np.bool_);allowed[S]=True;t=f.tree_
 if not hasattr(f,'_recourse_positive'):f._recourse_positive=np.argmax(t.value[:,0],axis=1)==1
 z,leaf,visited=pruned_kernel(x,allowed,scale,metric=='weighted_l1',t.children_left,t.children_right,t.feature,t.threshold,f._recourse_positive,t.max_depth+2)
 return (z if leaf>=0 else None),visited
def rec(base,method,x,z,f,scale,metric,**kw):
 yes=z is not None and f.predict(z[None])[0]==1
 return dict(**base,method=method,success=int(yes),cost=float(costs(x,z,scale,metric)) if yes else np.nan,
 changed=len(c.changed_feature_indices(x,z)) if yes else np.nan,**kw)

