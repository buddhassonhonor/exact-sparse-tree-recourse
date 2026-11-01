"""Directed grid rounding, grid reoptimization, semantic direction sensitivity."""
from mechanism_core import *

POLICIES={'credit_default':{'limit_bal':1,'pay_amt1':1,'pay_amt2':1,'pay_amt3':1,'pay_amt4':1,'pay_amt5':1,'pay_amt6':1},
 'adult_income':{'education_num':1,'capital_gain':1,'hours_per_week':1},
 'german_credit':{'duration':-1,'credit_amount':-1,'installment_rate':-1},'bank_marketing':{'balance':1}}

def inward(l,u):
 ll=l.astype('float32');uu=u.astype('float32')
 ll=np.where(ll.astype(float)<l,np.nextafter(ll,np.float32(np.inf)),ll)
 uu=np.where(uu.astype(float)>u,np.nextafter(uu,np.float32(-np.inf)),uu)
 return ll,uu

def task(name,seed):
 folder=OUT/'practical_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}-{seed}.csv'
 if dest.exists():return
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,40);pos=tr[f.predict(tr)==1];rows=[]
 quantum=np.array([10.**(-next((p for p in range(9) if np.array_equal(np.round(tr[:,j].astype(float),p).astype('float32'),tr[:,j])),8)) for j in range(tr.shape[1])])
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,_=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S);e,_,_=exact(x,S,lb,ub,scale,protocol.metric)
   base=dict(dataset=name,seed=seed,protocol=protocol.name,query=int(qi))
   candidates={'exact':e}
   if e is not None:
    ch=c.changed_feature_indices(x,e);z=x.copy();z[ch]=(np.round(e[ch].astype(float)/quantum[ch])*quantum[ch]).astype('float32');candidates['nearest_round']=z
    z=x.copy();positive=e[ch]>x[ch];v=e[ch].astype(float)/quantum[ch];z[ch]=(np.where(positive,np.ceil(v),np.floor(v))*quantum[ch]).astype('float32');candidates['directed_round']=z
   else:candidates.update(nearest_round=None,directed_round=None)
   # Grid optimization permits a coordinate to stay at its observed off-grid value;
   # any actual change must be on the training-inferred measurement grid.
   lo=np.ceil(lb.astype(float)/quantum)*quantum;hi=np.floor(ub.astype(float)/quantum)*quantum
   zz=np.broadcast_to(x,lo.shape).copy()
   for j in S:
    cl=np.clip(np.round(float(x[j])/quantum[j])*quantum[j],lo[:,j],hi[:,j]);inside=(x[j]>=lb[:,j])&(x[j]<=ub[:,j]);zz[:,j]=np.where(inside,x[j],cl)
   good=np.all(zz>=lb,axis=1)&np.all(zz<=ub,axis=1)&(f.predict(zz)==1);candidates['grid_exact']=pick(x,zz,good,scale,protocol.metric)
   if name in POLICIES:
    policy=POLICIES[name];SS=np.array([j for j,n in enumerate(ds.feature_names) if n in policy],int)
    if protocol.max_changes:SS=c.choose_support(x,c.nearest_positive_train(x,pos),SS,protocol.max_changes)
    FF=np.setdiff1d(np.arange(len(x)),SS);sl=np.maximum(lb.astype(float),tr.min(axis=0));su=np.minimum(ub.astype(float),tr.max(axis=0))
    for j,n in enumerate(ds.feature_names):
     if n in policy:
      if policy[n]>0:sl[:,j]=np.maximum(sl[:,j],x[j]);su[:,j]=np.minimum(su[:,j],float(x[j])+scale[j])
      else:sl[:,j]=np.maximum(sl[:,j],float(x[j])-scale[j]);su[:,j]=np.minimum(su[:,j],x[j])
    ll,uu=inward(sl,su);z=projections(x,ll,uu);z[:,FF]=x[FF]
    good=np.all(ll<=uu,axis=1)&np.all(z>=ll,axis=1)&np.all(z<=uu,axis=1)&(f.predict(z)==1)
    candidates['semantic_direction']=pick(x,z,good,scale,protocol.metric)
   for m,z in candidates.items():rows.append(rec(base,m,x,z,f,scale,protocol.metric))
 pd.DataFrame(rows).to_csv(dest,index=False)

def credit_case():
 ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context('credit_default',7,5,200);pos=tr[f.predict(tr)==1];x=te[1726]
 S,_=support(ds,x,pos,b.PROTOCOLS[2]);F=np.setdiff1d(np.arange(len(x)),S);rows=[]
 for margin in [0.,.001,.01,.05]:
  l=lb.astype(float).copy();u=ub.astype(float).copy();l[np.isfinite(l)]+=np.broadcast_to(margin*scale,l.shape)[np.isfinite(l)];u[np.isfinite(u)]-=np.broadcast_to(margin*scale,u.shape)[np.isfinite(u)]
  ll,uu=inward(l,u);z=projections(x,ll,uu);z[:,F]=x[F];good=np.all(ll<=uu,axis=1)&np.all(z>=ll,axis=1)&np.all(z<=uu,axis=1)&(f.predict(z)==1)
  e=pick(x,z,good,scale,'weighted_l1');rows.append(dict(margin=margin,success=e is not None,cost=float(costs(x,e,scale,'weighted_l1')) if e is not None else None,
   changes={ds.feature_names[j]:[float(x[j]),float(e[j])] for j in c.changed_feature_indices(x,e)} if e is not None else {}))
 (OUT/'credit_margin_case.json').write_text(json.dumps(rows,indent=2))

if __name__=='__main__':
 for d in b.DATASET_NAMES:
  for s in b.DEFAULT_SEEDS[:5]:task(d,s)
 pd.concat([pd.read_csv(OUT/'practical_units'/f'{d}-{s}.csv') for d in b.DATASET_NAMES for s in b.DEFAULT_SEEDS[:5]],ignore_index=True).to_csv(OUT/'practical.csv',index=False)
 credit_case()
