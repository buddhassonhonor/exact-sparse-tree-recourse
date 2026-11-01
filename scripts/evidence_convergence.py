"""Anytime DiCE budget curves, shared queries/seeds, with feasibility-first control."""
from evidence_external import *

def priority_loss(self,cfs,desired_range,desired_class):
 scores=np.asarray(self.predict_fn_scores(cfs));yes=np.argmax(scores,axis=1)==int(desired_class)
 return (~yes).astype(float)*self._invalid_penalty

def task(name):
 folder=OUT/'convergence_units';folder.mkdir(exist_ok=True);dest=folder/f'{name}.csv'
 if dest.exists():return
 seed=7;ds,tr,te,yt,ye,scale,f,leaves,lb,ub,ids=context(name,seed,5,3);pos=tr[f.predict(tr)==1];names=[f'x{j}' for j in range(tr.shape[1])]
 df=pd.DataFrame(tr.astype(float),columns=names);df['outcome']=yt
 dd=dice_ml.Data(dataframe=df,continuous_features=names,outcome_name='outcome',continuous_features_precision={n:8 for n in names});model=dice_ml.Model(model=f,backend='sklearn');rows=[]
 for protocol in b.PROTOCOLS[1:]:
  for qi in ids:
   x=te[qi];S,_=support(ds,x,pos,protocol);F=np.setdiff1d(np.arange(len(x)),S);lo=tr.min(axis=0);hi=tr.max(axis=0);lo[F]=x[F];hi[F]=x[F]
   E=bounded_exact(x,S,lb,ub,lo,hi,scale,protocol.metric)
   max_delta=np.maximum(np.abs((lo.astype(float)-x)/scale),np.abs((hi.astype(float)-x)/scale))
   upper=float(max_delta.sum() if protocol.metric=='weighted_l1' else np.linalg.norm(max_delta))
   for mode in ['default','proximity_aligned','feasibility_first']:
    for budget,iters in [(2.,100),(10.,500),(30.,1500)]:
     base=dict(dataset=name,seed=seed,depth=5,protocol=protocol.name,query=int(qi),budget_seconds=budget,max_iterations=iters,mode=mode,exact_success=int(E is not None),exact_cost=float(costs(x,E,scale,protocol.metric)) if E is not None else np.nan)
     if E is None:
      rows.append(rec(base,'dice_anytime',x,None,f,scale,protocol.metric,seconds=0.,status='certified_infeasible_prescreen',returned_success=0));continue
     exp=dice_ml.Dice(dd,model,method='genetic');exp._raw_query=x.astype(float);exp._scale=scale;exp._metric=protocol.metric
     if mode!='default':exp.compute_proximity_loss=types.MethodType(aligned_proximity,exp)
     if mode=='feasibility_first':exp.compute_yloss=types.MethodType(priority_loss,exp);exp._invalid_penalty=upper+1.
     original=exp.predict_fn_scores;best=[None,np.inf];evaluations=[0]
     def monitor(input_instance,*args,**kwargs):
      out=original(input_instance,*args,**kwargs);z=np.asarray(input_instance,dtype='float32')
      if z.ndim==2 and z.shape[1]==len(x):
       evaluations[0]+=len(z);valid=(f.predict(z)==1)&np.all(z[:,F]==x[F],axis=1)&np.all(z[:,S]>=lo[S],axis=1)&np.all(z[:,S]<=hi[S],axis=1)
       candidate=pick(x,z,valid,scale,protocol.metric)
       if candidate is not None:
        cost=float(costs(x,candidate,scale,protocol.metric))
        if cost<best[1]:best[:]=[candidate,cost]
      return out
     exp.predict_fn_scores=monitor
     method='genetic_default' if mode=='default' else 'genetic_aligned'
     returned,status,elapsed,internal=dice_run(exp,method,x,S,names,lo,hi,scale,protocol.metric,seed+int(qi),budget,iters)
     candidates=[z for z in [returned,best[0]] if z is not None]
     z=pick(x,np.asarray(candidates),np.ones(len(candidates),bool),scale,protocol.metric) if candidates else None
     rows.append(rec(base,'dice_anytime',x,z,f,scale,protocol.metric,seconds=elapsed,status=status,returned_success=int(returned is not None),evaluations=evaluations[0]))
 pd.DataFrame(rows).to_csv(dest,index=False);print(name,flush=True)
if __name__=='__main__':
 with parallel_config(backend='loky',inner_max_num_threads=1):
  Parallel(n_jobs=4)(delayed(task)(d) for d in NAMES)
 pd.concat([pd.read_csv(OUT/'convergence_units'/f'{d}.csv') for d in NAMES],ignore_index=True).to_csv(OUT/'convergence.csv',index=False)
