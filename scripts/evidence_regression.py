"""Binomial random-intercept logistic model; dataset-cluster bootstrap intervals.

Laplace-integrated normal dataset intercepts; dose cells are collapsed to counts.
Bootstrap whole datasets to preserve all split/query/dose dependence within them.
"""
from mechanism_core import *
import scipy.special as sps
from scipy.optimize import minimize
from joblib import Parallel,delayed,parallel_config
OUT=ROOT/'exp/evidence_20261101';OUT.mkdir(exist_ok=True)
def fit_model(X,y,n,group,weights=None,start=None):
 D=int(group.max()+1);weights=np.ones(D) if weights is None else weights
 def objective(params):
  eta=X@params[:-1];variance=np.exp(2*params[-1]);u=np.zeros(D)
  for _ in range(35):
   p=sps.expit(eta+u[group]);grad=np.bincount(group,weights=y-n*p,minlength=D)-u/variance
   curv=np.bincount(group,weights=n*p*(1-p),minlength=D)+1/variance
   step=np.clip(grad/curv,-3.,3.);u+=step
   if np.max(np.abs(step))<1e-8:break
  e=eta+u[group];p=sps.expit(e);ll=np.bincount(group,weights=y*e-n*np.logaddexp(0,e),minlength=D)
  h=np.bincount(group,weights=n*p*(1-p),minlength=D)+1/variance
  integrated=ll-.5*u*u/variance-params[-1]-.5*np.log(h)
  return -float(weights@integrated)+1e-6*float(params[:-1]@params[:-1])
 initial=np.r_[np.zeros(X.shape[1]),0.] if start is None else start
 result=minimize(objective,initial,method='L-BFGS-B',bounds=[(-15,15)]*X.shape[1]+[(-5,3)],options={'maxiter':500,'ftol':1e-10,'gtol':1e-5})
 return result.x,result.success,float(result.fun)
def boot(i,X,y,n,group,start):
 D=int(group.max()+1);rng=np.random.default_rng(17300+i);weights=np.bincount(rng.integers(0,D,D),minlength=D)
 return fit_model(X,y,n,group,weights,start)
def run(data,label):
 keys=['dataset','seed','depth','immutable_fraction','k','positive_leaves','d']
 g=data.groupby(keys).agg(failures=('failure','sum'),trials=('failure','size')).reset_index();g['log_leaves']=np.log(g.positive_leaves);g['log_d']=np.log(g.d)
 features=['immutable_fraction','k','depth','log_leaves','log_d'];raw=g[features].to_numpy(float);mu=raw.mean(axis=0);sd=raw.std(axis=0)
 X=np.c_[np.ones(len(g)),(raw-mu)/sd];y=g.failures.to_numpy(float);n=g.trials.to_numpy(float);group=pd.Categorical(g.dataset).codes.astype(np.int64)
 point,success,fun=fit_model(X,y,n,group)
 with parallel_config(backend='loky',inner_max_num_threads=1):
  draws=Parallel(n_jobs=4)(delayed(boot)(i,X,y,n,group,point) for i in range(300))
 ok=[v[0] for v in draws if v[1]];assert len(ok)>=270,(label,len(ok));draw=np.asarray(ok)
 rows=[]
 for j,name in enumerate(['Intercept']+features):
  low,high=np.quantile(draw[:,j],[.025,.975]);rows.append(dict(model=label,term=name,beta=point[j],low=low,high=high,odds_ratio=np.exp(point[j]),OR_low=np.exp(low),OR_high=np.exp(high)))
 pd.DataFrame(rows).to_csv(OUT/f'logistic_{label}.csv',index=False)
 meta=dict(model='normal dataset random-intercept binomial logistic; Laplace likelihood',uncertainty='300 dataset-cluster bootstrap resamples',converged=bool(success),bootstrap_converged=len(ok),observations=len(data),dose_cells=len(g),D=int(group.max()+1),random_sd=float(np.exp(point[-1])),centering=dict(zip(features,mu)),scaling=dict(zip(features,sd)),objective=fun)
 (OUT/f'logistic_{label}.json').write_text(json.dumps(meta,indent=2));print(label,meta,flush=True)
def main():
 d=pd.read_csv(OUT/'dose.csv');d=d[d.exact_success==1].copy();d['failure']=1-d.repair_success
 run(d,'feasible');run(d[d.direct==0],'active')
if __name__=='__main__':main()
