from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,str(Path('script').resolve()))
from followup_study import *
class Tree:
 def predict(self,z):
  z=np.asarray(z,dtype='float32');u,v=z[:,0],z[:,1]
  return (((v>.25)&(u<=.25))|((v<=.25)&(u>1))).astype(int)
f=Tree();lb=np.array([[-np.inf,b.succ32(.25)],[b.succ32(1),-np.inf]],dtype='float32')
ub=np.array([[.25,np.inf],[np.inf,.25]],dtype='float32');S=np.array([0]);scale=np.ones(2)
pos=np.array([[0,.5],[2.**20,0]],dtype='float32')
failed=0;valid=0
for u in np.linspace(0,.125,33).astype('float32'):
 x=np.array([u,0],dtype='float32');raw=projections(x,lb,ub);r=c.apply_support(x,raw[np.argmin(costs(x,raw,scale,'scaled_l2'))],S)
 anchor=c.apply_support(x,c.nearest_positive_train(x,pos),S)
 out,yes=bisect(f,x,r,anchor[None],S);failed+=int(not yes.any())
 z,v=valid_candidates(f,x,raw,S);valid+=int(v.any())
x=np.zeros(2,dtype='float32');r=x.copy();target=np.array([[2.**20,0]],dtype='float32')
z,yes=bisect(f,x,r,target,S,8)
ratio=float(z[0,0]/float(b.succ32(1)))
assert failed==33 and valid==33 and np.isclose(ratio,2.**20/(256*float(b.succ32(1))))
first,opt,tr=line_oracle(f,x,r,target,S,lb,ub,scale,'scaled_l2')
assert opt[0] is not None and abs(opt[0][0]-b.succ32(1))<1e-6
report=dict(query_count=33,LF_PRR1_coverage=0,Exact_coverage=1,bisection_cost_ratio=ratio,interval_oracle_cost=float(opt[0][0]))
Path('exp/followup_20261101/worst_case_audit.json').write_text(json.dumps(report,indent=2))
print(report)

