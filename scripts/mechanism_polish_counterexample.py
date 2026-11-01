"""Constructed leaf-complete pool whose bisection outputs miss the best leaf."""
from mechanism_core import *
from sklearn.tree import _tree

def main():
 # (feature,threshold,left,right), integer terminal values are class labels.
 good=(1,-.001,0,(1,100.,1,0))
 low=(0,1.,0,(0,1.5,(1,40.,0,(1,60.,1,0)),(0,2.,0,(0,3.,good,0))))
 upper=(0,.25,(1,.25,1,0),0);spec=(2,.25,low,upper)
 model=DecisionTreeClassifier().fit(np.array([[0.,0.,0.],[1.,1.,1.]]),[0,1]);dtype=model.tree_.__getstate__()['nodes'].dtype
 nodes=[];values=[]
 def build(part,depth=0):
  i=len(nodes);nodes.append({});values.append([1.,0.] if part==0 else [0.,1.])
  if isinstance(part,int):nodes[i]=dict(left_child=-1,right_child=-1,feature=-2,threshold=-2.,impurity=0.,n_node_samples=1,weighted_n_node_samples=1.,missing_go_to_left=0);return i,depth
  j,t,l,r=part;li,ld=build(l,depth+1);ri,rd=build(r,depth+1)
  nodes[i]=dict(left_child=li,right_child=ri,feature=j,threshold=t,impurity=0.,n_node_samples=2,weighted_n_node_samples=2.,missing_go_to_left=0);return i,max(ld,rd)
 _,h=build(spec);arr=np.zeros(len(nodes),dtype=dtype)
 for i,v in enumerate(nodes):
  for k,val in v.items():
   if k in dtype.names:arr[k][i]=val
 tree=_tree.Tree(3,np.array([2],np.intp),1);tree.__setstate__(dict(max_depth=h,node_count=len(arr),nodes=arr,values=np.array(values).reshape(-1,1,2)))
 model.tree_=tree;box=c.get_leaf_boxes(model);leaves=np.array([n for n in box if np.argmax(tree.value[n,0])==1]);lb,ub=b.implementable_bound_matrices(box,leaves,3)
 x=np.zeros(3,np.float32);S=np.array([0,1]);scale=np.ones(3);pool=np.array([[0,0,.5],[1.25,50,0],[2.5,100,0]],np.float32)
 assert set(model.apply(pool))==set(leaves)
 targets=pool.copy();targets[:,2]=0;r=x.copy();rep,good=bisect(model,x,r,targets,S)
 E,_,_=exact(x,S,lb,ub,scale,'scaled_l2');EP=polish(x,targets[model.predict(targets)==1],model,leaves,lb,ub,S,scale,'scaled_l2');RP=polish(x,rep[good],model,leaves,lb,ub,S,scale,'scaled_l2')
 assert np.array_equal(E,EP) and costs(x,RP,scale,'scaled_l2')>costs(x,E,scale,'scaled_l2')
 data=dict(query=x.tolist(),anchors=pool.tolist(),support=S.tolist(),spec=spec,exact=E.tolist(),endpoint_polish=EP.tolist(),output_polish=RP.tolist(),exact_cost=float(costs(x,E,scale,'scaled_l2')),output_polish_cost=float(costs(x,RP,scale,'scaled_l2')))
 (OUT/'polish_counterexample.json').write_text(json.dumps(data,indent=2));print(data['exact_cost'],data['output_polish_cost'])

if __name__=='__main__':main()
