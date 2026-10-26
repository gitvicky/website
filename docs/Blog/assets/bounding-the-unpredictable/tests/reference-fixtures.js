import * as iv from '../math/interval.js';
import * as p from '../math/polynomial.js';
import * as nn from '../math/neural.js';
import * as tr from '../math/transformer.js';
const scalar=[];
for(const name of ['quadratic','exp','composed'])for(const domain of [[0,1],[-1.2,.8],[.2,.2]])for(const degree of [1,2,4,8]){
  const m=p.taylor(name,domain,degree);scalar.push({name,domain,degree,p:m.p,r:m.r,centre:m.centre,bound:m.bound});
}
const networks=[];
for(const depth of [1,2])for(const activation of ['relu','square'])for(const domain of [[-1,1],[.1,.1],[-.5,.7]])networks.push({depth,activation,domain,results:nn.compareNetwork(domain,depth,activation)});
const attention=[];
for(const sequence of [[0,1],[2,3],[0,0]])for(const token of [0,1])for(const radius of [0,.2,.7])for(const temperature of [.25,1,2])for(const method of ['Interval','Dependencies + simplex','Subdivide + dependencies']){
  const result=tr.experiment(sequence,token,radius,temperature,method);attention.push({sequence,token,radius,temperature,method,result});
}
console.log(JSON.stringify({scalar,networks,attention}));
