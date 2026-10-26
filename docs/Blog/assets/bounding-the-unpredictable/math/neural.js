import * as iv from './interval.js';
import {AffineContext} from './affine.js';
import * as poly from './polynomial.js';
import {reluLines,squareLines} from './relaxations.js';
export const MODEL={
  first:{W:[[1],[-1],[.5]],b:[-.2,.1,.3]},
  second:{W:[[.7,-.4,.3],[-.2,.8,.5],[.4,.2,-.6]],b:[.1,-.1,.05]},
  output:{W:[[1,-.8,.5]],b:[.05]}
};
export function layers(depth=1){if(![1,2].includes(depth))throw new RangeError('Supported hidden depths: 1 or 2.');return depth===1?[MODEL.first]:[MODEL.first,MODEL.second];}
const act=(a,activation)=>activation==='relu'?iv.relu(a):iv.square(a);
export function forward(x,depth=1,activation='relu') {
  let v=[x];for(const {W,b}of layers(depth)){v=W.map((r,i)=>r.reduce((s,w,j)=>s+w*v[j],b[i])).map(z=>activation==='relu'?Math.max(0,z):z*z);}
  return MODEL.output.W[0].reduce((s,w,i)=>s+w*v[i],MODEL.output.b[0]);
}
export function ibp(domain,depth=1,activation='relu') {
  let v=[domain];const trace=[];
  for(const {W,b}of layers(depth)){const pre=iv.linear(W,v,b);v=pre.map(a=>act(a,activation));trace.push({pre,post:v});}
  return {bound:iv.linear(MODEL.output.W,v,MODEL.output.b)[0],trace};
}
export function affineNetwork(domain,depth=1,activation='relu') {
  const c=new AffineContext();let x=[c.input(domain)];const trace=[];
  for(const {W,b}of layers(depth)){x=c.linear(W,x,b).map(v=>activation==='relu'?c.relu(v):c.square(v));trace.push(x.map(v=>c.range(v)));}
  return {bound:c.range(c.linear(MODEL.output.W,x,MODEL.output.b)[0]),trace,generators:c.next};
}
export function crown(domain,depth=1,activation='relu',alpha=.5) {
  const hidden=layers(depth),trace=ibp(domain,depth,activation).trace;
  const backward=upper=>{
    let coeff=MODEL.output.W[0].slice(),bias=MODEL.output.b[0];
    for(let k=hidden.length-1;k>=0;k--){
      const {W,b}=hidden[k];
      coeff=coeff.map((a,j)=>{const lines=activation==='relu'?reluLines(trace[k].pre[j],alpha):squareLines(trace[k].pre[j],alpha);const [m,c]=lines[(a>=0)===upper?'upper':'lower'];bias+=a*c;return a*m;});
      bias+=coeff.reduce((s,a,j)=>s+a*b[j],0);
      coeff=W[0].map((_,i)=>coeff.reduce((s,a,j)=>s+a*W[j][i],0));
    }
    const range=iv.add(iv.scale(domain,coeff[0]),iv.I(bias));return range[upper?1:0];
  };
  return iv.I(backward(false),backward(true));
}
export function polynomialNetwork(domain,depth=1,activation='square') {
  if(activation==='relu')return exactRelu(domain,depth);
  let p=[[0,1]];
  for(const {W,b}of layers(depth))p=W.map((row,i)=>row.reduce((s,w,j)=>poly.plus(s,poly.scale(p[j],w)),[b[i]])).map(q=>poly.times(q,q));
  const out=MODEL.output.W[0].reduce((s,w,j)=>poly.plus(s,poly.scale(p[j],w)),[MODEL.output.b[0]]);
  return {bound:poly.bernstein(out,domain).bound,polynomial:out,pieces:1};
}
export function exactRelu(domain,depth=1) {
  let segments=[{domain,forms:[[0,1]]}];
  for(const {W,b}of layers(depth)) {
    const next=[];
    for(const cell of segments){
      const pre=W.map((row,i)=>row.reduce((s,w,j)=>poly.plus(s,poly.scale(cell.forms[j],w)),[b[i]]));
      const cuts=[...cell.domain];for(const [c,a=0]of pre){const z=-c/a;if(a!==0&&z>cell.domain[0]&&z<cell.domain[1])cuts.push(z);}
      cuts.sort((a,b)=>a-b);
      for(let i=0;i<cuts.length-1;i++){const d=iv.I(cuts[i],cuts[i+1]),c=iv.mid(d);next.push({domain:d,forms:pre.map(p=>poly.evaluate(p,c)>=0?p:[0,0])});}
    }
    segments=next;
  }
  const vals=segments.flatMap(cell=>cell.domain.map(x=>MODEL.output.W[0].reduce((s,w,j)=>s+w*poly.evaluate(cell.forms[j],x),MODEL.output.b[0])));
  return {bound:iv.I(Math.min(...vals),Math.max(...vals)),pieces:segments.length};
}
export function sensitivity(domain,depth=1,activation='relu',local=false) {
  const {trace}=ibp(domain,depth,activation);let d=[iv.I(1)];
  layers(depth).forEach(({W},k)=>{
    d=iv.linear(local?W:W.map(r=>r.map(Math.abs)),d).map((v,j)=>{
      const [l,u]=trace[k].pre[j];
      const derivative=activation==='relu'?(u<=0?iv.I(0):l>=0?iv.I(1):iv.I(0,1)):iv.scale([l,u],2);
      return iv.mul(v,local?derivative:iv.I(Math.max(Math.abs(derivative[0]),Math.abs(derivative[1]))));
    });
  });
  const derivative=iv.linear(local?MODEL.output.W:MODEL.output.W.map(r=>r.map(Math.abs)),d)[0];
  return iv.add(iv.I(forward(iv.mid(domain),depth,activation)),iv.mul(derivative,iv.I(-iv.radius(domain),iv.radius(domain))));
}
export function compareNetwork(domain,depth=1,activation='relu',alpha=.5) {
  if(!['relu','square'].includes(activation))throw new RangeError('Unsupported activation.');
  const methods=[['IBP',()=>ibp(domain,depth,activation).bound],['Affine / zonotope',()=>affineNetwork(domain,depth,activation).bound],
    [activation==='relu'?'Piecewise enumeration':'Polynomial / Bernstein',()=>polynomialNetwork(domain,depth,activation).bound],
    ['CROWN-style',()=>crown(domain,depth,activation,alpha)],['Lipschitz',()=>sensitivity(domain,depth,activation)],['Local Jacobian',()=>sensitivity(domain,depth,activation,true)]];
  return methods.map(([name,f])=>{const start=performance.now();return iv.result(name,f(),start);});
}
