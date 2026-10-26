import test from 'node:test';
import assert from 'node:assert/strict';
import * as iv from '../math/interval.js';
import {AffineContext,geometry} from '../math/affine.js';
import * as poly from '../math/polynomial.js';
import * as nn from '../math/neural.js';
import * as tr from '../math/transformer.js';
import {mccormick,mccormickSlice,refine} from '../math/relaxations.js';
const inside=(a,x,tol=1e-9)=>assert.ok(a[0]-tol<=x&&x<=a[1]+tol,`${x} outside ${a}`);
const points=(a,n=81)=>Array.from({length:n},(_,i)=>a[0]+(a[1]-a[0])*i/(n-1));
let seed=1234567;const rng=()=>((seed=(1664525*seed+1013904223)>>>0)/2**32);
test('interval edge cases, invalid domains and overflow fail explicitly',()=>{
  assert.deepEqual(iv.square([-1,1]),[0,1]);assert.deepEqual(iv.sub([0,1],[0,1]),[-1,1]);
  assert.deepEqual(iv.div([2,4],[2,2]),[1,2]);assert.deepEqual(iv.div([2,4],[-2,-1]),[-4,-1]);
  for(const f of [()=>iv.I(1,0),()=>iv.I(NaN),()=>iv.div([1,2],[-1,1]),()=>iv.exp([700,1000]),()=>iv.mul([1e308,1e308],[2,2]),()=>iv.partition([0,1],0),()=>refine([0,1],0)])assert.throws(f,RangeError);
  assert.deepEqual(iv.natural('quadratic',[0,1]),[0,1]);assert.deepEqual(iv.exactScalar('quadratic',[0,1]),[0,.25]);
  assert.deepEqual(iv.natural('quadratic',[0,1],'rewritten'),[0,.25]);
});
test('generated interval, centred and subdivision containment diagnostics',()=>{
  for(let trial=0;trial<150;trial++){
    const a=-2+3*rng(),b=a+2*rng(),X=[a,b];
    for(const name of ['cancel','quadratic','square','exp','sin','composed']){
      const bounds=[iv.natural(name,X),iv.centred(name,X),iv.hull(iv.partition(X,8).map(x=>iv.natural(name,x)))];
      for(const x of points(X,17))for(const bound of bounds)inside(bound,iv.scalar(name,x));
    }
  }
});
test('affine dependencies, nonlinear products and repeated affine maps',()=>{
  const c=new AffineContext(),x=c.input([-1,2]),y=c.input([.2,.7]);
  assert.deepEqual(c.range(c.add(x,c.scale(x,-1))),[0,0]);
  const z=c.linear([[2,1],[-1,3]],[x,y]);const back=c.linear([[3/7,-1/7],[1/7,2/7]],z);
  inside(c.range(back[0]),-1);inside(c.range(back[0]),2);
  for(const a of points([-1,2]))for(const b of points([.2,.7],13))inside(c.range(c.mul(x,y)),a*b);
  for(const a of points([-1,2]))inside(c.range(c.square(x)),a*a);
  for(const nonlinear of [false,true])for(const constraint of [false,true]){
    const g=geometry(37,.6,1,.7,nonlinear,constraint);
    for(const p of g.boundary)p.forEach((v,i)=>inside(c.range(g.forms[i]),v));
  }
});
test('Taylor remainder including nonlinear composition and off-centre expansions',()=>{
  for(const name of ['quadratic','exp','sin','composed'])for(const domain of [[0,1],[-1.4,.8],[.2,.2]])for(const degree of [1,2,4,8])for(const fraction of [0,.5,1]){
    const m=poly.taylor(name,domain,degree,domain[0]+(domain[1]-domain[0])*fraction);
    for(const x of points(domain)){inside(m.r,iv.scalar(name,x)-m.value(x),1e-8);inside(m.bound,iv.scalar(name,x),1e-8);}
  }
});
test('Bernstein convex hull and Chebyshev analytic remainder',()=>{
  assert.deepEqual(poly.bernstein([0,1,-1],[0,1]).coefficients,[0,.5,0]);
  for(const domain of [[-1,1],[0,2],[.7,.7]])for(const n of [1,2,5,8]){
    const m=poly.chebyshevExp(domain,n);for(const x of points(domain)){inside(m.r,Math.exp(x)-m.value(x));inside(m.bound,Math.exp(x));}
  }
  const refined=iv.hull(iv.partition([0,1],16).map(d=>poly.bernstein([0,1,-1],d).bound));
  assert.ok(iv.width(refined)<.26);
});
test('McCormick inequalities and complete refinement cover',()=>{
  const X=[-.7,1.2],Y=[-.2,2];for(const x of points(X))for(const y of points(Y,11))inside(mccormick(x,y,X,Y),x*y);
  assert.deepEqual(mccormickSlice([0,1]),[0,.5]);
  for(const engine of ['Interval','McCormick LP']){
    const r=refine([0,1],32,engine);for(const x of points([0,1]))inside([r.lower[0],r.upper[0]],x*(1-x));
    const cells=r.cells.slice().sort((a,b)=>a.domain[0]-b.domain[0]);assert.equal(cells[0].domain[0],0);assert.equal(cells.at(-1).domain[1],1);
    cells.slice(1).forEach((c,i)=>assert.equal(c.domain[0],cells[i].domain[1]));
    r.history.slice(1).forEach((h,i)=>assert.ok(iv.width(h.bound)<=iv.width(r.history[i].bound)+1e-12));
  }
});
test('all network methods enclose deterministic evaluations across activations/depths',()=>{
  for(const activation of ['relu','square'])for(const depth of [1,2])for(const domain of [[-1,1],[.2,.2],[-.6,.3],[.7,1.5]])for(const alpha of [0,.5,1]){
    const results=nn.compareNetwork(domain,depth,activation,alpha);
    for(const r of results){assert.equal(r.certified,false);for(const x of points(domain,161))inside([r.lower[0],r.upper[0]],nn.forward(x,depth,activation));}
  }
});
test('attention projections, probabilities and logits enclose grid diagnostics',()=>{
  for(const sequence of [[0,1],[2,3],[0,0]])for(const token of [0,1])for(const radius of [0,.15,.7])for(const temperature of [.25,1,2])for(const method of ['Interval','Dependencies + simplex','Subdivide + dependencies']){
    const b=tr.experiment(sequence,token,radius,temperature,method);assert.equal(b.certified,false);
    for(const dx of [-radius,0,radius])for(const dy of [-radius,0,radius]){
      const X=sequence.map(i=>tr.EMBEDDINGS[i].slice());X[token][0]+=dx;X[token][1]+=dy;const f=tr.forward(X,temperature);
      for(const key of ['Q','K','V','S','A','H'])f[key].forEach((row,i)=>row.forEach((v,j)=>inside(b[key][i][j],v)));
      f.logits.forEach((v,i)=>inside(b.logits[i],v));
      if(b.gap>0)assert.equal(f.logits.indexOf(Math.max(...f.logits)),b.winner);
    }
    b.A.flat().forEach(a=>{assert.ok(a[0]>=0&&a[1]<=1);});
  }
  assert.throws(()=>tr.experiment([0,1],0,.1,0));assert.throws(()=>tr.experiment([0,1],0,.1,Infinity));
});
test('display semantics and zero-width metrics',()=>{
  assert.equal(iv.metrics([0,0],[0,0]).ratio,null);
  const r=tr.experiment([0,1],0,1,.25);assert.equal(r.status,'Inconclusive');
  const small=tr.experiment([0,1],0,0,1);assert.ok(small.gap>0);assert.equal(small.certified,false);
});
