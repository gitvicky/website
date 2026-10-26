import * as iv from './interval.js';
import {AffineContext} from './affine.js';
import {affineMcCormick} from './relaxations.js';
export const VOCAB=['sun','rain','wind','still'];
export const EMBEDDINGS=[[.8,.2],[-.5,.7],[.1,-.8],[-.3,-.2]];
export const WEIGHTS={Q:[[.8,-.3],[.2,.7]],K:[[.6,.4],[-.5,.9]],V:[[1,.2],[-.3,.8]],O:[[1,-.4,-.7,.2],[.3,1,-.2,-.8]],bias:[.12,-.02,.02,-.08]};
const transpose=W=>W[0].map((_,i)=>W.map(r=>r[i]));
const mm=(X,W)=>X.map(x=>transpose(W).map(r=>r.reduce((s,w,j)=>s+w*x[j],0)));
export const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
export function forward(X,temperature=1) {
  if(!Number.isFinite(temperature)||temperature<=0)throw new RangeError('Positive temperature required.');
  const Q=mm(X,WEIGHTS.Q),K=mm(X,WEIGHTS.K),V=mm(X,WEIGHTS.V);
  const S=Q.map(q=>K.map(k=>q.reduce((s,v,i)=>s+v*k[i],0)/(Math.sqrt(2)*temperature)));
  const A=S.map(row=>{const p=sigmoid(row[0]-row[1]);return [p,1-p];});
  const H=A.map(a=>[0,1].map(j=>a[0]*V[0][j]+a[1]*V[1][j]));
  const logits=mm([H[1]],WEIGHTS.O)[0].map((v,i)=>v+WEIGHTS.bias[i]);return {Q,K,V,S,A,H,logits};
}
export function inputBox(sequence=[0,1],token=0,radius=.1) {
  if(sequence.length!==2||sequence.some(i=>!Number.isInteger(i)||i<0||i>=4)||![0,1].includes(token)||!Number.isFinite(radius)||radius<0||radius>1)throw new RangeError('Two vocabulary tokens and a radius in [0,1] required.');
  return sequence.map((s,i)=>EMBEDDINGS[s].map(v=>iv.I(v-(i===token?radius:0),v+(i===token?radius:0))));
}
export function boundBox(X,temperature=1,method='Interval') {
  if(!Number.isFinite(temperature)||temperature<=0)throw new RangeError('Positive finite temperature required.');
  X.flat().forEach(a=>iv.I(...a));
  const Q=X.map(x=>iv.linear(transpose(WEIGHTS.Q),x)),K=X.map(x=>iv.linear(transpose(WEIGHTS.K),x)),V=X.map(x=>iv.linear(transpose(WEIGHTS.V),x));
  let S=Q.map(q=>K.map(k=>iv.scale(q.reduce((s,a,i)=>iv.add(s,iv.mul(a,k[i])),iv.I(0)),1/(Math.sqrt(2)*temperature))));
  let differences;
  if(method!=='Interval'){
    const ctx=new AffineContext(),ax=X.map(row=>row.map(a=>ctx.input(a))),aq=ax.map(row=>ctx.linear(transpose(WEIGHTS.Q),row)),ak=ax.map(row=>ctx.linear(transpose(WEIGHTS.K),row));
    const as=aq.map(q=>ak.map(k=>ctx.scale(q.reduce((s,a,i)=>ctx.add(s,ctx.mul(a,k[i])),ctx.constant(0)),1/(Math.sqrt(2)*temperature))));
    S=S.map((row,i)=>row.map((natural,j)=>{
      const aa=ctx.range(as[i][j]),mc=iv.scale(aq[i].reduce((s,a,k)=>iv.add(s,affineMcCormick(ctx,a,ak[j][k])),iv.I(0)),1/(Math.sqrt(2)*temperature));
      return iv.intersect(iv.intersect(natural,aa),mc);
    }));
    // Build score differences before concretization to retain shared Q and K factors.
    differences=aq.map((q,i)=>{
      const diff=ctx.range(ctx.scale(q.reduce((s,a,k)=>ctx.add(s,ctx.mul(a,ctx.add(ak[0][k],ctx.scale(ak[1][k],-1)))),ctx.constant(0)),1/(Math.sqrt(2)*temperature)));
      const naive=iv.sub(S[i][0],S[i][1]);return iv.intersect(diff,naive);
    });
  }else differences=S.map(row=>iv.sub(row[0],row[1]));
  const A=differences.map(d=>{const p=iv.I(sigmoid(d[0]),sigmoid(d[1]));return [p,iv.sub(iv.I(1),p)];});
  const H=A.map(row=>[0,1].map(j=>{
    const independent=iv.add(iv.mul(row[0],V[0][j]),iv.mul(row[1],V[1][j]));
    if(method==='Interval')return independent;
    // For fixed p>=0, h is increasing in both values; endpoint optimization over p is exact for this relaxation.
    const lo=row[0].map(p=>p*V[0][j][0]+(1-p)*V[1][j][0]),hi=row[0].map(p=>p*V[0][j][1]+(1-p)*V[1][j][1]);
    return iv.I(Math.min(...lo),Math.max(...hi));
  }));
  const logits=iv.linear(transpose(WEIGHTS.O),H[1],WEIGHTS.bias);return {X,Q,K,V,S,A,H,logits};
}
export function experiment(sequence=[0,1],token=0,radius=.1,temperature=1,method='Interval') {
  const start=performance.now(),X=inputBox(sequence,token,radius),nominal=forward(sequence.map(i=>EMBEDDINGS[i]),temperature);
  const sub=method==='Subdivide + dependencies'?4:1,cells=[];
  for(const a of iv.partition(X[token][0],sub))for(const b of iv.partition(X[token][1],sub)){
    const box=X.map(row=>row.slice());box[token]=[a,b];cells.push(boundBox(box,temperature,method));
  }
  const merged={};for(const key of ['X','Q','K','V','S','A','H'])merged[key]=cells[0][key].map((r,i)=>r.map((_,j)=>iv.hull(cells.map(c=>c[key][i][j]))));
  merged.logits=cells[0].logits.map((_,i)=>iv.hull(cells.map(c=>c.logits[i])));
  const winner=nominal.logits.indexOf(Math.max(...nominal.logits)),gap=merged.logits[winner][0]-Math.max(...merged.logits.filter((_,i)=>i!==winner).map(a=>a[1]));
  return {...iv.result(method,merged.logits,start,{operations:cells.length}),...merged,nominal,winner,gap,
    status:gap>0?'Positive numerical margin — illustrative':'Inconclusive'};
}
