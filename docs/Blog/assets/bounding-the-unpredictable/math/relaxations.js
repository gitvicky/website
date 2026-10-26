import * as iv from './interval.js';
export function reluLines(domain,alpha=.5) {
  if(alpha<0||alpha>1)throw new RangeError('ReLU lower slope must lie in [0,1].');
  const [l,u]=domain;
  if(l>=0)return {lower:[1,0],upper:[1,0]};
  if(u<=0)return {lower:[0,0],upper:[0,0]};
  return {lower:[alpha,0],upper:[u/(u-l),-u*l/(u-l)]};
}
export function squareLines([l,u],alpha=.5){const t=l+(u-l)*alpha;return {lower:[2*t,-t*t],upper:[l+u,-l*u]};}
export function mccormick(x,y,X,Y) {
  const [l,u]=X,[v,w]=Y;
  const lo=Math.max(l*y+v*x-l*v,u*y+w*x-u*w),hi=Math.min(u*y+v*x-u*v,l*y+w*x-l*w);
  return iv.I(Math.min(lo,hi),Math.max(lo,hi)); // Hull if endpoint roundoff inverts a zero-width slice.
}
// Optimize the four McCormick planes on the line y=1-x. This is a tiny LP,
// not a general-purpose optimizer: envelope extrema occur at ends or line crossings.
export function mccormickSlice(X) {
  const [l,u]=X,v=1-u,w=1-l;
  const lower=[[v-l,l-l*v],[w-u,u-u*w]],upper=[[v-u,u-u*v],[w-l,l-l*w]];
  const candidates=lines=>{const xs=[l,u],den=lines[0][0]-lines[1][0];if(den!==0){const x=(lines[1][1]-lines[0][1])/den;if(x>l&&x<u)xs.push(x);}return xs;};
  return iv.I(Math.min(...candidates(lower).map(x=>Math.max(...lower.map(([a,b])=>a*x+b)))),Math.max(...candidates(upper).map(x=>Math.min(...upper.map(([a,b])=>a*x+b)))));
}
export function refine(domain,leaves,engine='Interval') {
  if(!Number.isInteger(leaves)||leaves<1||leaves>128)throw new RangeError('Use 1–128 leaves.');
  const start=performance.now(),bound=d=>engine==='Interval'?iv.natural('quadratic',d):mccormickSlice(d);
  const cells=[{domain,bound:bound(domain)}],history=[{leaves:1,bound:cells[0].bound}];
  for(let k=1;k<leaves;k++){
    // Best-first split by local bound width; every leaf is retained, so the cover stays complete.
    const idx=cells.reduce((best,c,i)=>iv.width(c.bound)>iv.width(cells[best].bound)?i:best,0),old=cells.splice(idx,1)[0];
    for(const d of iv.partition(old.domain,2))cells.push({domain:d,bound:bound(d)});
    history.push({leaves:cells.length,bound:iv.hull(cells.map(c=>c.bound))});
  }
  return {...iv.result(engine+' refinement',iv.hull(cells.map(c=>c.bound)),start,{operations:2*leaves-1,refinements:leaves-1}),cells,history};
}
// McCormick plane bounds retain affine dependencies in the original input factors.
export function affineMcCormick(ctx,a,b){
  const A=ctx.range(a),B=ctx.range(b),[l,u]=A,[v,w]=B;
  const plane=(p,q,k)=>ctx.range(ctx.add(ctx.add(ctx.scale(b,p),ctx.scale(a,q)),ctx.constant(k)));
  const ll=plane(l,v,-l*v),uu=plane(u,w,-u*w),ul=plane(u,v,-u*v),lu=plane(l,w,-l*w),natural=iv.mul(A,B);
  const lo=Math.max(natural[0],ll[0],uu[0]),hi=Math.min(natural[1],ul[1],lu[1]);
  return lo<=hi?iv.I(lo,hi):natural;
}
