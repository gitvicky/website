/** Real-arithmetic inclusion formulas evaluated in binary64, WITHOUT directed rounding. */
export function I(lo, hi = lo) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo > hi) throw new RangeError('Expected a finite, nonempty interval.');
  return [lo, hi];
}
export const add = (a,b) => I(a[0]+b[0],a[1]+b[1]);
export const neg = a => I(-a[1],-a[0]);
export const sub = (a,b) => add(a,neg(b));
export function mul(a,b) { const v=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]; return I(Math.min(...v),Math.max(...v)); }
export const scale = (a,s) => mul(a,I(s));
export function div(a,b) { if (b[0]<=0 && b[1]>=0) throw new RangeError('Division by an interval containing zero.'); return mul(a,I(1/b[1],1/b[0])); }
export const square = a => I(a[0]<=0 && a[1]>=0 ? 0 : Math.min(a[0]**2,a[1]**2),Math.max(a[0]**2,a[1]**2));
export function power(a,n) { if (!Number.isInteger(n)||n<0) throw new RangeError('Nonnegative integer power required.'); if(n===0)return I(1); return n%2 ? I(a[0]**n,a[1]**n) : I(square(a)[0]**(n/2),square(a)[1]**(n/2)); }
export const exp = a => I(Math.exp(a[0]),Math.exp(a[1]));
export const relu = a => I(Math.max(0,a[0]),Math.max(0,a[1]));
export const mid = a => a[0]+(a[1]-a[0])/2;
export const radius = a => (a[1]-a[0])/2;
export const width = a => a[1]-a[0];
// Intersections can invert at point domains after floating-point cancellation.
// Fall back to the union hull; this is conservative but is NOT directed rounding.
export function intersect(a,b){const l=Math.max(a[0],b[0]),u=Math.min(a[1],b[1]);return l<=u?I(l,u):I(Math.min(a[0],b[0]),Math.max(a[1],b[1]));}
export const hull = arr => I(Math.min(...arr.map(a=>a[0])),Math.max(...arr.map(a=>a[1])));
export function sin(a) {
  if(width(a)>=2*Math.PI)return I(-1,1);
  const v=[Math.sin(a[0]),Math.sin(a[1])];
  for(let k=Math.ceil((a[0]-Math.PI/2)/Math.PI);k<=Math.floor((a[1]-Math.PI/2)/Math.PI);k++)v.push(k%2===0?1:-1);
  return I(Math.min(...v),Math.max(...v));
}
export function partition(a,n) {
  I(...a); if(!Number.isInteger(n)||n<1||n>256)throw new RangeError('Use 1–256 subdomains.');
  return Array.from({length:n},(_,i)=>I(a[0]+width(a)*i/n,i===n-1?a[1]:a[0]+width(a)*(i+1)/n));
}
export const linear = (W,x,b=W.map(()=>0)) => W.map((row,i)=>row.reduce((s,w,j)=>add(s,scale(x[j],w)),I(b[i])));
export function result(method,bounds,start,extra={}) {
  const vector=typeof bounds[0]==='number'?[bounds]:bounds;
  vector.forEach(a=>I(...a));
  return {method,lower:vector.map(a=>a[0]),upper:vector.map(a=>a[1]),certified:false,
    assumptions:['Fixed documented model; finite box domain; formulas sound over real arithmetic.', 'Binary64 arithmetic and Math transcendental functions are not outward rounded.'],
    computationMs:performance.now()-start,...extra};
}
export function metrics(a,exact) {
  const w=width(a),e=exact?width(exact):null;
  return {width:w,excess:e===null?null:w-e,ratio:e>0?w/e:null};
}
export const scalar = (name,x) => name==='cancel'?0:name==='quadratic'?x*(1-x):name==='square'?x*x-x:name==='exp'?Math.exp(x):name==='sin'?Math.sin(x):Math.exp(x*(1-x));
export function exactScalar(name,a) {
  if(name==='cancel')return I(0);
  if(name==='exp')return exp(a);
  if(name==='sin')return sin(a);
  const v=[scalar(name,a[0]),scalar(name,a[1])];
  if(a[0]<=.5&&a[1]>=.5)v.push(scalar(name,.5));
  return I(Math.min(...v),Math.max(...v));
}
export function natural(name,a,form='natural') {
  if(name==='cancel')return form==='rewritten'?I(0):sub(a,a);
  if(name==='quadratic')return form==='rewritten'?sub(I(.25),square(sub(a,I(.5)))):mul(a,sub(I(1),a));
  if(name==='square')return form==='rewritten'?sub(square(sub(a,I(.5))),I(.25)):sub(square(a),a);
  if(name==='exp')return exp(a);
  if(name==='sin')return sin(a);
  return exp(natural('quadratic',a));
}
export function centred(name,a) {
  const c=mid(a),d=name==='cancel'?I(0):name==='quadratic'?sub(I(1),scale(a,2)):name==='square'?sub(scale(a,2),I(1)):name==='exp'?exp(a):name==='sin'?sin(add(a,I(Math.PI/2))):mul(exp(natural('quadratic',a)),sub(I(1),scale(a,2)));
  return add(I(scalar(name,c)),mul(d,sub(a,I(c))));
}
