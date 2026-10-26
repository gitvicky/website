import * as iv from './interval.js';
export const factorial=n=>{let v=1;for(let i=2;i<=n;i++)v*=i;return v;};
export const choose=(n,k)=>k<0||k>n?0:factorial(n)/(factorial(k)*factorial(n-k));
export const evaluate=(p,x)=>p.reduceRight((s,c)=>s*x+c,0);
export const plus=(p,q)=>Array.from({length:Math.max(p.length,q.length)},(_,i)=>(p[i]||0)+(q[i]||0));
export const times=(p,q)=>{const r=Array(p.length+q.length-1).fill(0);p.forEach((a,i)=>q.forEach((b,j)=>r[i+j]+=a*b));return r;};
export const scale=(p,s)=>p.map(x=>x*s);
// Substitute x=a+(b-a)t, then t^k = sum_{i=k}^n C(i,k)/C(n,k) B_{i,n}(t).
export function bernstein(p,domain) {
  const [a,b]=domain,n=p.length-1;
  const power=Array.from({length:n+1},(_,k)=>p.reduce((s,c,j)=>j<k?s:s+c*choose(j,k)*a**(j-k)*(b-a)**k,0));
  const coefficients=Array.from({length:n+1},(_,i)=>power.reduce((s,c,k)=>k>i?s:s+c*choose(i,k)/choose(n,k),0));
  return {coefficients,bound:iv.I(Math.min(...coefficients),Math.max(...coefficients))};
}
export const intervalPolynomial=(p,a)=>p.reduceRight((s,c)=>iv.add(iv.mul(s,a),iv.I(c)),iv.I(0));
export function tmAdd(a,b){return {p:plus(a.p,b.p),r:iv.add(a.r,b.r)};}
export function tmMul(a,b,domain,degree){
  const full=times(a.p,b.p),keep=full.slice(0,degree+1),tail=full.map((x,i)=>i>degree?x:0);
  const ra=bernstein(a.p,domain).bound,rb=bernstein(b.p,domain).bound;
  return {p:keep,r:iv.add(bernstein(tail,domain).bound,iv.add(iv.mul(ra,b.r),iv.add(iv.mul(rb,a.r),iv.mul(a.r,b.r))))};
}
export function taylor(name,domain,degree,centre=iv.mid(domain)) {
  if(!Number.isInteger(degree)||degree<1||degree>10)throw new RangeError('Taylor degree must be 1–10.');
  if(!Number.isFinite(centre))throw new RangeError('Finite expansion centre required.');
  const t=iv.sub(domain,iv.I(centre)),h=Math.max(Math.abs(t[0]),Math.abs(t[1]));let p,r;
  if(name==='quadratic') {
    const full=[centre*(1-centre),1-2*centre,-1];p=full.slice(0,degree+1);
    r=degree>=2?iv.I(0):iv.scale(iv.square(t),-1);
  } else if(name==='composed') {
    // exp(q(c)+v), q=x(1-x). Truncate polynomial products and retain every discarded term.
    const q=centre*(1-centre),v={p:[0,1-2*centre,-1],r:iv.I(0)};
    let term={p:[1],r:iv.I(0)},sum=term;
    for(let k=1;k<=degree;k++){term=tmMul(term,v,t,degree);sum=tmAdd(sum,{p:scale(term.p,1/factorial(k)),r:iv.scale(term.r,1/factorial(k))});}
    const vr=bernstein(v.p,t).bound,abs=Math.max(Math.abs(vr[0]),Math.abs(vr[1]));
    const error=Math.exp(q+Math.max(0,vr[1]))*abs**(degree+1)/factorial(degree+1);
    p=scale(sum.p,Math.exp(q));r=iv.add(iv.scale(sum.r,Math.exp(q)),iv.I(-error,error));
  } else {
    p=Array.from({length:degree+1},(_,k)=>(name==='exp'?Math.exp(centre):Math.sin(centre+k*Math.PI/2))/factorial(k));
    const M=name==='exp'?Math.exp(Math.max(domain[1],centre)):1;
    const error=M*h**(degree+1)/factorial(degree+1);r=iv.I(-error,error);
  }
  return {p,r,centre,bound:iv.add(bernstein(p,t).bound,r),value:x=>evaluate(p,x-centre)};
}
export function chebyshevExp(domain,degree) {
  if(!Number.isInteger(degree)||degree<1||degree>10)throw new RangeError('Chebyshev degree must be 1–10.');
  const c=iv.mid(domain),h=iv.radius(domain),N=degree+1;
  if(h===0)return {p:[Math.exp(c)],r:iv.I(0),bound:iv.I(Math.exp(c)),value:()=>Math.exp(c)};
  // Interpolation at roots of T_N; exact-arithmetic coefficient construction via a DCT.
  const theta=Array.from({length:N},(_,j)=>Math.PI*(j+.5)/N),ys=theta.map(t=>Math.exp(c+h*Math.cos(t)));
  const coeff=Array.from({length:N},(_,k)=>ys.reduce((s,y,j)=>s+y*Math.cos(k*theta[j]),0)*(k===0?1:2)/N);
  let T0=[1],T1=[0,1],p=[coeff[0]];
  for(let k=1;k<N;k++){p=plus(p,scale(T1,coeff[k]));const next=plus(times([0,2],T1),scale(T0,-1));T0=T1;T1=next;}
  const error=Math.exp(domain[1])*h**N/(2**(N-1)*factorial(N)),r=iv.I(-error,error);
  return {p,r,bound:iv.add(bernstein(p,[-1,1]).bound,r),value:x=>evaluate(p,(x-c)/h)};
}
export function polynomialExperiment(name,domain,degree,count,centreFraction=.5,method='Taylor + Bernstein') {
  const start=performance.now();const pieces=iv.partition(domain,count).map(d=>{
    let m=method==='Chebyshev + remainder'?chebyshevExp(d,degree):taylor(name,d,degree,d[0]+iv.width(d)*centreFraction);
    if(method==='Taylor + interval polynomial')m={...m,bound:iv.add(intervalPolynomial(m.p,iv.sub(d,iv.I(m.centre))),m.r)};
    return {...m,domain:d};
  });
  return {...iv.result(method,iv.hull(pieces.map(p=>p.bound)),start,{operations:count,diagnostic:'Analytic remainder formula; floating-point coefficients and remainder are illustrative.'}),pieces};
}
