import * as iv from './interval.js';
// Symbols are local to one experiment, never recycled inside an expression.
export class AffineContext {
  constructor(){this.next=0;}
  constant(c){return {c,g:new Map()};}
  input(a){const x=this.constant(iv.mid(a));x.g.set(this.next++,iv.radius(a));return x;}
  add(a,b){const g=new Map(a.g);for(const [i,v]of b.g)g.set(i,(g.get(i)||0)+v);return {c:a.c+b.c,g};}
  scale(a,s){return {c:a.c*s,g:new Map([...a.g].map(([i,v])=>[i,s*v]))};}
  noise(a,r,shift=0){const g=new Map(a.g);if(r)g.set(this.next++,r);return {c:a.c+shift,g};}
  radius(a){return [...a.g.values()].reduce((s,v)=>s+Math.abs(v),0);}
  range(a){const r=this.radius(a);return iv.I(a.c-r,a.c+r);}
  mul(a,b){const lin=this.add(this.scale(a,b.c),this.scale(b,a.c));lin.c=a.c*b.c;return this.noise(lin,this.radius(a)*this.radius(b));}
  square(a){const r=this.radius(a),lin=this.scale(a,2*a.c);lin.c=a.c*a.c;return this.noise(lin,r*r/2,r*r/2);}
  relu(a){const [l,u]=this.range(a);if(l>=0)return a;if(u<=0)return this.constant(0);const s=u/(u-l),r=-u*l/(2*(u-l));return this.noise(this.scale(a,s),r,r);}
  linear(W,x,b=W.map(()=>0)){return W.map((row,i)=>row.reduce((s,w,j)=>this.add(s,this.scale(x[j],w)),this.constant(b[i])));}
}
export function convexHull(points) {
  const pts=points.slice().sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  if(pts.length<3)return pts;
  const cross=(a,b,c)=>(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]);
  const half=p=>{const h=[];for(const x of p){while(h.length>1&&cross(h.at(-2),h.at(-1),x)<=0)h.pop();h.push(x);}return h;};
  return half(pts).slice(0,-1).concat(half(pts.slice().reverse()).slice(0,-1));
}
export function zonotopeVertices(forms) {
  const ids=[...new Set(forms.flatMap(a=>[...a.g.keys()]))];
  if(ids.length>12)throw new RangeError('Geometry demo supports at most 12 generators.');
  return convexHull(Array.from({length:2**ids.length},(_,bits)=>forms.map(a=>a.c+ids.reduce((s,id,j)=>s+(a.g.get(id)||0)*(bits&(1<<j)?1:-1),0))));
}
export function geometry(angle,shear,wx,wy,nonlinear=false,constraint=false) {
  const c=new AffineContext(),a=angle*Math.PI/180;
  const W=[[Math.cos(a),Math.cos(a)*shear-Math.sin(a)],[Math.sin(a),Math.sin(a)*shear+Math.cos(a)]];
  const x=c.input(iv.I(-wx,wx)),y=constraint?c.scale(x,wy/wx):c.input(iv.I(-wy,wy));
  let z=c.linear(W,[x,y]);if(nonlinear)z=[z[0],c.add(z[1],c.scale(c.square(z[0]),.4))];
  const f=(u,v)=>{const p=W[0][0]*u+W[0][1]*v,q=W[1][0]*u+W[1][1]*v;return [p,q+(nonlinear?.4*p*p:0)];};
  // Boundary of the actual mapped rectangle (or constrained diagonal), for these invertible shear maps.
  const boundary=constraint?Array.from({length:81},(_,i)=>f(wx*(2*i/80-1),wy*(2*i/80-1))):
    [[-wx,-wy],[wx,-wy],[wx,wy],[-wx,wy]].flatMap((p,i,all)=>Array.from({length:41},(_,j)=>f(p[0]+(all[(i+1)%4][0]-p[0])*j/40,p[1]+(all[(i+1)%4][1]-p[1])*j/40)));
  let box=iv.linear(W,[iv.I(-wx,wx),iv.I(-wy,wy)]);if(nonlinear)box=[box[0],iv.add(box[1],iv.scale(iv.square(box[0]),.4))];
  return {W,forms:z,box,polygon:zonotopeVertices(z),boundary,generators:c.next};
}
