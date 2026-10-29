/* Outward binary64 intervals for an exported quadratic Taylor enclosure.
   The mathematical target is the stored-weight network, at fixed query vertices.
   Every basic arithmetic primitive is padded outward; square roots are checked
   by enclosed squares, so no libm accuracy promise is assumed. */
(function(root){
  'use strict';
  const bits=new DataView(new ArrayBuffer(8)), single=new DataView(new ArrayBuffer(4));
  function up(x){
    if(Number.isNaN(x))throw Error('Invalid interval arithmetic');
    if(x===Infinity)return x;
    if(x===0)return Number.MIN_VALUE;
    bits.setFloat64(0,x,true);
    let lo=bits.getUint32(0,true),hi=bits.getUint32(4,true);
    if(x>0){lo=(lo+1)>>>0;if(lo===0)hi=(hi+1)>>>0;}
    else{if(lo===0)hi=(hi-1)>>>0;lo=(lo-1)>>>0;}
    bits.setUint32(0,lo,true);bits.setUint32(4,hi,true);return bits.getFloat64(0,true);
  }
  const down=x=>-up(-x);
  function next32(x,positive){
    if(x===0)return positive?2**-149:-(2**-149);
    single.setFloat32(0,x,true);let n=single.getUint32(0,true);
    n=(n+((x>0)===positive?1:-1))>>>0;single.setUint32(0,n,true);return single.getFloat32(0,true);
  }
  function f32down(x){const y=Math.fround(x);return y>x?next32(y,false):y;}
  function f32up(x){const y=Math.fround(x);return y<x?next32(y,true):y;}
  const add=(a,b)=>[down(a[0]+b[0]),up(a[1]+b[1])];
  const sub=(a,b)=>[down(a[0]-b[1]),up(a[1]-b[0])];
  function mul(a,b){const v=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]];return [down(Math.min(...v)),up(Math.max(...v))];}
  function div(a,b){if(b[0]<=0&&b[1]>=0)throw Error('Interval division by zero');return mul(a,[down(1/b[1]),up(1/b[0])]);}
  function square(a){return [a[0]<=0&&a[1]>=0?0:Math.max(0,down(Math.min(a[0]*a[0],a[1]*a[1]))),up(Math.max(a[0]*a[0],a[1]*a[1]))];}
  function rootBound(x,upper){
    if(x===0)return 0;
    if(x<0||!Number.isFinite(x))throw Error('Invalid squared magnitude');
    if(x<2**-500){const y=rootBound(x*2**512,upper)*2**-256;return upper?up(y):Math.max(0,down(y));}
    if(x>2**500){const y=rootBound(x*2**-512,upper)*2**256;return upper?up(y):down(y);}
    let y=Math.sqrt(x),steps=0;
    // Prove the endpoint with outward products instead of trusting sqrt.
    while(upper?down(y*y)<x:up(y*y)>x){y=upper?up(y):down(y);if(++steps>16)throw Error('Browser square-root arithmetic failed validation');}
    return y;
  }
  const sqrt=a=>[rootBound(Math.max(0,a[0]),false),rootBound(a[1],true)];
  function selfTest(){
    if(up(1)!==1+Number.EPSILON||down(1)!==1-Number.EPSILON/2||up(-Number.MIN_VALUE)!==0||Number.MIN_VALUE*2===0||Math.fround(2**-149)===0)throw Error('Browser does not satisfy required IEEE-754 arithmetic');
    const r=sqrt([2,2]);if(!(r[0]*r[0]<=2&&r[1]*r[1]>=2))throw Error('Square-root validation failed');
  }
  function safeHalf(p,c){return Math.max(0,Math.min(p.maximum_half_width,down(c-p.min),down(p.max-c)));}
  function encodedBox(model,centers,halves){
    if(centers.length!==2||halves.length!==2)throw Error('Expected two parameter centres and half-widths');
    const delta=[];
    for(let i=0;i<2;i++){
      const p=model.parameters[i],c=centers[i],h=halves[i];
      if(!Number.isFinite(c)||!Number.isFinite(h)||c<p.min||c>p.max||h<0||h>safeHalf(p,c))throw Error('Requested uncertainty box leaves certified coverage');
      const physical=[down(c-h),up(c+h)],shifted=sub(physical,[p.offset,p.offset]);
      const frozen=div(shifted,[p.scale,p.scale]),real=div(shifted,p.real_scale);
      const legacy=x=>Math.fround(Math.fround(x-Math.fround(p.offset))/Math.fround(p.scale));
      const legacyLo=legacy(f32down(physical[0])),legacyHi=legacy(f32up(physical[1]));
      // Intersect two independently proven covers: the local normalization
      // enclosure and the archived parent encoded rectangle. The physical
      // request was checked to lie within its inward-rounded parent coverage.
      const lo=Math.max(p.encoded_min,Math.min(frozen[0],real[0],legacyLo));
      const hi=Math.min(p.encoded_max,Math.max(frozen[1],real[1],legacyHi));
      if(lo>hi)throw Error('Empty normalization enclosure');
      delta.push(sub([lo,hi],[p.encoded_center,p.encoded_center]));
    }
    return delta;
  }
  function enclose(model,lo,hi,radius,centers,halves){
    const d=encodedBox(model,centers,halves),monomials=[[1,1],d[0],d[1],square(d[0]),mul(d[0],d[1]),square(d[1])];
    const n=model.vertices,lower=new Float64Array(n),upper=new Float64Array(n),width=new Float64Array(n),nominal=new Float32Array(n*3);
    const mid=centers.map((c,i)=>(c-model.parameters[i].offset)/model.parameters[i].scale-model.parameters[i].encoded_center);
    const m=[1,mid[0],mid[1],mid[0]*mid[0],mid[0]*mid[1],mid[1]*mid[1]];
    let maximumUpper=0,maximumWidth=0,maximumEstimate=0;
    for(let v=0;v<n;v++){
      let nearest=[0,0],furthest=[0,0];
      for(let c=0;c<3;c++){
        const node=v*3+c,offset=node*6;let value=[lo[offset],hi[offset]],estimate=lo[offset]/2+hi[offset]/2;
        for(let j=1;j<6;j++){value=add(value,mul([lo[offset+j],hi[offset+j]],monomials[j]));estimate+=(lo[offset+j]/2+hi[offset+j]/2)*m[j];}
        value=add(value,[-radius[node],radius[node]]);nominal[node]=estimate;
        const a=Math.max(value[0],-value[1],0),b=Math.max(Math.abs(value[0]),Math.abs(value[1]));
        nearest=add(nearest,square([a,a]));furthest=add(furthest,square([b,b]));
      }
      lower[v]=Math.max(0,down(rootBound(Math.max(0,nearest[0]),false)*1000));
      upper[v]=up(rootBound(furthest[1],true)*1000);width[v]=up(upper[v]-lower[v]);
      maximumUpper=Math.max(maximumUpper,upper[v]);maximumWidth=Math.max(maximumWidth,width[v]);
      maximumEstimate=Math.max(maximumEstimate,Math.hypot(nominal[v*3],nominal[v*3+1],nominal[v*3+2])*1000);
    }
    if(!Number.isFinite(maximumUpper)||!Number.isFinite(maximumWidth))throw Error('Nonfinite enclosure');
    return {lower,upper,width,nominal,maximumUpper,maximumWidth,maximumEstimate,delta:d};
  }
  root.MastuIntervals={up,down,add,sub,mul,div,square,sqrt,f32down,f32up,safeHalf,encodedBox,enclose,selfTest};
})(globalThis);
