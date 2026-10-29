'use strict';
importScripts('uq-interval.js');
let model,lower,upper,remainder;
onmessage=({data})=>{
  try{
    if(data.type==='init'){
      MastuIntervals.selfTest();model=data.model;lower=new Float64Array(data.lower);upper=new Float64Array(data.upper);remainder=new Float64Array(data.remainder);
      if(lower.length!==model.vertices*18||upper.length!==lower.length||remainder.length!==model.vertices*3)throw Error('Invalid uncertainty packet dimensions');
      postMessage({type:'ready'});return;
    }
    const start=performance.now(),out=MastuIntervals.enclose(model,lower,upper,remainder,data.centers,data.halves);
    postMessage({type:'bounds',id:data.id,lower:out.lower.buffer,upper:out.upper.buffer,width:out.width.buffer,field:out.nominal.buffer,
                 maximumUpper:out.maximumUpper,maximumWidth:out.maximumWidth,maximumEstimate:out.maximumEstimate,
                 delta:out.delta,milliseconds:performance.now()-start},[out.lower.buffer,out.upper.buffer,out.width.buffer,out.nominal.buffer]);
  }catch(error){postMessage({type:'error',message:error.message});}
};
