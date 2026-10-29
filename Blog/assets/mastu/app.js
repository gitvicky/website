'use strict';
if(new URLSearchParams(location.search).get('embed')==='1')document.body.classList.add('embedded');
const $=id=>document.getElementById(id);
function fail(message){$('loading').hidden=false;$('load-message').textContent=message;const spinner=document.querySelector('.loader');if(spinner)spinner.hidden=true;$('compute-status').textContent='Unavailable';$('controls').disabled=true;}
async function loadJSON(url){const response=await fetch(url);if(!response.ok)throw Error(`Could not load ${url} (${response.status}). The validated model assets must be exported before hosting.`);return response.json();}
async function loadBinary(spec){
  let data;
  if(spec.parts){
    const parts=await Promise.all(spec.parts.map(loadBinary));
    const combined=new Uint8Array(spec.bytes);
    let offset=0;
    for(const part of parts){combined.set(new Uint8Array(part),offset);offset+=part.byteLength;}
    data=combined.buffer;
  }else{
    const response=await fetch(`assets/${spec.file}`);
    if(!response.ok)throw Error(`Could not load model asset: ${spec.file}`);
    data=await response.arrayBuffer();
  }
  if(data.byteLength!==spec.bytes)throw Error(`Incomplete model asset: ${spec.file}`);
  if(globalThis.crypto?.subtle){const hash=Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),b=>b.toString(16).padStart(2,'0')).join('');if(hash!==spec.sha256)throw Error(`Model checksum mismatch: ${spec.file}`);}
  return data;
}
async function start(){
  const faces=await document.fonts.load('12px "Latin Modern Roman"');
  if(!faces.length)throw Error('Required Latin Modern Roman display font is unavailable.');
  const model=await loadJSON('assets/manifest.json');
  if(model.schema!==1||!model.validation?.passed)throw Error('This model export has not passed its independent validation checks.');
  if(!Number.isInteger(model.vertices)||model.vertices<=0||!Number.isInteger(model.rank)||model.rank<0)throw Error('Invalid model dimensions');
  const names=['positions','normals','triangles','mean','basis'];
  const buffers=await Promise.all(names.map(name=>loadBinary(model.files[name])));
  const assets=Object.fromEntries(names.map((name,i)=>[name,buffers[i]]));
  if(assets.positions.byteLength!==model.vertices*12||assets.normals.byteLength!==model.vertices*12||assets.mean.byteLength!==model.vertices*12||assets.basis.byteLength!==model.rank*model.vertices*12||assets.triangles.byteLength!==model.triangles*12)throw Error('Model assets have inconsistent dimensions');
  const viewer=new MastuViewer($('scene'),new Float32Array(assets.positions),new Float32Array(assets.normals),new Uint32Array(assets.triangles),model.colour_max_mm);
  $('scene').addEventListener('viewer-error',e=>fail(e.detail));
  $('reset-camera').disabled=false;$('reset-camera').onclick=()=>viewer.reset();
  $('mesh-info').textContent=`${model.triangles.toLocaleString()} display triangles · Steady state`;
  $('source-count').textContent=model.native_source_nodes.toLocaleString();
  $('legend-max').textContent=model.colour_max_mm.toFixed(0);
  const worst=model.validation.worst;
  $('validation-summary').textContent=`Compared with the original network at ${model.validation.points} separate parameter points: worst relative vector L₂ error ${(worst.relative_vector_l2*100).toFixed(4)}%; largest vector difference ${worst.maximum_vector_error_mm.toFixed(4)} mm. These checks use the quantised spatial basis shipped with this page.`;
  const controls=[['alpha',1e6,4],['temperature',1,2]];
  const values=model.parameters.map(p=>p.initial);
  controls.forEach(([id,scale,decimals],i)=>{
    const p=model.parameters[i],range=$(id),number=$(`${id}-value`);
    for(const input of [range,number]){input.min=p.min*scale;input.max=p.max*scale;input.step=(p.max-p.min)*scale/1000;}
    range.value=p.initial*scale;number.value=(p.initial*scale).toFixed(decimals);
    $(`${id}-min`).textContent=(p.min*scale).toFixed(decimals===4?2:0);$(`${id}-max`).textContent=(p.max*scale).toFixed(decimals===4?2:0);
    const update=input=>{let v=Number(input.value);if(input.value===''||!Number.isFinite(v))return;v=Math.max(Number(range.min),Math.min(Number(range.max),v));values[i]=Math.max(p.min,Math.min(p.max,v/scale));range.value=v;if(input!==number)number.value=v.toFixed(decimals);queue();};
    range.addEventListener('input',()=>update(range));number.addEventListener('change',()=>{update(number);number.value=(values[i]*scale).toFixed(decimals);});
  });
  const worker=new Worker('worker.js');
  let id=0,inFlight=false,ready=false,wanted=null;
  function queue(){wanted={id:++id,parameters:values.slice(),start:performance.now()};$('compute-status').textContent='Updating…';send();}
  function send(){if(ready&&!inFlight&&wanted){inFlight=true;worker.postMessage({type:'predict',id:wanted.id,parameters:wanted.parameters});}}
  worker.onerror=event=>fail(`Browser prediction failed: ${event.message}`);
  worker.onmessage=({data})=>{
    if(data.type==='error'){fail(data.message);return;}
    if(data.type==='ready'){ready=true;$('controls').disabled=false;queue();return;}
    inFlight=false;
    if(wanted&&data.id===wanted.id){
      const renderStart=performance.now();viewer.update(new Float32Array(data.field));
      const total=performance.now()-wanted.start;
      $('maximum').textContent=data.maximum.toFixed(2);$('saturation').hidden=data.maximum<=model.colour_max_mm;
      $('compute-status').textContent=`${Math.round(data.milliseconds)} ms predict · ${Math.round(total)} ms update`;
      $('compute-status').title=`CPU prediction: ${data.milliseconds.toFixed(1)} ms; WebGL submission: ${(performance.now()-renderStart).toFixed(1)} ms. Update time includes Worker messaging and draw submission, not GPU completion.`;
      $('loading').hidden=true;wanted=null;
    }
    send();
  };
  $('quantity').addEventListener('change',()=>{const q=$('quantity').value;viewer.quantity={magnitude:0,x:1,y:2,z:3}[q];$('legend-title').textContent=q==='magnitude'?'Displacement magnitude · mm':`Displacement u_${q} · mm`;$('legend-min').textContent=q==='magnitude'?'0':String(-model.colour_max_mm);viewer.draw();});
  $('reset-parameters').onclick=()=>{model.parameters.forEach((p,i)=>{values[i]=p.initial;const [name,scale,decimals]=controls[i];$(name).value=p.initial*scale;$(`${name}-value`).value=(p.initial*scale).toFixed(decimals);});queue();};
  worker.postMessage({type:'init',model,mean:assets.mean,basis:assets.basis},[assets.mean,assets.basis]);
  window.addEventListener('pagehide',()=>worker.terminate(),{once:true});
}
start().catch(error=>fail(error.message));
