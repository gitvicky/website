'use strict';
importScripts('response.js');
let model, mean, basis;
onmessage = ({data}) => {
  try {
    if (data.type === 'init') {
      model = data.model;
      mean = new Float32Array(data.mean);
      basis = new Float32Array(data.basis);
      postMessage({type: 'ready'});
      return;
    }
    const start = performance.now();
    const field = MastuResponse.evaluate(model, mean, basis, data.parameters);
    let maximum = 0;
    for (let i = 0; i < field.length; i += 3) maximum = Math.max(maximum, Math.hypot(field[i], field[i+1], field[i+2]) * 1000);
    postMessage({type: 'prediction', id: data.id, field: field.buffer, maximum, milliseconds: performance.now() - start}, [field.buffer]);
  } catch (error) {
    postMessage({type: 'error', message: error.message});
  }
};
