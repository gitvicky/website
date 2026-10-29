/* Pure numerical kernel; shared by the Worker and the independent parity test. */
(function (root) {
  'use strict';
  function chebyshev(x, degree) {
    const out = new Float64Array(degree + 1);
    out[0] = 1;
    if (degree) out[1] = x;
    for (let k = 2; k <= degree; k++) out[k] = 2 * x * out[k - 1] - out[k - 2];
    return out;
  }
  function evaluate(manifest, mean, basis, values) {
    const point = values.map((v, i) => {
      const p = manifest.parameters[i];
      if (!Number.isFinite(v) || v < p.min || v > p.max) throw Error('Parameter outside exported domain');
      return 2 * (v - p.min) / (p.max - p.min) - 1;
    });
    if (point.length !== 2) throw Error('Expected two parameters');
    const a = chebyshev(point[0], manifest.degree), b = chebyshev(point[1], manifest.degree);
    const weights = new Float64Array(manifest.rank);
    for (let i = 0; i <= manifest.degree; i++) {
      for (let j = 0; j <= manifest.degree; j++) {
        const c = manifest.coefficients[i][j], w = a[i] * b[j];
        for (let r = 0; r < weights.length; r++) weights[r] += w * c[r];
      }
    }
    const size = mean.length, accumulated = new Float64Array(mean);
    // Traverse each spatial mode contiguously to avoid strided reads across
    // the entire model on every output component. Preserve FP64 summation.
    for (let r = 0; r < weights.length; r++) {
      const weight = weights[r], offset = r * size;
      for (let i = 0; i < size; i++) accumulated[i] += weight * basis[offset + i];
    }
    return new Float32Array(accumulated);
  }
  root.MastuResponse = {evaluate, chebyshev};
})(globalThis);
