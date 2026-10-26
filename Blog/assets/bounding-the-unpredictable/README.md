# Bounding the Unpredictable

Companion code for `Blog/bounding_the_unpredictable.qmd`. The article follows the existing Quarto theme and navigation; its nine SVG experiments need only a modern browser with ES modules. No npm installation or remote model service is required.

## Run

From the website repository root:

```sh
node --test Blog/assets/bounding-the-unpredictable/tests/math.test.js
python3 Blog/assets/bounding-the-unpredictable/tests/reference.py
quarto render Blog/bounding_the_unpredictable.qmd --no-clean
quarto render Blog/blog.qmd --no-clean
quarto render Blog/assets/bounding-the-unpredictable/VERIFICATION.md --no-clean
python3 -m http.server 8765 --directory docs --bind 127.0.0.1
```

Open `http://127.0.0.1:8765/Blog/bounding_the_unpredictable.html`. Serve over HTTP; opening an ES-module page directly with `file://` is not supported. Quarto renders into the existing `docs/` GitHub Pages output. Article resources are declared in its YAML so transitive module imports are copied. Rendering is local; publishing still follows the repository's normal Git/deployment workflow.

The reading path follows a research investigation: establish a baseline, diagnose lost information, test a targeted change, and compare tightness with cost. Reproducible Taylor-degree and fixed-slope CROWN counterexamples show that a more elaborate method need not give a tighter enclosure. A separate disclosure distinguishes guaranteed monotone improvements from hypotheses. It centres on three experiments: dependency, the small network and attention. Six supplementary experiments and the full mathematical derivations are collapsed by default. Named dropdowns explain each method’s retained information, potential improvement in bounds, and computational trade-offs, including conceptual solver formulations and separate statistical guarantees. Output intervals and widths are shown first; additional metrics and internal matrices can be expanded.

## Modules

- `math/interval.js`: checked finite intervals, arithmetic, analytic scalar extrema, mean-value forms, partitions and the shared result interface.
- `math/affine.js`: shared-symbol arithmetic, affine transforms, conservative products, square/ReLU residuals and zonotope geometry. The constrained geometry example eliminates the equality ε₂ = ε₁ analytically.
- `math/polynomial.js`: coefficient arithmetic, power-to-Bernstein conversion, model multiplication with discarded-term bounds, Taylor models, nonlinear exponential composition and Chebyshev interpolation of exp with an analytic remainder.
- `math/relaxations.js`: ReLU/square lines, McCormick envelopes, an analytically optimized dependent bilinear slice, and complete-cover best-first subdivision.
- `math/neural.js`: documented fixed weights, forward evaluation, IBP, affine/zonotope propagation, backward CROWN-style substitution, polynomial propagation, exact-form ReLU region enumeration, Lipschitz and local Jacobian bounds.
- `math/transformer.js`: fixed embeddings/projections, two-token attention, interval and dependency-aware engines, row-sum-aware aggregation, 16-box refinement and logit margin calculation.
- `main.js`: accessible controls, independent reset handlers, SVG figures, numerical tables, explanations and statuses. Numerical kernels do not depend on the DOM.
- `tests/math.test.js`: Node's built-in test runner; deterministic property diagnostics and edge cases.
- `tests/reference.py`: independent 60-digit Decimal calculations, with separately written model weights and forward expressions. It obtains only browser-kernel outputs from `reference-fixtures.js`.

## Result contract

Every displayed method uses or derives from:

```ts
type BoundResult = {
  method: string;
  lower: number[];
  upper: number[];
  certified: boolean; // always false in this implementation
  assumptions: string[];
  computationMs: number;
  operations?: number;
  diagnostic?: string;
};
```

Lower/upper arrays are componentwise. The transformer shows each logit's width separately. Scalar excess and width ratios appear only when an analytic range is available; ratios are undefined for zero true width. Network timings cover each bounding method; transformer timing includes its nominal forward pass. Counts are domain-bound evaluations, not counts of arithmetic instructions.

All formulas target real arithmetic. JavaScript does not perform directed rounding here. Analytic truncation remainders do **not** account for coefficient error or floating-point evaluation error. The Python tolerance is diagnostic, not added padding that makes bounds rigorous. See `VERIFICATION.md`.

## Supported reduced experiments

1. Quadratic range versus natural intervals.
2. Three dependency examples, equivalent expressions, mean-value and subdivided intervals.
3. Two-dimensional affine and quadratic geometry; optional equal-factor input constraint. The nonlinear boundary retains polynomial factors analytically; the polygon is an affine enclosure.
4. Quadratic, exponential, sine, and exp(x(1-x)) Taylor models; centres, degrees and subdivision.
5. Taylor plus interval or Bernstein range evaluation; Chebyshev plus analytic interpolation remainder for exp only. The quadratic comparison has no approximation remainder for degree ≥ 2.
6. ReLU supporting lines and objective direction; bilinear McCormick slice; a computed small-network comparison.
7. Complete-cover subdivision using natural intervals or an analytic McCormick LP slice. No pruning, general LP solver, or SDP backend.
8. Scalar-input, width-three networks with one or two hidden layers and ReLU/square activations. Affine arithmetic and zonotopes share a row because they share an implementation. Piecewise enumeration is used for ReLU; polynomial/Bernstein for square.
9. Two embeddings of dimension two; four artificial output tokens; no training, residuals, masking, normalization or positional encodings. A selected input token varies in a two-dimensional L∞ box. The three engines are interval, affine/McCormick/score-difference/simplex, and a 4 × 4 domain cover using the second engine.

The weights and embeddings are documented in the numerical modules and exported as `MODEL`, `WEIGHTS` and `EMBEDDINGS`. They are decimal constants interpreted in binary64 by JavaScript.

## Extend

Add a kernel and containment/reference checks before registering a new control or method. Preserve input-factor identities through dependency-aware operations. Include every truncated term in a model remainder. Keep a region cover complete during refinement. Reject unsupported or nonfinite inputs instead of displaying stale results. Do not set `certified: true` without an end-to-end validated arithmetic/optimization implementation and its documented assumptions.

Potential extensions are a validated WebAssembly arithmetic backend; per-neuron optimized CROWN slopes; direct logit-margin propagation; general constrained/polynomial zonotope operations; verified solver proof output; and larger attention blocks. These are not implemented or required to operate the reduced examples on this page.
