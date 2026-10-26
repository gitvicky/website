---
title: "Bounding the Unpredictable — verification report"
---

Date: 26 October 2026. Scope: the new interactive article and its companion numerical modules.

## Guarantee audit

**No browser calculation is machine-certified.** All `BoundResult` objects have `certified: false`, including point domains, analytic-form reference ranges evaluated numerically, and positive transformer margins. Every experiment displays the floating-point limitation. A failed logit inequality is `Inconclusive`; a successful numerical inequality is `Positive numerical margin — illustrative`.

The paper-and-pencil formulas are inclusions under real arithmetic with a nonempty finite domain and the stated fixed model. Taylor and Chebyshev remainders bound mathematical approximation/truncation error. They do not bound floating-point coefficient, transcendental, summation or endpoint error. ReLU region enumeration is exact as a real-arithmetic procedure; its browser execution is still floating point.

A pair of enclosures that should intersect mathematically can be disjoint at a point domain due to cancellation. The implementation falls back to their hull (or the original natural product enclosure) instead of treating an inverted interval as meaningful. McCormick point slices similarly take the hull of computed edges if they invert. These robustness fallbacks do not establish directed-rounding soundness.

Samples are used only in regression diagnostics and drawings. They are never used to justify a certificate, choose an approximation-error bound, or prune a verification region.

## Mathematical checks

`node --test Blog/assets/bounding-the-unpredictable/tests/math.test.js`: nine test groups pass. They cover:

- Known scalar ranges, equivalent expression forms, derivative bounds and subdivision over 150 deterministic generated domains.
- Degenerate intervals, invalid/empty domains, division across zero, overflow, invalid subdivision budgets and invalid attention temperatures.
- Shared-symbol cancellation, multiplication/squaring, repeated affine maps, and constrained/nonlinear geometry.
- Taylor models at orders 1, 2, 4 and 8 with centred/endpoint expansions, including exp(x(1-x)) and degenerate domains.
- Bernstein coefficients/subdivision and Chebyshev analytic-remainder containment diagnostics.
- McCormick inequalities, refinement coverage and monotone refinement widths for the supported example.
- Every network method for both depths and activations, including three relaxation slopes.
- All attention stages and output margins across sequences, perturbed positions, radii, temperatures and three engines.
- Zero-width metric semantics and positive/inconclusive status handling.

`python3 Blog/assets/bounding-the-unpredictable/tests/reference.py`: **88,776 independent containment checks pass**, using 60-digit Decimal arithmetic and an explicit diagnostic tolerance of 1e-9 relative to max(1, |value|). Reference formulas and model weights are independently written; softmax uses shifted exponentials instead of the JavaScript sigmoid expression. Coverage includes polynomial residuals, network outputs, and attention Q/K/V, scores, probabilities, value aggregation and logits at corners and seeded interior points. This is a finite regression suite, not formal verification.

## Scope and limitations

- All nine experiments compute their displayed results from controls. No precomputed or fabricated robustness outcomes.
- Constrained zonotopes use one analytically eliminable equality. Polynomial zonotopes use the exact quadratic image in the geometry experiment. Neither is a general set library.
- The CROWN-style implementation uses fixed chosen lower slopes and IBP pre-activation bounds. It does not reproduce all of DeepPoly, CROWN-IBP training, or alpha-CROWN optimization.
- LP is an analytic one-dimensional McCormick slice. General SDP, MILP, SMT and star-set solvers are not implemented. Refinement retains every leaf rather than performing branch pruning.
- Chebyshev interpolation is supported for exp, where an explicit derivative remainder is available. Bernstein coefficient conversion can become ill-conditioned at high degree; UI degrees are capped at eight.
- The network is scalar-input and width three. Square activation is a transparent polynomial benchmark, not a claim about commonly trained language models.
- The attention head has no residual/normalization/position/masking stack. It verifies neither a pretrained model nor arbitrary natural-language substitutions. Componentwise logit bounds can lose more dependence than direct margins.
- Input parameter ranges and refinement budgets are capped. Browser runtimes are illustrative and exclude rendering unless explicitly stated.
- The existing site relies on externally loaded fonts and MathJax. The experiments themselves have no external numerical or rendering dependency.

## Reading structure

The article follows a baseline–diagnosis–intervention–comparison progression and presents three core experiments and retains six supplementary experiments in expandable sections. Named dropdowns restore the interval, affine/set, Taylor, polynomial, relaxation, solver, sensitivity and attention mathematics, with each method’s reasoning, tightening mechanism and cost. Randomized smoothing and conformal prediction remain separate guarantee families. Extra metrics and attention matrices are also collapsed by default. Numerical kernels and guarantee semantics are unchanged. The research narrative includes two counterexamples checked against those kernels: the centred quadratic Taylor model on [0,1] has bounds [0,0.25] at degree one and [0,0.5] at degree two with Bernstein range evaluation; the one-hidden-layer ReLU network on [-0.2,0.2] gives IBP [-0.09,0.25] and fixed-slope CROWN [-0.09,0.29], to displayed precision. The additional disclosure states the conditions for non-worsening intersections, nested subdivision and added relaxation constraints.

## Site checks

Quarto renders the article and blog listing successfully. The nine experiments mount with one live SVG each. MathJax reports zero equation errors. Browser console inspection found no application errors or warnings.

Browser QA exercised every slider with a keyboard increment, every select option in its supported benchmark, and every experiment reset. The attention demo displayed both a positive numerical margin (dependencies, radius 0.1, default sequence) and Inconclusive (radius 1). Invalid reversed scalar domains display an error and reset restores the experiment. Chebyshev is disabled for the quadratic benchmark.

Responsive checks used viewport overrides of 390 × 844, 768 × 1024 and 1280 × 1000. At each size the document scroll width equals its client width; wide numerical tables scroll within their own keyboard-focusable region. A discovered intrinsic-width overflow was fixed with an explicit article width and mobile padding override. SVGs resize with their containers, controls have labels and keyboard focus styles, and plots are accompanied by numerical tables and line-style legends. This is a desktop browser breakpoint check, not testing on physical phone hardware or a full screen-reader audit.

Generated resources are copied into docs/Blog/assets/bounding-the-unpredictable. The blog card links to the rendered article. Existing search entries are preserved, with additions for this article and report plus the updated blog listing. Pre-existing unrelated workspace changes were retained.

Publication is outside this local implementation run; there is no claim that the public website has been updated.
