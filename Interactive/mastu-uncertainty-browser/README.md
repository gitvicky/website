# Browser-local MAST-U interactive

A static viewer for the existing `gitvicky/website` Quarto site. Visitors need
WebGL2 and JavaScript; no API, Python runtime, GPU server, npm build, or third-party
CDN is used at runtime. The original trained model is approximated offline.

**Phase-one model validation has passed.** The degree-sixteen export used 289
fitting points and passed 173 validation checks (105 regression cases and 68
fresh cases). Worst relative vector L2 error was **0.004095%**; maximum vector
error was **0.07642 mm**, below the unchanged 0.1 mm limit. These are empirical
comparisons with the original network at display vertices.

The model assets are in `web/assets/`. Scientific evaluation passed in GPU job
**39600776**. That job then exposed a packaging-directory bug, which has been
corrected locally; regenerating the archive does not require another GPU run.
The failed degree-four and degree-eight reports are retained under
`Expts/data_cache/mastu_browser/validation_degree{4,8}_job*.json`.

## Generate the validated model

From the repository root:

```bash
sbatch --output=Expts/ablations/logs/mastu_browser_%j.out Interactive/mastu/export.slurm \
  --regression-report Expts/data_cache/mastu_browser/validation_degree8_job39600119.json
```

The script restores `mild-python` and its saved normalizers, uses the matched
70-case corpus's parameter ranges, and accumulates all 29,183,754 native source
nodes before applying the global operator. It compares the cached encoding at
the reference parameters with the saved full-native latent state. A mismatch
above `2e-4` relative L2 aborts the export.

The display is a material-separated simplification of the existing native
cutaway: approximately 120,000 triangles / 64,000 vertices. It is a display
approximation; it does not preserve all native geometry or serve as a volume
integration mesh. The exact network is evaluated at its display vertices,
including the trained material-specific bypass and nonlinear readout.

Two hundred eighty-nine tensor Chebyshev fitting points (degree sixteen in each parameter)
are compressed into shared spatial vector modes. The delivered mean and modes
are float32. The coefficient polynomials and browser accumulation use float64;
the output vectors are float32. Parameter domain limits come from the actual
corpus, in K^-1 and K. No extrapolation is permitted. Thermal expansion is shown
as multiples of 10^-6 K^-1 for readability.

The delivered response model passed **173 checks**: all 105 cases from the
previous attempts, plus **64 fresh seeded interior points and four fresh off-grid
edge points**. The fresh seed is `20261030`, separate from the `20261028` and
`20261029` sets used to select the higher degree. Acceptance remains:

- maximum relative vector L2 error <= 0.5%, and
- maximum per-vertex Euclidean vector error <= 0.1 mm.

These are empirical comparisons with the network over the **display vertices**.
They are not a uniform certificate, MOOSE error bound, statistical confidence
interval, or full-volume maximum. A failed export writes diagnostics under
`Expts/data_cache/mastu_browser/validation.json` and does not create the public
`manifest.json`. Increase `--degree` if needed; keep independent validation
points distinct from fitting points. Selecting degree based on these checks
makes them validation data; reserve an additional test set for publication.

Successful output is `web/assets/{manifest.json,validation.json,*.bin}`. The
manifest records dimensions and SHA-256 checksums. Large arrays are split
losslessly into files of at most 32 MiB for ordinary GitHub hosting. The current
174-mode model requires approximately 133 MiB of initial assets; subsequent
slider updates run locally. Commit the extracted files, rather than the ZIP. A successful export also
creates `Interactive/mastu/mastu-browser.zip`, containing the viewer, validated
model, Quarto page and this guide. The report identifies the
checkpoint, preprocessing, source geometry, native inference and parameter
corpus. Browser loading verifies asset sizes and, in secure contexts, checksums.

## Preview and host on the Quarto website

After successful export:

```bash
python -m http.server 8080 --directory Interactive/mastu/web
```

Visit `http://localhost:8080`. Files must be served by HTTP(S), not `file://`.
The controls stay disabled if the validated model is missing or corrupt.

In a checkout of `https://github.com/gitvicky/website`, copy this `web/` directory
to `Blog/assets/mastu/`, and copy `mastu.qmd` to `Blog/mastu.qmd`. Add the following
entry to the existing `project.resources` list in `_quarto.yml`:

```yaml
    - Blog/assets/mastu/**
```

Run `quarto render`. The existing site publishes `docs/`, so confirm the complete
`docs/Blog/assets/mastu/` directory is included in the generated output. Link to
`Blog/mastu.html` or use the full standalone interactive at
`Blog/assets/mastu/index.html`. The supplied Quarto page embeds the viewer with
`?embed=1`; the full page contains explanation and the validation report link.
Commit and publish with the website's established workflow.

The viewer reconstructs a new displacement vector field in a Web Worker on each
slider update and submits it to WebGL2. At most one prediction is active; stale
responses are discarded and the newest requested parameters run next. Camera
motion remains independent of inference. The displayed timing is measured on
the visitor's hardware; it includes messaging and drawing submission, not GPU
completion. The fixed colour scale is determined from fitting snapshots;
values above it saturate and trigger an explicit notice. Shape amplification
changes geometry only; displacement colours and reported values retain mm.

## Parameter uncertainty interactive

`web/uncertainty.html` provides a second browser-local interactive, using the
original model's full-native outward-rounded Taylor certificate. It does not
reuse enclosures for the phase-one compact approximation.

Controls vary the thermal-expansion and wall-temperature centres and their
uncertainty half-widths within the paper's parent rectangle:

- nominal thermal expansion: `1.4328253200801555e-5 K^-1`, maximum half-width `1e-7 K^-1`;
- nominal wall temperature: `863.978759765625 K`, maximum half-width `5 K`.

The numerical physical domain is rounded inward from the exact stored centre
plus/minus half-width. Moving a centre reduces its allowed half-width; requests
outside certified coverage are rejected by the worker as well as constrained
by the UI. Colour modes show lower magnitude, upper magnitude, enclosure width,
or the nominal Taylor estimate. Fixed scales allow comparison between boxes.

The offline exporter checks cached native Taylor states against their archived
SHA-256 values and verifies the saved weights, corpus, captured kernel tables
and query geometry. It exports outward binary64 quadratic coefficient intervals
and a rigorous remainder at all 64,018 material-routed display vertices. Every
packet is checked against `ValidatedNativeDecoder` over the parent rectangle.
No new global encoding or surrogate retraining is required.

Online, interval addition, multiplication, division and squaring pad every
binary64 primitive outward. Square-root endpoints are verified with enclosing
squares, including scaled handling of subnormals. Normalization includes the
frozen calibration, exact-real MinMax denominator and legacy FP32 endpoint map,
intersected with the independently known parent encoded rectangle. Query
locations and material IDs remain fixed.

Bounds enclose the original stored-weight mathematical neural operator at the
**display vertices**, not all points in each triangle. Interpolated colours do
not assert continuous spatial coverage. They do not bound MOOSE error, provide
probabilistic confidence intervals, certify every original FP32/TF32 execution,
or certify the separate phase-one approximation. The nominal view is a Taylor
estimate. The parent-box Taylor remainder is retained for nested boxes, so
width need not vanish at zero input half-width. This is conservative and may be
wider than a fresh offline certificate for the smaller box.

Reproduce the offline packet with:

```bash
sbatch --output=Expts/ablations/logs/mastu_browser_uq_%j.out \
  Interactive/mastu/export_uncertainty.slurm
```

It reuses `tmp/mastu_interval_measure_vveysn_m/after/`, the completed certificate
whose source report hash also appears in the paper figure provenance. Outputs
are `web/assets/uq-{manifest,validation}.json` and the binary64 coefficient and
remainder files. Source scientific caches and manuscript figures are unchanged.

After browser checks pass, build the uncertainty-only deployment archive:

```bash
.venv/bin/python -m Interactive.mastu.package --uncertainty-only
```

Extract `mastu-uncertainty-browser.zip` and merge `web/` into the existing
website's `Blog/assets/mastu/`. Keep the existing Quarto resource entry
`Blog/assets/mastu/**`; render and publish the site as before. Add this alongside
the publication's existing Paper / Blog / Interactive links in `research.qmd`:

```html
<a href="Blog/assets/mastu/uncertainty.html" class="btn-blog"
   target="_blank" rel="noopener">Parameter uncertainty</a>
```

The direct URL is `/Blog/assets/mastu/uncertainty.html`. The separate nominal
interactive remains at `/Blog/assets/mastu/index.html`. The optional
`uncertainty.qmd` embeds the second page if a Quarto wrapper is preferred.
The uncertainty-only ZIP includes the shared geometry and fonts but omits the
large phase-one response basis. It is intended to be merged with the existing
viewer directory; its footer links back to that nominal interactive.

The scientific legend and parameter symbols use the bundled Latin Modern Roman
OpenType faces, located with `kpsewhich`; a missing face stops loading clearly.
The font's GUST licence is included. No PDF figures are modified.

## Checks

```bash
.venv/bin/python -m unittest discover -s Interactive/mastu/tests -v
```

Install `quickjs` in a temporary test location and add it to `PYTHONPATH` to also
run the JavaScript parity check. Tests cover independent polynomial fields,
constant fields, out-of-domain inputs, physical error units, float32 export
parity and JavaScript syntax. The separate GPU export validates against the
actual saved neural operator; synthetic tests do not replace it.

## References

- Website layout and output: [website source](https://github.com/gitvicky/website/blob/main/_quarto.yml).
- Browser-local scientific interaction inspiration: [An atlas of 2D crystals](https://whitead.github.io/symdjs/).
- Requested visual reference: [Introducing Nova](https://zenithon.ai/research/introducing-nova).
  That page returned HTTP 403 during inspection, so no claims about its
  implementation or performance are made here.

Browser QA also passed in headless Chromium on the actual display mesh with a
clearly labelled **synthetic software test field**: WebGL2, Worker updates, both
sliders, rapid input changes, component selection, shape amplification, parameter
reset, desktop/mobile layout, iframe embedding and missing-export handling.
This verifies the application; actual-model scientific validation remains the
separate GPU export job. To repeat QA with Playwright in an isolated environment:

```bash
.venv/bin/python Interactive/mastu/tests/browser_check.py
```

The browser check creates and removes its own local HTTP server and temporary
fixture; it does not write a synthetic model into `web/assets/`.

For the delivered validated model, run the same browser checks with `--real`:

```bash
.venv/bin/python Interactive/mastu/tests/browser_check.py --real
```

This also compares sampled actual Worker outputs against NumPy reconstruction
from the distributed model files. The basis is traversed contiguously for
responsive updates while retaining float64 accumulation and float32 output.

The delivered-model desktop/mobile browser checks passed, including actual
Worker/NumPy agreement at 193 sampled components. On the headless test host,
a reference update took about 603 ms (355 ms prediction); visitor performance
will depend on device hardware. Camera interaction remains independent of the
prediction Worker. A separate parameter-uncertainty page is now available; its original-model target and certified coverage are documented above.

The uncertainty export completed as GPU job **39605188**. All 64,018 query
packets passed the native decoder comparison. Twelve numerical tests passed,
and the actual-model browser passed desktop/mobile, mode switching, nested and
zero-width boxes, endpoint coverage, iframe and missing-asset checks. Browser
bounds matched the independent interval reference at 193 actual vertices; the
nominal estimate was contained at every vertex. Initial bound computation took
about 183 ms on the headless test host; performance depends on hardware and
load. The uncertainty page downloads about 23 MB of geometry and coefficient
data, without the phase-one 174-mode basis. Nothing has been published.
