/* The explorer explains recorded work; it never runs or approximates a plasma solver. */
(() => {
  'use strict';
  const root = document.getElementById('tk-explorer');
  if (!root) return;
  const $ = id => root.querySelector(`#${id}`);
  const evidence = globalThis.TK_EXPLORER_DATA;
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const topics = [
    {id:'shape', title:'What holds the plasma?', x:50, y:10, input:'Magnet settings and the pressure of the plasma.', solve:'Find a magnetic shape that balances the plasma pressure.', output:'A map of magnetic surfaces: the geometry used by the other models.', link:'technical/backends.html#equilibrium-code',
      mastu:{state:'done', status:'Equilibrium computed', work:'A free-boundary solver computes a MAST-U-like plasma shape. A common data adapter carries that shape into the toolkit.', limit:'The static equilibrium route exists. Two-way feedback between this solver and the evolving transport loop is planned.', claims:['C-019','C-020']},
      iter:{state:'done', status:'Prescribed geometry in use', work:'The ITER benchmark reads an existing magnetic equilibrium. The rest of the simulation uses its magnetic surfaces to organise the plasma.', limit:'The benchmark uses prescribed geometry; this is not a live free-boundary equilibrium solve.', link:'technical/backends.html#transport-code', claims:[]}},
    {id:'heating', title:'What puts energy in?', x:18, y:31, input:'Heating settings, local temperature, density and magnetic geometry.', solve:'Work out where energy is deposited and how it drives electric current.', output:'Heating and current profiles, passed to the plasma evolution model.', link:'technical/physics-models.html#gaussian-rf',
      mastu:{state:'limited', status:'Prescribed source models', work:'Analytic heating profiles supply energy to model scenarios. Their totals are normalised on the simulation grid.', limit:'The ITER electron-cyclotron heating surrogate is not established for MAST-U.', claims:[]},
      iter:{state:'done', status:'Guarded heating coupled', work:'A fast electron-cyclotron heating model runs inside the transport loop, with a simpler heating profile available as fallback.', limit:'Its evidence is restricted to the studied equilibrium and launch settings. Short coupled checks are not whole-machine validation.', link:'technical/surrogates.html', claims:['C-168','C-171']}},
    {id:'transport', title:'Where does energy go?', x:82, y:31, input:'Temperature and density gradients, together with magnetic geometry.', solve:'Estimate how turbulence moves heat and particles between neighbouring plasma layers.', output:'Transport coefficients: the rates used to evolve temperature and density.', link:'technical/surrogates.html',
      mastu:{state:'limited', status:'Model built; accuracy limit missed', work:'A trained ensemble provides fast turbulence responses. A guard selects a fallback where the learned response cannot be used.', limit:'The recorded closed-loop comparison missed the accuracy target. Later reference comparison was not evaluable; the limitation remains.', claims:['C-179']},
      iter:{state:'done', status:'Guarded transport integrated', work:'An ensemble replaces expensive turbulence calls within its allowed conditions. The coupled model returns evolving plasma profiles.', limit:'The saved comparison and the current continuous-closure convergence study have different scopes. Neither certifies arbitrary ITER conditions.', claims:['C-178','C-305']}},
    {id:'edge', title:'What happens at the edge?', x:18, y:65, input:'Plasma state and the chosen edge conditions.', solve:'Set the outer pressure and temperature boundary for the hotter core.', output:'Boundary values that constrain the core evolution.', link:'technical/coupling-loop.html',
      mastu:{state:'planned', status:'Prescribed boundary; prediction deferred', work:'The MAST-U route keeps a prescribed edge boundary so other parts of the modelling can be studied.', limit:'A predictive MAST-U pedestal model is deferred. The ITER pedestal model is not a substitute for one.', claims:[]},
      iter:{state:'done', status:'Guarded pedestal coupled', work:'A learned pedestal model sets the edge conditions inside the transport loop. A fallback handles inputs outside its contract.', limit:'Coupled checks establish the implementation within their scope. Pedestal and equilibrium physical discrepancy remain unquantified.', link:'technical/verification.html', claims:['C-173']}},
    {id:'guard', title:'When is the surrogate used?', x:82, y:65, input:'The model input, the ensemble predictions and the declared limits.', solve:'Check allowed inputs, model agreement and physical output conditions.', output:'An accepted fast response or a fallback response, plus a record of that choice.', link:'technical/guard.html',
      mastu:{state:'limited', status:'Guard active; limits retained', work:'Guarded execution keeps model limits explicit. Verification records preserve the failed and non-evaluable comparisons.', limit:'A guard cannot make an inaccurate model accurate. MAST-U closed-loop limits remain part of the accepted scope.', claims:['C-179']},
      iter:{state:'limited', status:'Verification is bounded', work:'Verification asks which properties hold over stated input regions. The packaged model separates proved, unresolved and out-of-domain cases.', limit:'Ordinary guard acceptance does not imply a formal certificate. Local robustness is proved for only part of the selected state set.', link:'technical/verification.html', claims:['C-326','C-284']}},
    {id:'stability', title:'Could the plasma become unstable?', x:50, y:89, input:'A saved magnetic equilibrium and plasma pressure/current profiles.', solve:'In a separate calculation, test whether small disturbances grow.', output:'Growth and mode-location diagnostics for a stability study.', link:'technical/jorek.html',
      mastu:{state:'planned', status:'Outside the demonstrated MAST-U route', work:'The current documented stability branch is an ITER study. The map shows where a stability calculation would connect conceptually.', limit:'No completed MAST-U stability calculation or stability feedback is claimed here.', claims:[]},
      iter:{state:'limited', status:'Offline study; grid check incomplete', work:'Saved ITER states feed a separate stability code. Time-step checks passed for the studied modes.', limit:'The finer-grid check remains incomplete. No fine-grid stability verdict, trained stability surrogate or feedback into the core loop is established.', claims:[]}},
    {id:'system', title:'How are the profiles updated?', x:50, y:49, input:'Geometry, heating, transport rates and edge conditions.', solve:'Advance the plasma profiles together, checking numerical residuals and recording model choices.', output:'Updated profiles feed the next step; saved results support checks, comparisons and uncertainty studies.', link:'technical/coupling-loop.html',
      mastu:{state:'limited', status:'Components connected; key gaps remain', work:'Common data interfaces connect equilibrium and transport components. The work exposes which links exist and which still need evidence.', limit:'Static equilibrium is not yet a two-way live coupling. Accuracy and reference limitations prevent a full-system validation claim.', claims:['C-020','C-179']},
      iter:{state:'done', status:'Numerical convergence gate passed', work:'The continuous-closure ensemble reaches the tested endpoint across all declared maximum-step settings. The coupled model now has a passing numerical baseline.', limit:'Numerical convergence asks whether time refinement changes the answer. It does not establish experimental accuracy. Full current-baseline uncertainty propagation remains incomplete.', link:'technical/solver-learning.html#full-ensemble-convergence', claims:['C-305','C-306','C-327']}}
  ].filter(topic => topic.id !== 'stability');
  const machines = {
    mastu:{name:'MAST-U', summary:'The MAST-U work starts with a computed magnetic equilibrium and uses it in transport modelling. The geometry is static during the transport calculation. These are simulated cases, not reconstructions of experimental shots.', impactTitle:'What the MAST-U work provides', impact:'The equilibrium and data interfaces provide a starting point for coupled MAST-U modelling. The transport comparison also identifies a problem: the surrogate did not meet the accuracy target in the recorded coupled run. Improving that accuracy and adding live equilibrium feedback remain necessary before this can support a validated evolving-plasma calculation.'},
    iter:{name:'ITER', summary:'The ITER benchmark connects heating, turbulent transport and edge conditions through a core plasma solver. It uses a prescribed magnetic equilibrium. The diagram shows what each module receives and returns during a plasma update.', impactTitle:'What the ITER work provides', impact:'The coupled ITER model now passes the stated time-step refinement check across the tested ensemble. This gives a numerical baseline for studying the effect of the surrogate models inside the simulation. Experimental accuracy and complete uncertainty propagation still need to be assessed.'}
  };
  const scope = {
    mastu: [
      ['Starting point', 'A simulated MAST-U-like free-boundary equilibrium from FreeGSNKE, imported through IMAS. These cases are not experimental shot reconstructions.'],
      ['Evolving quantities', 'Transport calculations update plasma temperature and density profiles. The imported magnetic geometry stays static; this is not a demonstrated live equilibrium–transport feedback loop.'],
      ['Prescribed inputs', 'Magnetic geometry, analytic heating sources and edge boundary settings. The transport surrogate is guarded, but the recorded coupled comparison missed its accuracy target.'],
      ['Outside the demonstrated scope', 'Predictive pedestal modelling, transfer of the ITER EC surrogate, live equilibrium feedback, and a completed MAST-U stability calculation.']
    ],
    iter: [
      ['Starting point', 'A core-transport benchmark with a prescribed CHEASE equilibrium and configured initial profiles. This is a modelling scenario, not an experimental reconstruction.'],
      ['Evolving quantities', 'TORAX advances the enabled electron/ion temperature, density and current-diffusion channels. Evolving the poloidal-flux channel does not imply a new free-boundary equilibrium solve.'],
      ['Coupled responses', 'Guarded EC heating/current drive, turbulent transport and pedestal conditions feed the core equations. The passing transport baseline uses the frozen continuous closure.'],
      ['Outside the demonstrated scope', 'Live free-boundary geometry updates, online stability feedback, experimental accuracy and complete uncertainty propagation for the current baseline.']
    ]
  };
  const interfaces = {
    shape: {
      mastu: ['Machine and coil description, with prescribed equilibrium pressure/current inputs.', 'Compute a static free-boundary equilibrium with FreeGSNKE and import it through IMAS.', 'Magnetic surfaces and geometry mapped onto the transport grid; held static during evolution.'],
      iter: ['A prescribed CHEASE equilibrium for the benchmark.', 'Read and map the equilibrium geometry; no live free-boundary solve occurs.', 'Flux-surface geometry and metric factors supplied to TORAX.']
    },
    heating: {
      mastu: ['Prescribed source powers, deposition shapes and simulation-grid volume weights.', 'Construct analytic heating profiles and normalise their integrals.', 'Volumetric heat sources for the configured scenario; no established MAST-U EC surrogate.'],
      iter: ['Local temperature, density, magnetic geometry and EC launch settings.', 'Evaluate the guarded structured EC response, or its prescribed fallback, and reconstruct deposition.', 'Cell-integrated deposition profiles and current drive, combined with other enabled sources.']
    },
    edge: {
      mastu: ['Prescribed edge location, temperature and density settings.', 'Apply the scenario boundary; no predictive pedestal response is demonstrated.', 'Outer-core boundary conditions for the transport calculation.'],
      iter: ['Pedestal-model inputs, plasma state, composition and temperature-ratio assumptions.', 'Evaluate the guarded EPEDNN response or fallback; map pedestal pressure and width to core boundary quantities.', 'Pedestal-top location and boundary temperature/density constraints.']
    }
  };
  const names = {shape:'Equilibrium', heating:'Heating and current drive', transport:'Turbulent transport', edge:'Pedestal and boundary', guard:'Guard and fallback', stability:'Offline stability', system:'Core transport solver'};
  const physics = {
    shape: {
      equation:'j × B = ∇p; &nbsp; ∇ · B = 0',
      meaning:'Magnetic force balances the pressure gradient in a static equilibrium. The axisymmetric problem is solved through the Grad–Shafranov equation; it is distinct from evolving heat and particles.',
      quantities:'The equilibrium supplies poloidal flux ψ, safety factor q, flux-surface volume derivative V′ = dV/dρ and metric factors. The transport mesh uses a normalised toroidal-flux radius ρ. Mapping this coordinate to geometric radius is part of the interface, not an interchangeable choice.',
      coupling:'The intended feedback is profiles → pressure/current → equilibrium → transport geometry → new profiles. For MAST-U, FreeGSNKE supplies a static free-boundary solution through IMAS; its live two-way coupling is planned. ITER uses a prescribed CHEASE equilibrium.',
      link:'technical/backends.html#equilibrium-code'
    },
    heating: {
      equation:'P<sub>abs</sub> = ∫ Q dV; &nbsp; I<sub>CD</sub> = ∫ j<sub>CD</sub> dA',
      meaning:'A deposition profile must conserve the total it represents. Volumetric heating Q is integrated with volume weights; current density is integrated with the area measure required by its current convention.',
      quantities:'Electron and ion heating Qe and Qi have units W m⁻³; particle sources have units m⁻³ s⁻¹. Profiles are stored on cells. The ITER structured EC response predicts absorbed power, driven current and deposition moments, then reconstructs cell-integrated profiles.',
      coupling:'Density, temperature and magnetic geometry affect deposition. Heating changes those profiles on the next solve. EC current drive also contributes to current evolution. The Gaussian fallback preserves its prescribed integral; it is not a new ray-tracing calculation.',
      link:'technical/surrogates.html#ec-structured'
    },
    transport: {
      equation:'R/L<sub>X</sub> = −R d(ln X)/dr; &nbsp; χ<sub>e,i</sub>, D<sub>e</sub> [m² s⁻¹], v<sub>e</sub> [m s⁻¹]',
      meaning:'Turbulent fluxes depend on local gradients and magnetic geometry. The radius used in a gradient must match the model convention; a flux coordinate and a midplane distance are not automatically equivalent.',
      quantities:'Features include normalised gradients, safety factor, magnetic shear, collisionality and temperature ratio as required by the selected model. The evolving temperatures and density live at cell centres; diffusivities and convection live on faces between cells.',
      coupling:'This creates a nonlinear dependency: the profiles determine transport, while transport determines the profiles. The learned TGLF response is evaluated inside that coupling, with an explicit fallback. A held-out flux error, a coupled profile error and numerical convergence answer different questions.',
      link:'technical/surrogates.html'
    },
    edge: {
      equation:'p = e(n<sub>e</sub>T<sub>e</sub> + Σ<sub>i</sub> n<sub>i</sub>T<sub>i</sub>) &nbsp; for T in eV',
      meaning:'The pedestal is the narrow edge region with a steep pressure gradient. Its top conditions supply a boundary to the core transport calculation. Pressure alone does not determine both temperature and density.',
      quantities:'An interface must map predicted pedestal pressure and width to a radial location, density and electron/ion temperatures using declared composition and temperature-ratio assumptions. Temperatures in eV require the elementary-charge conversion for pressure in pascals.',
      coupling:'ITER uses the ported EPEDNN response within its allowed inputs and a prescribed boundary fallback. MAST-U retains prescribed edge conditions. This boundary treatment is not a resolved scrape-off-layer/divertor model and does not itself simulate an edge-localised-mode crash.',
      link:'technical/surrogates.html#pedestal-in-torax'
    },
    guard: {
      equation:'accept = (x ∈ D) ∧ (σ ≤ τ) ∧ (y ∈ Y) ∧ finite(y, σ)',
      meaning:'D is the declared input region, σ is ensemble spread, τ its threshold, and Y the allowed output set. This is the conceptual ensemble guard. A single-network route has no ensemble-spread test.',
      quantities:'The guard returns a served response and a decision mask at each evaluation point. Ordinary guard margins do not enlarge a formal proof box. Ensemble spread is a disagreement signal, not automatically a calibrated confidence interval.',
      coupling:'A hard switch can make the nonlinear transport problem nonsmooth. The passing ITER baseline instead uses its explicitly frozen continuous closure; it is a changed closure, not evidence that earlier hard-switch failures passed. Accepted-state masks also do not count every internal solver evaluation.',
      link:'technical/solver-learning.html#continuous-closure'
    },
    stability: {
      equation:'E<sub>n</sub>(t) ∝ exp(2γ<sub>n</sub>t); &nbsp; γ<sub>n</sub> = ½ d(ln E<sub>n</sub>)/dt',
      meaning:'In a linear growth window, perturbation energy grows at twice the amplitude growth rate. A toroidal mode number n labels the Fourier harmonic; the analysis also checks where the perturbation is localised.',
      quantities:'An adapter maps saved equilibrium and profile data into JOREK units and boundary inputs. Native magnetic/kinetic energies and perturbations support growth-rate and mode-location analysis. The chosen fitting window and unit normalisation are part of the result.',
      coupling:'This is an offline branch from saved ITER states, not a stability response called in the core loop. Time-step refinement and spatial-grid refinement are separate requirements. An incomplete grid study cannot establish a converged stability verdict or a nonlinear crash prediction.',
      link:'technical/jorek.html#adapter'
    },
    system: {
      equation:'F(u<sub>new</sub>; u<sub>old</sub>, Δt) = 0; &nbsp; u = (T<sub>e</sub>, T<sub>i</sub>, n<sub>e</sub>, ψ)',
      meaning:'TORAX advances coupled heat, particle and current-diffusion channels. Implicit stepping requires a nonlinear solve because coefficients and sources depend on the unknown new profiles. A small solver update is not by itself a small equation residual.',
      quantities:'On a fixed geometry, the finite-volume idea is ΔEi/Δt = Pi + Φin − Φout: stored energy changes through sources and fluxes across cell faces. Adjacent face contributions cancel in a global balance. The implementation audits its actual channel equations, geometry factors and source terms.',
      coupling:'The current ITER baseline checks native and physical residuals, rolls back rejected trials, and compares endpoint quantities under maximum-step refinement. Fixed-step differentiation and local implicit-root derivatives have evidence; reverse-mode differentiation through adaptive retry decisions and complete current-baseline uncertainty propagation remain outside the established result.',
      link:'technical/solver-learning.html#full-ensemble-convergence'
    }
  };
  let machine = null;
  let current = 0;
  let physicsOpen = true;
  let flowIndex = 0;
  let flowTimer = null;
  let flowPlaying = false;
  let flowMotion = null;
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const exchanges = [
    {from:'system', to:'system', title:'Start from the accepted plasma', payload:'Te(ρ), Ti(ρ), ne(ρ), ψ(ρ) at time t',
      explanation:'Keep the accepted state unchanged while constructing a trial for the next time step. It supplies the starting profiles for the nonlinear calculation.'},
    {from:'shape', to:'system', title:'Supply the magnetic geometry', payload:'Flux surfaces, q, V′ and geometric factors',
      iter:'Read the prescribed CHEASE equilibrium. These geometric factors tell the transport solver how radial fluxes and cell volumes fit together.',
      mastu:'Import the static FreeGSNKE equilibrium through IMAS. This is an existing geometry input, not a new live free-boundary solve at every step.'},
    {from:'system', to:'heating', title:'Send the plasma conditions to the source model', payload:'Temperature, density, geometry and heating settings',
      explanation:'The heating module receives the plasma conditions and actuator settings needed by the chosen source prescription.'},
    {from:'heating', to:'system', title:'Return heat and current sources', payload:'Electron/ion heating Qe, Qi; configured current drive jCD',
      iter:'The guarded EC response returns deposition profiles and current drive within its contract, with a prescribed fallback. Other enabled sources also enter the core equations.',
      mastu:'Prescribed source profiles supply the modelled heating. The ITER EC surrogate is not transferred to MAST-U by this walkthrough.'},
    {from:'system', to:'transport', title:'Ask how heat and particles move', payload:'Profile gradients, q, magnetic shear and model-specific features',
      explanation:'Evaluate transport using the current trial profiles. A change in temperature or density changes the gradients, so transport must be reevaluated during the solve.'},
    {from:'transport', to:'guard', title:'Check the candidate transport response', payload:'Candidate χe, χi, De, ve; domain and validity information',
      explanation:'Before the response is served, check its input domain and output constraints. Ensemble agreement is available only on ensemble routes.'},
    {from:'guard', to:'system', title:'Return the served transport coefficients', payload:'Guarded face coefficients and model-choice diagnostics',
      iter:'The current passing ITER baseline uses its frozen continuous closure near admissibility boundaries. Rejected candidates contribute no surrogate response; the fallback remains explicit.',
      mastu:'Serve the allowed model response or its fallback. This routing does not remove the recorded MAST-U transport accuracy limitation.'},
    {from:'system', to:'edge', title:'Provide the edge-model inputs', payload:'Plasma state, composition and prescribed edge settings',
      iter:'Supply the state and scenario quantities needed by the pedestal response. Pressure must be mapped to temperature and density with declared assumptions.',
      mastu:'Read the prescribed edge settings for this scenario. A predictive MAST-U pedestal closure remains deferred.'},
    {from:'edge', to:'system', title:'Return the core boundary conditions', payload:'Pedestal-top location, boundary temperature and density',
      iter:'The guarded pedestal response, or its prescribed fallback, constrains the outer core boundary. This is not a resolved divertor calculation.',
      mastu:'The prescribed boundary constrains the core evolution. It does not provide a learned or self-consistent MAST-U pedestal prediction.'},
    {from:'system', to:'system', title:'Solve for a new trial plasma', payload:'Updated trial Te, Ti, ne and ψ; equation residuals',
      explanation:'Combine geometry, sources, transport and boundaries in the implicit equations. Nonlinear iterations revisit the modules as the trial profiles change. A trial is not yet an accepted time update.'},
    {from:'system', to:'system', title:'Check before advancing time', payload:'Residuals, balance, finite profiles and acceptance decision',
      iter:'This illustration follows an accepted proposal: only after the declared gates pass does the new state replace the old one. A rejected proposal instead restores the complete old state and retries under the baseline policy.',
      mastu:'This illustrates an accepted update, not a newly validated MAST-U trajectory. An unconverged trial must not be treated as evidence of a successful plasma prediction.'},
    {from:'system', to:'heating', title:'Feed the updated plasma into the next step', payload:'Accepted profiles at t + Δt become the next inputs',
      explanation:'The new temperatures, density and current profile change the next source and transport evaluations. Repeat the coupled calculation with the scenario geometry and edge treatment; stability analysis remains offline.'}
  ];
  const visited = {mastu:new Set(), iter:new Set()};
  const symbol = {done:'●', limited:'◐', planned:'○'};

  function torus(kind) {
    const hole = kind === 'mastu' ? 16 : 36;
    const width = kind === 'mastu' ? 75 : 91;
    return `<svg viewBox="0 0 220 220" xmlns="http://www.w3.org/2000/svg" fill="none"><ellipse cx="110" cy="111" rx="${width}" ry="70" stroke="var(--ex-line)" stroke-width="1"/><ellipse cx="110" cy="111" rx="${width}" ry="45" stroke="var(--ex-ink)" stroke-width="2"/><ellipse cx="110" cy="111" rx="${hole}" ry="45" stroke="var(--ex-ink)" stroke-width="2"/>${[.25,.5,.75].map(f=>`<ellipse cx="110" cy="111" rx="${hole+(width-hole)*f}" ry="${45+25*Math.sin(f*Math.PI)}" stroke="var(--ex-muted)" opacity=".55"/>`).join('')}<path d="M110 23 V198 M20 111 H200" stroke="var(--ex-line)" stroke-dasharray="3 6"/>${[0,1,2,3,4,5,6,7].map(i=>{const a=i*Math.PI/4,x=110+width*Math.cos(a),y=111+45*Math.sin(a);return `<ellipse cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" rx="9" ry="28" stroke="var(--ex-muted)" opacity=".65" transform="rotate(${i*45},${x.toFixed(1)},${y.toFixed(1)})"/>`;}).join('')}<ellipse cx="110" cy="111" rx="${(width+hole)/2}" ry="46" stroke="var(--ex-ink)" stroke-width="3" opacity=".8"/></svg>`;
  }
  root.querySelectorAll('[data-art]').forEach(el => el.innerHTML = torus(el.dataset.art));
  function claim(id) {
    const c = evidence?.claims[id];
    if (!c) return `<p class="ex-claim"><a href="claims.html#${id.toLowerCase()}">${id}: inspect the recorded claim</a></p>`;
    return `<div class="ex-claim"><a href="claims.html#${id.toLowerCase()}">${id} · ${escape(c.status)}</a><p>${escape(c.display || `${c.value} ${c.unit}`)}</p><small>${escape(c.statement)}</small></div>`;
  }
  function progress() {
    $('ex-progress-text').textContent = `${visited[machine].size} / ${topics.length} topics viewed`;
  }
  function chooseTopic(index, focus = false) {
    current = (index + topics.length) % topics.length;
    const topic = topics[current], detail = topic[machine];
    visited[machine].add(topic.id);
    root.querySelectorAll('.ex-station').forEach((button, i) => {
      button.setAttribute('aria-pressed', String(i === current));
    });
    $('ex-detail-title').textContent = `${names[topic.id]}: ${topic.title}`;
    const theory = physics[topic.id];
    const [input, solve, output] = interfaces[topic.id]?.[machine] || [topic.input, topic.solve, topic.output];
    $('ex-detail-content').innerHTML = `<span class="ex-status ${detail.state}">${escape(machines[machine].name)} · ${symbol[detail.state]} ${escape(detail.status)}</span><p>${escape(detail.work)}</p><div class="ex-flow"><div><b>Inputs</b><span>${escape(input)}</span></div><div><b>Calculation</b><span>${escape(solve)}</span></div><div><b>Outputs</b><span>${escape(output)}</span></div></div><p class="ex-limit"><strong>Current limitations</strong><br>${escape(detail.limit)}</p><details class="ex-physics"><summary>Physics: equations and coupling</summary><p class="ex-equation">${theory.equation}</p><p>${escape(theory.meaning)}</p><h5>Quantities exchanged</h5><p>${escape(theory.quantities)}</p><h5>Coupling to the other models</h5><p>${escape(theory.coupling)}</p><a href="${topic.id === 'shape' && machine === 'iter' ? detail.link : theory.link}">Derivation and implementation</a></details><details><summary>Recorded evidence and method</summary>${detail.claims.map(claim).join('')}<p><a href="${detail.link || topic.link}">Method and limitations</a></p></details>`;
    const physicsPanel = root.querySelector('.ex-physics');
    physicsPanel.open = physicsOpen;
    physicsPanel.addEventListener('toggle', () => { physicsOpen = physicsPanel.open; });
    progress();
    if (focus) $('ex-detail-title').focus({preventScroll:true});
    updateHash();
  }
  function updateHash() {
    const url = new URL(location.href);
    url.hash = `explore-${machine}-${topics[current].id}`;
    try { history.replaceState(null, '', url); } catch { /* Local file previews may deny history updates. */ }
  }
  function chooseMachine(id, index = 0, focus = true) {
    pauseFlow();
    machine = id;
    root.dataset.machine = id;
    $('ex-mission').hidden = false;
    root.querySelectorAll('[data-machine]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.machine === id)));
    $('ex-machine-label').textContent = machines[id].name;
    $('ex-machine-summary').textContent = machines[id].summary;
    $('ex-scope-content').innerHTML = scope[id].map(([label, description]) => `<div><dt>${escape(label)}</dt><dd>${escape(description)}</dd></div>`).join('');
    $('ex-guard-scope').textContent = id === 'iter'
      ? 'ITER baseline: the transport response uses a continuous closure near admissibility boundaries. The binary switches below explain the hard guard, not that continuous weighting policy.'
      : 'MAST-U: guard acceptance selects an allowed response; it does not resolve the failed transport accuracy comparison. A fallback also has its own modelling assumptions.';
    $('ex-core-art').innerHTML = '';
    root.querySelectorAll('.ex-station').forEach(button => button.remove());
    topics.forEach((topic, i) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'ex-station';
      button.dataset.topic = topic.id;
      button.dataset.state = topic[id].state;
      button.style.left = `${topic.x}%`;
      button.style.top = `${topic.y}%`;
      button.setAttribute('aria-controls', 'ex-detail');
      button.setAttribute('aria-label', `${topic.title}. ${topic[id].status}`);
      button.innerHTML = `<small>${symbol[topic[id].state]} ${({shape:'Equilibrium',heating:'Heating',transport:'Transport',edge:'Edge',guard:'Guard',stability:'Stability',system:'Core solver'})[topic.id]}</small>${topic.title}`;
      button.addEventListener('click', () => { pauseFlow(); chooseTopic(i, true); });
      $('ex-map').appendChild(button);
    });
    $('ex-impact-title').textContent = machines[id].impactTitle;
    $('ex-impact-text').textContent = machines[id].impact;
    renderEvidence();
    chooseTopic(index);
    flowIndex = 0;
    renderExchange();
    if (focus) {
      $('ex-mission-title').focus({preventScroll:true});
      $('ex-mission-title').scrollIntoView({block:'start', behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth'});
    }
  }

  function pauseFlow(freeze = true) {
    clearTimeout(flowTimer);
    flowTimer = null;
    flowPlaying = false;
    if (freeze) flowMotion?.pause();
    $('ex-flow-play').textContent = flowIndex === exchanges.length-1 ? 'Replay exchanges' : 'Play exchanges';
    $('ex-flow-play').setAttribute('aria-pressed', 'false');
  }
  function renderExchange() {
    const step = exchanges[flowIndex];
    const source = topics.find(topic => topic.id === step.from);
    const target = topics.find(topic => topic.id === step.to);
    const names = {system:'Plasma solver', shape:'Magnetic geometry', heating:'Heating', transport:'Transport', edge:'Edge boundary', guard:'Guard'};
    $('ex-flow-position').textContent = `Exchange ${flowIndex+1} of ${exchanges.length} · ${machines[machine].name}`;
    $('ex-flow-route').textContent = step.from === step.to ? `Within the ${names[step.from].toLowerCase()}` : `${names[step.from]} → ${names[step.to]}`;
    $('ex-flow-title').textContent = step.title;
    $('ex-flow-explanation').textContent = step[machine] || step.explanation;
    $('ex-flow-payload').textContent = step.payload;
    $('ex-flow-accepted').textContent = flowIndex < 10 ? 'uⁿ · unchanged' : 'uⁿ⁺¹ · accepted example';
    $('ex-flow-trial').textContent = flowIndex < 9 ? 'Current iterate u⁽ᵏ⁾' : flowIndex === 9 ? 'New trial u⁽ᵏ⁺¹⁾' : flowIndex === 10 ? 'Checks passed in this example' : 'Next iterate starts from uⁿ⁺¹';
    $('ex-flow-scope').textContent = machine === 'iter'
      ? 'ITER: prescribed geometry, guarded heating and pedestal, and the current continuous transport closure. The acceptance stage illustrates the passing research baseline.'
      : 'MAST-U: static imported geometry, prescribed sources and edge conditions. Live equilibrium feedback is planned; the accepted update here is conceptual, not a validation result.';
    $('ex-flow-back').disabled = flowIndex === 0;
    $('ex-flow-next').disabled = flowIndex === exchanges.length-1;
    if (!flowPlaying) $('ex-flow-play').textContent = flowIndex === exchanges.length-1 ? 'Replay exchanges' : 'Play exchanges';
    root.querySelectorAll('.ex-station').forEach(button => {
      button.classList.toggle('ex-flow-sender', button.dataset.topic === step.from);
      button.classList.toggle('ex-flow-receiver', button.dataset.topic === step.to);
    });
    flowMotion?.cancel();
    const a = {x:source.x*6, y:source.y*5.6}, b = {x:target.x*6, y:target.y*5.6};
    let route = '';
    if (step.from !== step.to) {
      const dx = b.x-a.x, dy = b.y-a.y;
      const trim = Math.min(104/Math.abs(dx), 37/Math.abs(dy));
      route = `M${a.x+dx*trim},${a.y+dy*trim} L${b.x-dx*trim},${b.y-dy*trim}`;
    } else if (flowIndex > 0) {
      route = `M${a.x+105},${a.y-15} C${a.x+160},${a.y-95} ${a.x+160},${a.y+95} ${a.x+105},${a.y+15}`;
    }
    $('ex-transfer-path').setAttribute('d', route);
    $('ex-transfer-dot').style.display = route ? '' : 'none';
    const length = route ? $('ex-transfer-path').getTotalLength() : 0;
    const frames = route ? Array.from({length:25}, (_, i) => {
      const point = $('ex-transfer-path').getPointAtLength(length*i/24);
      return {transform:`translate(${point.x}px, ${point.y}px)`};
    }) : [];
    $('ex-transfer-dot').style.transform = frames.at(-1)?.transform || 'none';
    flowMotion = route && !reducedMotion.matches
      ? $('ex-transfer-dot').animate(frames, {duration:2200, fill:'forwards'})
      : null;
  }
  function scheduleExchange() {
    flowTimer = setTimeout(() => {
      if (!flowPlaying) return;
      flowIndex++;
      renderExchange();
      if (flowIndex === exchanges.length-1) pauseFlow(false);
      else scheduleExchange();
    }, 4500);
  }
  function playFlow() {
    if (flowPlaying) { pauseFlow(); return; }
    if (flowIndex === exchanges.length-1) { flowIndex = 0; renderExchange(); }
    flowPlaying = true;
    $('ex-flow-play').textContent = 'Pause exchanges';
    $('ex-flow-play').setAttribute('aria-pressed', 'true');
    if (!reducedMotion.matches) flowMotion?.play();
    scheduleExchange();
  }
  function moveExchange(delta) {
    pauseFlow();
    flowIndex = Math.max(0, Math.min(exchanges.length-1, flowIndex+delta));
    renderExchange();
  }

  function guard() {
    const accepted = ['ex-domain','ex-agreement','ex-physical'].every(id => $(id).checked);
    $('ex-route').textContent = accepted ? 'Surrogate' : 'Fallback';
    $('ex-route').style.borderColor = accepted ? 'var(--ex-accent)' : 'var(--ex-warm)';
    $('ex-route').style.color = accepted ? 'var(--ex-accent)' : 'var(--ex-warm)';
    $('ex-guard-result').textContent = accepted ? 'All demo checks pass: use the fast response and record the choice.' : 'A demo check fails: use the fallback and record why the fast model was rejected.';
  }
  function renderEvidence() {
    const container = $('ex-evidence-content');
    if (machine === 'mastu') {
      $('ex-evidence-title').textContent = 'MAST-U: equilibrium and transport checks';
      container.innerHTML = `<p>The static equilibrium was computed and imported into the toolkit. In the coupled transport comparison, the error exceeded the acceptance limit.</p><div class="ex-flow"><div><b>Equilibrium</b><span>Compute a static MAST-U-like equilibrium.</span></div><div><b>Data interface</b><span>Translate magnetic geometry onto the simulation grid.</span></div><div><b>Transport comparison</b><span>The recorded error exceeds the unchanged acceptance limit.</span></div></div>${claim('C-019')}${claim('C-179')}<p class="ex-limit">A later MAST-U reference comparison could not be evaluated, so it does not resolve the accuracy limitation.</p><a href="technical/verification.html#mastu-scope">Scope of the MAST-U results</a>`;
      return;
    }
    $('ex-evidence-title').textContent = 'Does a smaller time step change the answer?';
    if (!evidence?.convergence?.length) {
      container.innerHTML = '<p>The evidence charts are generated by the blog export. <a href="claims.html#c-306">Read the convergence claim</a>.</p>';
      return;
    }
    const rows = evidence.convergence;
    container.innerHTML = `<p>A simulation advances in time steps. Here, each case is rerun with smaller allowed steps. The chart shows the largest change across the checked outputs for each case.</p><div class="ex-evidence-controls"><label>Refinement <select id="ex-refinement"><option value="coarse_to_middle_percent">Coarse → middle</option><option value="middle_to_fine_percent">Middle → fine</option></select></label><label>Inspect case <input type="range" id="ex-case" min="0" max="${rows.length-1}" value="0" aria-label="Case to inspect"></label></div><div id="ex-chart"></div><p id="ex-chart-result" class="ex-result" role="status"></p><p class="ex-caption">Measured maximum relative endpoint change (%) across fusion power, thermal energy and axis temperatures. This is maximum-step sensitivity, not an exact-solution error bound. The dashed line marks the declared acceptance limit.</p>${claim('C-305')}${claim('C-306')}<details><summary>Read all chart values as a table</summary><table class="ex-data-table"><thead><tr><th>Case</th><th>Coarse → middle (%)</th><th>Middle → fine (%)</th></tr></thead><tbody>${rows.map(r=>`<tr><td>${r.case}</td><td>${r.coarse_to_middle_percent.toPrecision(4)}</td><td>${r.middle_to_fine_percent.toPrecision(4)}</td></tr>`).join('')}</tbody></table><a href="figures/m5_convergence.csv">Download the recorded CSV</a></details><h3 style="margin-top:32px">Local robustness checks</h3><p>This is a separate check of how much the model output can change near a selected input. Each tile represents one state from the ITER version 2 study; the symbols distinguish proved bounds, unresolved checks and inputs outside the allowed domain.</p><div class="ex-proof-row" id="ex-proof"></div>${claim('C-326')}<p class="ex-caption">✓ Proved · ? Unknown · × Outside domain. Grouped by outcome, not the original state order.</p>`;
    const counts = evidence.claims['C-326'].value;
    $('ex-proof').innerHTML = counts.map((count,i)=>Array.from({length:count},()=>`<span class="${['proved','unknown','outside'][i]}" aria-label="${['Proved','Unknown','Outside domain'][i]}">${['✓','?','×'][i]}</span>`).join('')).join('');
    $('ex-refinement').addEventListener('change', drawChart);
    $('ex-case').addEventListener('input', drawChart);
    drawChart();
  }
  function drawChart() {
    const rows = evidence.convergence, key = $('ex-refinement').value, selected = Number($('ex-case').value);
    const width = Math.max(240, $('ex-chart').clientWidth), height = 300;
    const left = 46, right = 16, top = 28, bottom = 45;
    // The protocol's fixed 1% threshold is checked against C-306 in the export tests.
    const limit = 1, ymax = 1.1, plotHeight = height-top-bottom, plotWidth = width-left-right;
    const y = value => top + plotHeight*(1-value/ymax);
    const step = plotWidth/rows.length;
    const grid = [0,.25,.5,.75,1].map(v=>`<line x1="${left}" y1="${y(v)}" x2="${width-right}" y2="${y(v)}" stroke="${v===limit ? 'var(--ex-active)' : 'var(--ex-line)'}" stroke-dasharray="${v===limit ? '6 5' : '0'}"/><text x="${left-8}" y="${y(v)+4}" text-anchor="end">${v}</text>`).join('');
    const tickEvery = width < 480 ? 8 : 4;
    const bars = rows.map((row,i)=>`<rect x="${left+i*step+1}" y="${y(row[key])}" width="${Math.max(1,step-2)}" height="${Math.max(.8,plotHeight*row[key]/ymax)}" fill="${i===selected ? 'var(--ex-active)' : 'var(--ex-ink)'}"><title>Case ${row.case}: ${row[key].toPrecision(4)}%</title></rect>${i%tickEvery===0 ? `<text x="${left+(i+.5)*step}" y="${height-25}" text-anchor="middle">${row.case}</text>` : ''}`).join('');
    $('ex-chart').innerHTML = `<svg class="ex-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Recorded endpoint changes by case; all lie below the one percent acceptance line."><text x="${left}" y="17" class="ex-chart-label">Maximum endpoint change (%)</text>${grid}${bars}<text x="${width-right}" y="${y(limit)-8}" text-anchor="end">Acceptance limit: 1%</text><text x="${width/2}" y="${height-6}" text-anchor="middle">Original ensemble case</text></svg>`;
    const row = rows[selected];
    $('ex-chart-result').textContent = `Case ${row.case}: ${row[key].toPrecision(4)}% maximum change · ${row[key] < limit ? 'below' : 'at or above'} the acceptance limit.`;
  }
  root.querySelectorAll('.ex-machine').forEach(button => button.addEventListener('click', () => chooseMachine(button.dataset.machine)));
  root.querySelectorAll('[data-section]').forEach(button => button.addEventListener('click', () => {
    pauseFlow();
    const heading = $(button.dataset.section);
    heading.focus({preventScroll:true});
    heading.scrollIntoView({block:'start', behavior:reducedMotion.matches ? 'instant' : 'smooth'});
  }));
  $('ex-flow-play').addEventListener('click', playFlow);
  $('ex-flow-back').addEventListener('click', () => moveExchange(-1));
  $('ex-flow-next').addEventListener('click', () => moveExchange(1));
  $('ex-flow-restart').addEventListener('click', () => { pauseFlow(); flowIndex = 0; renderExchange(); });
  document.addEventListener('visibilitychange', () => { if (document.hidden) pauseFlow(); });
  window.addEventListener('pagehide', () => pauseFlow());
  reducedMotion.addEventListener('change', () => { pauseFlow(); if (machine) renderExchange(); });
  $('ex-next').addEventListener('click', () => { pauseFlow(); chooseTopic(current+1, true); });
  $('ex-prev').addEventListener('click', () => { pauseFlow(); chooseTopic(current-1, true); });
  $('ex-reset').addEventListener('click', () => { pauseFlow(); visited[machine].clear(); chooseTopic(0, true); });
  ['ex-domain','ex-agreement','ex-physical'].forEach(id => $(id).addEventListener('change', guard));
  function restoreHash() {
    const match = location.hash.match(/^#explore-(mastu|iter)-([a-z]+)$/);
    const index = match ? topics.findIndex(topic => topic.id === (match[2] === 'stability' ? 'system' : match[2])) : -1;
    if (index >= 0) chooseMachine(match[1], index, false);
  }
  window.addEventListener('hashchange', restoreHash);
  let lastWidth = 0;
  new ResizeObserver(([entry]) => {
    if (entry.contentRect.width === lastWidth) return;
    lastWidth = entry.contentRect.width;
    if ($('ex-chart')) drawChart();
  }).observe(root);
  guard();
  restoreHash();
})();
