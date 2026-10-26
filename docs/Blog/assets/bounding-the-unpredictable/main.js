import * as iv from './math/interval.js';
import {geometry} from './math/affine.js';
import * as poly from './math/polynomial.js';
import {reluLines,mccormick,refine} from './math/relaxations.js';
import * as nn from './math/neural.js';
import * as tr from './math/transformer.js';
const fmt=x=>!Number.isFinite(x)?'—':Math.abs(x)<1e-11?'0':Math.abs(x)>=10000||Math.abs(x)<.0001?x.toExponential(2):Number(x.toFixed(4)).toString();
const pair=a=>`[${fmt(a[0])}, ${fmt(a[1])}]`;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const range=(key,label,min,max,step,value)=>({key,label,min,max,step,value});
const select=(key,label,options,value=options[0][0])=>({key,label,options,value});
const domainControls=[range('lo','Input lower bound',-1.5,1.5,.05,0),range('hi','Input upper bound',-1,2,.05,1)];
const funcs=[['quadratic','x(1 − x)'],['cancel','x − x'],['square','x² − x']];
const curves=[['quadratic','x(1 − x)'],['exp','exp(x)'],['sin','sin(x)'],['composed','exp(x(1 − x))']];
const status='<p class="bound-status"><strong>Illustrative bounds.</strong> Browser rounding is not validated. <a href="#arithmetic-note">What this means</a></p>';
function mount(id,title,statement,controls,render){
  const root=document.getElementById(id);if(!root)return;
  const defaults=Object.fromEntries(controls.map(c=>[c.key,c.value]));let state={...defaults};
  root.classList.add('bound-demo');root.setAttribute('aria-labelledby',id+'-title');
  root.innerHTML=`<div class="bound-head"><div><p class="bound-eyebrow">Explore · live calculation</p><h3 id="${id}-title">${title}</h3></div><button type="button" aria-label="Reset ${title}">Reset</button></div><p class="bound-detail">${statement}</p><div class="bound-controls">${controls.map(c=>`<label for="${id}-${c.key}">${c.label}${c.options?`<select id="${id}-${c.key}" data-key="${c.key}">${c.options.map(([v,l])=>`<option value="${v}">${l}</option>`).join('')}</select>`:`<output for="${id}-${c.key}" data-value="${c.key}">${fmt(c.value)}</output><input id="${id}-${c.key}" data-key="${c.key}" type="range" min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}">`}</label>`).join('')}</div><div class="bound-render"></div>${status}<p class="visually-hidden" role="status" aria-live="polite"></p>`;
  const output=root.querySelector('.bound-render');let scheduled=false;
  function update(){
    scheduled=false;
    if(id==='bound-demo-5'){
      const incompatible=state.fn!=='exp';
      root.querySelector('option[value="Chebyshev + remainder"]').disabled=incompatible;
      if(incompatible&&state.method==='Chebyshev + remainder'){
        state.method='Taylor + Bernstein';root.querySelector('[data-key="method"]').value=state.method;
      }
    }
    for(const el of root.querySelectorAll('[data-value]'))el.textContent=fmt(state[el.dataset.value]);
    try{if('lo'in state)iv.I(state.lo,state.hi);output.innerHTML=render(state);root.dataset.status='ready';root.querySelector('[role="status"]').textContent='Calculation updated. Numerical results follow the plot.';}
    catch(e){output.innerHTML=`<p class="bound-error" role="alert">${esc(e.message)} Adjust the controls or reset this experiment.</p>`;root.dataset.status='error';}
  }
  root.addEventListener('input',e=>{const key=e.target.dataset.key;if(!key)return;const c=controls.find(c=>c.key===key);state[key]=c.options?e.target.value:Number(e.target.value);if(!scheduled){scheduled=true;requestAnimationFrame(update);}});
  const sync=()=>{for(const el of root.querySelectorAll('[data-key]'))el.value=state[el.dataset.key];update();};
  root.querySelector('button').addEventListener('click',()=>{state={...defaults};sync();});sync();
}
function table(headers,rows,caption='',selected=-1){return `<div class="bound-scroll" tabindex="0" role="region" aria-label="${esc(caption||'Numerical comparison')}"><table>${caption?`<caption>${caption}</caption>`:''}<thead><tr>${headers.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map((r,i)=>`<tr${i===selected?' class="bound-selected"':''}>${r.map((v,j)=>j===0?`<th scope="row">${v}</th>`:`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;}
const asResult=(method,f,operations=1)=>{const start=performance.now();return iv.result(method,f(),start,{operations});};
function resultsTable(results,exact=null,selected=''){
  const selectedRow=results.findIndex(r=>r.method===selected);
  const primary=table(['Method','Output interval','Width'],results.map(r=>{
    const a=[r.lower[0],r.upper[0]];return [r.method,pair(a),fmt(iv.width(a))];
  }),exact?`Analytic reference: ${pair(exact)}.`:'Computed enclosures; exact range unavailable.',selectedRow);
  const detail=table(['Method',...(exact?['Excess width','Width / true width']:[]),'Time (ms)'],results.map(r=>{
    const m=iv.metrics([r.lower[0],r.upper[0]],exact);
    return [r.method,...(exact?[fmt(m.excess),m.ratio===null?'undefined':fmt(m.ratio)]:[]),r.computationMs.toFixed(3)];
  }),'Excess width subtracts the true range width. Browser timings are indicative.',selectedRow);
  return primary+`<details><summary>More metrics and timings</summary>${detail}</details>`;
}
function chart(domain,series,bands=[],xlabel='x',ylabel='output'){
  const W=660,H=280,L=62,R=18,T=24,B=42,span=domain[1]-domain[0],X=span===0?[domain[0]-.5,domain[1]+.5]:domain;
  const lines=series.map(s=>({...s,points:s.points||Array.from({length:181},(_,i)=>{const x=domain[0]+span*i/180;return [x,s.f(x)];})}));
  const values=lines.flatMap(s=>s.points.map(p=>p[1])).concat(bands.flatMap(b=>b.range));
  let lo=Math.min(...values),hi=Math.max(...values);const pad=Math.max((hi-lo)*.12,.05);lo-=pad;hi+=pad;
  const x=v=>L+(v-X[0])/(X[1]-X[0])*(W-L-R),y=v=>H-B-(v-lo)/(hi-lo)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(ylabel+' versus '+xlabel+'; numerical values are in the following table')}"><rect x="${L}" y="${T}" width="${W-L-R}" height="${H-T-B}" fill="white"/>`;
  for(let i=0;i<=4;i++){const v=lo+(hi-lo)*i/4;svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e0e4e6"/><text x="${L-8}" y="${y(v)+4}" text-anchor="end">${fmt(v)}</text>`;}
  bands.forEach((b,i)=>{svg+=`<rect x="${x(domain[0])}" y="${y(b.range[1])}" width="${Math.max(1,x(domain[1])-x(domain[0]))}" height="${Math.max(.8,y(b.range[0])-y(b.range[1]))}" fill="${b.color||'#b9d0dd'}" fill-opacity=".22" stroke="${b.color||'#315f78'}" stroke-dasharray="${i?'2 4':'7 4'}"/>`;});
  lines.forEach((s,i)=>{svg+=`<polyline points="${s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}" fill="none" stroke="${s.color||'#202b30'}" stroke-width="${s.width||2}" ${s.dash?`stroke-dasharray="${s.dash}"`:''}/>`;});
  if(span===0)lines.forEach(s=>{const p=s.points[0];svg+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="3" fill="${s.color||'#202b30'}"/>`;});
  for(let i=0;i<=4;i++){const v=X[0]+(X[1]-X[0])*i/4;svg+=`<text x="${x(v)}" y="${H-B+20}" text-anchor="middle">${fmt(v)}</text>`;}
  svg+=`<text x="${W-R}" y="${H-3}" text-anchor="end">${esc(xlabel)}</text><text x="${L}" y="15">${esc(ylabel)}</text></svg>`;
  return svg+`<div class="bound-legend">${lines.map(s=>`<span>${s.dash?'┄':'━'} ${s.label}</span>`).concat(bands.map(b=>`<span>▧ ${b.label} ${pair(b.range)}</span>`)).join('')}</div>`;
}
function scalarPlot(name,d,bounds){return chart(d,[{label:'Function (solid)',f:x=>iv.scalar(name,x)}],bounds);}
mount('bound-demo-1','How much room does a bound leave?','A single input x varies over the selected interval. The analytic reference checks the endpoints and the stationary point x = ½.',domainControls,s=>{
  const d=[s.lo,s.hi],exact=iv.exactScalar('quadratic',d),r=asResult('Natural interval',()=>iv.natural('quadratic',d));
  return scalarPlot('quadratic',d,[{label:'Interval enclosure',range:[r.lower[0],r.upper[0]]},{label:'Analytic range',range:exact,color:'#8d4930'}])+resultsTable([r],exact)+'<p class="bound-explanation">On [0, 1], the two factors cannot both be 1 at the same x. The natural interval calculation forgets this restriction, giving [0, 1] for a function that reaches only ¼.</p>';
});
mount('bound-demo-2','One variable, several appearances','Try x − x, then switch to its simplified expression. Every method sees the same input interval.',[...domainControls,select('fn','Function',funcs),select('form','Expression form',[['natural','Original expression'],['rewritten','Completed square / simplified']]),range('n','Equal subdomains',1,64,1,8)],s=>{
  const d=[s.lo,s.hi],r=[asResult('Natural interval',()=>iv.natural(s.fn,d,s.form)),asResult('Mean-value',()=>iv.centred(s.fn,d)),asResult('Subdivided interval',()=>iv.hull(iv.partition(d,s.n).map(a=>iv.natural(s.fn,a,s.form))),s.n)];
  const forms={quadratic:['x(1 − x)','¼ − (x − ½)²'],cancel:['x − x','0'],square:['x² − x','(x − ½)² − ¼']};
  return `<p class="bound-detail">Evaluating <strong>${forms[s.fn][s.form==='natural'?0:1]}</strong> on ${s.n} subdomains.</p>`+scalarPlot(s.fn,d,[{label:'Natural',range:[r[0].lower[0],r[0].upper[0]]},{label:'Subdivided',range:[r[2].lower[0],r[2].upper[0]],color:'#8d4930'}])+resultsTable(r,iv.exactScalar(s.fn,d))+'<p class="bound-explanation">Mean-value bounds use the derivative over the whole interval. Subdivision tightens the calculation by working on smaller intervals.</p>';
});
mount('bound-demo-3','Keep the shape, keep the dependence','Rotate and shear a rectangle, then optionally apply (u, v) ↦ (u, v + 0.4u²).',[range('angle','Rotation (degrees)',-90,90,5,35),range('shear','Shear',-1.5,1.5,.1,.5),range('wx','First input half-width',.1,1.5,.1,1),range('wy','Second input half-width',.1,1.5,.1,.6),select('nonlinear','Map',[['no','Affine'],['yes','Affine then quadratic']]),select('constraint','Input constraint',[['no','Independent factors'],['yes','ε₂ = ε₁']])],s=>{
  const start=performance.now(),g=geometry(s.angle,s.shear,s.wx,s.wy,s.nonlinear==='yes',s.constraint==='yes'),elapsed=performance.now()-start;
  const box=[[g.box[0][0],g.box[1][0]],[g.box[0][1],g.box[1][0]],[g.box[0][1],g.box[1][1]],[g.box[0][0],g.box[1][1]]];
  const all=box.concat(g.polygon,g.boundary),extent=Math.max(...all.flat().map(Math.abs))*1.12||1,px=v=>330+v/extent*220,py=v=>245-v/extent*220;
  const polygon=(p,fill,stroke,dash='')=>`<polygon points="${p.map(([x,y])=>`${px(x)},${py(y)}`).join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width="2" stroke-dasharray="${dash}"/>`;
  return `<svg viewBox="0 0 660 490" role="img" aria-label="Mapped reachable boundary, correlated enclosure and enclosing interval rectangle"><line x1="70" x2="590" y1="245" y2="245" stroke="#d5dade"/><line x1="330" x2="330" y1="15" y2="475" stroke="#d5dade"/>${polygon(box,'#edf0f2','#657986','7 5')}${polygon(g.polygon,'#c7dbe566','#315f78','3 3')}${polygon(g.boundary,'#c2b5a522','#7c422e')}<text x="590" y="263">u</text><text x="342" y="20">v</text><text x="72" y="480">Equal axis scales · extent ±${fmt(extent)}</text></svg><div class="bound-legend"><span>┄ Interval box</span><span>··· Affine enclosure</span><span>━ Reachable boundary (analytically parametrized)</span></div>`+table(['Quantity','Value'],[['Transform',g.W.map(row=>pair(row)).join('; ')],['Interval coordinate ranges',g.box.map(pair).join(' × ')],['Affine generators',g.generators],['Computation',elapsed.toFixed(3)+' ms']])+`<p class="bound-explanation">${s.constraint==='yes'?'The equality constraint reduces the input to a line: one shared factor represents both coordinates.':'Shared generator directions preserve the rotated rectangle exactly under the affine map.'} ${s.nonlinear==='yes'?'The quadratic image keeps a dependent squared factor; the affine enclosure replaces its nonlinear residual by a new independent generator.':'The interval box forgets which pairs of output coordinates can occur together.'}</p>`;
});
mount('bound-demo-4','A polynomial, plus everything left out','The band is Pₖ(x − c) + R on each subdomain. The remainder includes truncation during composition.',[...domainControls,select('fn','Function',curves,'exp'),range('degree','Retained degree',1,8,1,3),range('centre','Centre within each cell',0,1,.1,.5),range('n','Subdomains',1,16,1,1)],s=>{
  const d=[s.lo,s.hi],r=poly.polynomialExperiment(s.fn,d,s.degree,s.n,s.centre);
  const piece=x=>r.pieces.find(p=>x>=p.domain[0]&&x<=p.domain[1])||r.pieces.at(-1),p=x=>piece(x).value(x);
  const sample=Array.from({length:301},(_,i)=>d[0]+iv.width(d)*i/300),err=Math.max(...sample.map(x=>Math.abs(iv.scalar(s.fn,x)-p(x))));
  return chart(d,[{label:'Function',f:x=>iv.scalar(s.fn,x)},{label:'Polynomial',f:p,color:'#315f78',dash:'7 4'},{label:'Remainder envelope',f:x=>p(x)+piece(x).r[0],color:'#8d4930',dash:'2 4'},{label:'Envelope upper edge',f:x=>p(x)+piece(x).r[1],color:'#8d4930',dash:'2 4'}])+resultsTable([r],iv.exactScalar(s.fn,d))+table(['Error quantity','Value'],[['Largest analytic remainder radius',fmt(Math.max(...r.pieces.flatMap(p=>p.r.map(Math.abs))))],['Observed error on 301 points (diagnostic only)',fmt(err)],['Polynomial degree / subdomains',`${s.degree} / ${s.n}`]])+'<p class="bound-explanation">A small approximation error does not make the output range narrow: the function itself may vary substantially. Moving the expansion centre changes the largest distance to a domain endpoint; subdivision recentres each local model.</p>';
});
mount('bound-demo-5','Where should a polynomial be bounded?','Compare range evaluation separately from approximation. Chebyshev is supported for exp(x), with an analytic interpolation remainder.',[...domainControls,select('fn','Benchmark',[['exp','exp(x)'],['quadratic','x(1 − x)']]),range('degree','Degree',2,8,1,4),range('n','Subdomains',1,32,1,1),select('method','Plot approximation',[['Taylor + Bernstein','Taylor / Bernstein'],['Taylor + interval polynomial','Taylor / interval'],['Chebyshev + remainder','Chebyshev (exp only)']])],s=>{
  const d=[s.lo,s.hi],methods=['Taylor + interval polynomial','Taylor + Bernstein',...(s.fn==='exp'?['Chebyshev + remainder']:[])];
  const r=methods.map(m=>poly.polynomialExperiment(s.fn,d,s.degree,s.n,.5,m)),chosen=r.find(x=>x.method===s.method)||r[1];
  const piece=x=>chosen.pieces.find(p=>x>=p.domain[0]&&x<=p.domain[1])||chosen.pieces.at(-1);
  return chart(d,[{label:'Function',f:x=>iv.scalar(s.fn,x)},{label:chosen.method,f:x=>piece(x).value(x),color:'#315f78',dash:'6 4'}],[{label:'Selected enclosure',range:[chosen.lower[0],chosen.upper[0]]}])+resultsTable(r,iv.exactScalar(s.fn,d),chosen.method)+`<p class="bound-explanation">${s.fn==='quadratic'?'For this quadratic, degree ≥ 2 has zero mathematical approximation remainder. The unsplit Bernstein coefficients on [0, 1] are (0, ½, 0); subdivision tightens their convex hull. Chebyshev is omitted for this benchmark.':'Both approximations include a real-arithmetic remainder formula. Chebyshev interpolation uses roots of Tₙ₊₁, then bounds its polynomial with Bernstein coefficients. Neither floating-point construction is machine certified.'}</p>`;
});
mount('bound-demo-6','Relax the graph, bound the objective','The ReLU view bounds y − βx. The bilinear view restricts the McCormick envelope to y = 1 − x.',[range('lo','Input lower bound',-1.5,0,.1,-.5),range('hi','Input upper bound',0,2,.1,1),select('kind','Graph',[['relu','ReLU'],['bilinear','Bilinear, y = 1 − x']]),range('alpha','ReLU lower slope α',0,1,.05,.5),range('beta','Objective coefficient β',-1,1,.1,.5)],s=>{
  const d=[s.lo,s.hi],lines=reluLines(d,s.alpha),f=x=>s.kind==='relu'?Math.max(0,x):x*(1-x);
  const low=x=>s.kind==='relu'?lines.lower[0]*x+lines.lower[1]:mccormick(x,1-x,d,[1-d[1],1-d[0]])[0];
  const up=x=>s.kind==='relu'?lines.upper[0]*x+lines.upper[1]:mccormick(x,1-x,d,[1-d[1],1-d[0]])[1];
  const candidates=[d[0],d[1],iv.mid(d),...(s.kind==='relu'&&d[0]<0&&d[1]>0?[0]:[])];
  // McCormick slice has its two plane intersections at the interval midpoint.
  const start=performance.now(),relaxed=iv.I(Math.min(...candidates.map(x=>low(x)-s.beta*x)),Math.max(...candidates.map(x=>up(x)-s.beta*x)));
  const critical=s.kind==='relu'?0:(1-s.beta)/2,exactCandidates=[...d,...(critical>=d[0]&&critical<=d[1]?[critical]:[])],exact=iv.hull(exactCandidates.map(x=>iv.I(f(x)-s.beta*x)));
  const natural=iv.sub(s.kind==='relu'?iv.relu(d):iv.natural('quadratic',d),iv.scale(d,s.beta));
  const r=[iv.result('Linear relaxation',relaxed,start),asResult('Interval objective',()=>natural)];
  return chart(d,[{label:'Graph',f},{label:'Lower relaxation',f:low,color:'#315f78',dash:'6 4'},{label:'Upper relaxation',f:up,color:'#8d4930',dash:'2 4'}])+resultsTable(r,exact)+`<p class="bound-explanation">The plot shows the graph; the table bounds <strong>f(x) − ${fmt(s.beta)}x</strong>. ${s.kind==='relu'?'Changing α changes a valid supporting line and can help one objective while hurting another.':'The equality y = 1 − x is retained while optimizing the four planes. These bounds solve this small piecewise-linear problem analytically.'}</p><details><summary>Apply the same slope to a small ReLU network</summary>${resultsTable(nn.compareNetwork(d,1,'relu',s.alpha).filter(r=>['IBP','CROWN-style'].includes(r.method)),nn.exactRelu(d).bound)}</details>`;
});
mount('bound-demo-7','Spend computation on smaller questions','Split the leaf with the widest local bound. All leaves remain in the domain cover; no sample is used to discard a region.',[range('leaves','Leaf subdomains',1,64,1,8),select('engine','Bounding engine',[['Interval','Natural interval'],['McCormick LP','McCormick slice LP']])],s=>{
  const r=refine([0,1],s.leaves,s.engine),other=refine([0,1],s.leaves,s.engine==='Interval'?'McCormick LP':'Interval');
  return chart([1,Math.max(2,s.leaves)],[{label:r.method,points:r.history.map(h=>[h.leaves,iv.width(h.bound)])},{label:other.method,points:other.history.map(h=>[h.leaves,iv.width(h.bound)]),color:'#315f78',dash:'6 4'},{label:'True range width = ¼',f:()=>.25,color:'#8d4930',dash:'2 4'}],[],'leaf count','enclosure width')+resultsTable([r,other],[0,.25])+table(['Refinement accounting','Value'],[['Current numerical lower / upper bound',pair([r.lower[0],r.upper[0]])],['Leaf subdomains',s.leaves],['Splits',r.refinements],['Total bound evaluations',r.operations]])+'<p class="bound-explanation">The minimum of all leaf lower bounds and the maximum of all leaf upper bounds enclose the full range. The same refinement strategy works with either engine; their local relaxations differ. A remaining gap is an unresolved proof obligation, not a counterexample.</p>';
});
mount('bound-demo-8','Where does a network lose dependence?','Fixed weights; one uncertain scalar input; three neurons per hidden layer. The square activation gives a small polynomial network.',[range('centre','Input centre',-1,1,.1,.4),range('radius','Input half-width',0,1,.05,.2),select('depth','Hidden layers',[['1','One layer'],['2','Two layers']],'2'),select('activation','Activation',[['relu','ReLU'],['square','Square']]),select('method','Highlighted method',[['IBP','IBP'],['Affine / zonotope','Affine / zonotope'],['CROWN-style','CROWN-style'],['Polynomial','Polynomial / enumeration'],['Lipschitz','Lipschitz'],['Local Jacobian','Local Jacobian']],'CROWN-style')],s=>{
  const d=[s.centre-s.radius,s.centre+s.radius],depth=Number(s.depth),r=nn.compareNetwork(d,depth,s.activation),selected=s.method==='Polynomial'?r[2].method:s.method,chosen=r.find(x=>x.method===selected),trace=nn.ibp(d,depth,s.activation).trace;
  const affine=s.method==='Affine / zonotope'?nn.affineNetwork(d,depth,s.activation):null;
  const layerRows=trace.flatMap((layer,k)=>layer.pre.map((a,j)=>[`Layer ${k+1}, neuron ${j+1}`,pair(a),pair(affine?affine.trace[k][j]:layer.post[j])]));
  return `<p class="bound-detail"><strong>1 → 3${depth===2?' → 3':''} → 1</strong> · ${s.activation==='relu'?'ReLU':'Square'} hidden activations · linear output</p>`+chart(d,[{label:'Network output',f:x=>nn.forward(x,depth,s.activation)}],[{label:selected,range:[chosen.lower[0],chosen.upper[0]]}])+resultsTable(r,s.activation==='relu'?nn.exactRelu(d,depth).bound:null,selected)+`<details><summary>Inspect intermediate bounds</summary>${table(['Layer / neuron','Pre-activation (IBP)','Post-activation'+(affine?' (affine)':' (IBP)')],layerRows,'CROWN uses these IBP pre-activation intervals; its output objective propagates backwards.')}</details><p class="bound-explanation">Affine arithmetic and zonotopes share one implementation here. Increase the input half-width to see where each method starts losing precision.</p>`;
});
function matrix(label,nom,bounds){return table(['','Coordinate 1','Coordinate 2'],bounds.map((row,i)=>[`Token ${i+1}`,...row.map((a,j)=>`${fmt(nom[i][j])}<small>${pair(a)}</small>`)]),label+' — nominal value; interval underneath');}
mount('bound-demo-9','Can the next token stay the same?','Two tokens, two embedding coordinates, one attention head, four artificial output tokens. Output logits come from the second attention row.',[select('first','First token',tr.VOCAB.map((v,i)=>[String(i),v]),'0'),select('second','Second token',tr.VOCAB.map((v,i)=>[String(i),v]),'1'),select('token','Perturbed position',[['0','First token'],['1','Second token']]),range('radius','Embedding L∞ radius',0,1,.025,.1),range('temp','Attention temperature',.25,2,.05,1),select('method','Bounding method',[['Interval','Interval'],['Dependencies + simplex','Dependencies + row sum'],['Subdivide + dependencies','Subdivision + dependencies']])],s=>{
  const seq=[Number(s.first),Number(s.second)],r=tr.experiment(seq,Number(s.token),s.radius,s.temp,s.method),m=r.nominal;
  const chartRows=tr.VOCAB.map((v,i)=>[v,fmt(m.logits[i]),pair(r.logits[i]),fmt(iv.width(r.logits[i]))]);
  const min=Math.min(...r.lower),max=Math.max(...r.upper),span=Math.max(max-min,.1),px=v=>120+(v-min)/span*480;
  const bars=`<svg viewBox="0 0 660 190" role="img" aria-label="Four output logit intervals with nominal values as dots">${tr.VOCAB.map((v,i)=>{const y=30+i*37;return `<text x="95" y="${y+4}" text-anchor="end">${v}${i===r.winner?' *':''}</text><line x1="${px(r.lower[i])}" x2="${px(r.upper[i])}" y1="${y}" y2="${y}" stroke="#315f78" stroke-width="5"/><circle cx="${px(m.logits[i])}" cy="${y}" r="4" fill="#8d4930"/>`;}).join('')}<text x="120" y="183">${fmt(min)}</text><text x="600" y="183" text-anchor="end">${fmt(max)} · logit</text></svg>`;
  return `<p class="bound-status"><strong>${r.status}.</strong> Nominal winner: <strong>${tr.VOCAB[r.winner]}</strong>. Margin (winner’s lower bound − largest competing upper bound): <strong>${fmt(r.gap)}</strong>. ${r.gap>0?'The margin is positive numerically; this is not a software certificate.':'These bounds do not settle whether the winning token can change.'}</p>`+bars+table(['Token','Nominal logit','Logit interval','Width'],chartRows,'* marks the nominal greedy token',r.winner)+`<details><summary>Inspect embeddings and attention matrices</summary><div class="bound-matrices">${matrix('Input embeddings',seq.map(i=>tr.EMBEDDINGS[i]),r.X)}${matrix('Attention scores S',m.S,r.S)}${matrix('Attention probabilities A',m.A,r.A)}${matrix('Attention output H',m.H,r.H)}</div></details><details><summary>Inspect query, key and value projections</summary><div class="bound-matrices">${matrix('Q',m.Q,r.Q)}${matrix('K',m.K,r.K)}${matrix('V',m.V,r.V)}</div></details><p class="bound-detail">${r.operations} input boxes evaluated · ${r.computationMs.toFixed(3)} ms for bounds and nominal evaluation. This is a continuous embedding perturbation, not a paraphrase set.</p><p class="bound-explanation">Dependencies retain shared inputs and attention weights that sum to one. Subdivision repeats this calculation on 16 smaller input boxes.</p>`;
});
