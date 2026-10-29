import './response.js';
import './viewer.js';
import './uq-interval.js';
import {startPrediction} from './app.js';
import {startUncertainty} from './uq-app.js';

if (new URLSearchParams(location.search).get('embed') === '1') {
  document.body.classList.add('embedded');
}

const tabs = Array.from(document.querySelectorAll('[role="tab"]'));
const started = new Set();
const initializers = {prediction: startPrediction, uncertainty: startUncertainty};

function select(name) {
  for (const tab of tabs) {
    const active = tab.id === `tab-${name}`;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
    document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
  }
  if (!started.has(name)) {
    started.add(name);
    initializers[name]();
  }
}

function activate(tab) {
  const name = tab.id.slice(4);
  if (location.hash !== `#${name}`) history.pushState(null, '', `#${name}`);
  select(name);
}

tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activate(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[next].focus();
    activate(tabs[next]);
  });
});

function selectFromURL() {
  select(location.hash === '#uncertainty' ? 'uncertainty' : 'prediction');
}
window.addEventListener('hashchange', selectFromURL);
window.addEventListener('popstate', selectFromURL);
selectFromURL();
