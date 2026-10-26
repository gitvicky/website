"""Independent 60-digit Decimal reference. Diagnostic validation, not formal certification.
Only the JS bound outputs are imported; forward formulas and weights are reimplemented here.
Run with Python 3.10+ and Node.js on PATH. No third-party packages.
"""
from decimal import Decimal as D, getcontext
from pathlib import Path
import json
import subprocess
import random

getcontext().prec = 60
root = Path(__file__).resolve().parent
fixtures = json.loads(subprocess.check_output(['node', str(root / 'reference-fixtures.js')], text=True))
checks = 0

def decimal(x):
    return D(str(x))

def contain(bound, value):
    global checks
    checks += 1
    lo, hi = map(decimal, bound)
    # Explicit diagnostic tolerance for independent decimal inputs and binary64 outputs.
    tol = D('1e-9') * max(D(1), abs(value))
    assert lo - tol <= value <= hi + tol, (bound, str(value))

def grid(domain, count=61):
    a, b = map(decimal, domain)
    return [a + (b-a)*D(i)/D(count-1) for i in range(count)]

for case in fixtures['scalar']:
    for x in grid(case['domain']):
        q=x*(1-x)
        y=q if case['name']=='quadratic' else x.exp() if case['name']=='exp' else q.exp()
        contain(case['bound'], y)
        t=x-decimal(case['centre'])
        approx=sum(decimal(v)*t**i if i else decimal(v) for i,v in enumerate(case['p']))
        contain(case['r'], y-approx)

FIRST = ([[1],[-1],[.5]], [-.2,.1,.3])
SECOND = ([[.7,-.4,.3],[-.2,.8,.5],[.4,.2,-.6]], [.1,-.1,.05])

def network(x, depth, activation):
    values=[x]
    for weights,bias in [FIRST,SECOND][:depth]:
        z=[decimal(b)+sum(decimal(w)*v for w,v in zip(row,values)) for row,b in zip(weights,bias)]
        values=[max(D(0),v) if activation=='relu' else v*v for v in z]
    return D('.05')+sum(w*v for w,v in zip(map(D,['1','-.8','.5']),values))

for case in fixtures['networks']:
    for x in grid(case['domain'],101):
        y=network(x,case['depth'],case['activation'])
        for result in case['results']:
            contain([result['lower'][0],result['upper'][0]],y)

EMBEDDINGS = [[D('.8'),D('.2')],[D('-.5'),D('.7')],[D('.1'),D('-.8')],[D('-.3'),D('-.2')]]
Q=[[D('.8'),D('-.3')],[D('.2'),D('.7')]]
K=[[D('.6'),D('.4')],[D('-.5'),D('.9')]]
V=[[D('1'),D('.2')],[D('-.3'),D('.8')]]
O=[[D('1'),D('-.4'),D('-.7'),D('.2')],[D('.3'),D('1'),D('-.2'),D('-.8')]]

def mm(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]

def transformer(x,temp):
    q,k,v=mm(x,Q),mm(x,K),mm(x,V)
    scores=[[sum(a*b for a,b in zip(qrow,krow))/(D(2).sqrt()*temp) for krow in k] for qrow in q]
    # Independent softmax via shifted exponentials, not the JS sigmoid implementation.
    probs=[]
    for row in scores:
        mx=max(row); e=[(s-mx).exp() for s in row]; denom=sum(e)
        probs.append([z/denom for z in e])
    h=mm(probs,v)
    logits=[a+b for a,b in zip(mm([h[1]],O)[0],map(D,['.12','-.02','.02','-.08']))]
    return dict(Q=q,K=k,V=v,S=scores,A=probs,H=h,logits=logits)

rng=random.Random(4861)
for case in fixtures['attention']:
    radius=decimal(case['radius'])
    offsets=[(a*radius,b*radius) for a in [-1,0,1] for b in [-1,0,1]]
    offsets += [(decimal(rng.uniform(-1,1))*radius,decimal(rng.uniform(-1,1))*radius) for _ in range(8)]
    for dx,dy in offsets:
        x=[EMBEDDINGS[i].copy() for i in case['sequence']]
        x[case['token']][0]+=dx;x[case['token']][1]+=dy
        f=transformer(x,decimal(case['temperature']));r=case['result']
        for key in ['Q','K','V','S','A','H']:
            for i,row in enumerate(f[key]):
                for j,value in enumerate(row):contain(r[key][i][j],value)
        for interval,value in zip(r['logits'],f['logits']):contain(interval,value)
        if r['gap']>0:
            assert max(range(4),key=lambda i:f['logits'][i])==r['winner']
        assert r['certified'] is False

print(f'PASS: {checks:,} independent Decimal containment checks; 60-digit precision; diagnostic tolerance 1e-9 relative.')
print('Coverage: Taylor/composition residuals; both network depths and activations; attention projections, scores, probabilities, aggregation, logits and positive margins.')
print('These finite diagnostics are not a proof of outward-rounded soundness.')
