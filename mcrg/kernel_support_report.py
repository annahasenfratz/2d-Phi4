#!/usr/bin/env python3
"""Kernel sums, stride-two polyphase sums, and verified receptive-field widths."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from mcrg.blocking import perfect_kernel,PRODUCTION_KERNEL,PRODUCTION_KERNEL_5X5
def matrix(name):
 if name=='average': return np.full((2,2),perfect_kernel().matrix.sum()/4),range(0,2)
 k=perfect_kernel(PRODUCTION_KERNEL_5X5 if name=='perfect5' else PRODUCTION_KERNEL).matrix;r=k.shape[0]//2;return k,range(-r,r+1)
def width(name,n):
 k,off=matrix(name);s={0}
 for _ in range(n):s={2*x+y for x in s for y in off}
 return max(s)-min(s)+1
out={}
for name in ('average','perfect5','perfect7'):
 k,off=matrix(name);c={(a,b):0. for a in range(2) for b in range(2)}
 for i,x in enumerate(off):
  for j,y in enumerate(off):c[x%2,y%2]+=float(k[i,j])
 out[name]={'Ksum':float(k.sum()),'polyphase':{f'C{a}{b}':c[a,b] for a,b in c},'support_widths':[width(name,n) for n in range(1,5)]}
Path('mcrg/results/kernel_support_report.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# Blocking-kernel support and polyphase report','','| kernel | Ksum | C00 | C01 | C10 | C11 | widths n=1,2,3,4 |','|---|---:|---:|---:|---:|---:|---|']
for n,x in out.items():
 c=x['polyphase'];lines.append(f"| {n} | {x['Ksum']:.12g} | {c['C00']:.12g} | {c['C01']:.12g} | {c['C10']:.12g} | {c['C11']:.12g} | {x['support_widths']} |")
Path('mcrg/results/kernel_support_report.md').write_text('\n'.join(lines)+'\n')
