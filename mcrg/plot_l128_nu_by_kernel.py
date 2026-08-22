#!/usr/bin/env python3
"""Two-panel ν plot matching the native full-result convention."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt

PAIRS=("128→64","64→32","32→16","16→8")

def main():
    p=argparse.ArgumentParser();p.add_argument('--average',type=Path,required=True);p.add_argument('--perfect7',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    datasets=[('matched 2×2',json.loads(a.average.read_text())),('perfect7',json.loads(a.perfect7.read_text()))]
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8),sharey=True,constrained_layout=True)
    for ax,(title,data) in zip(axes,datasets):
        for k in (3,5,7):
            rows=[r for r in data['results'] if r['sector']=='even' and len(r['operators'])==k]
            ax.errorbar(range(4),[r['exponents']['nu'] for r in rows],yerr=[r['bootstrap']['exponent_std'] for r in rows],marker='o',capsize=2,label=f'E1–E{k}')
        ax.axhline(1.,color='black',ls='--',lw=.8);ax.set(title=title,xlabel='blocking pair',xticks=range(4),xticklabels=PAIRS);ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel(r'$\nu$');a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output);plt.close(fig)
if __name__=='__main__':main()
