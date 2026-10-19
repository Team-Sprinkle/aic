#!/usr/bin/env python3
"""Plot held-out executed-chunk error across bounded 8/4 and 12/6 fits."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eight',type=Path,nargs=2,required=True)
    p.add_argument('--twelve',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,(title,phases) in zip(axes,(('8 predicted / 4 executed',a.eight),
                                      ('12 predicted / 6 executed',a.twelve))):
        offset=0
        for phase,root in enumerate(phases,1):
            history=json.loads((root/'history.json').read_text())
            x=[offset+r['update'] for r in history]
            sfp=[r['validation']['sfp_other']['executed_mean_mm_p95'] for r in history]
            sc=[r['validation']['sc_other']['executed_mean_mm_p95'] for r in history]
            ax.plot(x,sfp,color='#2563eb',label='SFP held-out p95' if phase==1 else None)
            ax.plot(x,sc,color='#ea580c',label='SC held-out p95' if phase==1 else None)
            ax.axvline(x[-1],color='gray',lw=.7,ls='--')
            summary=json.loads((root/'summary.json').read_text())
            selected=next(r for r in history if r['update']==summary['best_update'])
            ax.scatter([offset+selected['update']]*2,
                       [selected['validation'][f]['executed_mean_mm_p95'] for f in ('sfp_other','sc_other')],
                       color=['#2563eb','#ea580c'],marker='x',s=50,zorder=3)
            ax.text(x[-1],ax.get_ylim()[1] if phase>1 else max(sfp+sc),f' phase {phase}',
                    fontsize=8,rotation=90,va='top')
            offset+=history[-1]['update']
        ax.set_title(title);ax.set_xlabel('Cumulative update count')
        ax.grid(alpha=.25);ax.legend(loc='upper right',fontsize=8)
    axes[0].set_ylabel('Held-out executed-chunk mean translation error, p95 (mm)')
    fig.suptitle('Episode-grouped development curves; × marks selected checkpoint in each phase')
    fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=180)
    print(a.output)


if __name__=='__main__':main()
