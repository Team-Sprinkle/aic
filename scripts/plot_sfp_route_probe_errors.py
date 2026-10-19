#!/usr/bin/env python3
"""Plot frozen development waypoint errors against route time and SFP request."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--predictions',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Preserve previous plot')
    rows=[json.loads(line) for line in (args.cache/'rows.jsonl').open()]
    rows=[r for r in rows if r['split']=='validation']
    errors=np.asarray(json.loads(args.predictions.read_text())['errors_mm'],float)
    if len(rows)!=len(errors):raise ValueError('Prediction and development rows mismatch')
    times=np.asarray([r['time_s'] for r in rows])
    targets=np.asarray([[int(np.argmax(r['task'][4:9])),int(np.argmax(r['task'][2:4]))]
                        for r in rows])
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    edges=[0,1,2,3,4,5,5.5,8,12,16]
    centers=[];medians=[];p95=[];counts=[]
    for lo,hi in zip(edges[:-1],edges[1:]):
        mask=(times>=lo)&(times<hi)
        if not mask.any():continue
        centers.append((lo+hi)/2);medians.append(np.median(errors[mask]))
        p95.append(np.quantile(errors[mask],.95));counts.append(mask.sum())
    axes[0].plot(centers,medians,'o-',label='median')
    axes[0].plot(centers,p95,'o-',label='p95')
    axes[0].axhline(20,color='red',ls='--',label='20 mm gate')
    axes[0].axvline(5.5,color='gray',ls=':',label='CheatCode stage boundary')
    axes[0].set(xlabel='Seconds since first teacher command',ylabel='Next-waypoint error (mm)',
                title='Held-out error by route time')
    axes[0].grid(alpha=.25);axes[0].legend(fontsize=8)
    labels=[];values=[]
    for card in range(5):
        for port in range(2):
            mask=(targets[:,0]==card)&(targets[:,1]==port)
            labels.append(f'{card}/{port}')
            values.append(np.quantile(errors[mask],.95))
    axes[1].bar(labels,values)
    axes[1].axhline(20,color='red',ls='--')
    axes[1].set(xlabel='Requested card/port',ylabel='Next-waypoint p95 error (mm)',
                title='One held-out reset per requested target')
    axes[1].grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(args.output,dpi=160)
    print(json.dumps(dict(output=str(args.output),time_counts=list(map(int,counts)),
                          target_p95_mm=dict(zip(labels,map(float,values))))))


if __name__=='__main__':main()
