#!/usr/bin/env python3
"""Create the matched RPDP comparison and held-out trajectory visualizations."""
import argparse,json
from pathlib import Path
import torch


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 variants=('pose_dp','rpdp_local','aic_rpdp');metrics={v:json.loads((a.root/'training'/v/'metrics.json').read_text()) for v in variants}
 table=[]
 for v in variants:
  d=metrics[v]['reports']['development'];base=metrics[v]['reports']['baselines']['persistence']
  table.append({'variant':v,'parameters':metrics[v]['parameters'],'best_update':metrics[v]['best_update'],
   'endpoint_translation_median_mm':d['endpoint_translation_error_mm']['median'],
   'endpoint_translation_p95_mm':d['endpoint_translation_error_mm']['p95'],
   'endpoint_lateral_median_mm':d['endpoint_lateral_error_mm']['median'],
   'endpoint_lateral_p95_mm':d['endpoint_lateral_error_mm']['p95'],
   'endpoint_orientation_median_deg':d['endpoint_orientation_error_deg']['median'],
   'persistence_translation_median_mm':base['endpoint_translation_error_mm']['median'],
   'persistence_lateral_p95_mm':base['endpoint_lateral_error_mm']['p95'],
   'zero_pose_translation_median_mm':metrics[v]['reliance']['zero_pose']['endpoint_translation_error_mm']['median']})
 selected=min(table,key=lambda r:(r['endpoint_lateral_median_mm'],r['endpoint_translation_median_mm']))['variant']
 report={'schema_version':1,'matched_variants':table,'selected_for_live_diagnostic':selected,
         'selection_rule':'lowest held-out endpoint lateral median, then translation median; final split remains sealed',
         'bc_live_gate':'nonzero autonomous insertion with acceptable force before DPPO actor updates',
         'reserved_final_opened':False,'dppo_actor_updates_started':False}
 (a.root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 lines=['# Matched PoseDP/RPDP offline comparison','', '| variant | params | best update | endpoint xyz median / p95 (mm) | lateral median / p95 (mm) | orientation median (deg) | zero-pose xyz median (mm) |','|---|---:|---:|---:|---:|---:|---:|']
 for r in table: lines.append(f"| {r['variant']} | {r['parameters']:,} | {r['best_update']} | {r['endpoint_translation_median_mm']:.3f} / {r['endpoint_translation_p95_mm']:.3f} | {r['endpoint_lateral_median_mm']:.3f} / {r['endpoint_lateral_p95_mm']:.3f} | {r['endpoint_orientation_median_deg']:.3f} | {r['zero_pose_translation_median_mm']:.3f} |")
 lines += ['',f"Selected for the live diagnostic: **{selected}**.",'','The selection uses development episodes only. The reserved final split is sealed.']
 (a.root/'comparison.md').write_text('\n'.join(lines)+'\n')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(3,4,figsize=(14,10))
 for row,v in enumerate(variants):
  x=torch.load(a.root/'training'/v/'heldout_predictions.pt',map_location='cpu',weights_only=False);pred=x['development_prediction'];truth=x['development_truth'];current=x['development_current']
  chosen=torch.linspace(0,len(pred)-1,12).round().long()
  for col,idx in enumerate(chosen):
   ax=axes[row,col%4] if row<3 else axes[-1,col%4]
   if col>=4: continue
   # Four representative trajectories per model in port-frame x/y.
   i=int(chosen[col*3]);t=torch.cat((current[i:i+1],truth[i]),0)[:,:3]*1000;pr=torch.cat((current[i:i+1],pred[i]),0)[:,:3]*1000
   ax.plot(t[:,0],t[:,1],'-o',label='expert',ms=3);ax.plot(pr[:,0],pr[:,1],'-x',label='predicted',ms=4)
   ax.set_title(f'{v} example {i}');ax.set_xlabel('port x (mm)');ax.set_ylabel('port y (mm)');ax.axis('equal');ax.grid(alpha=.2)
 axes[0,0].legend();fig.tight_layout();fig.savefig(a.root/'heldout_future_trajectories.png',dpi=180);plt.close(fig)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
