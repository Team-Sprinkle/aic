#!/usr/bin/env python3
"""One fixed equal-weight far/near RGB orientation fusion screen, no rollout."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

root=Path('/home/nas/chmin/aic_archives/2026-10-18_SC_explicit_continuation_v1')
output=root/'fixed_far_near_orientation_fusion_screen';output.mkdir(exist_ok=False)
truth_rows=json.loads((root/'close_reestimation_audit_fresh4_v4/records.json').read_text())
truth={(r['episode'],r['time_s']):np.array(r['truth_pose']) for r in truth_rows}
predictions=json.loads((root/'final80_frozen_rollout_generalization/summary.json').read_text())['results']
results=[]
for episode in sorted({r['episode'] for r in predictions}):
    means={}
    for phase in ['acquisition_150mm','hold_25mm']:
        rows=[r for r in predictions if r['episode']==episode and r['phase']==phase]
        means[phase]=Rotation.from_matrix([r['predicted_rotation'] for r in rows]).mean().as_matrix()
    labels=[truth[(r['episode'],r['time_s'])][:3,:3] for r in predictions if r['episode']==episode]
    drift=max(float(np.degrees(Rotation.from_matrix(labels[0].T@r).magnitude())) for r in labels)
    assert drift<.001,'Static-grasp assumption must be reported, not silently applied to slip data'
    target=Rotation.from_matrix(labels).mean().as_matrix()
    fused=Rotation.from_matrix([means['acquisition_150mm'],means['hold_25mm']]).mean().as_matrix()
    def score(r):
        # Actual port-relative rotation expected after perfect estimated alignment.
        relative=r.T@target
        return dict(orientation_deg=float(np.degrees(Rotation.from_matrix(relative).magnitude())),
            axial_tilt_deg=float(np.degrees(np.arccos(np.clip(relative[2,2],-1,1)))),
            rotation_vector_deg=np.degrees(Rotation.from_matrix(relative).as_rotvec()).tolist(),
            corner_axial_excess_proxy_mm=float(10.8*abs(relative[2,0])+3.65*abs(relative[2,1])+2.25*(1-relative[2,2])))
    results.append(dict(episode=episode,near_only=score(means['hold_25mm']),equal_far_near=score(fused),
        truth_drift_deg=drift,far_near_prediction_difference_deg=float(np.degrees(Rotation.from_matrix(means['acquisition_150mm'].T@means['hold_25mm']).magnitude()))))
near=np.array([r['near_only']['orientation_deg'] for r in results]);fused=np.array([r['equal_far_near']['orientation_deg'] for r in results])
corner_improved=sum(r['equal_far_near']['corner_axial_excess_proxy_mm']<r['near_only']['corner_axial_excess_proxy_mm'] for r in results)
material=bool(np.percentile(fused,95)<=.85*np.percentile(near,95) and corner_improved>=3 and np.all(fused<=near+.1))
report=dict(results=results,near_episode_p50_p95_deg=np.percentile(near,[50,95]).tolist(),
    fused_episode_p50_p95_deg=np.percentile(fused,[50,95]).tolist(),corner_proxy_improved_episodes=corner_improved,
    material_improvement_gate=material,gate='At least15% lower episodep95, cornerproxy improves>=3/4, no episode worsens>.1degree',
    same_stronger_model_used_at_both_distances=True,old_biased_RGB_model_not_averaged=True,
    fusion_weights_fixed_before_screen=[.5,.5],geometry_for_scoring_only=True,
    automatic_deployment=False,real_gripper_slip_not_covered=True)
(output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
