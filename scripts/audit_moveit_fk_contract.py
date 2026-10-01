#!/usr/bin/env python3
"""Compare measured-joint FK to controller TCP before blaming action labels."""
import argparse,bisect,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy
from aic_teacher_official.expert_generator.moveit_py_backend import MoveItPyBackendConfig,MoveItPyPlanningBackend


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--streams',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    cfg=MoveItPyBackendConfig(workspace_dir=str(Path(__file__).resolve().parents[1]),use_sim_time=False)
    backend=MoveItPyPlanningBackend(cfg);mi=MoveItPy(node_name='aic_fk_contract_audit',config_dict=backend._moveit_config_dict());state=RobotState(mi.get_robot_model());out=[]
    for source in a.streams:
        r=json.loads(source.read_text());controller=r['streams']['/aic_controller/controller_state'];joints=r['streams']['/joint_states'];errors=[];degrees=[];lags=[];samples=[]
        for i in np.linspace(0,len(controller)-1,128).astype(int):
            t,c=controller[i];j=bisect.bisect_right(joints,t,key=lambda x:x[0])-1
            if j<0:continue
            jt,q=joints[j];names=dict(zip(q['name'],q['position']))
            state.set_joint_group_active_positions(cfg.planning_group,np.array([names[k] for k in r['joint_order']]));state.update();pose=state.get_pose(cfg.end_effector_link)
            pred=np.array([getattr(pose.position,k) for k in 'xyz']);actual=np.array([c['tcp_pose']['position'][k] for k in 'xyz'])
            error=float(np.linalg.norm(pred-actual)*1000)
            rot=Rotation.from_quat([getattr(pose.orientation,k) for k in 'xyzw']);truth=Rotation.from_quat([c['tcp_pose']['orientation'][k] for k in 'xyzw']);deg=float((rot.inv()*truth).magnitude()*180/np.pi)
            errors.append(error);degrees.append(deg);lags.append((t-jt)/1e6);samples.append(dict(sim_time=t/1e9,translation_mm=error,orientation_deg=deg,joint_age_ms=(t-jt)/1e6))
        out.append(dict(source=str(source),samples=samples,translation_mm=dict(zip(['median','p95','max'],map(float,[np.median(errors),np.quantile(errors,.95),max(errors)]))),orientation_deg_p95=float(np.quantile(degrees,.95)),joint_age_ms_max=max(lags)))
    a.output.write_text(json.dumps(dict(schema='aic_fk_controller_contract/v1',episodes=out),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in out]))


if __name__=='__main__':main()
