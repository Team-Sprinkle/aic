"""Explicit public-only deployment settings shared by the new live tests.

Scene paths are launcher inputs only. They must not be read by the policy.
The approved bundle's intended configuration needs corrected-launcher live
revalidation; old host-only settings were not sufficient evidence.
"""
from pathlib import Path
import os


def environment(repo,root,bundle,case,*,policy,domain='90',actor=None,recovery='none'):
 env={k:v for k,v in os.environ.items() if not k.startswith('AIC_')}
 env.update(PATH=str(Path.home()/'bin')+':'+env['PATH'],DOCKER_HOST='unix:///run/user/1008/docker.sock',
  OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',AIC_TRANSPORT_EVAL_GPU='3',AIC_TRANSPORT_EVAL_POLICY_GPU='4',
  AIC_TRANSPORT_EVAL_ROS_DOMAIN=str(domain),AIC_TRANSPORT_EVAL_SCENE_DIR=case['scene'],
  AIC_TRANSPORT_EVAL_MAX_TRANSLATION_DELTA='.06',AIC_TRANSPORT_EVAL_MAX_SIMULATION_SEC='180' if actor else '115',
  AIC_TRANSPORT_EVAL_MAX_RUNTIME_SEC='300',AIC_TRANSPORT_EVAL_ORACLE_CONTACTS='true',AIC_AUDIT_CAPTURE_HZ='20',
  AIC_TRANSPORT_EVAL_POLICY=policy,AIC_TRANSPORT_EVAL_TEACHER_PREFIX='false',
  AIC_SC_REQUESTED_HISTORY_LOCATOR_CHECKPOINT=bundle['sc_coarse'],AIC_SC_NATIVE_POSE_CHECKPOINT=bundle['sc_native_refiner'],
  AIC_SC_NATIVE_POSE_MAX_REPROJECTION_PX='2.5',AIC_SFP_REQUESTED_LOCATOR_CHECKPOINT=bundle['sfp_coarse'],
  AIC_SFP_REQUESTED_BACKBONE_CHECKPOINT=bundle['backbone'],AIC_SFP_RULE_NATIVE_POSE_CHECKPOINT=bundle['sfp_native_refiner'],
  AIC_SFP_RULE_MOUNT_PRIORS=bundle['mount_priors'],AIC_SFP_NATIVE_POSE_MAX_REPROJECTION_PX='2.',AIC_SFP_REQUESTED_VIEW_THRESHOLD='.95',
  AIC_SFP_RULE_PRIORS=str(root/'SFP_rule_training_priors_global_region_v2.json'),
  AIC_SFP_RULE_BOARD_CHECKPOINT='/home/nas/chmin/aic_archives/2026-10-05_sfp_slot_local_probe_v1/best.pt',
  AIC_SFP_RULE_CROP_CHECKPOINT='/home/nas/chmin/aic_archives/2026-10-07_sfp_diverse_crop_fit_v1/best.pt',
  AIC_RULE_TRANSPORT_TEMPLATE='/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/common_corridor_candidate_live_v3.json',
  # Preserve the screened rule transport gains. The shared recovery wrapper
  # explicitly sets ONLY its insertion/supervisor controller to the audited90.
  AIC_SHARED_PORT_TCP_CARTESIAN_STIFFNESS='[240,240,240,50,50,50]',
  AIC_SHARED_PORT_TCP_CARTESIAN_DAMPING='[50,50,50,20,20,20]',
  AIC_JOINT_NATIVE_REFINER_CHECKPOINT=str(root/'joint_refiner_full_v2_continue/best.pt'),
  AIC_SC_CANDIDATE_ASSOCIATION_CHECKPOINT=str(root/'SC_rail_crop_association_history_v1/best.pt'),
  AIC_SC_CANDIDATE_GATE_MODE='association_replaces_presence')
 if actor:
  env.update(AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT=str(actor),
   AIC_LOCKED_GRASP_CHECKPOINT='/home/nas/chmin/aic_archives/2026-10-14_transport_handoff_execution_v1/sc_grasp_translation_observation_v1.pt',
   AIC_SFP_GRASP_CHECKPOINT=str(root/'SFP_native_grasp_keypoints_v1/best.pt'),
   AIC_SFP_GRASP_APPROVAL=str(root/'SFP_free_approach_grasp_approval_v1.json'),
   AIC_SHARED_RECOVERY_MODE=recovery,AIC_SHARED_RECOVERY_SIM_BUDGET_S='180',AIC_SHARED_RECOVERY_EXPLORATION_SEED='20261016')
 return env


def effective_environment(audit,env):
 import json
 actual=dict(x.split('=',1) for x in json.load((Path(audit)/'container_environment.json').open()) if '=' in x)
 keys=('AIC_SC_NATIVE_POSE_CHECKPOINT','AIC_SFP_REQUESTED_VIEW_THRESHOLD','AIC_SFP_NATIVE_POSE_MAX_REPROJECTION_PX')
 if env.get('AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT'):keys+=('AIC_SFP_GRASP_CHECKPOINT','AIC_SFP_GRASP_APPROVAL','AIC_SHARED_RECOVERY_MODE','AIC_OFFICIAL_TEACHER_LOCAL_ACTOR_CHECKPOINT')
 for k in keys:
  if actual.get(k)!=env[k]:raise ValueError('Effective deployment setting mismatch: '+k)
 return {k:actual[k] for k in keys}
