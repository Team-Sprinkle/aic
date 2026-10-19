"""Bounded causal teacher continuation from a frozen actor's actual state.

Actor transport/local prefix is autonomous. Subsequent oracle commands are
privileged expert labels only and never count as autonomous insertions.
"""
from .RunGuardedRuleThenLockedTipLocal import RunGuardedRuleThenLockedTipLocal
from .RuleThenOracleTipInsertionDiagnostic import RuleThenOracleTipInsertionDiagnostic


class RuleActorThenOracleContinuationDiagnostic(RunGuardedRuleThenLockedTipLocal):
    def insert_cable(self,task,get_observation,move_robot,send_feedback,**kwargs):
        self.local.max_simulation_sec=12.
        completed=super().insert_cable(task,get_observation,move_robot,send_feedback,**kwargs)
        if not completed or not self.rule_final_valid or self.rule_failure:return False
        observation=get_observation()
        if observation is None:return False
        self._trace(event='teacher_takeover',sim_time=self._control_sim_time(observation),
            state=self.assembler.assemble_ros(observation).tolist(),
            action_owner='teacher',privileged_geometry=True,actor_generated_prefix=True,
            autonomous_success_claim=False,teacher_prefix_in_expert_bc=False)
        return RuleThenOracleTipInsertionDiagnostic.run_oracle_local(
            self,task,get_observation,move_robot,send_feedback,**kwargs)
