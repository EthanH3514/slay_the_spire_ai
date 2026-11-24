"""Inference helpers from two_act_construction_pilot.py."""

import json
from spire_ai.stage_reward_targets import Boundary
from spire_ai.normal_run_scope import NormalClearEnvironment
from spire_ai.policy_api import HierarchicalPolicy, policy_input_from_state
class ObservedEnvironment(NormalClearEnvironment):
    def __init__(self, game):
        super().__init__(game)
        self.boundaries = []

    def reset(self,**kwargs):
        self.boundaries=[];return super().reset(**kwargs)

    def step(self,action):
        public=self._current_public_input();before=dict(public.observation['run'])
        combat=HierarchicalPolicy.specialist_for(public).value=='combat'
        result=super().step(action);after=policy_input_from_state(result.state)
        battle=json.loads(self.session.public_battle_result()) if combat else None
        self.boundaries.append(Boundary(len(self.boundaries),before,dict(after.observation['run']),
            battle,result.terminated,result.truncated))
        return result
