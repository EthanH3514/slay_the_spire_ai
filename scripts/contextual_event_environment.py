"""Inference helpers from contextual_event_environment.py."""

from dataclasses import replace
from integrated_two_act_training import EventEnvironment as Parent
from potion_replacement_combat import _with_dependencies
from spire_ai.fixed_combat import make_fixed_environment
from spire_ai.policy_api import policy_input_from_state
from spire_ai.contextual_event_options import enrich
from spire_ai.event_effect_followups import followups, verify_followups
class ContextualEventEnvironment(Parent):
    def __init__(self, encoder, *, module_dir):
        super().__init__(encoder, make_fixed_environment(module_dir, session_search=True))
        self._event_followups = ()

    def _state(self,state):
        p=policy_input_from_state(state);q=enrich(p)
        return state if p is q else replace(state,observation=q.observation,legal_actions=q.legal_actions)

    def reset(self,**kwargs):
        self._event_followups=();return self._state(super().reset(**kwargs))

    def _read_state(self):return self._state(super()._read_state())

    def _current_public_input(self):return enrich(super()._current_public_input())

    def step(self,action):
        public=self._current_public_input();tail=self._event_followups
        if public.observation['screen']['type']=='EVENT':self._event_followups=followups(public,action)
        check=lambda t,p:verify_followups(t,p,tail,catalog=self.encoder.catalog)
        result=_with_dependencies(Parent.step,verify_deck=check)(self,action)
        if self._current_public_input().observation['screen']['type']!='CARD_SELECT':self._event_followups=()
        return replace(result,state=self._state(result.state))
