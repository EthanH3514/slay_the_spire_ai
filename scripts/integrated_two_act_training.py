"""Inference helpers from integrated_two_act_training.py."""

import json
from two_act_construction_pilot import ObservedEnvironment
from spire_ai.event_target_selection import build_targets, task_for, verify_deck, public_selector


class EventEnvironment(ObservedEnvironment):
    def __init__(self, encoder, game):
        super().__init__(game)
        self.encoder = encoder

    def step(self, action):
        public = self._current_public_input()
        transition = None
        if task_for(public) is not None:
            menu = build_targets(public, self.encoder,
                                 supplement=json.loads(self.session.public_run_supplement()),
                                 selector=public_selector(self))
            matches = [i for a, i in menu.action_targets if a == action]
            if len(matches) != 1: raise ValueError('event replay target is ambiguous')
            transition = menu.transitions[matches[0]]
        result = super().step(action)
        if transition is not None:
            verify_deck(transition, self._current_public_input())
        return result
