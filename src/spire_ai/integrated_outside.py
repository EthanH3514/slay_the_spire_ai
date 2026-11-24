"""Explicit ownership for the deployed outside specialists.

Only the existing construction learner changes during PPO. Event and boss
heads are frozen artifacts; unknown event operations retain the base actor.
"""
from .event_candidate_policy import EventCandidatePolicy, applicable
from .boss_relic_policy import BossRelicPolicy
from .event_target_selection import task_for, choose as choose_target
from .event_terminal_proof import choose as guarded_choice
from .conditional_removal import candidate_groups as removable
from .event_candidate_policy import BOTTLES
from .rule_coverage import _validate_input
from .policy_api import _reject_forbidden_policy_keys

VERSION = 'integrated-outside-specialists-v1'


class IntegratedOutside:
    def __init__(self, construction, encoder, model, base, *, event, boss):
        self.construction = construction
        self.encoder, self.model, self.base = encoder, model, base
        self.event, self.boss = event, boss

    @classmethod
    def load(cls, event_path, boss_path, construction, encoder, model, base):
        event = EventCandidatePolicy(event_path, encoder)
        boss = BossRelicPolicy(boss_path, encoder)
        return cls(construction, encoder, model, base, event=event, boss=boss)

    def choose(self, public, supplement, *, rng):
        _validate_input(public)
        _reject_forbidden_policy_keys(supplement, reject_seeds=True)
        screen = public.observation['screen']['type']
        if 'combat' in public.observation:
            raise ValueError('combat retains its own controller')
        if task_for(public) is not None:
            action, detail, _ = choose_target(public, self.construction, supplement=supplement)
            owner = 'shared_event_target'
        elif screen == 'BOSS_RELIC_REWARDS':
            action = self.boss.choose(public)
            detail, owner = dict(self.boss.last_decision), 'boss_candidate'
        elif applicable(public):
            # These are documented exclusions of the historical projection,
            # not swallowed inference/encoding failures. Keep the trained base
            # actor for these menus and record their coverage explicitly.
            inv = public.observation['inventory']
            reason = ('bottled_inventory_outside_partial_event_projection'
                      if BOTTLES & {r['id'] for r in inv['relics']} else
                      'no_removable_card' if not removable(inv['deck'], self.encoder.catalog) else None)
            if reason is None:
                action = self.event.choose(public)
                detail, owner = dict(self.event.last_decision), 'event_candidate'
            else:
                action, detail = guarded_choice(public, supplement, self.encoder, self.model, rng=rng)
                detail, owner = dict(detail, specialist_unavailable=reason), 'guarded_base_event'
        elif screen == 'EVENT':
            action, detail = guarded_choice(public, supplement, self.encoder, self.model, rng=rng)
            owner = 'guarded_base_event'
        else:
            action, detail = self.base(public, supplement, self.encoder, self.model, None,
                                      eligible=False, arm='control', rng=rng)
            owner = 'unchanged_base'
        if action not in public.legal_actions:
            raise ValueError('integrated outside controller returned an illegal action')
        return action, dict(detail, outside_dispatch=VERSION, outside_owner=owner)
