"""Inference helpers from local_event_environment.py."""

import copy
import json
from dataclasses import replace
from contextual_event_environment import ContextualEventEnvironment
from spire_ai.environment import EnvironmentState
from spire_ai.run_options import public_run_options
from spire_ai.local_event_context import attach
from spire_ai.local_event_selectors import enrich as enrich_selector
from spire_ai.policy_api import policy_input_from_state
from spire_ai.identifiers import canonical_identifier
def remaining_effects(before, action, after, tail):
    """Display order is not execution order. Remove only observed paid costs.

    Costs can precede a target window even when printed after the target clause.
    Compare the total declared cost for that field to the observed transition;
    modified/ambiguous costs stay pending and fail the existing strict verifier.
    This does not infer a phase from an event name or silently waive deck effects.
    """
    if before.observation['screen']['type'] != 'EVENT' or after.observation['screen']['type'] != 'CARD_SELECT':
        return tail
    fields = {'lose_hp': 'hp', 'damage': 'hp', 'lose_gold': 'gold', 'lose_max_hp': 'max_hp'}
    option = public_run_options(before)[before.legal_actions.index(action)]
    declared = option.get('event_choice', {}).get('display_order', ())
    paid = set()
    for field in set(fields.values()):
        costs = [x for x in declared if fields.get(x.get('kind')) == field]
        if not costs or any(type(x.get('amount')) is not int for x in costs):
            continue
        a, b = before.observation['player'].get(field), after.observation['player'].get(field)
        relics = {canonical_identifier(r['id'], 'relic') for r in before.observation.get('inventory',{}).get('relics',())}
        def amount(op):
            n = op['amount']
            if field == 'hp' and n > 0:
                if op['kind'] == 'damage' and n <= 5 and 'torii' in relics:
                    n = 1
                if 'tungsten_rod' in relics:
                    n = max(0, n-1)
            return n
        if type(a) is int and type(b) is int and a - b == sum(amount(x) for x in costs):
            paid.add(field)
    return tuple(x for x in tail if fields.get(x.get('kind')) not in paid)


class LocalEventEnvironment(ContextualEventEnvironment):
    def _event_public(self, public):
        export = getattr(self.session,'public_event_context',None)
        if export is None:return public
        public=attach(public,json.loads(export()))
        if public.observation['screen'].get('event_origin',{}).get('event_name')=='Match and Keep':
            from spire_ai.match_and_keep import attach as attach_match
            public=attach_match(public,json.loads(self.session.public_match_and_keep()))
        if public.observation['screen']['type']=='CARD_SELECT':
            public=enrich_selector(public,self.session.get_actions_text(),catalog=self.encoder.catalog)
        return public

    def _current_public_input(self):
        return self._event_public(super()._current_public_input())

    def _state(self, state):
        state=super()._state(state)
        public=self._event_public(policy_input_from_state(state))
        # EnvironmentState is consumed by the existing sanitizing factory,
        # which deep-copies descendants. A no-event overlay can return its
        # owned/frozen PolicyInput unchanged (e.g. a shop removal selector).
        # Materialize the public wire view at this boundary, not MappingProxy.
        return replace(state,observation=public.as_dict()['observation'],legal_actions=public.legal_actions)

    def step(self, action):
        before = self._current_public_input()
        result = super().step(action)
        self._event_followups = remaining_effects(before, action,
            self._current_public_input(), self._event_followups)
        return result

    def step_replacement(self, plan):
        from spire_ai.stage_reward_targets import Boundary
        from spire_ai.potion_replacement import plans
        # Event search cannot turn a reward transaction into a shop swap.
        before=self._current_public_input()
        if before.observation.get('combat') is not None:
            # ReplacementEnvironment.step dispatches here for a combat refill
            # commitment too. Its raw executor owns validation and both actions;
            # the surrounding step already records the combat boundary.
            return super().step_replacement(plan)
        if before.observation['screen']['type'] != 'COMBAT_REWARD':
            raise ValueError('event replacement requires an already visible reward')
        if plan not in plans(before):
            raise ValueError('stale event replacement')
        # The raw executor sees the same public offers without our additional
        # event-origin provenance. Validate the outer plan, then bind that view.
        raw=replace(plan,fingerprint=self.game._current_public_input().fingerprint())
        result=super().step_replacement(raw)
        after=self._current_public_input()
        self.boundaries.append(Boundary(len(self.boundaries),dict(before.observation['run']),
            dict(after.observation['run']),None,result.terminated,result.truncated))
        return replace(result,state=self._state(result.state))

    def clone_exact(self):
        # Parent replacement wrapper's constructor takes a game, whereas the
        # event wrapper's constructor takes an encoder and opens a new runtime.
        # Do not call that constructor when cloning an already running event.
        expected = {"game", "last_replacement", "encoder", "boundaries", "_event_followups"}
        if set(vars(self)) != expected:
            raise ValueError("unreviewed event wrapper state in exact clone")
        self._check_complete()
        clone = object.__new__(type(self))
        clone.game = self.game.clone_exact()
        clone.encoder = self.encoder  # Read-only catalogue/schema, no model RNG.
        clone.last_replacement = copy.deepcopy(self.last_replacement)
        clone.boundaries = copy.deepcopy(self.boundaries)
        clone._event_followups = copy.deepcopy(self._event_followups)
        return clone

    def event_checkpoint(self):
        # Fresh read catches out-of-band native mutations as well as step changes.
        state = self.game._read_state()
        p = self._current_public_input()
        return EnvironmentState(p.observation, p.legal_actions,
            audit_state=copy.deepcopy(state.audit_state),
            info=dict(step=self.game._step, boundaries=len(self.boundaries),
                pending_effects=copy.deepcopy(self._event_followups),
                replacement=copy.deepcopy(self.last_replacement)))

    def event_supplement(self):
        return json.loads(self.session.public_run_supplement())
