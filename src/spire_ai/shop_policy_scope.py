"""Current shop scope: buy cards/relics, remove cards, never replace potions.

The historical replacement executor stays available to sealed experiments.
New whole-run callers use ShopRestrictedEnvironment. Ordinary potion purchases
with a free slot and committed combat/reward replacements remain available.
"""
from dataclasses import replace

from .actions import Action, ActionKind
from .environment import InvalidActionError
from .policy_api import PolicyInput, policy_input_from_state
from .potion_replacement import key
from .potion_replacement_environment import ReplacementEnvironment

VERSION = 'shop-no-potion-replacement-v1'
LEARNING_KINDS = ('leave', 'buy_card', 'buy_remove', 'buy_relic')


def is_shop(public):
    return public.observation.get('combat') is None and public.observation.get('screen', {}).get('type') == 'SHOP'


def constrain(public):
    if not is_shop(public):
        return public
    slots = public.observation.get('inventory', {}).get('potions', ())
    free = any(key(p['id']) == 'empty_potion_slot' for p in slots)
    blocked = {a for a in public.legal_actions if a.kind == ActionKind.DISCARD_POTION}
    if not free:
        blocked.update(Action.from_dict(o['action']) for o in
                       public.observation['screen'].get('decision_options', ()) if o.get('kind') == 'buy_potion')
    if not blocked:
        return public
    actions = tuple(a for a in public.legal_actions if a not in blocked)
    if not actions:
        raise ValueError('shop restriction requires a remaining action, including leave')
    observation = public.as_dict()['observation']
    screen = observation['screen']
    if 'decision_options' in screen:
        screen['decision_options'] = [o for o in screen['decision_options']
                                      if Action.from_dict(o['action']) not in blocked]
    return PolicyInput(observation, actions)


def learning_choices(state):
    """First shop-learning scope; this does not rank or label the choices."""
    from .shop_transactions import choices
    return tuple(c for c in choices(state) if c.kind in LEARNING_KINDS)


def encode_learning_decision(encoder, state):
    """Keep the shared representation; select the declared learning candidates."""
    decision = encoder.encode_decision(state)
    indices = [i for i, choice in enumerate(decision.choices) if choice.kind in LEARNING_KINDS]
    result = replace(decision, choices=tuple(decision.choices[i] for i in indices),
                     candidates=tuple(decision.candidates[i] for i in indices))
    if result.choices != learning_choices(state):
        raise ValueError('shop candidate/feature scope mismatch')
    return result


def _state(state):
    public = policy_input_from_state(state)
    filtered = constrain(public)
    if filtered is public:
        return state
    # Preserve all environment-only audit data, never move it into observation.
    return replace(state, observation=filtered.observation, legal_actions=filtered.legal_actions)


class ShopRestrictedEnvironment(ReplacementEnvironment):
    outside_policy_contract = dict(version=VERSION, shop_potion_replacement=False,
                                   shop_learning_kinds=LEARNING_KINDS,
                                   ordinary_potion_purchase_with_free_slot=True)

    def reset(self, **kwargs):
        return _state(super().reset(**kwargs))

    def _read_state(self):
        return _state(self.game._read_state())

    def _current_public_input(self):
        return constrain(self.game._current_public_input())

    def replacement_choices(self):
        if is_shop(self.game._current_public_input()):
            return ()
        return super().replacement_choices()

    def step(self, action):
        self._check_complete()
        public = self.game._current_public_input()
        if is_shop(public) and action not in constrain(public).legal_actions:
            raise InvalidActionError('shop potion replacement is disabled')
        result = super().step(action)
        return replace(result, state=_state(result.state))

    def step_replacement(self, plan):
        self._check_complete()
        if is_shop(self.game._current_public_input()):
            raise InvalidActionError('shop potion replacement is disabled')
        result = super().step_replacement(plan)
        return replace(result, state=_state(result.state))
