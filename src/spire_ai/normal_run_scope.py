"""New ordinary-clear runs: no campfire Recall, with the existing shop scope.

Historical entries retain their original menus. If Recall was the ONLY campfire
option, expose the simulator's existing rewardless Skip command. This is explicit
in the ruleset; do not silently take Recall or grant the ruby key to escape it.
"""
from dataclasses import replace

from .actions import Action, ActionKind
from .environment import InvalidActionError
from .policy_api import PolicyInput, policy_input_from_state, _thaw_public_value
from .run_options import public_run_options
from .shop_policy_scope import ShopRestrictedEnvironment

VERSION = 'ordinary-clear-no-recall-v1'
SKIP = Action(ActionKind.PROCEED)


def is_campfire(public):
    return public.observation.get('combat') is None and public.observation.get('screen', {}).get('type') == 'REST'


def constrain(public):
    if not is_campfire(public):
        return public
    options = public_run_options(public)
    blocked = {action for action, option in zip(public.legal_actions, options, strict=True)
               if option['kind'] == 'recall'}
    if not blocked:
        return public
    actions = tuple(a for a in public.legal_actions if a not in blocked)
    observation = public.as_dict()['observation']
    # Native options contain immutable nested action maps. Detach them fully;
    # a shallow dict copy leaves a mappingproxy inside this mutable list.
    observation['screen']['decision_options'] = [_thaw_public_value(o) for a, o in
        zip(public.legal_actions, options, strict=True) if a not in blocked]
    if not actions:
        actions = (SKIP,)
        observation['screen']['decision_options'] = [dict(action=SKIP.as_dict(), kind='leave')]
    return PolicyInput(observation, actions)


def _state(state):
    public = policy_input_from_state(state)
    scoped = constrain(public)
    if scoped is public:
        return state
    return replace(state, observation=scoped.observation, legal_actions=scoped.legal_actions)


class NormalClearEnvironment(ShopRestrictedEnvironment):
    outside_policy_contract = dict(ShopRestrictedEnvironment.outside_policy_contract,
        version=VERSION, goal='ordinary_act3_clear', campfire_recall=False,
        rewardless_skip_when_only_recall=True)

    def reset(self, **kwargs):
        return _state(super().reset(**kwargs))

    def _read_state(self):
        return _state(super()._read_state())

    def _current_public_input(self):
        return constrain(super()._current_public_input())

    def step(self, action):
        self._check_complete()
        public = super()._current_public_input()
        if is_campfire(public):
            if action not in constrain(public).legal_actions:
                raise InvalidActionError('campfire Recall is disabled for ordinary clears')
            if action == SKIP and action not in public.legal_actions:
                from .simulator_environment import _SimulatorOption
                # GameContext::chooseCampfireOption(6) only regains map control.
                # As with committed potion replacement, register the native
                # command for this validated action; reading the successor
                # rebuilds the native menu. Never change keys or game RNG.
                self.game._options = (*self.game._options,
                                      _SimulatorOption(SKIP, '6', '[Skip]'))
        result = super().step(action)
        return replace(result, state=_state(result.state))

    def step_replacement(self, plan):
        result = super().step_replacement(plan)
        return replace(result, state=_state(result.state))
