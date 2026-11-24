"""Inference helpers from terminal_tree_controller.py."""

import json
import time
import tree_standalone_candidate as legacy
from spire_ai.environment import EnvironmentError
OBJECTIVE = 'combat-survival-resource-v2-terminal-hp'


def require_capability(environment):
    module = environment.module
    if (not environment.session_search or not legacy.compatible_versions(module)
            or not legacy.compatible_session(module)
            or getattr(module, 'public_standalone_tree_search_abi', None) != legacy.VERSION
            or getattr(module, 'public_terminal_objective_abi', None) != OBJECTIVE
            or getattr(module, 'public_event_offer_abi', None) != legacy.EVENT_ROOT_MODEL):
        raise ValueError('explicit terminal V2 standalone/event module required')


def validate_result(result, public, budget, config):
    if result.get('combat_objective') != OBJECTIVE:
        raise EnvironmentError('terminal V2 result objective mismatch')
    return legacy.validate_result(dict(result, combat_objective=legacy.COMBAT_OBJECTIVE_VERSION), public, budget, config)


class TerminalSampler(legacy.StandaloneSampler):
    def __init__(self, environment):
        require_capability(environment)
        super().__init__(environment)

    def native_tree_search(self, public, seeds, budget, *, trace=False):
        env = self._environment
        require_capability(env)
        if (budget.get('combat_objective') != OBJECTIVE or budget.get('tree_algorithm') != legacy.VERSION
                or budget.get('root_model') != legacy.EVENT_ROOT_MODEL):
            raise ValueError('terminal V2 request contract mismatch')
        started = time.perf_counter()
        current = env._prepare_combat_sampling(public, conditioned=True)
        limits = dict(budget)
        if limits['max_seconds'] is not None:
            limits['max_seconds'] = max(0., limits['max_seconds']-(time.perf_counter()-started))
        self.request = (public, seeds, dict(limits))
        self.result = json.loads(env.session.search_session_standalone_tree(seeds,
            legacy.session_projection_json(current), json.dumps([a.as_dict() for a in current.legal_actions]),
            json.dumps(limits), legacy.metadata_json(), trace))
        return self.result


class TerminalSearch(legacy.StandaloneSearch):
    def configure_request(self, public, limits):
        return dict(limits, combat_objective=OBJECTIVE)

    def validate_native(self, result, public):
        return validate_result(result, public, self.budget, self.tree_config)


class TerminalPolicy(legacy.StandalonePolicy):
    search_class = TerminalSearch
