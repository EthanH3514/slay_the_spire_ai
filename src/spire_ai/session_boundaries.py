"""Versioned, opt-in fixed flat controller with no rule fallback on refusal.

Keeps old experiments/classes untouched. The parent flat aggregator is reused
with its strict *failure handling*, while this capability enforces the separately
declared event model. It never exposes the live session or audit RNG to scoring.
"""
from dataclasses import asdict, replace
import json
import math
import time

from .combat_search import CombatSearchPolicy, SearchBudget
from .environment import EnvironmentError
from .frozen_evaluation import ResourceAlignedSearchPolicy, _SearchRule
from .native_search import compatible_session, compatible_versions, metadata_json, session_projection_json

VERSION = "public-session-boundaries-v1"
MODEL = "public-event-offer-v2-discovery"


def selection_budget(public):
    count = len(public.legal_actions)
    if not 1 <= count <= 1024:
        raise EnvironmentError("session_boundary_action_limit")
    large = public.observation.get("combat", {}).get("input_state") == "CARD_SELECT" and count > 128
    return SearchBudget(max_branch_steps=96 * count, max_samples=2 * count, max_seconds=30.,
                        max_value_evaluations=2 * count) if large else SearchBudget()


def contract(sampling_seed=19000):
    if type(sampling_seed) is not int or not 0 <= sampling_seed < 2**64:
        raise ValueError("sampling_seed must be unsigned 64-bit")
    return dict(version=VERSION, root_model=MODEL, sampling_seed=sampling_seed,
                default_budget=asdict(SearchBudget()), large_selection_threshold=128,
                max_candidates=1024, large_samples_per_candidate=2, large_horizon=48,
                large_steps_per_candidate=96, construction_work_per_candidate=128000,
                construction_work_min=8000000, construction_work_max=131072000,
                random_words=1000000, large_max_seconds=30., fallback_policy="refuse", actor_training_labels=False)


class BoundarySampler:
    conditional_roots = True

    def __init__(self, environment):
        if (not environment.session_search or not compatible_session(environment.module)
                or not compatible_versions(environment.module)
                or getattr(environment.module, "public_session_boundaries_abi", None) != VERSION
                or hasattr(environment.module.SimulatorSession, "_test_fixture")):
            raise ValueError("explicit boundary-repair module with fixture API OFF required")
        self._environment = environment
        self.last_boundary = None

    def native_search(self, public, seeds, budget, *, trace=False):
        env = self._environment
        started = time.perf_counter()
        current = env._prepare_combat_sampling(public, conditioned=True)
        expected_budget = asdict(selection_budget(current))
        if ({k:v for k,v in budget.items() if k != "max_seconds"}
                != {k:v for k,v in expected_budget.items() if k != "max_seconds"}):
            raise EnvironmentError("session_boundary_budget")
        seconds=budget.get("max_seconds")
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0 <= seconds <= expected_budget["max_seconds"]:
            raise EnvironmentError("session_boundary_deadline")
        limits = dict(budget, root_model=MODEL, session_boundaries=VERSION)
        if limits["max_seconds"] is not None:
            limits["max_seconds"] = max(0., limits["max_seconds"] - (time.perf_counter()-started))
        result = json.loads(env.session.search_session_public(
            seeds, session_projection_json(current), json.dumps([a.as_dict() for a in current.legal_actions]),
            json.dumps(limits), metadata_json(), trace))
        if (result.get("root_model") != MODEL or result.get("session_boundaries") != VERSION
                or result.get("resource_fallback") is not None):
            raise EnvironmentError("session_boundary_result_contract")
        self.last_boundary = dict(version=VERSION, root_model=MODEL, budget=limits)
        return result


class BoundarySearch(CombatSearchPolicy):
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.algorithm_version = VERSION

    def rank(self, public):
        self.budget = selection_budget(public)
        self.sampler.last_boundary = None
        try:
            result = super().rank(public)
            return replace(result, resampler_version=MODEL)
        finally:
            if self.sampler.last_boundary is not None:
                self.last_backend["boundary_contract"] = self.sampler.last_boundary


class BoundaryPolicy(ResourceAlignedSearchPolicy):
    def __init__(self, environment, *, sampling_seed=19000, catalog=None):
        contract(sampling_seed)
        sampler = BoundarySampler(environment)
        super().__init__(sampler, budget=SearchBudget(), sampling_seed=sampling_seed, catalog=catalog)
        rule = self.search.rollout
        self.search = BoundarySearch(sampler, sampling_seed=sampling_seed, rollout=rule)
        self.fast.combat.rule = _SearchRule(self.search, rule)
