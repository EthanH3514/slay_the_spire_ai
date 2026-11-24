"""Opt-in C++ history-tree policy. Python owns contracts/transport, not search."""
from dataclasses import dataclass

from .combat_search import CombatSearchPolicy
from .frozen_evaluation import ResourceAlignedSearchPolicy, _SearchRule

TREE_SEARCH_VERSION = "public-history-tree-v1"
GUARDED_TREE_VERSION = "public-history-tree-v2"
PAIRED_TREE_VERSION = "public-history-tree-v3"
COMBAT_OBJECTIVE_VERSION = "combat-survival-resource-v1"
OBJECTIVE_TREE_VERSION = "public-history-tree-v4-objective"
EFFICIENT_TREE_VERSION = "public-history-tree-v5-nomination"
VERIFICATION_PROPOSALS = 64


@dataclass(frozen=True, slots=True)
class TreeConfig:
    training_rollouts: int = 64
    training_horizon: int = 16
    max_depth: int = 12
    max_nodes: int = 2048
    arena_bytes: int = 8 * 1024 * 1024
    observation_bytes: int = 1024 * 1024

    def __post_init__(self):
        for name, maximum in (("training_rollouts", 4096), ("training_horizon", 512),
                              ("max_depth", 64), ("max_nodes", 65536),
                              ("arena_bytes", 128 * 1024 * 1024), ("observation_bytes", 4 * 1024 * 1024)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError("invalid history tree limit: " + name)


class HistoryTreeSearchPolicy(CombatSearchPolicy):
    def __init__(self, sampler, *, tree_config=None, **kwargs):
        if not getattr(sampler, "conditional_roots", False):
            raise ValueError("history tree requires the public session root service")
        super().__init__(sampler, **kwargs)
        self.tree_config = TreeConfig() if tree_config is None else tree_config
        if type(self.tree_config) is not TreeConfig:
            raise ValueError("expected a versioned TreeConfig")
        self.algorithm_version = TREE_SEARCH_VERSION


class ResourceAlignedTreePolicy(ResourceAlignedSearchPolicy):
    """Identical noncombat/potion routing, different explicit combat search entry."""
    search_type = HistoryTreeSearchPolicy
    def __init__(self, sampler, *, budget, sampling_seed, tree_config=None, catalog=None, root_model="native-seed-conditioned-v1"):
        super().__init__(sampler, budget=budget, sampling_seed=sampling_seed, catalog=catalog, root_model=root_model)
        rule = self.search.rollout
        self.search = self.search_type(sampler, budget=budget, sampling_seed=sampling_seed,
                                             rollout=rule, tree_config=tree_config, root_model=root_model)
        self.fast.combat.rule = _SearchRule(self.search, rule)


class GuardedHistoryTreeSearchPolicy(HistoryTreeSearchPolicy):
    """Separate candidate; shares v1 planning worlds for the policy comparison."""
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.algorithm_version = GUARDED_TREE_VERSION
        self.planning_seed_namespace = TREE_SEARCH_VERSION


class ResourceAlignedGuardedTreePolicy(ResourceAlignedTreePolicy):
    search_type = GuardedHistoryTreeSearchPolicy


class PairedHistoryTreeSearchPolicy(GuardedHistoryTreeSearchPolicy):
    """One frozen challenger, independently paired against the rule suffix."""
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.algorithm_version = PAIRED_TREE_VERSION
        self.verification_proposals = VERIFICATION_PROPOSALS


class ResourceAlignedPairedTreePolicy(ResourceAlignedTreePolicy):
    search_type = PairedHistoryTreeSearchPolicy


class ObjectivePairedTreeSearchPolicy(PairedHistoryTreeSearchPolicy):
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.algorithm_version = OBJECTIVE_TREE_VERSION
        self.combat_objective = COMBAT_OBJECTIVE_VERSION
        # Hold ALL proposal namespaces fixed for single-factor comparisons.
        self.verification_seed_namespace = PAIRED_TREE_VERSION


class EfficientPairedTreeSearchPolicy(ObjectivePairedTreeSearchPolicy):
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.algorithm_version = EFFICIENT_TREE_VERSION


class ResourceAlignedObjectiveTreePolicy(ResourceAlignedTreePolicy):
    search_type = ObjectivePairedTreeSearchPolicy


class ResourceAlignedEfficientTreePolicy(ResourceAlignedTreePolicy):
    search_type = EfficientPairedTreeSearchPolicy


class ResourceAlignedObjectiveFlatPolicy(ResourceAlignedSearchPolicy):
    """Compatibility control: shared objective is already the old flat ranker."""
    def __init__(self, sampler, **kwargs):
        super().__init__(sampler, **kwargs)
        self.search.combat_objective = COMBAT_OBJECTIVE_VERSION
