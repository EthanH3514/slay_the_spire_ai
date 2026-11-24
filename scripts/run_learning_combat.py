"""Inference helpers from run_learning_combat.py."""

from neural_prior import PriorSampler
from puct_exploration_trial import coefficient_config
PLANNING_SEED = 125101


class Sampler(PriorSampler):
    def native_tree_search(self, public, seeds, budget, *, trace=False):
        return super().native_tree_search(public, seeds, dict(budget, puct_exploration=coefficient_config(.5)), trace=trace)
