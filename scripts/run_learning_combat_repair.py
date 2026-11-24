"""Inference helpers from run_learning_combat_repair.py."""

from run_learning_combat import Sampler
SAMPLING='public-run-sampling-v1'


class RepairSampler(Sampler):
    def native_tree_search(self,public,seeds,budget,*,trace=False):
        return super().native_tree_search(public,seeds,dict(budget,run_sampling=SAMPLING),trace=trace)
