"""Inference helpers from run_learning_combat_offers.py."""

from run_learning_combat_repair import RepairSampler
OFFERS='public-offer-replay-v1'


class OfferSampler(RepairSampler):
    def native_tree_search(self,public,seeds,budget,*,trace=False):
        return super().native_tree_search(public,seeds,dict(budget,offer_replay=OFFERS),trace=trace)
