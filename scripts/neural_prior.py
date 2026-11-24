"""Inference helpers from neural_prior.py."""

import json
import time
from terminal_tree_controller import TerminalSampler, require_capability
from spire_ai.native_search import session_projection_json, metadata_json
VERSION='public-neural-prior-v1'


class PriorSampler(TerminalSampler):
    def native_tree_search(self,public,seeds,budget,*,trace=False):
        env=self._environment;require_capability(env)
        if getattr(env.module,'public_neural_prior_abi',None)!=VERSION:raise ValueError('neural prior ABI')
        if budget.get('neural_prior') not in ('probe','uniform','learned'):raise ValueError('neural prior mode')
        started=time.perf_counter();current=env._prepare_combat_sampling(public,conditioned=True);limits=dict(budget)
        if limits['max_seconds'] is not None:limits['max_seconds']=max(0.,limits['max_seconds']-(time.perf_counter()-started))
        self.request=(public,seeds,limits)
        self.result=json.loads(env.session.search_session_neural_prior(seeds,session_projection_json(current),
            json.dumps([a.as_dict() for a in current.legal_actions]),json.dumps(limits),metadata_json(),trace))
        return self.result
