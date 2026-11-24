"""Inference helpers from root_recommendation.py."""

from boundary_value import request as boundary_request, allocation, validate_native as validate_boundary
from tree_standalone_candidate import canonical
from value_leaf_escape import OBJECTIVE
VERSION='public-root-recommendation-v1'


VISITS_OBJECTIVE='completed-visits-then-distribution-objective-v1'


MODES=('score','visits')


def request(mode):
    if mode not in MODES: raise ValueError('unknown root recommendation mode')
    return dict(version=VERSION,mode=mode)


def receipt(mode):
    return dict(request(mode),scope='final-root-only',visits='completed-backups-not-distribution-atoms',
        tie_breaker='distribution-key-then-canonical-action',internal_selection_unchanged=True)


def select(rows,mode):
    request(mode)
    if not rows or any(type(r['visits']) is not int or r['visits']<=0 for r in rows):
        raise ValueError('root recommendation requires complete coverage')
    ordered=sorted(rows,key=lambda r:canonical(r['action']))
    if len({canonical(r['action']) for r in ordered})!=len(ordered): raise ValueError('duplicate root action')
    def key(r):
        score=tuple(r['distribution_key'])
        return (r['visits'],*score) if mode=='visits' else score
    return max(ordered,key=key)['action']


def validate_native(result,legal,mode,regime):
    t=result['tree']
    if (t.get('root_recommendation')!=receipt(mode) or
            t['recommendation']!=(VISITS_OBJECTIVE if mode=='visits' else OBJECTIVE)):
        raise ValueError('root recommendation receipt')
    # Reuse the sealed strict evidence/accounting oracle without weakening it.
    # This shallow READ-ONLY projection changes only what that old oracle expects
    # as its score-selected action/label. The actual returned action is separately
    # checked below and is never replaced by the oracle's action.
    projected=dict(result,action=select(result['root_evidence'],'score'),tree=dict(t,recommendation=OBJECTIVE))
    validate_boundary(projected,legal,'direct',regime)
    expected=select(result['root_evidence'],mode)
    if result['action']!=expected: raise ValueError('root recommendation selection mismatch')
    from spire_ai.actions import Action
    return Action.from_dict(expected)


def make_policy(env,mode,regime):
    from spire_ai.combat_search import SearchBudget
    from spire_ai.history_tree_search import TreeConfig
    from spire_ai.event_sampling import EVENT_ROOT_MODEL
    from value_leaf_escape import ValuePolicy,ValueSearch
    from run_learning_combat_offers import OfferSampler
    from run_learning_combat import PLANNING_SEED
    if getattr(env.module,'root_recommendation_abi',None)!=VERSION: raise ValueError('root recommendation ABI')
    config=request(mode);b=allocation(regime)
    class Search(ValueSearch):
        def configure_request(self,public,limits):
            return dict(super().configure_request(public,limits),boundary_value=boundary_request('direct',regime),
                        root_recommendation=config)
        def validate_native(self,result,public):
            return validate_native(result,[a.as_dict() for a in public.legal_actions],mode,regime),()
    class Policy(ValuePolicy): search_class=Search
    return Policy(OfferSampler(env),budget=SearchBudget(max_seconds=b['seconds'],max_branch_steps=b['steps'],
        max_samples=b['simulations'],max_value_evaluations=b['simulations'],rollout_horizon=16),
        tree_config=TreeConfig(training_rollouts=b['simulations']),sampling_seed=PLANNING_SEED,root_model=EVENT_ROOT_MODEL)
