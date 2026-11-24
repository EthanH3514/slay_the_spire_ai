"""Inference helpers from value_leaf_escape.py."""

import math
from spire_ai.actions import Action
from spire_ai.event_sampling import EVENT_ROOT_MODEL
from cold_leaf_check import ColdSearch, ColdPolicy
from escape_distribution import summarize_distributions
from tree_standalone_candidate import VERSION as TREE, canonical
VERSION='public-value-leaf-v2-escape'


OBJECTIVE='combat-survival-resource-v3-weighted-distribution'


def validate_native(r,legal,mode):
    if (mode not in ('terminal','learned','probe') or r.get('contract')!=TREE or r.get('combat_objective')!=OBJECTIVE
        or r.get('root_model')!=EVENT_ROOT_MODEL or r.get('refusal') or r.get('resource_fallback') or r.get('unavailable')):
        raise ValueError('value search refused/contract: '+str(r.get('refusal')))
    t=r['tree'];v=t['value_leaf'];p=t['neural_prior'];n=len(legal);complete=t['completed_simulations'];horizon=16 if mode=='learned' else 192
    if (v['version']!=VERSION or v['mode']!=mode or v['prediction_is_terminal_observation'] or not v['one_visit_per_simulation']
        or v['network_calls']>1024 or t['recommendation']!=OBJECTIVE or not t['frozen'] or t['version']!=TREE
        or t['simulation_horizon']!=min(horizon,8192//n) or t['simulation_cap']!=1024
        or t['covered_root_actions']!=n or t['root_action_count']!=n or t['flat_delegations'] or t['verification_simulations']
        or t['actor_mode']!='trained' or t['actor_counts']['calls']!=t['rule_leaf_steps']
        or p['mode']!='learned' or p['selection_calls']!=t['tree_policy_steps'] or p['action_pruning'] or p['pseudo_visits']
        or not p['root_schedule_before_world'] or p['retained_bytes']>p['byte_limit'] or p['action_slots']>p['slot_limit']):
        raise ValueError('value search behavior')
    if (not n<=complete<=r['values']<=r['samples']<=1024 or not 0<r['steps']<=8192
        or t['tree_policy_steps']+t['rule_leaf_steps']!=r['steps'] or t['unused_steps']!=8192-r['steps']
        or r['discarded']!=r['samples']-complete or t['interrupted_simulations']!=r['samples']-complete
        or not complete<=t['backed_up_edges']<=t['tree_policy_steps'] or t['rule_calls']!=r['policies']
        or not 0<=r['policies']<=r['steps'] or t['nodes']>2048 or t['retained_payload_bytes']>8*1024*1024
        or t['max_new_nodes_per_simulation']>3 or t['new_node_cap']!=3):raise ValueError('value search budget')
    sampling=r['root_sampling']
    if (sampling['probability_model']!=EVENT_ROOT_MODEL or not sampling['valid']
        or sampling['construction_work']>sampling['construction_work_limit'] or sampling['random_words']>sampling['random_word_limit']):
        raise ValueError('value root sampler')
    keys={};visits=predicted=observed=atoms=0
    for e in r['root_evidence']:
        name=canonical(e['action']);rows=e['distributions']
        if e['action'] not in legal or name in keys or e['visits']!=len(rows) or not rows:raise ValueError('value root evidence')
        key=summarize_distributions(rows)
        if len(e['distribution_key'])!=6 or any(not math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-10) for a,b in zip(key,e['distribution_key'])):
            raise ValueError('weighted objective mismatch')
        keys[name]=e['distribution_key'];visits+=e['visits']
        for row in rows:
            if row['source']=='model':predicted+=1;atoms+=63
            else:observed+=1;atoms+=1
    if (len(keys)!=n or visits!=complete or predicted!=t['predicted_simulations'] or observed!=t['observed_terminal_simulations']
        or atoms!=t['distribution_atoms'] or atoms>64512 or (mode!='learned' and predicted)
        or (mode=='learned' and v['network_calls']!=predicted) or (mode=='terminal' and v['network_calls'])):
        raise ValueError('model mass counted as observations')
    best=max(sorted(keys),key=lambda k:keys[k])
    if canonical(r['action'])!=best:raise ValueError('value recommendation')
    return Action.from_dict(r['action'])


class ValueSearch(ColdSearch):
    value_mode='learned'
    def configure_request(self,public,limits):
        return dict(super().configure_request(public,limits),neural_prior='learned',value_leaf=self.value_mode,combat_objective=OBJECTIVE)
    def validate_native(self,result,public):
        # Old CandidateEstimate represents integer empirical outcomes only.
        # Leave it empty; complete typed distributions are retained in native evidence.
        return validate_native(result,[a.as_dict() for a in public.legal_actions],self.value_mode),()


class ValuePolicy(ColdPolicy):search_class=ValueSearch
