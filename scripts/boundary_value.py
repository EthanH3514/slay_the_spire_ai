"""Inference helpers from boundary_value.py."""

import math
VERSION = 'public-boundary-value-v4'


MODES = ('control', 'direct')


def allocation(regime):
    if regime == 'count': return dict(simulations=1024, steps=16384, seconds=None)
    if regime == 'time': return dict(simulations=4096, steps=65536, seconds=2.0)
    if regime == 'gate': return dict(simulations=1024, steps=8192, seconds=None)
    if regime == 'tiny': return dict(simulations=32, steps=512, seconds=None)
    raise ValueError('unknown boundary allocation')


def request(mode, regime):
    if mode not in MODES: raise ValueError('unknown boundary mode')
    return dict(version=VERSION, mode=mode, limit=allocation(regime)['simulations'])


def validate_selection_accounting(r, regime):
    t = r['tree']
    gap = t['neural_prior']['selection_calls'] - t['tree_policy_steps']
    if gap == 0: return
    # Root scheduling intentionally precedes sampled-world construction. A
    # deadline can therefore interrupt AFTER selecting the next root action but
    # BEFORE executing it. Only that one unexecuted root selection is permitted:
    # all tree steps must still belong to completed/backed-up simulations.
    if not (regime == 'time' and r['stop'] == 'deadline' and gap == 1 and
            r['discarded'] == t['interrupted_simulations'] == 1 and
            r['samples'] == t['completed_simulations'] + 1 and
            r['values'] == t['completed_simulations'] and
            t['backed_up_edges'] == t['tree_policy_steps']):
        raise ValueError('boundary pending root selection accounting')


def validate_native(r, legal, mode, regime):
    from escape_distribution import summarize_distributions
    from combat_resource_utility import LEGACY,validate_version
    from tree_standalone_candidate import canonical, VERSION as TREE
    from spire_ai.event_sampling import EVENT_ROOT_MODEL
    from value_leaf_escape import OBJECTIVE, VERSION as VALUE_ABI
    from spire_ai.actions import Action
    b = allocation(regime); n = len(legal)
    utility=validate_version(r.get('resource_utility',LEGACY))
    if (r.get('contract') != TREE or r.get('combat_objective') != OBJECTIVE or
            r.get('root_model') != EVENT_ROOT_MODEL or r.get('refusal') or
            r.get('resource_fallback') or r.get('unavailable')):
        raise ValueError('boundary search refusal: '+str(r.get('refusal')))
    t = r['tree']; v = t['value_leaf']; p = t['neural_prior']; e = t['boundary_value']
    done = t['completed_simulations']; limit = b['simulations']
    if (any(e[k] != x for k,x in request(mode,regime).items()) or
            e['new_node_cap_unchanged'] != 3 or
            e['value_target'] != 'frozen-R2-greedy-not-current-search' or
            len(e['cutoff_depths']) != 17 or sum(e['cutoff_depths']) != e['boundary_cutoffs'] or
            not 0 <= e['boundary_cutoffs'] <= r['samples']):
        raise ValueError('boundary receipt')
    if (v['version'] != VALUE_ABI or v['mode'] != 'learned' or v['prediction_is_terminal_observation'] or
            not v['one_visit_per_simulation'] or v['call_limit'] != limit or v['network_calls'] > limit or
            t['recommendation'] != OBJECTIVE or not t['frozen'] or t['version'] != TREE or
            t['simulation_horizon'] != min(16,b['steps']//n) or t['simulation_cap'] != limit or
            t['covered_root_actions'] != n or t['root_action_count'] != n or
            t['flat_delegations'] or t['verification_simulations'] or t['actor_mode'] != 'trained' or
            t['actor_counts']['calls'] != t['rule_leaf_steps'] or
            p['mode'] != 'learned' or p['puct_c'] != .5 or
            p['action_pruning'] or p['pseudo_visits'] or not p['root_schedule_before_world'] or
            p['retained_bytes'] > p['byte_limit'] or p['action_slots'] > p['slot_limit']):
        raise ValueError('boundary search behavior')
    validate_selection_accounting(r,regime)
    if mode == 'direct' and (t['actor_counts']['calls'] or t['rule_leaf_steps']):
        raise ValueError('direct mode executed old actor')
    if mode == 'control' and e['boundary_cutoffs']: raise ValueError('control stopped at boundary')
    if (not n <= done <= r['values'] <= r['samples'] <= limit or not 0 < r['steps'] <= b['steps'] or
            t['tree_policy_steps']+t['rule_leaf_steps'] != r['steps'] or
            t['unused_steps'] != b['steps']-r['steps'] or r['discarded'] != r['samples']-done or
            t['interrupted_simulations'] != r['samples']-done or
            not done <= t['backed_up_edges'] <= t['tree_policy_steps'] or
            t['rule_calls'] != r['policies'] or not 0 <= r['policies'] <= r['steps'] or
            t['nodes'] > 2048 or t['retained_payload_bytes'] > 8*1024*1024 or
            t['max_new_nodes_per_simulation'] > 3 or t['new_node_cap'] != 3):
        raise ValueError('boundary work accounting')
    if regime in ('count','tiny') and (done != limit or r['stop'] != 'simulation_cap'):
        raise ValueError('not equal completed simulation counts')
    sampling = r['root_sampling']
    # A deadline invalidates ONLY the unfinished sampler, not earlier accepted
    # and backed-up worlds. Require its typed native cause and exact prefix.
    timed_sampler_stop = (regime == 'time' and r['stop'] == 'deadline' and
        e['root_failure_code'] == 'deadline' and not sampling['valid'] and
        p['selection_calls'] == t['tree_policy_steps'] + 1 and
        sampling['prepared_rounds'] == done)
    if (sampling['probability_model'] != EVENT_ROOT_MODEL or
            not ((sampling['valid'] and e['root_failure_code'] == '') or timed_sampler_stop) or
            sampling['construction_work'] > sampling['construction_work_limit'] or
            sampling['random_words'] > sampling['random_word_limit']):
        raise ValueError('boundary sampler')
    keys = {}; visits = predicted = observed = atoms = 0
    for edge in r['root_evidence']:
        name = canonical(edge['action']); rows = edge['distributions']
        if any(row.get('resource_utility',LEGACY)!=utility for row in rows):
            raise ValueError('mixed or missing resource utility evidence')
        if edge['action'] not in legal or name in keys or edge['visits'] != len(rows) or not rows:
            raise ValueError('boundary root coverage')
        key = summarize_distributions(rows)
        if len(edge['distribution_key']) != 6 or any(not math.isclose(a,c,rel_tol=1e-12,abs_tol=1e-10)
                for a,c in zip(key,edge['distribution_key'])):
            raise ValueError('boundary weighted objective')
        keys[name] = key; visits += len(rows)
        for row in rows:
            if row['source'] == 'model': predicted += 1; atoms += 63
            elif row['source'] == 'terminal': observed += 1; atoms += 1
            else: raise ValueError('boundary evidence source')
    if (len(keys) != n or visits != done or predicted != t['predicted_simulations'] or
            observed != t['observed_terminal_simulations'] or atoms != t['distribution_atoms'] or
            atoms > 63*limit or t['distribution_atom_limit'] != 63*limit or
            v['network_calls'] != predicted):
        raise ValueError('boundary model mass/terminal accounting')
    if canonical(r['action']) != max(sorted(keys),key=lambda k:keys[k]):
        raise ValueError('boundary recommendation mismatch')
    return Action.from_dict(r['action'])
