"""Inference helpers from allocation_probe.py."""

from run_learning_combat_history_guard import validate_receipt
from root_recommendation import validate_native as validate_original
ABI='public-allocation-probe-v1'


MODES=('control','lower_c','uniform')


def request(mode):
    if mode not in MODES:raise ValueError('undeclared allocation mode')
    return dict(version=ABI,mode=mode)


def receipt(mode):
    return dict(request(mode),coefficient=.1 if mode=='lower_c' else .5,uniform_prior=mode=='uniform',
        scope='root-and-internal-PUCT',weights_changed=False,backup_changed=False,final_selector_changed=False)


def validate_native(native,legal,mode):
    expected=receipt(mode)
    if native.get('refusal'):raise ValueError('allocation search refused: '+str(native['refusal']))
    tree=native['tree'];prior=tree['neural_prior']
    if tree.get('allocation_probe')!=expected or prior['puct_c']!=expected['coefficient']:
        raise ValueError('allocation request/receipt mismatch')
    # The frozen oracle hardcodes c=.5. Verify the actual experimental receipt
    # above, then adapt only this scalar in a read-only shallow projection. All
    # action/evidence/accounting checks still inspect the unchanged native result.
    projected=dict(native,tree=dict(tree,neural_prior=dict(prior,puct_c=.5)))
    action=validate_original(projected,legal,'visits','count')
    validate_receipt(native['root_sampling'])
    return action
