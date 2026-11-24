"""Inference helpers from autoplay_target_combat.py."""

ABI='public-autoplay-target-prefix-v1'


def validate_target_receipt(native,enabled):
    if native.get('forced'):return
    receipt=native['root_sampling'].get('autoplay_target_prefix')
    if not enabled:
        if receipt is not None:raise ValueError('disabled target conditioning receipt')
        return
    if not isinstance(receipt,dict) or receipt.get('version')!=ABI:raise ValueError('target conditioning receipt')
    if receipt['probability_model']!='native-card-rng-conditional-target-prefix-v1':raise ValueError('target probability model')
    if receipt['eligible']:
        if not receipt['prepared'] or not receipt['entropy_verified'] or receipt['gap']:raise ValueError('unverified target factorization')
        if not 1<=receipt['accepted_target_partitions']<=receipt['target_partitions']<=125:raise ValueError('target partition count')
        if not 1<=receipt['representative_seed_proposals']<=4096 or receipt['seed_attempts']<1:raise ValueError('target conditioning work')
    elif receipt['gap'] not in ('outside_reviewed_program','unconditioned_draw_prefix','unreviewed_additional_randomness'):
        raise ValueError('unknown target conditioning scope')
