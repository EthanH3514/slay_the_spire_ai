"""Opt-in route probability records for a future on-policy collector.

Uses the existing run/route network and exactly one categorical draw. It does
not train, alter routing scores, or make archived demonstrations on-policy.
"""
import hashlib
import json
import math

import torch

from .policy_api import _reject_forbidden_policy_keys
from .public_map import link_route_options
from .run_options import public_run_options
from .run_learning_model import tensorize

VERSION = 'public-route-behavior-record-v1'


def input_digest(public, supplement):
    _reject_forbidden_policy_keys(public.as_dict(), reject_seeds=True)
    _reject_forbidden_policy_keys(supplement, reject_seeds=True)
    return hashlib.sha256(json.dumps(dict(public=public.as_dict(), supplement=supplement),
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def route_input(public, supplement, encoder):
    if public.observation['screen']['type'] != 'MAP' or 'combat' in public.observation:
        raise ValueError('route behavior requires a public MAP decision')
    input_digest(public, supplement)
    options = [dict(o) for o in public_run_options(public)]
    if link_route_options(public.observation, options) != 'linked':
        raise ValueError('route options are not linked to the public graph')
    if any(o.get('destination') != linked.get('destination')
           for o, linked in zip(public_run_options(public), options, strict=True)):
        raise ValueError('route destination identity differs')
    return encoder.encode(public, run_public=supplement)


def sample_route(public, supplement, encoder, model, *, generator, behavior_id):
    """behavior_id identifies frozen weights; it is metadata, never a feature."""
    if not isinstance(behavior_id, str) or len(behavior_id) != 64 or any(c not in '0123456789abcdef' for c in behavior_id):
        raise ValueError('behavior checkpoint SHA256 required')
    if model.training:
        raise ValueError('collect with a frozen evaluation-mode behavior model')
    encoded = route_input(public, supplement, encoder)
    with torch.no_grad():
        dist, value = model.distribution(tensorize([encoded], encoder))
        selected = int(torch.multinomial(dist.probs[0], 1, generator=generator).item())
        logp = float(dist.log_prob(torch.tensor([selected])).item())
    action = encoded.decode(selected)
    record = dict(version=VERSION, behavior_id=behavior_id, schema_id=encoder.schema_id,
        fingerprint=public.fingerprint(), input_sha256=input_digest(public, supplement),
        action=action.as_dict(), action_row=selected, legal_actions=[a.as_dict() for a in public.legal_actions],
        probabilities=dist.probs[0].tolist(), old_log_prob=logp, value=float(value.item()),
        sampled=True, training_admitted=False)
    validate_record(record, public, supplement, encoder, expected_behavior_id=behavior_id)
    return action, record


def validate_record(record, public, supplement, encoder, *, expected_behavior_id):
    if (record['version'] != VERSION or not record['sampled'] or
            record['behavior_id'] != expected_behavior_id or record['schema_id'] != encoder.schema_id):
        raise ValueError('route behavior identity differs')
    encoded = route_input(public, supplement, encoder)
    if (record['fingerprint'] != public.fingerprint() or
            record['input_sha256'] != input_digest(public, supplement) or
            record['legal_actions'] != [a.as_dict() for a in public.legal_actions]):
        raise ValueError('route observation, supplement or legal-action order changed')
    selected, probs = record['action_row'], record['probabilities']
    if (type(selected) is not int or not 0 <= selected < len(public.legal_actions) or
            len(probs) != len(public.legal_actions) or
            any(not math.isfinite(p) or not 0 <= p <= 1 for p in probs) or
            abs(sum(probs)-1) > 2e-6 or probs[selected] <= 0 or
            any(p != 0 for p, allowed in zip(probs, encoded.scope_mask, strict=True) if not allowed)):
        raise ValueError('invalid route behavior probability/mask')
    if (encoded.decode(selected).as_dict() != record['action'] or
            not math.isfinite(record['old_log_prob']) or not math.isfinite(record['value']) or
            abs(math.log(probs[selected])-record['old_log_prob']) > 2e-6):
        raise ValueError('route action or likelihood differs')
