"""Inference helpers from combat_resource_utility.py."""

import math
LEGACY = 'combat-resource-utility-v1-capped'


LINEAR = 'combat-resource-utility-v2-linear'


def validate_version(version):
    if version not in (LEGACY, LINEAR):
        raise ValueError('unsupported combat resource utility')
    return version


def scalar(outcome, total, max_hp, version=LINEAR):
    validate_version(version)
    if (type(outcome) is not int or outcome not in (1, 2, 3)
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in (total, max_hp))
            or total < 0 or max_hp <= 0):
        raise ValueError('invalid terminal resource utility')
    if outcome == 2:
        return -3.
    resource = total / max_hp
    if not math.isfinite(resource):
        raise ValueError('nonfinite normalized resource')
    if version == LEGACY:
        resource = min(1., resource)
    return (2. if outcome == 1 else .5) + resource
