"""Inference helpers from potion_replacement_combat.py."""

from types import FunctionType
from spire_ai.potion_replacement import constrain, VERSION
class PolicyEnvironment:
    def __init__(self, environment): self.environment = environment
    def __getattr__(self, name): return getattr(self.environment,name)

    def _prepare_combat_sampling(self, expected_public=None, *, conditioned=False):
        current = self.environment._prepare_combat_sampling(None,conditioned=conditioned)
        eligible = constrain(current)
        if expected_public is not None and expected_public.fingerprint() != eligible.fingerprint():
            raise ValueError('replacement sampler input does not match the current root')
        return eligible


def decision_accounting(native, cap=16384):
    """Check physical work, then reuse the old decision-step accounting oracle.

    Counts/returns/visits are never rewritten. Only the old oracle's work fields
    are projected, because it predates two-primitive policy choices.
    """
    if native.get('forced'): return native
    r = native.get('potion_replacement',{}); tree = native['tree']
    extra = r.get('additional_primitive_steps')
    if (r.get('version') != VERSION or type(extra) is not int or extra < 0
            or r.get('completed_replacements') != extra or r.get('evaluated_between_actions') is not False
            or r.get('primitive_steps') != native['steps'] or r.get('primitive_step_cap') != cap
            or not 0 < native['steps'] <= cap or tree['unused_steps'] != cap-native['steps']
            or r.get('decision_steps') != tree['tree_policy_steps']+tree['rule_leaf_steps']
            or r['decision_steps']+extra != native['steps'] or extra > r['decision_steps']):
        raise ValueError('replacement primitive/decision accounting mismatch')
    return dict(native,steps=r['decision_steps'],tree=dict(tree,unused_steps=cap-r['decision_steps']))


def _with_dependencies(function, **dependencies):
    """Reuse sealed validators with instance-local dependencies, never globals."""
    result = FunctionType(function.__code__,dict(function.__globals__,**dependencies),
                          function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__ = None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    result.__annotations__ = dict(function.__annotations__)
    result.__dict__.update(function.__dict__)
    result.__qualname__, result.__module__, result.__doc__ = function.__qualname__, function.__module__, function.__doc__
    if hasattr(function, '__type_params__'):
        result.__type_params__ = function.__type_params__
    return result
