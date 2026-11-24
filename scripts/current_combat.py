"""Inference helpers from current_combat.py."""

from potion_replacement_combat import _with_dependencies
import capacity_boundary_value as boundary
def proof_validator(capacity):
    import terminal_proof,allocation_probe,root_recommendation
    def validate_boundary(native,legal,mode,regime):
        if native['stop']!='step_budget':
            return boundary.validate_native(native,legal,mode,regime,capacity=capacity)
        # Retain the original coupled potion-action step accounting, without
        # projecting or rewriting any tree/visit/outcome measurements.
        t=native['tree'];r=native['potion_replacement'];b=boundary.allocation('count')
        if (regime!='count' or r['primitive_step_cap']!=b['steps']
                or not 0<=b['steps']-r['primitive_steps']<2*t['simulation_horizon']
                or not len(legal)<=t['completed_simulations']<b['simulations']
                or native['samples']!=native['values'] or native['values']!=t['completed_simulations']
                or native['discarded'] or t['interrupted_simulations']):
            raise ValueError('capacity search step-cap stop is not a complete bounded prefix')
        bounded=_with_dependencies(boundary.validate_native,allocation=lambda _:b,
            request=lambda m,_:boundary.request(m,'count'))
        return bounded(native,legal,mode,'replacement_primitive_cap',capacity=capacity)
    recommendation=_with_dependencies(root_recommendation.validate_native,validate_boundary=validate_boundary)
    allocation=_with_dependencies(allocation_probe.validate_native,validate_original=recommendation)
    return _with_dependencies(terminal_proof.validate_native,validate_parent=allocation)
