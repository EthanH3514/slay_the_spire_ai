"""Repair the deployed two-menu guard without changing its trained artifacts."""
from .integrated_outside import IntegratedOutside,VERSION,applicable
from .conditional_removal import BOTTLES
from .rule_coverage import _validate_input
from .policy_api import _reject_forbidden_policy_keys
from .event_terminal_proof import choose as guarded_choice

REPAIR='canonical-bottled-event-guard-v1'


def alias_guard_needed(public,catalog):
    if not applicable(public):return False
    raw={r['id'] for r in public.observation['inventory']['relics']}
    canonical={entry['id'] for r in raw if (entry:=catalog.lookup('relic',r)) is not None}
    return bool(canonical&BOTTLES) and not bool(raw&BOTTLES)


class CanonicalEventOutside(IntegratedOutside):
    def choose(self,public,supplement,*,rng):
        if not alias_guard_needed(public,self.encoder.catalog):
            return super().choose(public,supplement,rng=rng)
        _validate_input(public);_reject_forbidden_policy_keys(supplement,reject_seeds=True)
        action,detail=guarded_choice(public,supplement,self.encoder,self.model,rng=rng)
        if action not in public.legal_actions:raise ValueError('canonical event guard returned an illegal action')
        return action,dict(detail,specialist_unavailable='bottled_inventory_outside_partial_event_projection',
            outside_dispatch=VERSION,outside_owner='guarded_base_event',dispatch_repair=REPAIR)
