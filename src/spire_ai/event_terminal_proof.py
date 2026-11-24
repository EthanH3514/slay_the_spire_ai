"""Public HP-loss proofs, independent of event preferences and reward values.

Reviewed event adapters certify execution order; the shared evaluator decides
whether mandatory HP loss is fatal. Display parsing alone is not certification.
Only proven deaths are removed, and only with a proven surviving alternative.
"""
from collections import Counter
from dataclasses import dataclass
import json

from .event_options import EVENT_OPTIONS_VERSION
from .policy_api import _reject_forbidden_policy_keys
from .rule_catalog import RuleCatalog
from .run_options import public_run_options

VERSION = 'public-event-bounded-terminal-proof-v2'


def signature(value):
    return json.dumps(value, default=dict, sort_keys=True, separators=(',', ':'))


# These are semantic adapters, not preferred-action IDs or HP thresholds.
# GameContext::chooseEventOption and original KnowingSkull/CursedTome bytecode
# both apply HP_LOSS before these rewards. Unknown events/effects stay unknown.
REWARDS_AFTER_HP_LOSS = {
    'Knowing Skull': (
        [], [dict(kind='gain_gold', amount=90)],
        [dict(kind='random_card', count=1, pool='colorless', rarity='unspecified')],
        [dict(kind='random_potion', count=1)],
    ),
    'Cursed Tome': ([], [dict(kind='random_relic', pool='cursed_tome_books', count=1)]),
}


@dataclass(frozen=True)
class Proof:
    status: str
    reason: str
    loss: int | None = None
    hp_after_loss: int | None = None


def hp_loss_proof(hp, loss, *, tungsten=False, revival_uncertain=False):
    """One certified HP_LOSS operation, not attack damage or display arithmetic."""
    if (type(hp) is not int or not 0 < hp <= 1000000 or
            type(loss) is not int or not 0 < loss <= 1000000):
        return Proof('unknown', 'unknown_or_invalid_hp_loss')
    realized = max(0, loss-int(tungsten))
    if realized < hp:
        return Proof('survives', 'survives_certified_immediate_hp_loss', realized, hp-realized)
    if revival_uncertain:
        return Proof('unknown', 'revival_requires_separate_semantics', realized)
    return Proof('dead', 'mandatory_hp_loss_before_reward', realized, 0)


def context(public, catalog):
    o = public.observation
    inventory = o.get('inventory', {})
    relics, potions = set(), set()
    if 'relics' not in inventory or 'potions' not in inventory:
        return None
    for kind, target in (('relic', relics), ('potion', potions)):
        for row in inventory[kind+'s']:
            if row.get('id') == 'EMPTY_POTION_SLOT': continue
            item = catalog.lookup(kind, row.get('id'))
            if item is None: return None
            target.add(item['id'])
    return dict(tungsten='TUNGSTEN_ROD' in relics,
                revival_uncertain='LIZARD_TAIL' in relics or 'FAIRY_POTION' in potions,
                reward_healing_possible='BLOODY_IDOL' in relics,
                active_healing_possible=bool(potions & {'BLOOD_POTION','FRUIT_JUICE','ENTROPIC_BREW'}))


def option_proof(name, option, hp, modifiers):
    unknown = Proof('unknown', 'uncertified_event_effects')
    if not isinstance(name, str) or name not in REWARDS_AFTER_HP_LOSS or option.get('kind') != 'event_choice': return unknown
    effect = option.get('event_choice', {})
    clauses = effect.get('unknown_clauses')
    if (effect.get('version') != EVENT_OPTIONS_VERSION or effect.get('event_name') != name or
            effect.get('status') != 'parsed' or not isinstance(clauses, (list, tuple)) or clauses or
            effect.get('fidelity') != 'display_contract_not_effect_certification'):
        return unknown
    costs, effects = list(effect.get('costs', ())), list(effect.get('effects', ()))
    display = effect.get('display_order', ())
    if Counter(map(signature, display)) != Counter(map(signature, costs+effects)):
        return unknown
    if not costs and effects == [dict(kind='leave')]:
        return Proof('survives', 'reviewed_leave_without_hp_cost', 0, hp)
    if (len(costs) != 1 or set(costs[0]) != {'kind', 'amount'} or costs[0]['kind'] != 'lose_hp' or
            signature(effects) not in {signature(x) for x in REWARDS_AFTER_HP_LOSS[name]}):
        return unknown
    return hp_loss_proof(hp, costs[0]['amount'], tungsten=modifiers['tungsten'],
                         revival_uncertain=modifiers['revival_uncertain'])


def repeated_menu_proofs(name, options, proofs, modifiers):
    """A bounded universal successor proof, not a sampled favorable future.

Knowing Skull's four public costs never decrease and its reward choices repeat
the menu. Gold/colorless cards cannot provide revival. Bloody Idol (including
Ceramic Fish gold triggering it) may heal: keep those branches unknown here.
Potion generation can provide revival and therefore has no successor proof.
"""
    if (name!='Knowing Skull' or modifiers is None or modifiers['revival_uncertain'] or
            modifiers['reward_healing_possible'] or modifiers['active_healing_possible'] or len(options)!=4 or
            any(p.status=='unknown' or p.loss is None for p in proofs)):
        return proofs
    effects=[list(o['event_choice']['effects']) for o in options]
    if Counter(map(signature,effects))!=Counter(map(signature,REWARDS_AFTER_HP_LOSS[name])):
        return proofs
    minimum_loss=min(p.loss for p in proofs)
    can_certify={signature(x) for x in REWARDS_AFTER_HP_LOSS[name][1:3]}
    result=[]
    for p,effect in zip(proofs,effects,strict=True):
        if (p.status=='survives' and signature(effect) in can_certify and p.hp_after_loss<=minimum_loss):
            p=Proof('forced_next_death','repeating_menu_all_next_choices_fatal',p.loss,p.hp_after_loss)
        result.append(p)
    return result


def analyze(public, *, catalog=None):
    """Returns a legal-action-aligned mask; unknown does not mean forbidden."""
    from dataclasses import asdict
    _reject_forbidden_policy_keys(public.as_dict(), reject_seeds=True)
    o = public.observation
    allowed = [True]*len(public.legal_actions)
    if o.get('screen', {}).get('type') != 'EVENT' or 'combat' in o:
        return dict(version=VERSION, allowed=allowed, changed=False, reason='not_outside_event', proofs=[])
    name = o['screen'].get('decision_context', {}).get('event_name')
    modifiers = context(public, catalog or RuleCatalog.bundled())
    hp = o.get('player', {}).get('hp')
    options = public_run_options(public)
    if len(options) != len(allowed): raise ValueError('event proof action alignment differs')
    proofs = []
    for action, option in zip(public.legal_actions, options, strict=True):
        if dict(option.get('action', {})) != action.as_dict(): raise ValueError('event proof action join differs')
        p = (Proof('unknown', 'inventory_or_hp_incomplete') if modifiers is None or type(hp) is not int or hp <= 0
             else option_proof(name, option, hp, modifiers))
        proofs.append(p)
    proofs=repeated_menu_proofs(name,options,proofs,modifiers)
    safe = any(p.status == 'survives' for p in proofs)
    if safe: allowed = [p.status not in ('dead','forced_next_death') for p in proofs]
    return dict(version=VERSION, allowed=allowed, changed=not all(allowed),
        reason='certified_survivor_present' if safe else 'no_certified_survivor_keep_original_choices',
        proofs=[asdict(p) for p in proofs])


def choose(public, supplement, encoder, model, candidate=None, *, eligible=False, arm='control', rng):
    """Opt-in drop-in outside dispatcher. Mask logits after normal full encoding.

Retained actions preserve their original relative probabilities. No retries or
rule-based preference; the actual masked likelihood is recorded for audit.
"""
    import torch
    from .run_learning_model import tensorize
    if candidate is not None or eligible or arm != 'control':
        raise ValueError('terminal guard expects normal sampled outside decisions')
    proof = analyze(public, catalog=encoder.catalog)
    encoded = encoder.encode(public, run_public=supplement)
    with torch.no_grad():
        original, value = model.distribution(tensorize([encoded], encoder))
        dist = original
        if proof['changed']:
            mask = torch.tensor(proof['allowed'], dtype=torch.bool) & torch.tensor(encoded.scope_mask)
            safe = any(allowed and p['status']=='survives' for allowed, p in zip(mask.tolist(), proof['proofs'], strict=True))
            if safe:
                dist = torch.distributions.Categorical(logits=original.logits.masked_fill(~mask.unsqueeze(0), -torch.inf))
            else:
                proof = dict(proof, changed=False, reason='survivor_excluded_by_scope')
        selected = int(torch.multinomial(dist.probs[0], 1, generator=rng))
    return encoded.decode(selected), dict(mode='sampled_with_event_terminal_proof' if proof['changed'] else 'unchanged_base_sampled',
        terminal_proof=proof, original_probabilities=original.probs[0].tolist(), probabilities=dist.probs[0].tolist(),
        old_log_prob=float(dist.log_prob(torch.tensor([selected]))), value=float(value.item()), action_row=selected)
