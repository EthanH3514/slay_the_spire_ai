"""Opt-in boss specialist using the existing run encoder and candidate head.

Historical and live inputs share an explicit partial projection. Native action
indices are mapped by relic identity, never reused as sorted dataset indices.
"""
from dataclasses import dataclass
from pathlib import Path

from .actions import Action, ActionKind
from .boss_relic_demonstrations import VERSION as ROW_VERSION, EXCLUDED_RELICS, encode_row
from .expert_cards import PARTIAL_FIELDS, card_key
from .expert_card_policy import _card
from .policy_api import PolicyInput
from .policy_scope import assess_policy_scope
from .rule_coverage import _validate_input
from .run_options import RUN_OPTIONS_VERSION, public_run_options

VERSION = 'boss-relic-candidate-head-v1'


@dataclass(frozen=True)
class BossProjection:
    row: dict
    encoded: object
    actions: tuple
    meanings: tuple

    def decode(self, index):
        if type(index) is not int or not 0 <= index < len(self.actions):
            raise ValueError('invalid projected boss action')
        return self.actions[index]


def project(public, encoder):
    _validate_input(public)
    obs=public.observation; run=obs.get('run',{}); screen=obs.get('screen',{})
    if screen.get('type')!='BOSS_RELIC_REWARDS' or 'combat' in obs:
        raise ValueError('boss relic decision required')
    act=run.get('act'); floor=run.get('floor'); ascension=run.get('ascension')
    if (run.get('character')!='IRONCLAD' or act not in (1,2) or floor!={1:17,2:34}[act]
            or type(ascension) is not int or not 0<=ascension<=20):
        raise ValueError('outside reviewed boss window')
    inv=obs.get('inventory',{})
    if not isinstance(inv.get('deck'),(tuple,list)) or not isinstance(inv.get('relics'),(tuple,list)):
        raise ValueError('missing boss inventory')
    deck=sorted((_card(c,encoder.catalog) for c in inv['deck']),key=card_key)
    relics=[]
    for r in inv['relics']:
        entry=encoder.catalog.lookup('relic',r.get('id'))
        if entry is None or entry['id'] in relics: raise ValueError('unknown or duplicate owned relic')
        relics.append(entry['id'])
    eligible=assess_policy_scope(obs,public.legal_actions).eligible_actions
    if len(set(eligible))!=len(eligible): raise ValueError('duplicate boss actions')
    local=PolicyInput(obs,eligible); choices={}; skip=None
    for action,row in zip(eligible,public_run_options(local),strict=True):
        if row['kind']=='boss_relic' and action.kind is ActionKind.CHOOSE:
            entry=encoder.catalog.lookup('relic',row.get('item_id'))
            if (entry is None or entry['tier']!='Boss' or entry['id'] in choices or
                    entry['id'] in relics or entry['id'] in EXCLUDED_RELICS):
                raise ValueError('invalid offered boss relic')
            choices[entry['id']]=action
        elif row['kind']=='leave' and action.kind in (ActionKind.PROCEED,ActionKind.CANCEL,ActionKind.CHOOSE):
            if skip is not None: raise ValueError('duplicate boss skip')
            skip=action
        else: raise ValueError('unknown boss choice')
    if len(choices) not in (2,3) or skip is None: raise ValueError('incomplete boss menu')
    if 'BLACK_BLOOD' in choices and 'BURNING_BLOOD' not in relics:
        raise ValueError('black blood without starter relic')
    names=sorted(choices); actions=tuple(choices[n] for n in names)+(skip,)
    canonical=[Action(ActionKind.CHOOSE,choice_index=i) for i in range(len(names))]+[Action(ActionKind.PROCEED)]
    options=[dict(kind='boss_relic',item_id=n,action=a.as_dict()) for n,a in zip(names,canonical)]
    options.append(dict(kind='leave',action=canonical[-1].as_dict()))
    partial=PolicyInput(dict(run=dict(character='IRONCLAD',act=act,floor=floor,ascension=ascension),
        player={},inventory=dict(deck=deck,relics=[dict(id=r) for r in sorted(relics)]),
        screen=dict(type='BOSS_RELIC_REWARDS',run_options_version=RUN_OPTIONS_VERSION,decision_options=options)),tuple(canonical))
    # Dummy skip target only satisfies the row validator; it is never a label.
    row=dict(version=ROW_VERSION,public=partial.as_dict(),input_sha256=partial.fingerprint(),
             target=len(names),partial_fields=list(PARTIAL_FIELDS))
    return BossProjection(row,encode_row(row,encoder),actions,tuple(names)+('skip',))


def candidate_features(model, batch):
    """Capture the exact existing head input without duplicating its encoder."""
    import torch
    captured=[]
    handle=model.run_head.register_forward_pre_hook(lambda _module,args:captured.append(args[0].detach().clone()))
    try:
        with torch.no_grad(): logits,_=model(batch)
    finally: handle.remove()
    if len(captured)!=1: raise ValueError('candidate encoder hook contract changed')
    features=captured[0]
    with torch.no_grad(): check=model.run_head(features).squeeze(-1).masked_fill(~batch.action_mask,-torch.inf)
    if not torch.equal(logits,check): raise ValueError('cached head scores differ')
    return features


class BossRelicPolicy:
    """Explicit boss-only inference; the caller retains all other specialists."""
    def __init__(self, path, encoder):
        import torch
        from .run_learning_model import make_model
        path=Path(path)
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['version']!=VERSION or saved['schema']!=encoder.schema_id:
            raise ValueError('boss checkpoint contract mismatch')
        self.encoder=encoder; self.model=make_model(encoder,width=saved['width'])
        self.model.load_state_dict(saved['model'],strict=True);self.model.eval();self.model.requires_grad_(False)
        self.last_decision=None

    def choose(self,public):
        import torch
        from .run_learning_model import tensorize
        projected=project(public,self.encoder)
        with torch.inference_mode(): dist,_=self.model.distribution(tensorize([projected.encoded],self.encoder))
        p=dist.probs[0];index=int(p.argmax());action=projected.decode(index)
        if action not in public.legal_actions: raise ValueError('mapped boss action is not legal')
        self.last_decision=dict(version=VERSION,input_sha256=projected.row['input_sha256'],
            meanings=projected.meanings,probabilities=p.tolist(),action=action.as_dict(),mode='greedy_partial_boss')
        return action
