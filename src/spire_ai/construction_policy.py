"""Opt-in complete construction menus and one shared preference controller.

Scores are within-menu preferences, not absolute gains or calibrated values.
Execution binds a compound choice to the current public instance; future RNG,
outcomes and source-game identity never enter this policy.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import torch

from .policy_api import PolicyInput,policy_input_from_state,_reject_forbidden_policy_keys
from .campfire_construction import CampfireConstructionEncoder,CampfireConstructionModel,tensorize
from .campfire_transactions import capture_campfire,transact as camp_transact,selection_action,verify_immediate as verify_camp
from .construction_model import SharedConstructionModel
from .run_choice_policy import project_choices,state_hash
from .shop_capture import packet,load_packet,choice_wire,removal_action,verify_immediate as verify_shop
from .shop_transactions import bind_root_action,transact as shop_transact

VERSION='shared-construction-controller-v1'


def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_from_checkpoint(saved):
    """Build the declared architecture directly; complete weights need no initializer."""
    encoder = CampfireConstructionEncoder()
    if saved['schema'] != encoder.schema_id:
        raise ValueError('construction checkpoint schema mismatch')
    width = saved['model']['preference.0.weight'].shape[0]
    model = CampfireConstructionModel(encoder, width=width)
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    return ConstructionPolicy(encoder, model)


def load_joint(folder,arm='joint_without_startup'):
    folder=Path(folder);receipt=json.loads((folder/'receipt.json').read_text())
    if receipt.get('complete') is not True:raise ValueError('completed joint fit required')
    paths=('report.json','plan.json','initial.pt',arm+'/selected-head.pt')
    for name in paths:
        if file_sha(folder/name)!=receipt['outputs'][name]:raise ValueError('joint input seal mismatch: '+name)
    report=json.loads((folder/'report.json').read_text());selected=report['results'][arm]
    initial=torch.load(folder/'initial.pt',map_location='cpu',weights_only=True)
    saved=torch.load(folder/arm/'selected-head.pt',map_location='cpu',weights_only=True)
    encoder=CampfireConstructionEncoder();common=SharedConstructionModel(encoder.common)
    if initial['schema']!=common.schema_id or saved['schema']!=common.schema_id or saved['best']!=selected['selected']:
        raise ValueError('joint selected schema/epoch mismatch')
    common.load_state_dict(initial['model']);common.preference.load_state_dict(saved['head'])
    if not saved['startup_enabled'] and common.preference[0].weight[:,common.startup_offset:].count_nonzero():
        raise ValueError('masked-startup checkpoint has nonzero startup weights')
    model=CampfireConstructionModel(encoder,warm_common=common);model.eval();model.requires_grad_(False)
    provenance=dict(version=VERSION,arm=arm,epoch=saved['best']['epoch'],schema=encoder.schema_id,
        initial_sha256=file_sha(folder/'initial.pt'),head_sha256=file_sha(folder/arm/'selected-head.pt'),
        model_sha256=state_hash(model),commitment_trained=False,production_default=False)
    return ConstructionPolicy(encoder,model,provenance)


@dataclass(frozen=True)
class Menu:
    kind:str
    fingerprint:str
    root:object
    encoded:object
    candidates:tuple
    choices:tuple
    inputs:dict

    def wire(self):
        return dict(version=VERSION,kind=self.kind,public=self.inputs,choices=list(self.choices))


class ConstructionPolicy:
    def __init__(self,encoder,model,provenance=None):
        self.encoder,self.model=encoder,model
        self.provenance=provenance or dict(version=VERSION,schema=encoder.schema_id,commitment_trained=False)

    def menu(self,public,*,supplement=None,shelf=None,reward_eligible=False):
        _reject_forbidden_policy_keys(public.observation,reject_seeds=True)
        screen=public.observation['screen']['type']
        inputs=dict(public=public.as_dict(),run_supplement=supplement)
        if screen=='REST':
            root=capture_campfire(public,run_public=supplement);encoded=self.encoder.encode_campfire(root)
            wires=tuple(dict(kind=c.kind,target_index=c.target_index,action=c.action.as_dict()) for c in encoded.choices)
            return Menu('campfire',public.fingerprint(),root,encoded,encoded.candidates,wires,inputs)
        if screen=='SHOP':
            value=packet(public,supplement,shelf);root=load_packet(value);encoded=self.encoder.encode_shop(root)
            return Menu('shop',public.fingerprint(),root,encoded,encoded.candidates,
                tuple(choice_wire(c) for c in encoded.choices),value)
        if screen=='CARD_REWARD' and reward_eligible:
            # Preserve the reviewed historical missing-field projection.
            projected=project_choices(public,self.encoder.common)
            candidates=tuple(self.encoder._wrap(c) for c in projected.choices)
            wires=tuple(dict(kind=op,action=a.as_dict()) for op,a in zip(projected.operations,projected.reward.actions,strict=True))
            return Menu('reward',public.fingerprint(),projected,None,candidates,wires,inputs)
        return None

    def scores(self,menu):
        with torch.no_grad():scores=self.model(tensorize(menu.candidates,self.encoder))
        if scores.shape!=(len(menu.choices),) or not torch.isfinite(scores).all():raise ValueError('invalid complete-menu scores')
        return scores

    def choose(self,menu):return int(self.scores(menu).argmax())


def execute(environment,menu,index,*,take=None):
    if type(index) is not int or not 0<=index<len(menu.choices):raise ValueError('invalid complete choice index')
    if environment._current_public_input().fingerprint()!=menu.fingerprint:raise ValueError('stale construction menu')
    step=take or (lambda a,detail:environment.step(a))
    detail=dict(controller=VERSION,menu=menu.kind,choice=menu.choices[index])
    if menu.kind=='campfire':
        choice=menu.encoded.choices[index];predicted=camp_transact(menu.root,choice)
        result=step(choice.action,detail)
        if choice.kind in ('smith','remove'):
            result=step(selection_action(menu.root,policy_input_from_state(result.state),choice),dict(detail,compound_target=True))
        verify_camp(predicted,policy_input_from_state(result.state),json.loads(environment.session.public_run_supplement()))
    elif menu.kind=='shop':
        choice=menu.encoded.choices[index];predicted=shop_transact(menu.root,choice)
        result=step(bind_root_action(menu.root,choice).action,detail)
        if choice.kind=='buy_remove':
            result=step(removal_action(menu.root,policy_input_from_state(result.state),choice.target_index),dict(detail,compound_target=True))
        verify_shop(predicted.after,policy_input_from_state(result.state),json.loads(environment.session.public_run_supplement()))
    elif menu.kind=='reward':
        result=step(menu.root.reward.actions[index],detail)
    else:raise ValueError('unknown construction menu')
    return result


def nominate(menu,scores,*,maximum=4):
    """Bounded diagnostics only. Untested legal choices remain unlabelled."""
    if maximum!=4:raise ValueError('pilot cap is four complete choices')
    def key(i):return hashlib.sha256(json.dumps([menu.fingerprint,menu.choices[i]],sort_keys=True).encode()).hexdigest()
    selected=[]
    kinds=[c['kind'] for c in menu.choices]
    if menu.kind=='shop':
        for kind in ('leave','buy_card','buy_remove','buy_relic'):
            available=[i for i,k in enumerate(kinds) if k==kind]
            if available:selected.append(min(available,key=key))
    elif menu.kind=='campfire':
        selected=[i for i,k in enumerate(kinds) if k=='rest']
        upgrades=[i for i,k in enumerate(kinds) if k=='smith']
        if upgrades:
            best=max(upgrades,key=lambda i:(float(scores[i]),-i));selected.append(best)
            other=[i for i in upgrades if i!=best]
            if other:selected.append(min(other,key=key))
        special=[i for i,k in enumerate(kinds) if k in ('remove','lift','dig')]
        if special:selected.append(min(special,key=key))
    else:raise ValueError('pilot requires shop or campfire')
    if not selected or len(selected)>maximum or len(selected)!=len(set(selected)):
        raise ValueError('invalid bounded nominations')
    return tuple(selected)
