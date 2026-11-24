"""Observed Act1/Act2 milestone returns for one fresh, continuous run.

Reward evidence is a label, never a feature. Already collected milestones do
not reward later actions. Errors and truncated prefixes are not zero-return
losses. The native game still distinguishes stage clears from ordinary victory.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math

VERSION='observed-two-act-milestone-return-v1'
BOSS_FLOORS={1:16,2:33}


@dataclass(frozen=True)
class RewardSchedule:
    act1:float=1.
    act2:float=2.

    def __post_init__(self):
        if (type(self.act1) not in (int,float) or type(self.act2) not in (int,float) or
            not math.isfinite(self.act1) or not math.isfinite(self.act2) or not 0<self.act1<self.act2):
            raise ValueError('positive increasing stage rewards required')

    def wire(self):return dict(version=VERSION,act1=float(self.act1),act2=float(self.act2),failure=0.,gamma=1.,gae_lambda=1.)

    @property
    def objective(self):
        digest=hashlib.sha256(json.dumps(self.wire(),sort_keys=True).encode()).hexdigest()[:16]
        return VERSION+'/'+digest


DEFAULT=RewardSchedule()


@dataclass(frozen=True)
class Boundary:
    index:int
    before:dict
    after:dict
    battle:dict | None
    terminated:bool
    truncated:bool


def from_row(row):
    return Boundary(row['index'],copy.deepcopy(row['public']['observation']['run']),
        copy.deepcopy(row['after_run']),copy.deepcopy(row.get('battle_result')),
        row['terminated'],row['truncated'])


def completed_act(boundary):
    before,battle=boundary.before,boundary.battle;act=before.get('act')
    if (act in BOSS_FLOORS and before.get('floor')==BOSS_FLOORS[act] and before.get('room_type')=='BOSS'
        and isinstance(battle,dict) and battle.get('version')=='public-battle-result-v1'
        and battle.get('outcome')==1 and battle.get('hp',0)>0):
        if boundary.terminated or boundary.truncated or boundary.after.get('outcome')!='UNDECIDED':
            raise ValueError('stage victory contradicts native terminal evidence')
        return act
    return None


@dataclass(frozen=True)
class StageEpisode:
    samples:tuple
    sample_indices:tuple
    boundaries:tuple
    schedule:RewardSchedule
    outcome:str
    split:str


def validate(e):
    if not isinstance(e,StageEpisode) or e.split!='train' or not isinstance(e.schedule,RewardSchedule):
        raise ValueError('fresh continuous TRAIN stage episode required')
    bs=e.boundaries
    if not bs or bs[0].before.get('act')!=1 or bs[0].before.get('floor')!=0:
        raise ValueError('joint return requires a fresh run, not a restored Act2 prefix')
    if (len(e.samples)!=len(e.sample_indices) or any(type(i)is not int or not 0<=i<len(bs) for i in e.sample_indices)
        or tuple(sorted(set(e.sample_indices)))!=e.sample_indices):
        raise ValueError('unaligned or repeated decision indices')
    milestones=[];cleared=0
    for i,b in enumerate(bs):
        if (not isinstance(b,Boundary) or type(b.index)is not int or b.index!=i or
            type(b.terminated)is not bool or b.truncated is not False or (b.terminated and i!=len(bs)-1)):
            raise ValueError('incomplete, reordered or truncated trajectory')
        if i and b.before!=bs[i-1].after:raise ValueError('run state discontinuity')
        if b.before.get('act')!=cleared+1 and not (cleared==1 and b.before.get('act')==1):
            raise ValueError('act progression lacks prior boss evidence')
        act=completed_act(b)
        if act is not None:
            if act!=cleared+1 or (act==2 and i!=len(bs)-1):raise ValueError('duplicated, out-of-order or nonfinal milestone')
            cleared=act;milestones.append((i,e.schedule.act1 if act==1 else e.schedule.act2))
    last=bs[-1]
    if last.terminated and last.after.get('outcome')=='PLAYER_LOSS':outcome='PLAYER_LOSS'
    elif not last.terminated and cleared==2 and completed_act(last)==2:outcome='ACT2_CLEAR'
    else:raise ValueError('need native loss or verified Act2 clear, not a timeout or Act1 cutoff')
    if e.outcome!=outcome:raise ValueError('stage outcome changed')
    return milestones


def complete_episode(samples,sample_indices,boundaries,*,schedule=DEFAULT,split='train'):
    bs=tuple(copy.deepcopy(boundaries))
    last=bs[-1] if bs else None
    outcome='ACT2_CLEAR' if last is not None and completed_act(last)==2 else 'PLAYER_LOSS'
    result=StageEpisode(tuple(samples),tuple(sample_indices),bs,schedule,outcome,split)
    validate(result);return result


def returns(e):
    milestones=validate(e)
    return tuple(float(sum(reward for boundary,reward in milestones if boundary>=index)) for index in e.sample_indices)


def targets(episodes,*,schedule=DEFAULT):
    import torch
    advantages=[];values=[]
    for e in episodes:
        if not isinstance(e,StageEpisode) or e.schedule!=schedule:raise ValueError('cannot mix reward objectives')
        rs=returns(e);values.extend(rs);advantages.extend(r-s.old_value for r,s in zip(rs,e.samples,strict=True))
    return torch.tensor(advantages,dtype=torch.float32),torch.tensor(values,dtype=torch.float32)
