"""Opt-in outcome learning for the existing shared construction head.

Frozen public candidate features are
cached once; PPO trains the existing preference head and existing state value
head only. No new network or simulated-action labels. The first contract uses
complete TRAIN episodes and undiscounted terminal +/-1 Monte Carlo returns.
It deliberately does not pretend historical greedy actions are on-policy data.
"""
from dataclasses import dataclass
import copy
import hashlib
import json
import math

import torch
from .advantage_normalization import normalize_advantages

from .campfire_construction import tensorize
from .construction_policy import Menu
from .deck_startup import COLUMNS
from .policy_api import _reject_forbidden_policy_keys
from .run_choice_policy import state_hash
from .training_contract import TrainingConfig, generalized_advantages

VERSION = 'shared-construction-terminal-ppo-interface-v1'
CRITIC_PREFIX = 'common.states.backbone.value_head.'


def tensor_hash(tensor):
    t = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((t.shape,t.dtype)).encode()+t.numpy().tobytes()).hexdigest()


def frozen_hash(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if name.startswith(('preference.', CRITIC_PREFIX)): continue
        h.update(name.encode()); h.update(tensor_hash(value).encode())
    return h.hexdigest()


@dataclass(frozen=True)
class CachedMenu:
    schema: str
    frozen_sha256: str
    fingerprint: str
    kind: str
    features: torch.Tensor
    choices: tuple
    sha256: str


def menu_hash(schema, frozen, fingerprint, kind, features, choices):
    return hashlib.sha256(json.dumps((schema,frozen,fingerprint,kind,
        tensor_hash(features),choices),sort_keys=True,separators=(',',':')).encode()).hexdigest()


@dataclass(frozen=True)
class Sample:
    menu: CachedMenu
    generation: int
    model_sha256: str
    ordinal: int
    selected: int
    old_log_prob: float
    old_value: float
    sampling: str = 'categorical'
    split: str = 'train'


@dataclass(frozen=True)
class Episode:
    samples: tuple
    outcome: str
    terminated: bool
    truncated: bool
    split: str


@dataclass(frozen=True)
class Act1Episode:
    """A real stage boundary, deliberately distinct from ordinary-run victory."""
    samples: tuple
    outcome: str
    before_run: dict
    after_run: dict
    battle_result: dict | None
    terminated: bool
    truncated: bool
    split: str


def complete_act1_episode(samples, *, before_run, after_run, battle_result,
                          terminated, truncated, split):
    if (split != 'train' or type(terminated) is not bool or
            truncated is not False or before_run.get('act') != 1):
        raise ValueError('actual complete Act1 TRAIN episode required')
    if terminated and after_run.get('outcome') == 'PLAYER_LOSS':
        outcome = 'PLAYER_LOSS'
    elif (not terminated and before_run.get('room_type') == 'BOSS' and
          before_run.get('floor') == 16 and isinstance(battle_result, dict) and
          battle_result.get('version') == 'public-battle-result-v1' and
          battle_result.get('outcome') == 1 and battle_result.get('hp', 0) > 0 and
          after_run.get('outcome') not in ('PLAYER_LOSS', 'PLAYER_VICTORY')):
        outcome = 'ACT1_CLEAR'
    else:
        raise ValueError('Act1 requires a real boss victory or native player loss')
    return Act1Episode(tuple(samples), outcome, copy.deepcopy(before_run),
        copy.deepcopy(after_run), copy.deepcopy(battle_result), terminated, truncated, split)


def episode_objective(episode):
    return 'act1_clear' if isinstance(episode, Act1Episode) else 'ordinary_clear'


def complete_episode(samples, *, outcome, terminated, truncated, split):
    if (not samples or split != 'train' or terminated is not True or truncated is not False
            or outcome not in ('PLAYER_VICTORY','PLAYER_LOSS')):
        raise ValueError('actual complete TRAIN episode required; errors/caps are not losses')
    return Episode(tuple(samples),outcome,terminated,truncated,split)


def terminal_targets(episodes):
    """Reuse GAE with gamma=lambda=1: all decisions receive the actual result.

    Fixed combat/route actions are collapsed, not labelled as learner actions.
    No per-click discount; a two-click upgrade is ONE construction decision.
    """
    advantages, targets = [], []
    for episode in episodes:
        if isinstance(episode, Act1Episode):
            checked = complete_act1_episode(episode.samples,before_run=episode.before_run,
                after_run=episode.after_run,battle_result=episode.battle_result,
                terminated=episode.terminated,truncated=episode.truncated,split=episode.split)
            if checked.outcome != episode.outcome: raise ValueError('stage outcome tampered')
        else:
            complete_episode(episode.samples,outcome=episode.outcome,terminated=episode.terminated,
                             truncated=episode.truncated,split=episode.split)
        samples=episode.samples; values=[s.old_value for s in samples]; n=len(samples)
        if not n: continue  # A real loss before the first learner decision has no actor label.
        reward=1. if episode.outcome in ('PLAYER_VICTORY','ACT1_CLEAR') else -1.
        adv,ret=generalized_advantages([0.]*(n-1)+[reward], values, values[1:]+[0.],
            [False]*(n-1)+[True], [False]*n, gamma=1., lam=1.)
        advantages.extend(adv); targets.extend(ret)
    return torch.tensor(advantages,dtype=torch.float32),torch.tensor(targets,dtype=torch.float32)


class ConstructionLearner:
    def __init__(self, policy, *, startup_enabled, config=None, objective='ordinary_clear', inference_only=False):
        if objective not in ('ordinary_clear','act1_clear'): raise ValueError('unknown construction objective')
        self.objective = objective
        if type(startup_enabled) is not bool: raise ValueError('explicit startup ablation required')
        self.encoder=policy.encoder
        # Owning a detached copy prevents accidental changes to an evaluation.
        self.model=copy.deepcopy(policy.model).cpu().eval(); self.model.requires_grad_(False)
        self.model.preference.requires_grad_(True)
        self.value_head=self.model.common.states.backbone.value_head
        self.value_head.requires_grad_(True)
        self.config=config or TrainingConfig(gamma=1.,gae_lambda=1.,progress_cap=0.)
        self.config.validate()
        if (self.config.gamma,self.config.gae_lambda,self.config.progress_cap)!=(1.,1.,0.):
            raise ValueError('v1 uses complete-episode terminal returns, no progress/discount shaping')
        self.startup_enabled=startup_enabled
        self.start=self.model.common.startup_offset; self.stop=self.start+6*len(COLUMNS)
        if not startup_enabled and self.model.preference[0].weight[:,self.start:self.stop].count_nonzero():
            raise ValueError('masked-startup initializer contains startup weights')
        self.parameters=tuple(p for p in self.model.parameters() if p.requires_grad)
        self.inference_only = inference_only
        self.optimizer=None if inference_only else torch.optim.Adam(self.parameters,lr=self.config.learning_rate)
        if inference_only: self.model.requires_grad_(False)
        self.frozen=frozen_hash(self.model); self.generation=0; self.ordinal=0
        self.snapshot=state_hash(self.model); self.failed=False

    def cache(self, menu):
        if (not isinstance(menu,Menu) or not menu.candidates or len(menu.candidates)!=len(menu.choices)
                or menu.kind not in ('reward','campfire','shop')):
            raise ValueError('complete observed construction menu required')
        _reject_forbidden_policy_keys(menu.inputs,reject_seeds=True)
        if any(c.before!=menu.candidates[0].before for c in menu.candidates):
            raise ValueError('candidates must share one public root')
        with torch.no_grad(): features=self.model.features(tensorize(menu.candidates,self.encoder)).detach().clone()
        if not self.startup_enabled: features[:,self.start:self.stop]=0
        choices=tuple(copy.deepcopy(menu.choices)); schema=self.encoder.schema_id
        h=menu_hash(schema,self.frozen,menu.fingerprint,menu.kind,features,choices)
        return CachedMenu(schema,self.frozen,menu.fingerprint,menu.kind,features,choices,h)

    def _check_menu(self, m):
        if (m.schema!=self.encoder.schema_id or m.frozen_sha256!=self.frozen or
                m.features.ndim!=2 or m.features.shape!=(len(m.choices),self.model.preference[0].in_features)
                or not torch.isfinite(m.features).all() or
                menu_hash(m.schema,m.frozen_sha256,m.fingerprint,m.kind,m.features,m.choices)!=m.sha256):
            raise ValueError('cached public menu changed')

    def distribution(self, menu):
        self._check_menu(menu)
        logits=self.model.preference(menu.features).squeeze(-1)
        # The first width columns are the shared BEFORE-state context. Neither
        # the chosen afterstate nor final outcomes enter this baseline.
        context=menu.features[0,:self.model.common.width]
        value=self.value_head(context).reshape(())
        if not torch.isfinite(logits).all() or not torch.isfinite(value):
            raise ValueError('non-finite construction actor/critic')
        return torch.distributions.Categorical(logits=logits),value

    def sample(self, menu, *, split, generator):
        if self.failed or split!='train': raise ValueError('sampling requires active TRAIN learner')
        with torch.no_grad():
            dist,value=self.distribution(menu)
            selected=int(torch.multinomial(dist.probs,1,generator=generator))
            log=float(dist.log_prob(torch.tensor(selected)))
        result=Sample(menu,self.generation,self.snapshot,self.ordinal,selected,log,float(value))
        self.ordinal+=1
        return result

    def _validate(self, episodes):
        if self.inference_only: raise ValueError('inference-only learner cannot update')
        if self.failed or state_hash(self.model)!=self.snapshot or frozen_hash(self.model)!=self.frozen:
            raise ValueError('learner weights changed outside the declared update')
        samples=[s for e in episodes for s in e.samples]
        if any(episode_objective(e) != self.objective for e in episodes):
            raise ValueError('cannot mix stage and ordinary-clear objectives')
        if not samples or len({s.ordinal for s in samples})!=len(samples):
            raise ValueError('empty or repeated behavior sample')
        for s in samples:
            if (s.split!='train' or s.sampling!='categorical' or s.generation!=self.generation
                    or s.model_sha256!=self.snapshot or type(s.selected) is not int
                    or not 0<=s.selected<len(s.menu.choices)
                    or not math.isfinite(s.old_log_prob) or not math.isfinite(s.old_value)):
                raise ValueError('stale/greedy/held-out behavior is not on-policy PPO')
            with torch.no_grad():
                dist,value=self.distribution(s.menu)
                if (abs(float(dist.log_prob(torch.tensor(s.selected)))-s.old_log_prob)>1e-6 or
                        abs(float(value)-s.old_value)>1e-6):
                    raise ValueError('recorded behavior probability/value mismatch')
        return samples

    def update(self, episodes, *, generator):
        """Same clipped PPO objective/KL guard as run_learning_rollout.py.

        The difference is the explicit full-episode terminal target and cached
        shared-head adapter. This is NOT yet a continuously running collector.
        """
        samples=self._validate(episodes); advantage,target=terminal_targets(episodes)
        if len(samples)>1: advantage=normalize_advantages(advantage)
        old_log=torch.tensor([s.old_log_prob for s in samples]); config=self.config
        logs=[]; stopped=False; before=self.snapshot
        try:
            for _ in range(config.epochs):
                order=torch.randperm(len(samples),generator=generator)
                for start in range(0,len(samples),config.minibatch_size):
                    ids=order[start:start+config.minibatch_size]; chosen=[samples[i] for i in ids.tolist()]
                    # Ragged legal menus; no padded/invalid actions are sampled.
                    distributions,values=zip(*(self.distribution(s.menu) for s in chosen))
                    logp=torch.stack([d.log_prob(torch.tensor(s.selected)) for d,s in zip(distributions,chosen)])
                    values=torch.stack(values); lr=logp-old_log[ids]; ratio=lr.exp()
                    if not torch.isfinite(ratio).all(): raise ValueError('non-finite PPO ratio')
                    kl=((ratio-1)-lr).mean()
                    if float(kl.detach())>config.target_kl: stopped=True; break
                    policy=-torch.minimum(ratio*advantage[ids],ratio.clamp(1-config.clip,1+config.clip)*advantage[ids]).mean()
                    value_loss=.5*(values-target[ids]).square().mean()
                    entropy=torch.stack([d.entropy() for d in distributions]).mean()
                    loss=policy+config.value_coef*value_loss-config.entropy_coef*entropy
                    if not torch.isfinite(loss): raise ValueError('non-finite construction loss')
                    self.optimizer.zero_grad(set_to_none=True);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(self.parameters,config.max_grad_norm,error_if_nonfinite=True)
                    self.optimizer.step()
                    logs.append(dict(loss=float(loss.detach()),policy_loss=float(policy.detach()),
                        value_loss=float(value_loss.detach()),entropy=float(entropy.detach()),
                        approx_kl=float(kl.detach()),gradient_norm=float(grad)))
                if stopped: break
            if not logs: raise ValueError('no optimizer update')
            if frozen_hash(self.model)!=self.frozen: raise ValueError('frozen feature encoder changed')
            if not all(torch.isfinite(p).all() for p in self.model.parameters()): raise ValueError('non-finite updated model')
            if not self.startup_enabled and self.model.preference[0].weight[:,self.start:self.stop].count_nonzero():
                raise ValueError('startup ablation changed')
        except BaseException:
            self.failed=True;raise  # Never silently resume a partially applied update.
        self.generation+=1;self.snapshot=state_hash(self.model)
        return dict(version=VERSION,generation=self.generation,episodes=len(episodes),samples=len(samples),
            wins=sum(e.outcome=='PLAYER_VICTORY' for e in episodes),
            stage_clears=sum(e.outcome=='ACT1_CLEAR' for e in episodes),objective=self.objective,
            outcome_classes=len({e.outcome for e in episodes}),target='undiscounted actual objective-boundary +/-1',
            optimizer_steps=len(logs),stopped_for_kl=stopped,encoder_unchanged=True,
            old_model_sha256=before,model_sha256=self.snapshot,
            metrics={k:sum(x[k] for x in logs)/len(logs) for k in logs[0]})
