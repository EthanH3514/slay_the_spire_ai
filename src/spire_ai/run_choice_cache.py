"""Bounded, weight-independent numeric feature cache for paired card BC.

Only existing public encodings are serialized, in ragged float32/bool tables.
No hidden RNG, network embeddings, model outputs, future labels or dummy actions.
Supervised targets/group IDs are separate metadata and never model arguments.
"""
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence

from .learning_model import TensorBatch
from .public_encoding import TABLE_COLUMNS
from .run_learning_encoding import EXTRA_COLUMNS
from .run_state_model import StateBatch
from .run_choice_model import ChoiceBatch, OPERATIONS

VERSION='public-run-choice-numeric-cache-v1'
MAX_PADDED_VALUES=2_000_000


class BatchMemoryBound(ValueError):
    pass


def pack_states(states,encoder):
    if not states or any(s.schema_id!=encoder.states.schema_id for s in states):
        raise ValueError('cache requires aligned encoded states')
    tables={}
    for name,columns in TABLE_COLUMNS.items():
        offsets=[0]
        values,known,extra,extra_known=[],[],[],[]
        for state in states:
            table=state.entities.tables[name]
            if table.columns!=columns: raise ValueError('cache column mismatch')
            offsets.append(offsets[-1]+len(table.values))
            values.extend(table.values);known.extend(table.known)
            if name in EXTRA_COLUMNS:
                supplement=state.entities.extra_tables[name]
                if supplement.columns!=EXTRA_COLUMNS[name] or len(supplement.values)!=len(table.values):
                    raise ValueError('cache supplement mismatch')
                extra.extend(supplement.values);extra_known.extend(supplement.known)
        item=dict(offsets=torch.tensor(offsets,dtype=torch.int64),
            values=torch.tensor(values,dtype=torch.float32).reshape(-1,len(columns)),
            known=torch.tensor(known,dtype=torch.bool).reshape(-1,len(columns)))
        if name in EXTRA_COLUMNS:
            item.update(extra_values=torch.tensor(extra,dtype=torch.float32).reshape(-1,len(EXTRA_COLUMNS[name])),
                extra_known=torch.tensor(extra_known,dtype=torch.bool).reshape(-1,len(EXTRA_COLUMNS[name])))
        if any(not torch.isfinite(t).all() for k,t in item.items() if k.endswith('values')):
            raise ValueError('nonfinite cache feature')
        tables[name]=item
    masks=[a and b for s in states for a,b in zip(s.entities.legal_mask,s.entities.scope_mask,strict=True)]
    return dict(tables=tables,action_mask=torch.tensor(masks,dtype=torch.bool),
        phases=torch.tensor([[float(s.phase==0),float(s.phase==1),float(s.opportunities_known)] for s in states],dtype=torch.float32))


def pack_examples(examples,encoder):
    if not examples: raise ValueError('nonempty cache chunk required')
    roots,afters,effects,refs,offsets,metadata=[],[],[],[],[0],[]
    for e in examples:
        choices=e['choices']
        if not choices: raise ValueError('empty choice group')
        root=choices[0].before
        if any(c.before!=root for c in choices): raise ValueError('choice roots differ')
        # Do not assume baseline/state encoders agree; prove it per cached row.
        if (e['baseline'].tables!=root.entities.tables or e['baseline'].extra_tables!=root.entities.extra_tables or
                e['baseline'].legal_mask!=root.entities.legal_mask or e['baseline'].scope_mask!=root.entities.scope_mask):
            raise ValueError('actor/root cache feature mismatch')
        roots.append(root)
        for c in choices:
            afters.append(c.after)
            effects.append([float(i==c.operation) for i in range(len(OPERATIONS))]+list(c.effects)+list(map(float,c.known)))
            refs.append((c.before_card_ref,c.after_card_ref))
        offsets.append(offsets[-1]+len(choices))
        row=e['row'];run=row['public']['observation']['run']
        target=row['target'];operations=list(e['operations'])
        labels={'take_card':'card','skip':'skip','singing_bowl':'singing_bowl'}
        if labels[operations[target]]!=row['label'] or operations.count('skip')!=1:
            raise ValueError('choice label/target alignment mismatch')
        metadata.append(dict(group=row['group'],input_sha256=row['input_sha256'],choice_index=row['choice_index'],
            act=row['act'],ascension=run['ascension'],boss=run.get('boss','<unknown>'),label=row['label'],
            target=target,operations=operations))
    return dict(version=VERSION,schema=encoder.schema_id,state_schema=encoder.states.schema_id,
        root=pack_states(roots,encoder),after=pack_states(afters,encoder),
        offsets=torch.tensor(offsets,dtype=torch.int64),effects=torch.tensor(effects,dtype=torch.float32),
        refs=torch.tensor(refs,dtype=torch.int64),metadata=metadata)


def validate_chunk(chunk,encoder):
    if (chunk.get('version')!=VERSION or chunk.get('schema')!=encoder.schema_id or
            chunk.get('state_schema')!=encoder.states.schema_id):
        raise ValueError('numeric cache schema mismatch')
    offsets=chunk['offsets'];n=len(chunk['metadata'])
    if (offsets.dtype!=torch.int64 or offsets.shape!=(n+1,) or int(offsets[0])!=0 or
            not (offsets[1:]>offsets[:-1]).all()):
        raise ValueError('invalid cache choice offsets')
    choices=int(offsets[-1])
    if chunk['effects'].shape!=(choices,15) or chunk['refs'].shape!=(choices,2):
        raise ValueError('invalid cache effects/references')
    if chunk['effects'].dtype!=torch.float32 or chunk['refs'].dtype!=torch.int64 or not torch.isfinite(chunk['effects']).all():
        raise ValueError('invalid cache dtype/value')
    for state,count in ((chunk['root'],n),(chunk['after'],choices)):
        if (set(state['tables'])!=set(TABLE_COLUMNS) or state['phases'].shape!=(count,3) or
                state['phases'].dtype!=torch.float32 or not torch.isfinite(state['phases']).all()):
            raise ValueError('invalid cached state tables/phases')
        for name,columns in TABLE_COLUMNS.items():
            table=state['tables'][name];off=table['offsets'];values=table['values']
            if (off.dtype!=torch.int64 or off.shape!=(count+1,) or int(off[0])!=0 or
                    not (off[1:]>=off[:-1]).all() or values.shape!=(int(off[-1]),len(columns)) or
                    values.dtype!=torch.float32 or table['known'].dtype!=torch.bool or table['known'].shape!=values.shape):
                raise ValueError('invalid cached feature table')
            if not torch.isfinite(values).all(): raise ValueError('nonfinite cached table')
            if name in EXTRA_COLUMNS:
                if (table['extra_values'].shape!=(values.shape[0],len(EXTRA_COLUMNS[name])) or
                        table['extra_known'].shape!=table['extra_values'].shape or
                        table['extra_values'].dtype!=torch.float32 or table['extra_known'].dtype!=torch.bool or
                        not torch.isfinite(table['extra_values']).all()):
                    raise ValueError('invalid cached supplement')
        if state['action_mask'].dtype!=torch.bool or state['action_mask'].shape!=(int(state['tables']['actions']['offsets'][-1]),):
            raise ValueError('invalid cached action mask')
    for i,m in enumerate(chunk['metadata']):
        size=int(offsets[i+1]-offsets[i])
        if type(m['target']) is not int or not 0<=m['target']<size or len(m['operations'])!=size:
            raise ValueError('invalid cache target')
        if (m['operations'].count('skip')!=1 or any(o not in OPERATIONS[:3] for o in m['operations']) or
                {'take_card':'card','skip':'skip','singing_bowl':'singing_bowl'}[m['operations'][m['target']]]!=m['label']):
            raise ValueError('invalid cache target operation')


def take_state(packed,index):
    # Clone each slice so a shuffle-buffer row cannot retain a whole shard's
    # storage. Memory is bounded by buffer rows, not all encountered shards.
    tables={}
    for name,table in packed['tables'].items():
        left,right=map(int,table['offsets'][index:index+2])
        tables[name]={k:v[left:right].clone() for k,v in table.items() if k!='offsets'}
    left,right=map(int,packed['tables']['actions']['offsets'][index:index+2])
    return dict(tables=tables,action_mask=packed['action_mask'][left:right].clone(),phase=packed['phases'][index].clone())


def records(chunk):
    for i,metadata in enumerate(chunk['metadata']):
        left,right=map(int,chunk['offsets'][i:i+2])
        yield dict(root=take_state(chunk['root'],i),afters=[take_state(chunk['after'],j) for j in range(left,right)],
            effects=chunk['effects'][left:right].clone(),refs=chunk['refs'][left:right].clone(),metadata=metadata)


def padded_cost(frames):
    base=extra=0
    for name,columns in TABLE_COLUMNS.items():
        width=max(f['tables'][name]['values'].shape[0] for f in frames)
        base+=len(frames)*width*len(columns)
        if name in EXTRA_COLUMNS: extra+=len(frames)*width*len(EXTRA_COLUMNS[name])
    return base,extra


def collate_states(frames,encoder):
    if max(padded_cost(frames))>MAX_PADDED_VALUES:
        raise BatchMemoryBound('padded cached states exceed batch budget; microbatch, never truncate')
    tables={}
    for name in TABLE_COLUMNS:
        rows=[f['tables'][name] for f in frames]
        tables[name]={k:pad_sequence([r[k] for r in rows],batch_first=True) for k in rows[0]}
        tables[name]['present']=pad_sequence([torch.ones(r['values'].shape[0],dtype=torch.bool) for r in rows],batch_first=True)
    entities=TensorBatch(encoder.base.schema_id,tables,
        pad_sequence([f['action_mask'] for f in frames],batch_first=True),torch.zeros(len(frames),dtype=torch.bool))
    return StateBatch(encoder.states.schema_id,entities,torch.stack([f['phase'] for f in frames]))


@dataclass
class TrainingBatch:
    actor: TensorBatch
    choice: ChoiceBatch
    sizes: tuple
    metadata: tuple


def collate(records_,encoder):
    if not records_: raise ValueError('nonempty training batch required')
    sizes=tuple(len(r['afters']) for r in records_)
    if max(padded_cost([r['root'] for r in records_ for _ in r['afters']]))>MAX_PADDED_VALUES:
        raise BatchMemoryBound('repeated choice roots exceed batch memory bound')
    root=collate_states([r['root'] for r in records_],encoder)
    owners=torch.repeat_interleave(torch.arange(len(records_)),torch.tensor(sizes))
    before=StateBatch(root.schema_id,TensorBatch(root.entities.schema_id,
        {name:{k:t.index_select(0,owners) for k,t in table.items()} for name,table in root.entities.tables.items()},
        root.entities.action_mask.index_select(0,owners),root.entities.route.index_select(0,owners)),root.phase_features.index_select(0,owners))
    after=collate_states([s for r in records_ for s in r['afters']],encoder)
    refs=torch.cat([r['refs'] for r in records_])
    choice=ChoiceBatch(encoder.schema_id,before,after,torch.cat([r['effects'] for r in records_]),refs[:,0],refs[:,1])
    if not root.entities.action_mask.any(1).all(): raise ValueError('no legal cached root actions')
    return TrainingBatch(root.entities,choice,sizes,tuple(r['metadata'] for r in records_))


def microbatches(records_,encoder):
    try:
        yield collate(records_,encoder)
    except BatchMemoryBound:
        if len(records_)==1: raise
        middle=len(records_)//2
        yield from microbatches(records_[:middle],encoder)
        yield from microbatches(records_[middle:],encoder)


def model_scores(model,batch,arm):
    if arm=='candidate_actor':
        values,_=model(batch.actor)
        return [values[i,:size] for i,size in enumerate(batch.sizes)]
    if arm=='effect_preference': return list(model(batch.choice).split(batch.sizes))
    raise ValueError('unknown paired training arm')
