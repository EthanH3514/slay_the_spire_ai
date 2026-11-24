"""Opt-in empty and duplicate-row paths for existing CPU float32 run models.

Only genuinely zero-length tables qualify. Unknown features and masked/padded
rows with a positive table length still execute the original encoder. The empty
result retains zero-gradient edges to precisely the original parameters: Adam
must keep updating existing momentum, rather than treating them as unused.
No parameters are replaced, no learned cache is stored, and state-dict keys and
optimizer parameter identities stay unchanged.
"""
import torch
from torch import nn
import numpy as np

from .learning_model import RowEncoder
from .run_learning_model import ExpandedRow


VERSION = 'cpu-static-row-compute-v1'
MODES = ('reference', 'empty', 'cards', 'empty_cards')
MIN_UNIQUE_ROWS = 128


class _EmptyRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shape, *inputs):
        ctx.save_for_backward(*inputs)
        return inputs[0].new_empty(shape)

    @staticmethod
    def backward(ctx, gradient):
        # Returning None would change optimizer participation after a table
        # becomes empty. Match the original graph's dense, explicit zero grads.
        return (None, *(torch.zeros_like(tensor) if needed else None
            for tensor, needed in zip(ctx.saved_tensors, ctx.needs_input_grad[1:], strict=True)))


class StaticRowEncoder(nn.Module):
    def __init__(self, source, *, empty=False, unique=False):
        super().__init__()
        if type(source) not in (RowEncoder, ExpandedRow):
            raise TypeError('known run row encoder required')
        self.numeric, self.categorical = source.numeric, source.categorical
        self.network = source.network
        self.expanded = isinstance(source, ExpandedRow)
        self.empty, self.unique = empty, unique
        if self.expanded:
            self.extra = source.extra
        self.train(source.training)

    def forward(self, table, embeddings):
        values = table['values']
        weight = self.network[0].weight
        forward = ExpandedRow.forward if self.expanded else RowEncoder.forward
        if (values.ndim != 3 or values.device.type != 'cpu' or
                values.dtype != torch.float32 or weight.dtype != torch.float32 or
                torch.is_autocast_cpu_enabled()):
            return forward(self, table, embeddings)
        if values.shape[1] != 0:
            if self.unique and values.shape[0]*values.shape[1] >= MIN_UNIQUE_ROWS:
                return self.unique_forward(table, embeddings, forward)
            return forward(self, table, embeddings)
        if not self.empty:
            return forward(self, table, embeddings)
        # Access all required arrays even for empty data, preserving malformed
        # input rejection instead of silently accepting missing unknown masks.
        for key in ('known', 'present'):
            if table[key].shape[:2] != values.shape[:2]:
                raise ValueError('empty row mask shape mismatch')
        if table['known'].shape != values.shape:
            raise ValueError('empty row known-mask shape mismatch')
        inputs = [*self.network.parameters()]
        if self.expanded:
            extra, known = table['extra_values'], table['extra_known']
            if (extra.shape != known.shape or extra.shape[:2] != values.shape[:2] or
                    extra.shape[-1]*2 != self.extra.in_features):
                raise ValueError('empty supplemental row shape mismatch')
            inputs.extend(self.extra.parameters())
            if extra.requires_grad:
                inputs.append(extra)
        # Repeated categorical kinds are intentionally retained as repeated
        # zero-gradient edges, just as separate embedding calls in the reference.
        inputs.extend(embeddings[kind].weight for _, kind in self.categorical)
        if values.requires_grad:
            inputs.append(values)
        return _EmptyRows.apply((values.shape[0], 0, self.network[0].out_features), *inputs)

    def unique_forward(self, table, embeddings, forward):
        """Exact bit-pattern keys include every input and every unknown mask.

        Network operations are row-independent (Linear/Tanh/LayerNorm, no
        dropout/BatchNorm). Positions are restored before all entity joins.
        Shared backward reductions may differ in floating point, not in the
        mathematical gradient. Never fold differentiable input features.
        """
        if any(t.requires_grad or t.device.type != 'cpu' for t in table.values()):
            return forward(self, table, embeddings)
        b, n = table['values'].shape[:2]
        flat = {}
        keys = []
        for key, tensor in table.items():
            if tensor.shape[:2] != (b, n):
                raise ValueError('card row attributes misaligned')
            array = np.ascontiguousarray(tensor.detach().numpy()).reshape(b*n, -1)
            flat[key] = array
            keys.append(array.view(np.uint8).reshape(b*n, -1))
        packed = np.concatenate(keys, axis=1)
        opaque = packed.view(np.dtype((np.void, packed.shape[1]))).reshape(-1)
        _, first, inverse = np.unique(opaque, return_index=True, return_inverse=True)
        if len(first) == b*n:
            return forward(self, table, embeddings)
        compact = {key: torch.from_numpy(array[first].copy()).reshape(1, len(first), *table[key].shape[2:])
                   for key, array in flat.items()}
        encoded = forward(self, compact, embeddings).reshape(len(first), -1)
        return encoded.index_select(0, torch.from_numpy(inverse)).reshape(b, n, -1)


def install(models, mode='empty'):
    """Install on an owned model pair; return a reversible benchmark handle."""
    backbones = (models['candidate_actor'], models['effect_preference'].states.backbone)
    if mode not in MODES:
        raise ValueError('unknown static-row computation mode')
    locations = [(m.rows, name, row) for m in backbones for name, row in m.rows.items()]
    if any(type(row) not in (RowEncoder, ExpandedRow) for _, _, row in locations):
        raise TypeError('only unwrapped reference row encoders can be installed')
    installed = []
    for rows, name, row in locations:
        empty, unique = mode in ('empty', 'empty_cards'), name == 'cards' and mode in ('cards', 'empty_cards')
        if empty or unique:
            rows[name] = StaticRowEncoder(row, empty=empty, unique=unique)
            installed.append((rows, name, row))
    return installed


def restore(handle):
    for rows, name, row in handle:
        if not isinstance(rows[name], StaticRowEncoder):
            raise ValueError('empty row wrapper changed before restore')
        rows[name] = row
