"""Training support for the released combat model (leaf_batch)."""

from learned_continuation_contract import canonical
from .leaf_numeric import ACTION_GROUPS
from .leaf_numeric import STATE_GROUPS
import torch


def pack(encoded):
    """Pack the public state and deduplicate identical candidate feature rows."""
    groups = [encoded["state"][name] for name in STATE_GROUPS]
    identities, inverse = {}, []
    for candidate in encoded["candidates"]:
        identity = canonical(candidate)
        if identity not in identities:
            identities[identity] = len(identities)
            groups.extend(candidate[name] for name in ACTION_GROUPS)
        inverse.append(identities[identity])
    rows = [row for group in groups for row in group]
    assignment = [index for index, group in enumerate(groups) for _ in group]
    return dict(
        categories=torch.tensor(
            [r["categories"] for r in rows], dtype=torch.long
        ).reshape(-1, 4),
        numbers=torch.tensor(
            [r["values"] + r["known"] for r in rows], dtype=torch.float32
        ).reshape(-1, 64),
        assignment=torch.tensor(assignment, dtype=torch.long),
        counts=torch.tensor([len(group) for group in groups], dtype=torch.float32),
        candidates=len(identities),
        inverse=torch.tensor(inverse, dtype=torch.long),
    )


def collate(packed):
    """Concatenate owned tensors, retaining state/group/candidate boundaries."""
    if not packed:
        raise ValueError("empty actor batch")
    categories, numbers, assignment, counts = [], [], [], []
    context, candidate_groups, candidate_states, inverse, lengths = [], [], [], [], []
    group_offset = candidate_offset = 0
    ns, na = len(STATE_GROUPS), len(ACTION_GROUPS)
    for state, item in enumerate(packed):
        groups, candidates = len(item["counts"]), item["candidates"]
        if candidates < 1 or groups != ns + na * candidates or len(item["inverse"]) < 1:
            raise ValueError("invalid packed actor groups")
        categories.append(item["categories"])
        numbers.append(item["numbers"])
        assignment.append(item["assignment"] + group_offset)
        counts.append(item["counts"])
        context.append(torch.arange(ns) + group_offset)
        candidate_groups.append(
            torch.arange(ns, groups).reshape(candidates, na) + group_offset
        )
        candidate_states.extend([state] * candidates)
        inverse.append(item["inverse"] + candidate_offset)
        lengths.append(len(item["inverse"]))
        group_offset += groups
        candidate_offset += candidates
    return dict(
        categories=torch.cat(categories),
        numbers=torch.cat(numbers),
        assignment=torch.cat(assignment),
        counts=torch.cat(counts),
        context=torch.stack(context),
        candidate_groups=torch.cat(candidate_groups),
        candidate_states=torch.tensor(candidate_states, dtype=torch.long),
        inverse=torch.cat(inverse),
        lengths=lengths,
    )


def prepare_public(encoded):
    """Entry point for non-training callers; do not silently discard a mask."""
    if not encoded.get("mask") or not all(encoded["mask"]):
        raise ValueError("expected already-legal actor candidates")
    return pack(encoded)
