"""Card multisets available to a policy, without inventing hidden draw order."""
from typing import Mapping


def opening_priority_ids(combat: Mapping) -> tuple[int, ...]:
    """Validate public membership, never reconstruct it from static card flags.

    Legacy captures may omit the entire schema. Partial/invalid declarations are
    errors, not permission to erase an opening constraint or fall back silently.
    """
    fields = ("opening_order_schema", "public_opening_priority_ids", "opening_order_uncertain")
    if not any(k in combat for k in fields):
        return ()
    if (type(combat.get(fields[0])) is not int or combat[fields[0]] != 1
            or type(combat.get(fields[2])) is not bool):
        raise ValueError("invalid public opening-order schema")
    ids = combat.get(fields[1])
    if (not isinstance(ids, (list, tuple)) or any(type(i) is not int or i < 0 for i in ids)
            or list(ids) != sorted(set(ids)) or (not ids and combat[fields[2]])):
        raise ValueError("invalid public opening-priority identities")
    if ids:
        pile_ids = [c.get("unique_id") for c in public_draw_cards(combat)]
        bottom = combat.get("public_known_bottom_ids", ())
        bottom = set(bottom) | {c.get("unique_id") for c in combat.get("draw_pile_known_bottom", ())}
        if any(pile_ids.count(i) != 1 or i in bottom for i in ids):
            raise ValueError("opening-priority identity not in its public draw segment")
    return tuple(ids)


def draw_order_sampling_reason(combat: Mapping) -> str | None:
    """Rules can act here; reconstructing an independent root is unsupported.

    Unknown multiset order alone is not a refusal. These public flags describe
    lost conditional top/bottom knowledge, e.g. insertion after Headbutt. Do not
    clear them or permanently block history: a later shuffle can resolve it.
    """
    opening_priority_ids(combat)
    if (combat.get("top_order_uncertain") or combat.get("bottom_order_uncertain")
            or combat.get("opening_order_uncertain")):
        return "uncertain public draw order"
    return None


def public_draw_cards(combat: Mapping) -> tuple[Mapping, ...]:
    if "draw_pile" in combat:  # Raw observation or Frozen Eye, not an additional pile.
        values = combat.get("draw_pile", ())
    else:
        values = tuple(combat.get("draw_pile_unordered", ()))
        top = combat.get("draw_pile_known_top")
        if isinstance(top, Mapping):
            values += (top,)
        values += tuple(combat.get("draw_pile_known_bottom", ()))
    return tuple(card for card in values if isinstance(card, Mapping))
