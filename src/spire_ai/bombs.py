"""Public per-instance Bomb state and structural admission."""
from dataclasses import dataclass
import re
from typing import Mapping

from .identifiers import canonical_identifier


BOMB_STATE_SCHEMA = 1


def is_bomb_power(identifier: object) -> bool:
    return re.fullmatch(r"the_bomb(?:_?[0-9]+)?", canonical_identifier(identifier, "power") or "") is not None


@dataclass(frozen=True, slots=True)
class PendingBomb:
    remaining_turns: int
    damage: int
    before_combust: bool | None = None


def bomb_projection(player: Mapping) -> list[tuple[object, object]] | None:
    """Drop backend instance IDs, never multiplicity, packet order or missing fields.

    None means the power list is missing/malformed; [] means known absence.
    Values are deliberately not coerced so reconciliation can detect bad exports.
    """
    powers = player.get("powers")
    if not isinstance(powers, (list, tuple)) or any(not isinstance(p, Mapping) for p in powers):
        return None
    return [(p.get("amount"), p.get("damage")) for p in powers if is_bomb_power(p.get("id"))]


def pending_bombs(player: Mapping) -> tuple[PendingBomb, ...]:
    records = bomb_projection(player)
    if records is None:
        raise ValueError("The Bomb requires a public structured power list; missing state is unknown")
    powers = player["powers"]
    combust = [(i, p) for i, p in enumerate(powers) if canonical_identifier(p.get("id"), "power") == "combust"]
    if records and (len(combust) > 1 or any(type(p.get("amount")) is not int or p["amount"] <= 0 for _, p in combust)):
        raise ValueError("The Bomb requires at most one valid public Combust instance to preserve order")
    result = []
    indices = [i for i, p in enumerate(powers) if is_bomb_power(p.get("id"))]
    for index, (remaining, damage) in zip(indices, records, strict=True):
        if type(remaining) is not int or not 1 <= remaining <= 3 or type(damage) is not int or damage <= 0:
            raise ValueError("The Bomb requires public remaining turns 1..3 and positive integer damage")
        result.append(PendingBomb(remaining, damage, index < combust[0][0] if combust else None))
    return tuple(result)


def bomb_combust_projection(player: Mapping) -> list[tuple[object, ...]] | None:
    """Compare public relative order without interpreting backend instance suffixes."""
    if bomb_projection(player) is None:
        return None
    result = []
    for power in player["powers"]:
        if is_bomb_power(power.get("id")):
            result.append(("the_bomb", power.get("amount"), power.get("damage")))
        elif canonical_identifier(power.get("id"), "power") == "combust":
            result.append(("combust", power.get("amount"), power.get("hp_loss")))
    return result


def bomb_scope_reason(combat: Mapping) -> str | None:
    """Validate the public prerequisite for scoped Bomb recipe admission.

    Inspect public records and every public pile, never latent draw order/queues.
    An exporter must explicitly attest complete instance state and a resolved
    decision boundary. Old exporters are unknown, not an empty Bomb state.
    Future effects can leave this envelope; callers must recheck each decision.
    """
    player = combat.get("player", {})
    powers = player.get("powers", ()) if isinstance(player, Mapping) else ()
    lists = ("hand", "draw_pile", "draw_pile_unordered", "draw_pile_known_bottom",
             "discard_pile", "exhaust_pile", "limbo")
    cards = [combat.get("draw_pile_known_top")]
    for name in lists:
        values = combat.get(name, ())
        if isinstance(values, (list, tuple)):
            cards.extend(values)
    has_card = any(isinstance(c, Mapping) and canonical_identifier(c.get("id"), "card") == "the_bomb"
                   for c in cards)
    has_power = isinstance(powers, (list, tuple)) and any(
        isinstance(p, Mapping) and is_bomb_power(p.get("id")) for p in powers)
    if not has_card and not has_power:
        return None
    if not isinstance(player, Mapping):
        return "The Bomb requires a structured public player state"
    try:
        packets = pending_bombs(player)
    except ValueError as error:
        return str(error)
    if type(combat.get("bomb_state_schema")) is not int or combat["bomb_state_schema"] != BOMB_STATE_SCHEMA:
        return "The Bomb requires explicit public instance-state schema 1; older or missing export is unknown"
    if combat.get("powers_complete") is not True:
        return "The Bomb requires a complete public power list; omitted flag powers are unknown"
    if combat.get("input_state") != "CARD_SELECT" and (
            combat.get("effects_resolved") is not True or combat.get("input_state") != "PLAYER_NORMAL"):
        return "The Bomb requires a resolved normal decision or a public selection window"
    if any(not isinstance(combat.get(name, ()), (list, tuple)) or
           any(not isinstance(c, Mapping) for c in combat.get(name, ())) for name in lists):
        return "The Bomb requires structured public card piles"
    if combat.get("limbo"):
        return "The Bomb requires empty public limbo at a resolved decision"
    if "draw_pile_known_top" in combat and not isinstance(combat["draw_pile_known_top"], Mapping):
        return "The Bomb requires a structured public known top"
    if any(p.damage > 2**31 - 1 for p in packets):
        return "The Bomb damage exceeds the native integer range"
    seen = set()
    for power in powers:
        identifier = canonical_identifier(power.get("id"), "power")
        if is_bomb_power(power.get("id")):
            continue
        amount = power.get("amount")
        if not identifier or identifier in seen or type(amount) is not int:
            return "The Bomb requires one integer record per companion power"
        seen.add(identifier)
        if not -(2**31) <= amount <= 2**31 - 1:
            return f"The Bomb companion power amount exceeds the native integer range: {identifier}"
        if identifier == "combust":
            hp_loss = power.get("hp_loss")
            # Keep self-loss explicit; it cannot be inferred from total damage.
            if type(hp_loss) is not int or not 1 <= hp_loss <= 2**31 - 1:
                return "The Bomb with Combust requires positive public hp_loss"
    return None


def bomb_pressure(combat: Mapping, additional: PendingBomb | None = None) -> float:
    """Discounted snapshot pressure, NOT predicted damage, protection or victory.

    Allocate each enemy's current HP/block budget in due order, preserving packet
    multiplicity and applying current Intangible per hit. Future block/status
    changes are unknown. No Strength/Weak/Vulnerable or kill bonus applies.
    This is a bounded heuristic on known public records, not a rollout shortcut.
    """
    prospective = (dict(combat, hand=tuple(combat.get("hand", ())) + ({"id": "the_bomb"},))
                   if additional is not None and isinstance(combat.get("hand", ()), (list, tuple)) else combat)
    if bomb_scope_reason(prospective) is not None:
        return 0.0
    try:
        packets = list(pending_bombs(combat.get("player", {})))
    except ValueError:
        return 0.0
    if additional is not None:
        if (not 1 <= additional.damage <= 2**31 - 1
                or not 1 <= additional.remaining_turns <= 3):
            return 0.0
        packets.append(additional)
    packets.sort(key=lambda p: p.remaining_turns)  # Stable: preserve equal-due instance order.
    monsters = combat.get("monsters", ())
    if not isinstance(monsters, (list, tuple)):
        return 0.0
    score = 0.0
    for monster in monsters:
        if not isinstance(monster, Mapping) or monster.get("is_gone") or monster.get("half_dead"):
            continue
        hp, block = monster.get("current_hp"), monster.get("block", 0)
        powers = monster.get("powers", ())
        if type(hp) is not int or type(block) is not int or hp <= 0 or block < 0:
            continue
        if not isinstance(powers, (list, tuple)) or any(not isinstance(p, Mapping) for p in powers):
            continue
        intangible = any(canonical_identifier(p.get("id"), "power") == "intangible"
                         and type(p.get("amount")) is int and p["amount"] > 0 for p in powers)
        for packet in packets:
            hit = min(1, packet.damage) if intangible else packet.damage
            absorbed = min(block, hit)
            block -= absorbed
            useful = min(hp, hit - absorbed)
            hp -= useful
            score += useful * (0.5 ** packet.remaining_turns)
    return score
