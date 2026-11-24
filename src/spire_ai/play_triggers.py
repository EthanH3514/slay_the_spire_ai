"""Public trigger state. Panache's displayed counter is not its damage amount."""
from typing import Mapping

from .bombs import bomb_scope_reason

from .identifiers import canonical_identifier


def panache_state(player: Mapping) -> tuple[int, int] | None:
    matches = [p for p in player.get("powers", ()) if isinstance(p, Mapping)
               and canonical_identifier(p.get("id"), "power") == "panache"]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Panache requires one public counter/damage record")
    remaining, damage = matches[0].get("amount"), matches[0].get("damage")
    if type(remaining) is not int or not 1 <= remaining <= 5 or type(damage) is not int or damage <= 0:
        raise ValueError("Panache requires public remaining amount 1..5 and positive damage; missing state is unknown")
    return remaining, damage


def play_trigger_state_reason(combat: Mapping) -> str | None:
    reason = bomb_scope_reason(combat)
    if reason is not None:
        return f"{reason} [THE_BOMB]"
    powers = combat.get("player", {}).get("powers", ())
    if not isinstance(powers, (list, tuple)) or any(not isinstance(p, Mapping) for p in powers):
        return "Public trigger state requires a structured power list"
    try:
        panache_state(combat.get("player", {}))
    except ValueError as error:
        return str(error)
    return None
