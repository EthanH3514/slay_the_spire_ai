"""Public pool metadata and scope checks; no RNG state or sampled identities."""
from typing import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES, DiscoveryGeneration, RandomDrawGeneration, RandomHandGeneration
from .identifiers import canonical_identifier
from .public_piles import public_draw_cards

COLORLESS_COMBAT_IDS = frozenset("""
MADNESS THINKING_AHEAD MIND_BLAST METAMORPHOSIS JACK_OF_ALL_TRADES SWIFT_STRIKE
GOOD_INSTINCTS MASTER_OF_STRATEGY MAGNETISM FINESSE DISCOVERY CHRYSALIS TRANSMUTATION
PANACEA PURITY ENLIGHTENMENT FORETHOUGHT FLASH_OF_STEEL HAND_OF_GREED MAYHEM
APOTHEOSIS SECRET_WEAPON PANACHE VIOLENCE DEEP_BREATH SECRET_TECHNIQUE BLIND
THE_BOMB IMPATIENCE DRAMATIC_ENTRANCE TRIP PANIC_BUTTON SADISTIC_NATURE DARK_SHACKLES
""".split())


def generation_pool(contract: RandomHandGeneration | RandomDrawGeneration | DiscoveryGeneration, catalog) -> tuple[Mapping, ...]:
    """Full-unlock base-game membership, not a prediction of the generated card."""
    if not isinstance(contract, (RandomHandGeneration, RandomDrawGeneration, DiscoveryGeneration)):
        raise ValueError("unknown random generation contract")
    cards = []
    for card in catalog.entries("card"):
        if card.get("rarity") not in {"COMMON", "UNCOMMON", "RARE"}:
            continue
        key = canonical_identifier(card["id"], "card")
        if contract in {RandomHandGeneration.IRONCLAD_ATTACK_TURN_ZERO, RandomDrawGeneration.IRONCLAD_ATTACK}:
            eligible = card.get("color") == "RED" and card.get("type") == "ATTACK" and key not in {"feed", "reaper"}
        elif contract is DiscoveryGeneration.IRONCLAD_THREE:
            eligible = card.get("color") == "RED" and key not in {"feed", "reaper"}
        elif contract is RandomDrawGeneration.IRONCLAD_SKILL:
            eligible = card.get("color") == "RED" and card.get("type") == "SKILL"
        else:
            eligible = card["id"] in COLORLESS_COMBAT_IDS
        if eligible:
            cards.append(card)
    return tuple(sorted(cards, key=lambda c: c["id"]))


def generation_support_reason(observation: Mapping, *, played: Mapping | None = None) -> str | None:
    combat = observation.get("combat", {})
    cards = list(combat.get("hand", ())) + list(public_draw_cards(combat)) + list(combat.get("discard_pile", ()))
    if played is not None:
        cards.append(played)
    contracts = set()
    powers = combat.get("player", {}).get("powers", ())
    if any(canonical_identifier(p.get("id"), "card") == "magnetism" for p in powers if isinstance(p, Mapping)):
        contracts.add(RandomHandGeneration.COLORLESS_TURN_START)
    pending_discovery = str(combat.get("card_select_task", "")).upper() == "DISCOVERY" and str(combat.get("input_state", "")).upper() == "CARD_SELECT"
    pending_codex = str(combat.get("card_select_task", "")).upper() == "CODEX" and str(combat.get("input_state", "")).upper() == "CARD_SELECT"
    if pending_discovery or pending_codex:
        contracts.add(DiscoveryGeneration.IRONCLAD_THREE)
    for card in cards:
        recipe = DEFAULT_CARD_RECIPES.lookup(card.get("id")) if isinstance(card, Mapping) else None
        if recipe:
            contracts.update(s.random_hand_generation for s in recipe.effects if s.random_hand_generation is not None)
            contracts.update(s.random_draw_generation for s in recipe.effects if s.random_draw_generation is not None)
            contracts.update(s.discovery_generation for s in recipe.effects if s.discovery_generation is not None)
            contracts.update(e.random_hand_generation for s in recipe.effects if s.turn_trigger
                             for e in s.turn_trigger.effects if e.random_hand_generation is not None)
    if not contracts:
        return None
    if contracts - {RandomHandGeneration.COLORLESS_BASE, RandomHandGeneration.COLORLESS_X_TURN_ZERO,
                    RandomHandGeneration.COLORLESS_TURN_START}:
        if str(observation.get("run", {}).get("character", "")).upper() != "IRONCLAD":
            return "random class-pool generation requires an explicit public Ironclad character"
    if contracts & {RandomHandGeneration.IRONCLAD_ATTACK_TURN_ZERO, DiscoveryGeneration.IRONCLAD_THREE}:
        damaged = combat.get("times_damaged")
        if type(damaged) is not int or damaged < 0:
            return "random attack generation requires public times_damaged for Blood for Blood"
    if any(canonical_identifier(p.get("id"), "card") == "master_reality" for p in powers if isinstance(p, Mapping)):
        return "random generation with Master Reality is not audited"
    if pending_discovery:
        from .discovery import discovery_candidates
        try:
            discovery_candidates(combat)
        except ValueError as error:
            return str(error)
    if pending_codex:
        from .discovery import ironclad_offer_candidates
        try:
            ironclad_offer_candidates(combat, source="Codex")
        except ValueError as error:
            return str(error)
    return None
