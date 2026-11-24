"""Conservative, forward-only STS1 history reconstruction for offline card BC.

This is a PARTIAL public observation contract, not a complete game-state replay.
Decision-time HP/gold/map/potions, relic counters and bottled assignments are
unavailable here. Never backfill them from end-of-floor or final-run summaries.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import re

from .actions import Action, ActionKind
from .environment import EnvironmentState
from .policy_api import policy_input_from_state
from .rule_catalog import RuleCatalog

VERSION = "expert-card-prefix-partial-v2"
PARTIAL_FIELDS = ("decision_hp", "decision_max_hp", "decision_gold", "public_map",
                  "potions", "relic_counters", "relic_acquisition_order",
                  "bottled_assignments", "persistent_card_misc")
# These pickups can change permanent cards without recording the affected
# instances in this historical format. This is a DATA fidelity barrier, not a
# statement that our game simulator or combat controller lacks these mechanics.
UNLOGGED_DECK_RELICS = {
    "WAR_PAINT", "WHETSTONE", "PANDORAS_BOX", "ASTROLABE", "EMPTY_CAGE",
    "TINY_HOUSE", "DOLLYS_MIRROR", "CALLING_BELL", "NECRONOMICON", "CURSED_KEY",
    "OMAMORI",
}
EGG_TYPES = {"MOLTEN_EGG": "ATTACK", "TOXIC_EGG": "SKILL", "FROZEN_EGG": "POWER"}
DEFERRED_RELICS = {"RUNIC_DOME", "PRISMATIC_SHARD"}
NO_CARD = {"SKIP", "Singing Bowl"}


class HistoryGap(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def integer(value):
    if type(value) not in (int, float) or not 0 <= value <= 10000 or int(value) != value:
        raise HistoryGap("invalid_integer")
    return int(value)


def act_for_floor(floor):
    return 1 if floor <= 17 else 2 if floor <= 34 else 3 if floor <= 51 else 4


def card(value, catalog):
    if not isinstance(value, str):
        raise HistoryGap("invalid_card_id")
    match = re.fullmatch(r"(.+?)(?:\+(\d+))?", value)
    if not match:
        raise HistoryGap("invalid_card_id")
    name, upgrades = match.groups()
    entry = catalog.lookup("card", name)
    if entry is None:
        raise HistoryGap("unknown_card:" + name)
    return {"id": entry["id"], "upgrades": int(upgrades or 0)}


def card_key(value):
    return value["id"], value["upgrades"]


def card_counts(cards):
    return Counter(card_key(c) for c in cards)


def obtained_card(value, relics, catalog):
    """Egg onObtainCard: upgrade matching unupgraded cards exactly once.

    Event metrics may be written BEFORE the obtain effect; reward/shop metrics
    may already include preview upgrades. Never upgrade an existing deck card,
    or upgrade Searing Blow a second time just because it can still upgrade.
    """
    result = card(value, catalog)
    if result["upgrades"] == 0:
        kind = catalog.lookup("card", result["id"])["type"]
        if any(EGG_TYPES.get(r) == kind for r in relics):
            result["upgrades"] = 1
    return result


def offer_key(choice, catalog):
    offered = list(choice["not_picked"])
    if choice["picked"] not in NO_CARD:
        offered.append(choice["picked"])
    return tuple(sorted(card_key(card(c, catalog)) for c in offered))


def ambiguous_egg_order(relic_names, card_names, catalog):
    eggs = {catalog.lookup("relic", name)["id"] for name in relic_names
            if catalog.lookup("relic", name) is not None}
    return any(obtained_card(name, eggs, catalog) != card(name, catalog) for name in card_names)


def remove_card(deck, value, catalog, *, base_id_only=False):
    target = card(value, catalog)
    if base_id_only:
        # Some event log helpers store cardID, not the upgraded metric ID.
        # Even when an unupgraded copy exists, do not silently prefer it if an
        # upgraded copy could also have been removed. Only a unique variant is
        # reconstructable from the already-known prefix.
        variants = {c["upgrades"] for c in deck if c["id"] == target["id"]}
        if "+" in value or len(variants) != 1:
            raise HistoryGap("ambiguous_event_removal_variant:" + value)
        target["upgrades"] = next(iter(variants))
    for index, current in enumerate(deck):
        if current == target:
            deck.pop(index)
            return
    raise HistoryGap("remove_missing_card:" + value)


def upgrade_card(deck, value, catalog):
    target = card(value, catalog)
    candidates = [c for c in deck if c["id"] == target["id"] and
                  (c["upgrades"] == target["upgrades"] if "+" in value else
                   c["upgrades"] == 0 or c["id"] == "SEARING_BLOW")]
    if not candidates or len({c["upgrades"] for c in candidates}) != 1:
        raise HistoryGap("ambiguous_or_missing_upgrade:" + value)
    candidates[0]["upgrades"] += 1


def public_choice(deck, relics, floor, ascension, choice, catalog):
    """Candidate ordering is independent of picked/not_picked partition."""
    offered = list(choice["not_picked"])
    picked = choice["picked"]
    if picked not in NO_CARD:
        offered.append(picked)
    candidates = sorted((card(c, catalog) for c in offered), key=card_key)
    if not candidates or len(candidates) > 30:
        raise HistoryGap("unsupported_offer_size")
    if len(set(map(card_key, candidates))) != len(candidates):
        raise HistoryGap("duplicate_offer_identity")
    if picked == "Singing Bowl" and "SINGING_BOWL" not in relics:
        raise HistoryGap("singing_bowl_not_observed")
    actions = [Action(ActionKind.CHOOSE, choice_index=i) for i in range(len(candidates))]
    options = [{"action": a.as_dict(), "kind": "card_reward", "card": c}
               for a, c in zip(actions, candidates, strict=True)]
    actions.append(Action(ActionKind.CANCEL))
    options.append({"action": actions[-1].as_dict(), "kind": "leave"})
    target = len(candidates) if picked == "SKIP" else None
    if "SINGING_BOWL" in relics:
        actions.append(Action(ActionKind.CHOOSE, choice_index=len(candidates)))
        options.append({"action": actions[-1].as_dict(), "kind": "max_hp", "amount": 2})
        if picked == "Singing Bowl":
            target = len(actions) - 1
    if picked not in NO_CARD:
        target = candidates.index(card(picked, catalog))
    observation = {"run": {"character": "IRONCLAD", "act": act_for_floor(floor),
                           "floor": floor, "ascension": ascension},
                   "player": {}, "inventory": {"deck": copy.deepcopy(sorted(deck, key=card_key)),
                                                "relics": [{"id": r} for r in sorted(relics)]},
                   "screen": {"type": "CARD_REWARD", "card_reward_options": options}}
    public = policy_input_from_state(EnvironmentState(observation, tuple(actions)))
    return {"public": public.as_dict(), "target": target,
            "label": "skip" if picked == "SKIP" else "singing_bowl" if picked == "Singing Bowl" else "card",
            "input_sha256": public.fingerprint(), "partial_fields": list(PARTIAL_FIELDS)}


def reconstruct(run, *, catalog=None, base_id_removal_events=(), inventory_barriers=()):
    """Return prefix rows and separate audits. Never use final summaries as input.

Same-floor inventory mutations remain excluded as decision rows. Distinct
combat reward acquisitions follow append order, but SKIP can be logged later
on room exit; do not reconstruct its timing around other card acquisitions. An
unlogged permanent deck mutation taints all following prefixes, not earlier ones.
"""
    catalog = catalog or RuleCatalog.bundled()
    required = ("card_choices", "event_choices", "campfire_choices", "relics_obtained",
                "items_purchased", "item_purchase_floors", "items_purged", "items_purged_floors",
                "boss_relics", "neow_bonus_log", "path_per_floor")
    if any(k not in run for k in required):
        raise HistoryGap("missing_history_tables")
    if run.get("character_chosen") != "IRONCLAD":
        raise HistoryGap("wrong_character")
    for flag in ("is_daily", "is_trial", "is_endless", "chose_seed", "is_beta"):
        if run.get(flag) is not False:
            raise HistoryGap("unsupported_or_unknown_run_flag:" + flag)
    ascension = integer(run["ascension_level"])
    if ascension > 20 or run.get("special_seed"):
        raise HistoryGap("unsupported_run_mode")
    deck = [card("Strike_R", catalog) for _ in range(5)] + [card("Defend_R", catalog) for _ in range(4)]
    deck.append(card("Bash", catalog))
    if ascension >= 10:
        deck.append(card("AscendersBane", catalog))
    relics = ["BURNING_BLOOD"]
    barriers = []
    rows, excluded = [], []

    def barrier(reason, floor):
        entry = {"floor": floor, "reason": reason}
        if entry not in barriers:
            barriers.append(entry)

    def gain_relic(name, floor):
        entry = catalog.lookup("relic", name)
        if entry is None:
            raise HistoryGap("unknown_relic:" + name)
        identifier = entry["id"]
        if identifier not in relics:
            if identifier == "BLACK_BLOOD" and "BURNING_BLOOD" in relics:
                relics.remove("BURNING_BLOOD")
            relics.append(identifier)
            if identifier in UNLOGGED_DECK_RELICS:
                barrier("unlogged_deck_effect:" + identifier, floor)
            if identifier in DEFERRED_RELICS:
                barrier("outside_project_scope:" + identifier, floor)

    def changes(removed, transformed, upgraded, obtained, *, base_id_removals=False):
        for name in removed:
            remove_card(deck, name, catalog, base_id_only=base_id_removals)
        for name in transformed:
            remove_card(deck, name, catalog)
        for name in upgraded:
            upgrade_card(deck, name, catalog)
        deck.extend(obtained_card(name, relics, catalog) for name in obtained)

    neow = run["neow_bonus_log"]
    if not isinstance(neow, dict) or any(k not in neow for k in
            ("cardsRemoved", "cardsTransformed", "cardsUpgraded", "cardsObtained", "relicsObtained")):
        raise HistoryGap("missing_neow_detail")
    if run.get("neow_bonus") == "BOSS_RELIC":
        if len(neow["relicsObtained"]) != 1:
            raise HistoryGap("ambiguous_neow_boss_swap")
        relics.remove("BURNING_BLOOD")
    try:
        if ambiguous_egg_order(neow["relicsObtained"], neow["cardsObtained"], catalog):
            barrier("neow_egg_card_order_unknown", 0)
        for name in neow["relicsObtained"]:
            gain_relic(name, 0)
        changes(neow["cardsRemoved"], neow["cardsTransformed"], neow["cardsUpgraded"], neow["cardsObtained"])
    except HistoryGap as error:
        barrier(str(error), 0)

    choices, events, fires, pickups, purchases, purges, bosses = (defaultdict(list) for _ in range(7))
    scheduled_barriers = defaultdict(list)
    for entry in inventory_barriers:
        if not isinstance(entry["reason"], str) or not entry["reason"]:
            raise HistoryGap("invalid_inventory_barrier")
        scheduled_barriers[integer(entry["floor"])].append(entry["reason"])
    for index, row in enumerate(run["card_choices"]):
        choices[integer(row["floor"])].append((index, row))
    for table, destination in (("event_choices", events), ("campfire_choices", fires), ("relics_obtained", pickups)):
        for row in run[table]:
            destination[integer(row["floor"])].append(row)
    for names_key, floors_key, destination in (("items_purchased", "item_purchase_floors", purchases),
                                               ("items_purged", "items_purged_floors", purges)):
        if len(run[names_key]) != len(run[floors_key]):
            raise HistoryGap("unaligned_history:" + names_key)
        for name, floor in zip(run[names_key], run[floors_key], strict=True):
            destination[integer(floor)].append(name)
    # Boss chests occur after Act1/2. These choices are not known before then.
    for index, boss in enumerate(run["boss_relics"]):
        if index > 1:
            raise HistoryGap("unexpected_boss_relic_count")
        bosses[17 + 17 * index].append(boss)

    for floor in sorted(set(choices) | set(events) | set(fires) | set(pickups) | set(purchases) |
                        set(purges) | set(bosses) | set(scheduled_barriers)):
        for reason in scheduled_barriers[floor]:
            barrier(reason, floor)
        floor_choices = choices[floor]
        # Record the pre-floor relic set: taking an egg on this floor must not
        # retroactively upgrade a reward picked BEFORE that relic. Reward IDs
        # already include any preview upgrade if the egg was taken first.
        prior_relics = tuple(relics)
        try:
            signatures = [offer_key(c, catalog) for _, c in floor_choices]
            duplicate_offers = len(set(signatures)) != len(signatures)
            if duplicate_offers and any(c["picked"] not in NO_CARD for _, c in floor_choices):
                barrier("ambiguous_repeated_reward_acquisition", floor)
        except (HistoryGap, KeyError, TypeError) as error:
            duplicate_offers = True
            barrier(str(error), floor)
        try:
            # Purchases have an ordered items_purchased stream. Other tables
            # only have floor numbers, so an egg and a raw event acquisition on
            # that same floor cannot be arbitrarily ordered. Reward metrics are
            # handled separately using prior_relics and their preview IDs.
            event_gains = [name for e in events[floor] for name in e.get("cards_obtained", [])]
            unsequenced_relics = [p["key"] for p in pickups[floor]]
            unsequenced_relics += [name for e in events[floor] for key in ("relics_obtained", "relics_lost")
                                  for name in e.get(key, [])]
            purchased_cards = [name for name in purchases[floor] if catalog.lookup("card", name)]
            if (ambiguous_egg_order(unsequenced_relics, event_gains + purchased_cards, catalog) or
                    ambiguous_egg_order(purchases[floor], event_gains, catalog)):
                barrier("same_floor_egg_card_order_unknown", floor)
        except (HistoryGap, KeyError, TypeError) as error:
            barrier(str(error), floor)
        if len({digest(e) for e in events[floor]}) != len(events[floor]):
            barrier("duplicate_event_record", floor)
        event_names = [e.get("event_name") for e in events[floor]]
        if len(set(event_names)) != len(event_names):
            barrier("repeated_same_floor_event", floor)
        # If an upgrade/removal could target a newly picked card, a floor-only
        # log cannot identify whether the operation preceded that acquisition.
        try:
            new_ids = {card(c["picked"], catalog)["id"] for _, c in floor_choices if c["picked"] not in NO_CARD}
            edited = list(purges[floor])
            edited.extend(c["data"] for c in fires[floor] if c["key"] in ("SMITH", "PURGE"))
            edited.extend(name for event in events[floor] for key in
                          ("cards_removed", "cards_transformed", "cards_upgraded") for name in event.get(key, []))
            if new_ids & {card(name, catalog)["id"] for name in edited}:
                barrier("interleaved_same_card_mutation", floor)
        except (HistoryGap, KeyError, TypeError) as error:
            barrier(str(error), floor)
        mutations = (bool(pickups[floor] or purchases[floor] or purges[floor] or bosses[floor]) or
                     any(e.get(k) for e in events[floor] for k in
                         ("cards_obtained", "cards_removed", "cards_transformed", "cards_upgraded", "relics_obtained", "relics_lost")) or
                     any(c["key"] in ("SMITH", "PURGE", "DIG") for c in fires[floor]))
        # A private multiset for ordered reward picks, separate from replaying
        # other same-floor effects below. Never expose a later pick to an earlier
        # row. Mutating floors are still excluded regardless of this order.
        reward_deck = copy.deepcopy(deck)
        has_card_pick = any(c["picked"] not in NO_CARD for _, c in floor_choices)
        for index, choice in floor_choices:
            reasons = []
            if barriers:
                reasons.append("prefix_tainted:" + barriers[0]["reason"])
            if floor == 0:
                reasons.append("neow_decision_order_not_verified")
            if duplicate_offers:
                reasons.append("repeated_offer_identity")
            if len(floor_choices) > 1 and choice["picked"] == "SKIP" and has_card_pick:
                reasons.append("deferred_skip_order_unknown")
            if mutations:
                reasons.append("same_floor_inventory_order_unknown")
            if act_for_floor(floor) > 3:
                reasons.append("act4_outside_pilot")
            # Library and similar event choices do not necessarily permit SKIP.
            # First pilot accepts combat rewards only; no guessed legal exits.
            room = run["path_per_floor"][floor - 1] if 1 <= floor <= len(run["path_per_floor"]) else None
            if room not in ("M", "E", "B"):
                reasons.append("not_verified_combat_reward")
            if not reasons:
                try:
                    example = public_choice(reward_deck, relics, floor, ascension, choice, catalog)
                    rows.append({"version": VERSION, "choice_index": index, "floor": floor,
                                 "act": act_for_floor(floor), **example})
                except (HistoryGap, KeyError, TypeError, ValueError) as error:
                    reasons.append("invalid_offer:" + str(error))
            if reasons:
                excluded.append({"choice_index": index, "floor": floor, "reasons": reasons})
            if choice["picked"] not in NO_CARD:
                try:
                    reward_deck.append(obtained_card(choice["picked"], prior_relics, catalog))
                except (HistoryGap, KeyError, TypeError) as error:
                    barrier(str(error), floor)
        # Advance only after constructing the pre-choice observation.
        try:
            for pickup in pickups[floor]:
                gain_relic(pickup["key"], floor)
            for name in purchases[floor]:
                if catalog.lookup("relic", name) is not None:
                    gain_relic(name, floor)
                elif catalog.lookup("potion", name) is None:
                    deck.append(obtained_card(name, relics, catalog))
            for name in purges[floor]:
                remove_card(deck, name, catalog)
            for fire in fires[floor]:
                if fire["key"] == "SMITH":
                    upgrade_card(deck, fire["data"], catalog)
                elif fire["key"] == "PURGE":
                    remove_card(deck, fire["data"], catalog)
                elif fire["key"] not in ("REST", "RECALL", "DIG", "LIFT"):
                    raise HistoryGap("unknown_campfire_action")
            selected = [c["picked"] for _, c in floor_choices if c["picked"] not in NO_CARD]
            for event in events[floor]:
                obtained = list(event.get("cards_obtained", []))
                # Vanilla Vampires logs five Bites, but omits the removal of
                # ALL starter Strikes (including upgraded ones). The importer
                # is Ironclad-only and excludes Prismatic Shard. Do not remove
                # other attacks whose names happen to contain "Strike".
                if event["event_name"] == "Vampires" and obtained:
                    if Counter(obtained) != Counter({"Bite": 5}) or event.get("cards_removed"):
                        raise HistoryGap("unexpected_vampires_record")
                    deck[:] = [c for c in deck if c["id"] != "STRIKE_RED"]
                # The Library logs the SAME card in both event and card choices.
                # Do not globally deduplicate equal IDs: two acquisitions can
                # legitimately obtain the same card on the same floor.
                if event["event_name"] == "The Library" and floor_choices:
                    if Counter(obtained) != Counter(selected):
                        raise HistoryGap("library_choice_event_mismatch")
                    obtained = []
                if event["event_name"] == "Duplicator" and obtained:
                    # This event logs cardID, not the copy's upgrade count.
                    # Infer only from a UNIQUE public pre-event variant; never
                    # recover the chosen variant from the final deck.
                    if len(obtained) != 1:
                        raise HistoryGap("unexpected_duplicator_count")
                    identifier = card(obtained[0], catalog)["id"]
                    variants = {card_key(c) for c in deck if c["id"] == identifier}
                    if len(variants) != 1:
                        raise HistoryGap("ambiguous_duplicator_variant")
                    name, level = next(iter(variants))
                    obtained = [name + ("+" + str(level) if level else "")]
                changes(event.get("cards_removed", []), event.get("cards_transformed", []),
                        event.get("cards_upgraded", []), obtained,
                        base_id_removals=event.get("event_name") in base_id_removal_events)
                for name in event.get("relics_lost", []):
                    entry = catalog.lookup("relic", name)
                    if entry is None or entry["id"] not in relics:
                        raise HistoryGap("removed_relic_not_present")
                    relics.remove(entry["id"])
                for name in event.get("relics_obtained", []):
                    gain_relic(name, floor)
            deck.extend(obtained_card(name, prior_relics, catalog) for name in selected)
            for boss in bosses[floor]:
                if "picked" not in boss:
                    raise HistoryGap("missing_boss_relic_pick")
                if boss["picked"] not in NO_CARD:
                    gain_relic(boss["picked"], floor)
        except (HistoryGap, KeyError, TypeError) as error:
            barrier(str(error), floor)

    # AUDIT ONLY. A mismatch never repairs earlier observations with final cards.
    try:
        final_match = card_counts(deck) == card_counts([card(x, catalog) for x in run["master_deck"]])
    except (KeyError, HistoryGap):
        final_match = None
    return {"version": VERSION, "rows": rows, "excluded": excluded, "barriers": barriers,
            "final_deck_matches": final_match,
            "unexplained_final_mismatch": final_match is False and not barriers,
            "partial_fields": list(PARTIAL_FIELDS)}


def encode_row(row, encoder):
    """Explicit partial-input encoding; no fabricated relic acquisition order."""
    from dataclasses import replace
    from .policy_api import PolicyInput
    from .public_encoding import FeatureTable
    if row["version"] != VERSION or row["partial_fields"] != list(PARTIAL_FIELDS):
        raise ValueError("expert partial-input contract mismatch")
    public = row["public"]
    policy_input = PolicyInput(public["observation"], tuple(Action.from_dict(a) for a in public["legal_actions"]))
    if policy_input.fingerprint() != row["input_sha256"]:
        raise ValueError("expert public-input fingerprint mismatch")
    decision = encoder.encode(policy_input)
    target = row["target"]
    if type(target) is not int or not 0 <= target < len(decision.legal_actions):
        raise ValueError("invalid expert target")
    decision.decode(target)
    table = decision.tables["relics"]
    index = table.columns.index("order")
    values, known = [list(x) for x in table.values], [list(x) for x in table.known]
    for value, mask in zip(values, known, strict=True):
        value[index], mask[index] = 0.0, False
    tables = dict(decision.tables)
    tables["relics"] = FeatureTable(table.columns, tuple(map(tuple, values)), tuple(map(tuple, known)))
    return replace(decision, tables=tables)
