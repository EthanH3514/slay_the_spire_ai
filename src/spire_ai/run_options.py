"""Versioned public run choices decoded at the native adapter boundary.

Only the pinned ConsoleSimulator printer grammar is recognized. Commands stay
inside the adapter; policy code sees typed options, not English keyword guesses.
Unknown semantics remain explicit and do not remove the original legal action.
"""
from __future__ import annotations

from functools import lru_cache
import re
from typing import Mapping, Sequence

from .actions import Action, ActionKind
from .reward_policy import native_reward_card
from .rule_catalog import RuleCatalog
from .public_map import link_route_options
from .event_options import event_choice


RUN_OPTIONS_VERSION = "native-public-run-options-v2"


@lru_cache(maxsize=1)
def _catalog() -> RuleCatalog:
    return RuleCatalog.bundled()


def native_card(label: str, deck: Sequence[Mapping] = ()) -> dict | None:
    card = native_reward_card("take " + label)
    if card is not None:
        return card
    # The static catalogue deliberately refuses ambiguous names such as Strike.
    # Resolve these only against identities already present in the public deck.
    matches = {}
    for item in deck:
        entry = _catalog().lookup("card", item.get("id"))
        if entry is None:
            continue
        upgrades = int(item.get("upgrades", 0))
        printed = entry["name"] + ("+" if upgrades else "")
        if printed == label:
            matches[entry["id"]] = {"id": entry["id"], "upgrades": upgrades}
    return next(iter(matches.values())) if len(matches) == 1 else None


def native_run_options(observation: Mapping, options: Sequence, text: str) -> dict:
    """Consume adapter-private option objects; return public semantic metadata."""
    screen_type = str(observation.get("screen", {}).get("type", "")).upper()
    deck = observation.get("inventory", {}).get("deck", ())
    context: dict = {}
    if screen_type == "CARD_SELECT":
        match = re.search(r"^Select Card to (.+)\. \((\d+)/(\d+)\)$", text, re.M)
        tasks = {"Upgrade": "upgrade", "Remove": "remove", "Transform": "transform",
                 "Transform and Upgrade": "transform_upgrade", "Bottle": "bottle",
                 "Obtain": "obtain", "Offer to the spirits": "offer"}
        if match and match[1] in tasks:
            context = {"selection_task": tasks[match[1]],
                       "selected_count": int(match[2]), "selection_count": int(match[3])}
    if screen_type == "EVENT":
        match = re.search(r"^Event - (.+): Select Action\.$", text, re.M)
        if match:
            context["event_name"] = match[1]

    rows = []
    for option in options:
        action, command, label = option.action, option.command or "", option.description.strip()
        row = {"action": action.as_dict(), "kind": "unknown"}
        if action.kind in {ActionKind.PROCEED, ActionKind.CANCEL}:
            row["kind"] = "leave"
        elif option.virtual_action == "open_card_reward":
            row["kind"] = "card_reward"
        elif screen_type == "MAP":
            rooms = {"MONSTER": "monster", "ELITE": "elite", "REST": "rest",
                     "SHOP": "shop", "EVENT": "event", "TREASURE": "treasure",
                     "BOSS": "boss", "Advance to Boss": "boss"}
            if label in rooms and command.isdigit():
                row.update(kind="route", room=rooms[label], x=int(command))
        elif screen_type == "REST":
            choices = {"[Rest]": "rest", "[Smith]": "smith", "[Recall]": "recall",
                       "[Lift]": "lift", "[Toke]": "remove", "[Dig]": "dig", "[Skip]": "leave"}
            row["kind"] = choices.get(label, "unknown")
        elif screen_type == "SHOP":
            match = re.fullmatch(r"(.+) (\d+)g", label)
            if match:
                name, price = match[1].strip(), int(match[2])
                item = command.split()[0]
                if item == "card":
                    card = native_card(name, deck)
                    if card:
                        row.update(kind="buy_card", card=card, price=price)
                elif item in {"relic", "potion"}:
                    row.update(kind="buy_" + item, item_id=name, price=price)
                elif command == "remove" and name == "Remove Card":
                    row.update(kind="buy_remove", price=price)
        elif screen_type == "CARD_SELECT" and context.get("selection_task"):
            card = native_card(label, deck)
            if card:
                row.update(kind="select_card", card=card, task=context["selection_task"])
        elif screen_type == "BOSS_RELIC_REWARDS":
            if label.startswith("Take "):
                row.update(kind="boss_relic", item_id=label[5:])
            elif label == "Skip Reward":
                row["kind"] = "leave"
        elif screen_type == "TREASURE":
            if re.fullmatch(r"Open (Small|Medium|Large|Boss) chest\.", label):
                row["kind"] = "open_chest"
            elif label == "Skip Reward":
                row["kind"] = "leave"
        elif screen_type == "COMBAT_REWARD":
            item = command.split()[0] if command else ""
            if item == "gold" and (match := re.fullmatch(r"take (\d+)g", label)):
                row.update(kind="gold", amount=int(match[1]))
            elif item in {"potion", "relic"}:
                row.update(kind=item, item_id=label.removesuffix("; lose sapphire key"))
            elif item in {"emeraldKey", "sapphireKey"}:
                row.update(kind="key", key=item)
        elif screen_type == "EVENT":
            # Deliberately small exact effect grammar. No fuzzy parsing of costs,
            # curses, random outcomes, fight rewards or phase-dependent effects.
            if re.fullmatch(r"\[Leave\](?: Nothing happens\.)?", label):
                row["kind"] = "leave"
            elif match := re.fullmatch(r"\[[^\]]+\] Heal (\d+) HP\.", label):
                row.update(kind="heal", amount=int(match[1]))
            elif match := re.fullmatch(r"\[[^\]]+\] Max HP \+(\d+)\.", label):
                row.update(kind="max_hp", amount=int(match[1]))
            elif re.fullmatch(r"\[[^\]]+\] Upgrade a card\.", label):
                row["kind"] = "smith"
            elif re.fullmatch(r"\[[^\]]+\] Remove a card from your deck\.", label):
                row["kind"] = "remove"
        if screen_type == "EVENT":
            row["event_choice"] = event_choice(context.get("event_name"), label)
            if row["kind"] == "unknown" and row["event_choice"]["status"] == "parsed":
                row["kind"] = "event_choice"
        if row["kind"] == "unknown":
            row["unknown_reason"] = "unparsed_native_choice"
        rows.append(row)
    if screen_type == "MAP":
        context["map_link_status"] = link_route_options(observation, rows)
    return {"run_options_version": RUN_OPTIONS_VERSION, "decision_options": rows,
            "decision_context": context}


def public_run_options(public) -> tuple[Mapping, ...]:
    screen = public.observation.get("screen", {})
    if screen.get("run_options_version") != RUN_OPTIONS_VERSION:
        raise ValueError("missing_structured_run_options")
    rows = screen.get("decision_options", ())
    if not isinstance(rows, (list, tuple)):
        raise ValueError("invalid_structured_run_options")
    mapped = {}
    for row in rows:
        if (not isinstance(row, Mapping) or not isinstance(row.get("kind"), str)
                or not isinstance(row.get("action"), Mapping)):
            raise ValueError("invalid_structured_run_option")
        action = Action.from_dict(row.get("action", {}))
        if action in mapped:
            raise ValueError("duplicate_structured_run_action")
        mapped[action] = row
    if any(action not in mapped for action in public.legal_actions):
        raise ValueError("incomplete_structured_run_options")
    return tuple(mapped[action] for action in public.legal_actions)
