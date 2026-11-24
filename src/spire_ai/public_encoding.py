"""Versioned numeric entity/candidate tables for a future learned no-SL policy.

No ML dependency or fitted normalization. PAD=0, UNK=1; missing numbers are zero
with a false known mask. This is a deliberately scoped representation, not a
claim of sufficient state, complete simulator fidelity, or a trained policy.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from .actions import Action, ActionKind
from .policy_api import PolicyInput, hidden_enemy_intents
from .policy_scope import assess_policy_scope
from .public_map import validate_public_map
from .rule_catalog import RuleCatalog
from .rule_coverage import _validate_input


ENCODING_VERSION = "public-entity-candidates-v2-combat-selections"
PAD, UNK = 0, 1

# Static vocabularies, never extended/fitted from trajectories. Power identifiers
# are the union of the pinned native PlayerStatus/MonsterStatus enum names.
POWER_NAMES = """
DOUBLE_DAMAGE DRAW_REDUCTION FRAIL INTANGIBLE VULNERABLE WEAK BIAS CONFUSED
CONSTRICTED ENTANGLED FASTING HEX LOSE_DEXTERITY LOSE_STRENGTH NO_BLOCK NO_DRAW
WRAITH_FORM BARRICADE BLASPHEMER CORRUPTION ELECTRO SURROUNDED MASTER_REALITY PEN_NIB
WRATH_NEXT_TURN AMPLIFY BLUR BUFFER COLLECT DOUBLE_TAP DUPLICATION ECHO_FORM
FREE_ATTACK_POWER REBOUND MANTRA ACCURACY AFTER_IMAGE BATTLE_HYMN BRUTALITY BURST
COMBUST CREATIVE_AI DARK_EMBRACE DEMON_FORM DEVA DEVOTION DRAW_CARD_NEXT_TURN
ENERGIZED ENVENOM ESTABLISHMENT EVOLVE FEEL_NO_PAIN FIRE_BREATHING FLAME_BARRIER
FOCUS FORESIGHT HELLO_WORLD INFINITE_BLADES JUGGERNAUT LIKE_WATER LOOP MAGNETISM
MAYHEM METALLICIZE NEXT_TURN_BLOCK NOXIOUS_FUMES OMEGA PANACHE PHANTASMAL
PLATED_ARMOR RAGE REGEN RITUAL RUPTURE SADISTIC STATIC_DISCHARGE THORNS
THOUSAND_CUTS TOOLS_OF_THE_TRADE VIGOR WAVE_OF_THE_HAND EQUILIBRIUM ARTIFACT
DEXTERITY STRENGTH THE_BOMB BLOCK_RETURN CHOKED CORPSE_EXPLOSION LOCK_ON MARK
POISON SHACKLED ANGRY BEAT_OF_DEATH CURIOSITY CURL_UP ENRAGE FADING FLIGHT
GENERIC_STRENGTH_UP MALLEABLE MODE_SHIFT SLOW SPORE_CLOUD THIEVERY TIME_WARP
INVINCIBLE REACTIVE SHARP_HIDE ASLEEP MINION MINION_LEADER PAINFUL_STABS REGROW
SHIFTING STASIS
""".split()
STATIC_VOCABS = {
    "screen": "BATTLE CARD_REWARD CARD_SELECT COMBAT_REWARD MAP REST SHOP EVENT TREASURE BOSS_RELIC_REWARDS".split(),
    "input": "PLAYER_NORMAL CARD_SELECT EXECUTING_ACTIONS NONE".split(),
    "room": "NONE MONSTER ELITE REST SHOP EVENT TREASURE BOSS BOSS_TREASURE".split(),
    "card_type": "ATTACK SKILL POWER STATUS CURSE".split(),
    "pile": "deck hand draw_pile draw_pile_unordered draw_pile_known_top draw_pile_known_bottom discard_pile exhaust_pile selection_candidates candidate".split(),
    "action": [kind.value for kind in ActionKind],
    "option": "unknown leave card_reward route rest smith recall lift remove dig buy_card buy_relic buy_potion buy_remove select_card boss_relic open_chest gold potion relic key heal max_hp event_choice play_card use_potion discard_potion end_turn combat_select".split(),
    "task": "EXHAUST_MANY EXHAUST_ONE HEADBUTT ARMAMENTS WARCRY FORETHOUGHT EXHUME SECRET_TECHNIQUE SECRET_WEAPON DUAL_WIELD DISCOVERY upgrade remove transform transform_upgrade bottle obtain offer GAMBLE CODEX LIQUID_MEMORIES_POTION".split(),
    "rarity": "BASIC COMMON UNCOMMON RARE SPECIAL CURSE BOSS unspecified".split(),
    "pool": "class colorless cursed_tome_books".split(),
    "selector": "curses strikes strikes_and_defends all_upgradeable".split(),
    "key": "emeraldKey rubyKey sapphireKey".split(),
    "event": """NEOW|Ominous Forge|Pleading Vagrant|Ancient Writing|Old Beggar|Big Fish|Bonfire Spirits|
        The Colosseum|Cursed Tome|Dead Adventurer|Designer In-Spire|Augmenter|Duplicator|Face Trader|Falling|
        Forgotten Altar|The Divine Fountain|Council of Ghosts|Golden Idol|Golden Shrine|Wing Statue|
        Knowing Skull|Lab|The Ssssserpent|Living Wall|Masked Bandits|Match and Keep|Mindbloom|
        Hypnotizing Colored Mushrooms|Mysterious Sphere|The Nest|N'loth|Note For Yourself|Purifier|
        Scrap Ooze|Secret Portal|Sensory Stone|Shining Light|The Cleric|The Joust|The Library|
        The Mausoleum|The Moai Head|The Woman in Blue|Tomb of Lord Red Mask|Transmorgrifier|
        Upgrade Shrine|Vampires(?)|We Meet Again!|Wheel of Change|Winding Halls|World of Goop""".split("|"),
    "power": POWER_NAMES,
    "operation": """leave heal heal_full lose_hp damage lose_max_hp gain_max_hp
        gain_gold lose_gold lose_all_gold gain_gold_range select_remove select_upgrade
        select_transform select_duplicate select_store remove_set upgrade_set random_upgrade
        random_transform card_offer random_card obtain_card add_curse random_curse obtain_relic
        random_relic random_potion lose_card lose_relic lose_potion lose_starter_relic
        gain_max_hp_fraction lose_max_hp_fraction damage_fraction disable_healing travel_to_boss
        begin_combat continue_event combat_reward_relic search_loot random_event_outcome
        displayed_chance""".split(),
}
TABLE_COLUMNS = {
    "global": ("screen_id", "boss_id", "room_id", "input_id", "task_id", "event_id", "act", "floor", "ascension",
               "hp", "max_hp", "gold", "turn", "energy", "block", "cards_played", "times_damaged",
               "emerald_key", "ruby_key", "sapphire_key", "powers_complete", "effects_resolved",
               "intents_hidden", "map_present", "event_fidelity_certified", "top_order_uncertain",
               "bottom_order_uncertain", "bottom_knowledge_tracked"),
    "cards": ("card_id", "pile_id", "position", "type_id", "upgrades", "base_cost", "cost_for_turn",
              "free_to_play_once", "base_damage", "special_data", "misc", "exhausts", "ethereal", "has_target"),
    "monsters": ("monster_id", "move_id", "hp", "max_hp", "block", "move_damage", "move_hits", "is_gone", "half_dead"),
    "powers": ("owner_ref", "order", "power_id", "amount", "damage", "hp_loss"),
    "relics": ("relic_id", "order", "counter"),
    "potions": ("potion_id", "slot"),
    "map_nodes": ("x", "y", "room_id", "burning_elite", "current", "start", "boss"),
    "map_edges": ("source_ref", "destination_ref"),
    "known_draw_top": ("order", "card_ref"),
    "actions": ("action_id", "option_id", "card_ref", "potion_ref", "monster_ref", "map_ref",
                "relic_id", "potion_id", "price", "amount", "task_id", "room_id", "key_id", "event_parsed",
                "event_partial", "unknown_clause_count", "effect_fidelity_certified", "selection_known"),
    "action_cards": ("action_ref", "order", "card_ref"),
    "event_effects": ("action_ref", "parent_ref", "operation_id", "cost", "amount", "count", "minimum",
                      "maximum", "numerator", "denominator", "offer_size", "rarity_id", "pool_id",
                      "card_id", "relic_id", "potion_id", "selector_id", "unspecified_details"),
}
SCALES = {"act": 4, "floor": 60, "ascension": 20, "hp": 100, "max_hp": 100, "gold": 1000,
          "turn": 20, "energy": 10, "block": 100, "cards_played": 20, "times_damaged": 20,
          "upgrades": 10, "base_cost": 5, "cost_for_turn": 5, "base_damage": 100,
          "special_data": 100, "misc": 100, "move_damage": 100, "move_hits": 10,
          "amount": 100, "damage": 100, "hp_loss": 10, "counter": 100, "price": 1000,
          "count": 10, "minimum": 1000, "maximum": 1000, "numerator": 100, "denominator": 100,
          "offer_size": 20, "unknown_clause_count": 10}


def _key(value):
    return "".join(char for char in str(value).lower() if char.isalnum())


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def _items(value):
    if not isinstance(value, (list, tuple)):
        raise ValueError("encoder expected a public sequence")
    return value


@dataclass(frozen=True, slots=True)
class FeatureTable:
    columns: tuple[str, ...]
    values: tuple[tuple[int | float, ...], ...]
    known: tuple[tuple[bool, ...], ...]


@dataclass(frozen=True, slots=True)
class EncodedDecision:
    schema_id: str
    tables: Mapping[str, FeatureTable]
    legal_actions: tuple[Action, ...]  # Decoder metadata, NEVER model features.
    legal_mask: tuple[bool, ...]
    scope_mask: tuple[bool, ...]
    unknown_categories: Mapping[str, int]
    gaps: tuple[str, ...]

    def decode(self, row: int) -> Action:
        if type(row) is not int or not 0 <= row < len(self.legal_actions):
            raise ValueError("invalid candidate row")
        if not self.legal_mask[row] or not self.scope_mask[row]:
            raise ValueError("candidate row excluded by legality or policy scope")
        return self.legal_actions[row]


class PublicEncoder:
    def __init__(self, catalog: RuleCatalog | None = None, *, max_actions=4096, max_rows=32768):
        if type(max_actions) is not int or type(max_rows) is not int or min(max_actions, max_rows) < 1:
            raise ValueError("positive integer encoder limits required")
        self.catalog = catalog or RuleCatalog.bundled()
        self.max_actions, self.max_rows = max_actions, max_rows
        vocab = dict(STATIC_VOCABS)
        for kind in ("card", "monster", "move", "encounter", "potion", "relic"):
            vocab[kind] = [entry["id"] for entry in self.catalog.entries(kind)]
        vocab["potion"].append("empty_potion_slot")
        self.vocabularies = MappingProxyType({kind: tuple(sorted({_key(v) for v in values}))
                                              for kind, values in vocab.items()})
        self._tokens = {kind: {name: i + 2 for i, name in enumerate(names)}
                        for kind, names in self.vocabularies.items()}
        schema = {"version": ENCODING_VERSION, "columns": TABLE_COLUMNS, "scales": SCALES,
                  "vocabularies": dict(self.vocabularies), "padding": PAD, "unknown": UNK}
        self.schema_id = hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()

    def describe(self) -> dict:
        return {"version": ENCODING_VERSION, "schema_id": self.schema_id, "columns": TABLE_COLUMNS,
                "scales": SCALES, "vocabularies": dict(self.vocabularies), "pad": PAD, "unknown": UNK,
                "max_actions": self.max_actions, "max_rows": self.max_rows,
                "normalization": "fixed_scales_no_fitting", "training_ready": False}

    def encode(self, public: PolicyInput) -> EncodedDecision:
        _validate_input(public)
        actions = public.legal_actions
        if len(actions) > self.max_actions or len(set(actions)) != len(actions):
            raise ValueError("candidate overflow or duplicate legal actions; no truncation allowed")
        if any(a.kind in {ActionKind.START, ActionKind.WAIT, ActionKind.REQUEST_STATE}
               or any(getattr(a, field) is not None for field in ("seed", "character", "ascension", "wait_frames"))
               for a in actions):
            raise ValueError("episode-control metadata is not a learned game action")
        scope = assess_policy_scope(public.observation, actions) if actions else None
        rows = {name: [] for name in TABLE_COLUMNS}
        unknown = Counter()
        gaps = {"effect_fidelity_not_certified", "representation_is_scoped_not_sufficient_state"}

        def numeric(value, name):
            if value is None:
                return 0.0, False
            if type(value) not in {int, float, bool}:
                raise ValueError("non-numeric public encoder field: " + name)
            number = float(value) / SCALES.get(name, 1)
            if not math.isfinite(number) or abs(number) > 3.402823466e38:
                raise ValueError("non-finite public encoder field: " + name)
            return number, True

        def category(kind, value):
            if value is None:
                return PAD, False
            if not isinstance(value, str):
                raise ValueError("non-string public category: " + kind)
            if kind in {"card", "monster", "move", "encounter", "potion", "relic"}:
                value = self.catalog.canonical_id(kind, value)
            token = self._tokens[kind].get(_key(value), UNK)
            if token == UNK:
                unknown[kind] += 1
            return token, token != UNK

        def add(table, fields):
            row = tuple(fields.get(name, (0.0, False)) for name in TABLE_COLUMNS[table])
            rows[table].append(row)
            if sum(map(len, rows.values())) > self.max_rows:
                raise ValueError("entity/relationship overflow; no truncation allowed")
            return len(rows[table])  # One-based refs; 0 is missing/padding.

        def ref(index, count, description):
            if index is None:
                return 0, False
            if type(index) is not int or not 0 <= index < count:
                raise ValueError("invalid public action reference: " + description)
            return index + 1, True

        obs = public.observation
        run, player, combat, inventory, screen = (_mapping(obs.get(k)) for k in
                                                 ("run", "player", "combat", "inventory", "screen"))
        context = _mapping(screen.get("decision_context"))
        battle_player = _mapping(combat.get("player"))
        fields = {name: numeric(value, name) for name, value in {
            "act": run.get("act"), "floor": run.get("floor"), "ascension": run.get("ascension"),
            "hp": player.get("hp"), "max_hp": player.get("max_hp"), "gold": player.get("gold"),
            "turn": combat.get("turn"), "energy": battle_player.get("energy"), "block": battle_player.get("block"),
            "cards_played": combat.get("cards_played_this_turn"), "times_damaged": combat.get("times_damaged"),
            "powers_complete": combat.get("powers_complete"), "effects_resolved": combat.get("effects_resolved"),
            "intents_hidden": hidden_enemy_intents(obs), "map_present": "map" in obs,
            "event_fidelity_certified": False,
            **{key: combat.get(key) for key in ("top_order_uncertain", "bottom_order_uncertain", "bottom_knowledge_tracked")},
            **{key + "_key": _mapping(obs.get("keys")).get(key) for key in ("emerald", "ruby", "sapphire")},
        }.items()}
        fields.update(screen_id=category("screen", screen.get("type")), boss_id=category("encounter", run.get("boss")),
                      room_id=category("room", run.get("room_type")), input_id=category("input", combat.get("input_state")),
                      task_id=category("task", combat.get("card_select_task", context.get("selection_task"))),
                      event_id=category("event", context.get("event_name")))
        add("global", fields)

        card_refs = {}
        def card_row(card, pile, index, *, ordered=True):
            card = _mapping(card)
            fields = {name: numeric(card.get(name), name) for name in TABLE_COLUMNS["cards"][4:]}
            fields.update(card_id=category("card", card.get("id")), pile_id=category("pile", pile),
                          position=numeric(index if ordered else None, "position"), type_id=category("card_type", card.get("type")))
            return add("cards", fields)

        draw_uid_refs = {}
        for pile in STATIC_VOCABS["pile"][:-1]:
            source = inventory.get("deck", ()) if pile == "deck" else combat.get(pile, ())
            if pile == "draw_pile_known_top":
                source = (source,) if isinstance(source, Mapping) else ()
            card_refs[pile] = [card_row(card, pile, i, ordered=pile not in {"deck", "draw_pile_unordered"})
                               for i, card in enumerate(_items(source))]
            if pile.startswith("draw_pile"):
                for card, row_ref in zip(source, card_refs[pile]):
                    uid = _mapping(card).get("unique_id")
                    if type(uid) is int:
                        if uid in draw_uid_refs:
                            raise ValueError("duplicate public draw-pile instance")
                        draw_uid_refs[uid] = row_ref
        top_ids = _items(combat.get("public_known_top_ids", ()))
        if any(type(uid) is not int or uid not in draw_uid_refs for uid in top_ids) or len(set(top_ids)) != len(top_ids):
            raise ValueError("invalid public known-top identity relationship")
        for order, uid in enumerate(top_ids):
            add("known_draw_top", {"order": (order, True), "card_ref": (draw_uid_refs[uid], True)})
        if "draw_pile" in combat and not any(_key(_mapping(r).get("id")) == "frozeneye"
                                                for r in inventory.get("relics", ())):
            raise ValueError("encoder refuses unrevealed draw order in direct PolicyInput")

        def powers(owner, values):
            for order, power in enumerate(_items(values)):
                power = _mapping(power)
                add("powers", {"owner_ref": (owner, True), "order": (order, True),
                    "power_id": category("power", power.get("id")),
                    **{name: numeric(power.get(name), name) for name in ("amount", "damage", "hp_loss")}})
        powers(1, battle_player.get("powers", ()))
        monsters = _items(combat.get("monsters", ()))
        for i, monster in enumerate(monsters):
            monster = _mapping(monster)
            if hidden_enemy_intents(obs) and any(k.startswith(("move_", "intent")) for k in monster):
                raise ValueError("encoder refuses concealed intent in direct PolicyInput")
            add("monsters", {"monster_id": category("monster", monster.get("id")),
                "move_id": category("move", monster.get("move_id")),
                **{name: numeric(monster.get(source), name) for name, source in (
                    ("hp", "current_hp"), ("max_hp", "max_hp"), ("block", "block"),
                    ("move_damage", "move_base_damage"), ("move_hits", "move_hits"),
                    ("is_gone", "is_gone"), ("half_dead", "half_dead"))}})
            powers(i + 2, monster.get("powers", ()))
        for i, relic in enumerate(_items(inventory.get("relics", ()))):
            add("relics", {"relic_id": category("relic", _mapping(relic).get("id")),
                           "order": (i, True), "counter": numeric(_mapping(relic).get("counter"), "counter")})
        potions = _items(inventory.get("potions", ()))
        for i, potion in enumerate(potions):
            add("potions", {"potion_id": category("potion", _mapping(potion).get("id")), "slot": (i, True)})

        map_refs = {}
        if "map" in obs:
            graph = validate_public_map(obs)
            point = lambda p: (p["x"], p["y"])
            starts = {point(p) for p in graph["starts"]}
            nodes = sorted(graph["nodes"], key=lambda n: (n["y"], n["x"]))
            for node in nodes:
                map_refs[point(node)] = add("map_nodes", {"x": (node["x"], True), "y": (node["y"], True),
                    "room_id": category("room", node["room"]), "burning_elite": (int(node["burning_elite"]), True),
                    "current": (int(point(node) == point(graph["position"])), True),
                    "start": (int(point(node) in starts), True), "boss": (int(point(node) == point(graph["boss_node"])), True)})
            for node in nodes:
                for destination in node["next"]:
                    add("map_edges", {"source_ref": (map_refs[point(node)], True),
                                      "destination_ref": (map_refs[point(destination)], True)})
        else:
            gaps.add("missing_public_map")

        options = {}
        for option in tuple(screen.get("decision_options", ())) + tuple(screen.get("card_reward_options", ())):
            action = Action.from_dict(_mapping(option.get("action")))
            if action in options:
                raise ValueError("duplicate public candidate metadata")
            options[action] = option

        def effects(action_ref, operations, parent=0, depth=0):
            if depth > 4:
                raise ValueError("event effect nesting exceeds encoder bound")
            for operation in _items(operations):
                op = _mapping(operation)
                kind = op.get("kind", "unknown")
                card_kind = kind in {"obtain_card", "add_curse", "lose_card"}
                relic_kind = kind in {"obtain_relic", "lose_relic"}
                effect_ref = add("event_effects", {"action_ref": (action_ref, True), "parent_ref": (parent, True),
                    "operation_id": category("operation", kind),
                    "cost": (int(kind.startswith("lose_") or kind in {"damage", "damage_fraction", "add_curse", "random_curse", "disable_healing"}), True),
                    **{name: numeric(op.get(name), name) for name in ("amount", "count", "minimum", "maximum", "numerator", "denominator", "offer_size")},
                    "rarity_id": category("rarity", op.get("rarity")), "pool_id": category("pool", op.get("pool")),
                    "card_id": category("card", op.get("id") if card_kind else None),
                    "relic_id": category("relic", op.get("id") if relic_kind else None),
                    "potion_id": category("potion", op.get("id") if kind == "lose_potion" else None),
                    "selector_id": category("selector", op.get("selector")),
                    "unspecified_details": (int(any(isinstance(v, str) and ("unspecified" in v or "unrevealed" in v) for v in op.values())), True)})
                if "outcome" in op and isinstance(op["outcome"], Mapping):
                    effects(action_ref, (op["outcome"],), effect_ref, depth + 1)

        for i, action in enumerate(actions):
            required_index = {ActionKind.CHOOSE: action.choice_index, ActionKind.PLAY_CARD: action.card_index,
                              ActionKind.USE_POTION: action.potion_index, ActionKind.DISCARD_POTION: action.potion_index}
            if action.kind in required_index and (type(required_index[action.kind]) is not int
                                                  or required_index[action.kind] < 0):
                raise ValueError("game action omitted its non-negative integer index")
            option = _mapping(options.get(action))
            kind = option.get("kind", action.kind.value if action.kind is not ActionKind.CHOOSE else "unknown")
            card = option.get("card")
            if not option.get("kind") and action.kind in {ActionKind.PROCEED, ActionKind.CANCEL}:
                kind = "leave"
            elif not option.get("kind") and isinstance(card, Mapping) and screen.get("type") == "CARD_REWARD":
                kind = "card_reward"
            selected_refs = []
            selection_known = False
            card_ref = (0, False)
            if action.kind is ActionKind.PLAY_CARD:
                hand_ref, _ = ref(action.card_index, len(card_refs["hand"]), "hand")
                if not hand_ref:
                    raise ValueError("play action omitted hand reference")
                card_ref = (card_refs["hand"][hand_ref - 1], True)
            elif isinstance(card, Mapping):
                card_ref = (card_row(card, "candidate", i), True)
                selection_known = True
            elif action.kind is ActionKind.CHOOSE and combat.get("input_state") == "CARD_SELECT":
                kind = "combat_select"
                task = combat.get("card_select_task")
                index = action.choice_index
                if "selection_options" in combat:
                    groups = _items(combat["selection_options"])
                    group_ref, _ = ref(index, len(groups), "selection group")
                    selected_pile = "selection_candidates" if task == "LIQUID_MEMORIES_POTION" else "hand"
                    for selected in _items(groups[group_ref - 1]):
                        local, _ = ref(selected, len(card_refs[selected_pile]), "selected card")
                        selected_refs.append(card_refs[selected_pile][local - 1])
                    selection_known = True  # Including the empty subset.
                else:
                    if task == "ARMAMENTS" and not combat.get("selection_candidates"):
                        raise ValueError("upgrade selection has no structured public candidates")
                    pile = ("selection_candidates" if "selection_candidates" in combat else
                            "discard_pile" if task == "HEADBUTT" else
                            "hand" if task in {"EXHAUST_ONE", "ARMAMENTS", "WARCRY"} else None)
                    if task == "CODEX" and index == 3:
                        if len(card_refs["selection_candidates"]) != 3:
                            raise ValueError("Codex skip requires three structured public candidates")
                        selection_known = True  # An explicit skip has no card reference.
                    elif pile:
                        local, _ = ref(index, len(card_refs[pile]), "selected card")
                        if not local:
                            raise ValueError("selection omitted choice index")
                        selected_refs = [card_refs[pile][local - 1]]
                        card_ref = (selected_refs[0], True)
                        selection_known = True
                    else:
                        gaps.add("unencoded_combat_selection")
            for order, selected in enumerate(selected_refs):
                add("action_cards", {"action_ref": (i + 1, True), "order": (order, True), "card_ref": (selected, True)})
            potion_ref = ref(action.potion_index, len(potions), "potion")
            target_ref = ref(action.target_index, len(monsters), "monster")
            destination = _mapping(option.get("destination"))
            map_ref = map_refs.get((destination.get("x"), destination.get("y")), 0)
            if destination and not map_ref:
                raise ValueError("candidate destination absent from public graph")
            event = _mapping(option.get("event_choice"))
            if kind == "unknown":
                gaps.add("unknown_candidate_semantics")
            add("actions", {"action_id": category("action", action.kind.value), "option_id": category("option", kind),
                "card_ref": card_ref, "potion_ref": potion_ref, "monster_ref": target_ref, "map_ref": (map_ref, bool(map_ref)),
                "relic_id": category("relic", option.get("item_id") if kind in {"relic", "buy_relic", "boss_relic"} else None),
                "potion_id": category("potion", option.get("item_id") if kind in {"potion", "buy_potion"} else None),
                "price": numeric(option.get("price"), "price"), "amount": numeric(option.get("amount"), "amount"),
                "task_id": category("task", option.get("task", combat.get("card_select_task", context.get("selection_task")))),
                "room_id": category("room", option.get("room")), "key_id": category("key", option.get("key")),
                "event_parsed": (int(event.get("status") == "parsed"), bool(event)),
                "event_partial": (int(event.get("status") == "partial"), bool(event)),
                "unknown_clause_count": numeric(len(event.get("unknown_clauses", ())) if event else None, "unknown_clause_count"),
                "effect_fidelity_certified": (0, True), "selection_known": (int(selection_known), True)})
            effects(i + 1, event.get("display_order", ()))
        tables = MappingProxyType({name: FeatureTable(TABLE_COLUMNS[name],
            tuple(tuple(cell[0] for cell in row) for row in values),
            tuple(tuple(cell[1] for cell in row) for row in values)) for name, values in rows.items()})
        return EncodedDecision(self.schema_id, tables, tuple(actions), (True,) * len(actions),
            tuple(action in scope.eligible_actions for action in actions) if scope else (),
            MappingProxyType(dict(unknown)), tuple(sorted(gaps)))


def collate_decisions(decisions: Sequence[EncodedDecision], *, max_padded_values=2_000_000) -> dict:
    """Pad only to this batch's maxima. Framework-neutral tensor-ready numbers.

    Decoder Actions, raw text/instance IDs, gap strings and unknown identifiers
    are deliberately excluded from this model payload. No silent truncation.
    """
    if type(max_padded_values) is not int or max_padded_values < 1:
        raise ValueError("positive integer padded-value budget required")
    if not decisions or len({d.schema_id for d in decisions}) != 1:
        raise ValueError("nonempty, identical encoder schemas required")
    sizes = {name: max(len(d.tables[name].values) for d in decisions) for name in TABLE_COLUMNS}
    if sum(sizes[name] * len(TABLE_COLUMNS[name]) for name in sizes) * len(decisions) > max_padded_values:
        raise ValueError("padded batch exceeds memory budget; split batch, do not truncate")
    tables = {}
    for name, width in sizes.items():
        ncolumns = len(TABLE_COLUMNS[name])
        values, known, present = [], [], []
        for decision in decisions:
            table = decision.tables[name]
            padding = width - len(table.values)
            values.append(table.values + ((0.0,) * ncolumns,) * padding)
            known.append(table.known + ((False,) * ncolumns,) * padding)
            present.append((True,) * len(table.values) + (False,) * padding)
        tables[name] = {"values": tuple(values), "known": tuple(known), "present": tuple(present)}
    width = sizes["actions"]
    masks = {name: tuple(getattr(d, name) + (False,) * (width - len(d.legal_actions)) for d in decisions)
             for name in ("legal_mask", "scope_mask")}
    masks["action_mask"] = tuple(tuple(a and b for a, b in zip(legal, scope))
                                 for legal, scope in zip(masks["legal_mask"], masks["scope_mask"]))
    return {"schema_id": decisions[0].schema_id, "tables": tables, **masks}
