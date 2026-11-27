"""Training support for the released combat model (leaf_numeric)."""

from .learned_continuation_contract import digest
from .learned_continuation_contract import policy_input
from spire_ai.public_encoding import POWER_NAMES
from spire_ai.public_encoding import STATIC_VOCABS
from spire_ai.rule_catalog import RuleCatalog
import copy
import math
import struct


VERSION = "public-leaf-entities-v1"


STATE_GROUPS = [
    "global",
    "hand",
    "draw",
    "discard",
    "exhaust",
    "held",
    "top_candidates",
    "selection_candidates",
    "monsters",
    "player_powers",
    "relics",
    "potions",
]


ACTION_GROUPS = ["action", "cards", "potion", "target"]


NUMBERS = 32


ROLES = [
    "global",
    "card",
    "monster",
    "player_power",
    "monster_power",
    "relic",
    "potion",
    "action",
]


FIELDS = {
    "global": [
        ("current_hp", 100),
        ("max_hp", 100),
        ("block", 100),
        ("energy", 10),
        ("panache_damage", 100),
        ("times_damaged", 20),
        ("selection_any", 1),
        ("selection_copies", 10),
        ("selection_max", 10),
        ("selection_pick", 10),
        ("uncertain_order", 1),
        ("opening_uncertain", 1),
        ("bottom_tracked", 1),
        ("powers_complete", 1),
        ("resolved", 1),
        ("limbo_empty", 1),
        ("bomb_schema", 1),
        ("opening_schema", 1),
    ],
    "card": [
        ("up", 10),
        ("base_cost", 5),
        ("cost", 5),
        ("free", 1),
        ("damage", 100),
        ("special", 100),
        ("playable", 1),
        ("order", 10),
    ],
    "monster": [
        ("current_hp", 100),
        ("max_hp", 100),
        ("block", 100),
        ("move_base_damage", 100),
        ("move_hits", 10),
        ("is_gone", 1),
        ("half_dead", 1),
    ],
    "player_power": [("amount", 100), ("damage", 100), ("hp_loss", 10), ("order", 10)],
    "monster_power": [("amount", 100)],
    "relic": [("order", 10)],
    "potion": [],
    "action": [("skip", 1), ("selected_count", 10), ("has_target", 1)],
}


def key(value):
    if not isinstance(value, str) or not value.isascii():
        raise ValueError("actor identifiers must be ASCII strings")
    return "".join(c for c in value.lower() if c.isalnum())


def schema():
    vocab = {f"role:{r}" for r in ROLES}
    catalog = RuleCatalog.bundled()
    for kind in ("card", "relic", "potion"):
        vocab.update(f'{kind}:{key(r["id"])}' for r in catalog.entries(kind))
    vocab.update("power:" + key(p) for p in POWER_NAMES)
    for kind, values in [
        ("input", ["PLAYER_NORMAL", "CARD_SELECT"]),
        ("task", STATIC_VOCABS["task"]),
        ("type", ["ATTACK", "SKILL", "POWER", "STATUS", "CURSE", "INVALID"]),
        ("action", ["play_card", "end_turn", "use_potion", "discard_potion", "choose"]),
    ]:
        vocab.update(kind + ":" + key(v) for v in values)
    vocab.add("potion:emptypotionslot")
    value = dict(
        version=VERSION,
        vocab=["<pad>", "<unknown>", *sorted(vocab)],
        fields=FIELDS,
        state_groups=STATE_GROUPS,
        action_groups=ACTION_GROUPS,
        numbers=NUMBERS,
        categories=4,
        max_actions=4096,
        max_rows=32768,
        width=32,
        embedding_width=8,
        numeric_transform="float32(value/fixed_scale),known-mask",
        pool="mean-plus-log1p-count",
        card_order="selected-card-order-only",
        power_order="public-player-acquisition-order",
    )
    value[
        "selection_readiness"
    ] = "CARD_SELECT may pause unresolved effects; preserve resolved flag"
    # JSON representation, not Python tuple/list identity, defines this schema.
    import json

    value = json.loads(json.dumps(value))
    value["id"] = digest(value)
    return value


def f32(value):
    if not math.isfinite(value) or abs(value) > 3.4028234e38:
        raise ValueError("numeric feature outside finite float32")
    return struct.unpack("<f", struct.pack("<f", value))[0]


def encode(view, legal, spec=None):
    spec = spec or schema()
    expected = {k: v for k, v in spec.items() if k != "id"}
    if spec["id"] != digest(expected) or spec["version"] != VERSION:
        raise ValueError("numeric schema changed")
    # The older sealed envelope only admitted resolved decision points. A native
    # CARD_SELECT legitimately pauses a queued action. Reuse its structural and
    # privacy checks on an owned validation copy, then retain the actual flag.
    if view.get("input") == "CARD_SELECT" and view.get("resolved") is False:
        validation = copy.deepcopy(view)
        validation["resolved"] = True
        envelope = policy_input(validation, legal)
        envelope["state"]["resolved"] = False
    else:
        envelope = policy_input(view, legal)
    state = envelope["state"]
    tokens = {s: i for i, s in enumerate(spec["vocab"])}
    unknown = {}
    row_count = 0

    def token(category, value):
        if value in (None, ""):
            return 0
        name = category + ":" + (value if category == "role" else key(value))
        if name not in tokens:
            unknown[name] = unknown.get(name, 0) + 1
        return tokens.get(name, 1)

    def row(role, cats, values):
        nonlocal row_count
        row_count += 1
        if row_count > spec["max_rows"]:
            raise ValueError("numeric row capacity")
        cat = [token("role", role)] + [token(k, v) for k, v in cats]
        cat += [0] * (4 - len(cat))
        numbers = [0.0] * NUMBERS
        known = [0.0] * NUMBERS
        for i, (name, scale) in enumerate(spec["fields"][role]):
            if name in values:
                x = values[name]
                if type(x) not in (int, float, bool):
                    raise ValueError("numeric public field must be scalar")
                numbers[i] = f32(x / scale)
                known[i] = 1.0
        return dict(categories=cat, values=numbers, known=known)

    def card(c, order=None):
        values = dict(c)
        if order is not None:
            values["order"] = order
        return row("card", [("card", c["id"]), ("type", c["type"])], values)

    def monster(m):
        rows = [row("monster", [], m)]
        rows += [
            row("monster_power", [("power", name)], dict(amount=amount))
            for name, amount in sorted(m["powers"].items())
        ]
        return rows

    groups = {name: [] for name in STATE_GROUPS}
    groups["global"] = [
        row(
            "global",
            [
                ("input", state["input"]),
                ("task", state["selection"]),
                ("type", state["selection_type"]),
            ],
            dict(state, **state["player"]),
        )
    ]
    for name in STATE_GROUPS[1:8]:
        groups[name] = [card(c) for c in state[name]]
    groups["monsters"] = [r for m in state["monsters"] for r in monster(m)]
    groups["player_powers"] = [
        row("player_power", [("power", p["id"])], dict(p, order=i))
        for i, p in enumerate(state["power_records"])
    ]
    groups["relics"] = [
        row("relic", [("relic", r)], dict(order=i))
        for i, r in enumerate(state["relics"])
    ]
    groups["potions"] = [row("potion", [("potion", p)], {}) for p in state["potions"]]
    candidates = []
    for c in envelope["candidates"]:
        candidates.append(
            {
                "action": [
                    row(
                        "action",
                        [("action", c["kind"])],
                        dict(
                            skip=c.get("skip", False),
                            selected_count=len(c["cards"]),
                            has_target=c["target"] is not None,
                        ),
                    )
                ],
                "cards": [card(card_row, i) for i, card_row in enumerate(c["cards"])],
                "potion": []
                if c["potion"] is None
                else [row("potion", [("potion", c["potion"])], {})],
                "target": [] if c["target"] is None else monster(c["target"]),
            }
        )
    return dict(
        schema_id=spec["id"],
        state=groups,
        candidates=candidates,
        mask=envelope["mask"],
        unknown=unknown,
        rows=row_count,
    )
