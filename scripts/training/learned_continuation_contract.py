"""Training support for the released combat model (learned_continuation_contract)."""

from learned_continuation_contract import canonical
from learned_continuation_contract import digest
import copy
import math


INPUT_VERSION = "public-native-leaf-input-v1"


FORBIDDEN = {
    "seed",
    "seed_numeric",
    "seed_text",
    "reset_seed",
    "audit_state",
    "backend_features",
    "rng",
    "rng_state",
    "rng_streams",
    "streams",
    "world_words",
    "source",
    "run_group",
    "root_evidence",
    "future",
    "label",
    "outcome",
}


VIEW_KEYS = set(
    "bomb_schema bottom_tracked character discard draw exhaust hand held input limbo_empty "
    "monsters opening_priority_ids opening_schema opening_uncertain player potions power_records "
    "powers_complete relics resolved selection selection_any selection_candidates selection_copies "
    "selection_max selection_options selection_pick selection_type times_damaged top_candidates uncertain_order".split()
)


CARD_KEYS = set("base_cost cost damage free id playable special type uid up".split())


PILES = (
    "hand",
    "draw",
    "discard",
    "exhaust",
    "held",
    "top_candidates",
    "selection_candidates",
)


MULTI = {"FORETHOUGHT", "EXHAUST_MANY", "GAMBLE", "LIQUID_MEMORIES_POTION"}


SELECTION_FIELDS = {
    "HEADBUTT": "discard",
    "WARCRY": "hand",
    "EXHAUST_ONE": "hand",
    **{
        k: "selection_candidates"
        for k in (
            "ARMAMENTS",
            "EXHUME",
            "SECRET_TECHNIQUE",
            "SECRET_WEAPON",
            "DUAL_WIELD",
            "DISCOVERY",
            "CODEX",
        )
    },
}


def public_only(value):
    if isinstance(value, dict):
        if set(value) & FORBIDDEN:
            raise ValueError("non-public/provenance channel in model input")
        for item in value.values():
            public_only(item)
    elif isinstance(value, list):
        for item in value:
            public_only(item)
    elif type(value) is float and not math.isfinite(value):
        raise ValueError("non-finite public input")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ValueError("input must be owned JSON data")


def require_keys(value, allowed, required=()):
    if (
        not isinstance(value, dict)
        or set(value) - set(allowed)
        or set(required) - set(value)
    ):
        raise ValueError("unexpected or missing public schema field")


def index(value, sequence):
    if type(value) is not int or not 0 <= value < len(sequence):
        raise ValueError("invalid public action reference")
    return value


def policy_input(view, legal):
    """Owned JSON envelope for a future shared Python/C++ tensor encoder.

    No EnvironmentState, recipe, teacher scores, prior action or sampled world is
    accepted. Candidate references are resolved, not used as continuous features.
    This is not yet the numeric encoder, native inference or a sufficient belief state.
    """
    public_only(view)
    public_only(legal)
    require_keys(view, VIEW_KEYS, VIEW_KEYS)
    if (
        view["input"] not in ("PLAYER_NORMAL", "CARD_SELECT")
        or view["character"] != "IRONCLAD"
    ):
        raise ValueError("leaf input scope")
    if not view["resolved"] or not view["powers_complete"] or not view["limbo_empty"]:
        raise ValueError("unresolved native public view")
    state = copy.deepcopy(view)
    # Actual copy IDs are unnecessary to the actor and cannot be an identity leak.
    state.pop("opening_priority_ids")
    for name in PILES:
        if not isinstance(state[name], list):
            raise ValueError("public pile must be a list")
        for card in state[name]:
            require_keys(card, CARD_KEYS, CARD_KEYS - {"playable"})
            if type(card["uid"]) is not int or card["uid"] != -1:
                raise ValueError("non-anonymous card copy ID")
            del card["uid"]
    require_keys(
        state["player"],
        "block current_hp energy max_hp panache_damage powers".split(),
        "block current_hp energy max_hp panache_damage powers".split(),
    )
    for monster in state["monsters"]:
        require_keys(
            monster,
            "block current_hp half_dead is_gone max_hp move_base_damage move_hits powers".split(),
            "block current_hp half_dead is_gone max_hp move_base_damage move_hits powers".split(),
        )
    for power in state["power_records"]:
        require_keys(power, ("id", "amount", "damage", "hp_loss"), ("id", "amount"))
    # Unknown draw order is a multiset; never learn an incidental sort/order.
    state["draw"].sort(key=canonical)
    state["top_candidates"].sort(key=canonical)
    if not isinstance(legal, list) or not legal or len(legal) > 4096:
        raise ValueError("legal-action capacity")
    candidates = []
    for action in legal:
        if not isinstance(action, dict):
            raise ValueError("action mapping required")
        kind = action.get("kind")
        links = dict(kind=kind, cards=[], potion=None, target=None)
        if view["input"] == "CARD_SELECT":
            require_keys(action, ("kind", "choice_index"), ("kind", "choice_index"))
            if kind != "choose":
                raise ValueError("selection action required")
            task = view["selection"]
            choice = action["choice_index"]
            if task in MULTI:
                selected = view["selection_options"][
                    index(choice, view["selection_options"])
                ]
                if not isinstance(selected, list) or len(set(selected)) != len(
                    selected
                ):
                    raise ValueError("invalid selection group")
                field = (
                    "selection_candidates"
                    if task == "LIQUID_MEMORIES_POTION"
                    else "hand"
                )
                links["cards"] = [
                    copy.deepcopy(state[field][index(i, state[field])])
                    for i in selected
                ]
            elif task in SELECTION_FIELDS:
                if task == "CODEX" and type(choice) is int and choice == 3:
                    links["skip"] = True
                else:
                    field = SELECTION_FIELDS[task]
                    links["cards"] = [
                        copy.deepcopy(state[field][index(choice, state[field])])
                    ]
            else:
                raise ValueError("unknown public selection task")
        elif kind == "play_card":
            require_keys(
                action, ("kind", "card_index", "target_index"), ("kind", "card_index")
            )
            links["cards"] = [
                copy.deepcopy(state["hand"][index(action["card_index"], state["hand"])])
            ]
        elif kind in ("use_potion", "discard_potion"):
            require_keys(
                action,
                ("kind", "potion_index", "target_index")
                if kind == "use_potion"
                else ("kind", "potion_index"),
                ("kind", "potion_index"),
            )
            links["potion"] = state["potions"][
                index(action["potion_index"], state["potions"])
            ]
        elif kind == "end_turn":
            require_keys(action, ("kind",), ("kind",))
        else:
            raise ValueError("unknown combat action")
        if "target_index" in action:
            links["target"] = copy.deepcopy(
                state["monsters"][index(action["target_index"], state["monsters"])]
            )
        candidates.append(links)
    if len({canonical(a) for a in legal}) != len(legal):
        raise ValueError("duplicate legal action")
    # Exclude backend indices from the features; they remain separate decoder data.
    state.pop("selection_options")
    result = dict(
        version=INPUT_VERSION,
        state=state,
        candidates=candidates,
        mask=[True] * len(candidates),
    )
    public_only(result)
    return result
