"""Training support for the released combat model (rich_combat)."""

from .combat_value import LEVELS
from .combat_value import QUANTILES
from .combat_value import SCALE
from .combat_value import features
from .combat_value import load_model as load_base
from .leaf_batch import collate as old_collate
from .leaf_batch import prepare_public
from .leaf_numeric import encode as old_encode
from .leaf_numeric import key
from .leaf_numeric import schema as old_schema
from .learned_continuation_contract import digest
from .learned_continuation_contract import public_only
from .learned_continuation_contract import require_keys
from .rich_intents import GROUPS
from spire_ai.public_encoding import POWER_NAMES
from spire_ai.rule_catalog import RuleCatalog
from torch import nn
from torch.nn import functional as F
import copy
import math
import torch


VERSION = "public-combat-policy-value-v2-rich"


PUBLIC = "public-combat-entities-v2"


MFIELDS = [
    ("current_hp", 100),
    ("max_hp", 100),
    ("block", 100),
    ("move_base_damage", 100),
    ("move_hits", 10),
    ("is_gone", 1),
    ("half_dead", 1),
    ("slot", 5),
]


RFIELDS = [
    ("order", 32),
    ("counter", 10),
    ("available", 1),
    ("attack_played", 1),
    ("skill_played", 1),
    ("power_played", 1),
]


GFIELDS = [
    ("turn", 20),
    ("ascension", 20),
    ("cards_played", 10),
    ("attacks_played", 10),
    ("skills_played", 10),
    ("discarded", 10),
]


def schema():
    c = RuleCatalog.bundled()
    value = dict(
        version=PUBLIC,
        monsters=["<unknown>"]
        + sorted({"invalid0", *(key(r["id"]) for r in c.entries("monster"))}),
        relics=["<unknown>"] + sorted({key(r["id"]) for r in c.entries("relic")}),
        intents=["<missing>"] + sorted(GROUPS),
        powers=sorted({key(p) for p in POWER_NAMES}),
        monster_fields=MFIELDS,
        relic_fields=RFIELDS,
        global_fields=GFIELDS,
        maximum_monsters=5,
        maximum_relics=256,
        dtype="float32",
        history="previous_observed_icon_only; missing on new public generation",
        missing="value_zero_with_known_mask",
        aggregation="per_enemy_nonlinear_then_mean_and_count",
        empty_monster="invalid0: gone, zero hp/max_hp/block/damage/hits, no powers; retains target slot",
    )
    # Artifact equality uses JSON's arrays, not Python tuple types.
    import json

    value = json.loads(json.dumps(value))
    value["id"] = digest(value)
    return value


class Encoder:
    """Compile immutable lookup tables once, not once per state/epoch."""

    def __init__(self, spec=None):
        self.spec = spec or schema()
        if self.spec != schema():
            raise ValueError("rich schema mismatch")
        self.tokens = {
            k: {v: i for i, v in enumerate(self.spec[k])}
            for k in ("monsters", "relics", "intents", "powers")
        }
        self.old_spec = old_schema()

    @staticmethod
    def numbers(row, fields):
        vals, masks = [], []
        for name, scale in fields:
            known = name in row and row[name] is not None
            x = row[name] if known else 0
            if (
                type(x) not in (bool, int, float)
                or not math.isfinite(x)
                or abs(x) > 1e9
            ):
                raise ValueError("rich numeric field")
            vals.append(float(x) / scale)
            masks.append(float(known))
        return vals + masks

    def encode(self, view, legal, rich, *, include_old=True):
        public_only(rich)
        required = {"version", "monsters", "relics", *(n for n, _ in GFIELDS)}
        require_keys(rich, required, required)
        if (
            rich["version"] != PUBLIC
            or not isinstance(rich["monsters"], list)
            or not isinstance(rich["relics"], list)
        ):
            raise ValueError("rich public envelope")
        if (
            not 0 < len(rich["monsters"]) <= 5
            or len(rich["monsters"]) != len(view["monsters"])
            or len(rich["relics"]) > 256
        ):
            raise ValueError("rich entity capacity/alignment")
        if sorted(r["id"] for r in rich["relics"]) != sorted(view["relics"]):
            raise ValueError("rich relic identity alignment")
        if {"runicdome", "prismaticshard"} & set(view["relics"]):
            raise ValueError("rich relic scope")
        mc, mn, rc, rn = [], [], [], []
        for i, (m, base) in enumerate(zip(rich["monsters"], view["monsters"])):
            require_keys(
                m,
                {"id", "slot", "intent", "previous_intent"},
                {"id", "slot", "intent", "previous_intent"},
            )
            if type(m["slot"]) is not int or m["slot"] != i:
                raise ValueError("rich monster slot")
            if (
                m["id"] not in self.tokens["monsters"]
                or m["intent"] not in self.tokens["intents"]
            ):
                raise ValueError(
                    f"unmapped rich monster/intent: slot={i}, id={m['id']}, intent={m['intent']}"
                )
            if m["id"] == "invalid0" and (
                not base["is_gone"]
                or base["half_dead"]
                or base["powers"]
                or any(
                    base[k] != 0
                    for k in (
                        "current_hp",
                        "max_hp",
                        "block",
                        "move_base_damage",
                        "move_hits",
                    )
                )
                or m["intent"] != "UNKNOWN"
                or m["previous_intent"] not in (None, "UNKNOWN")
            ):
                raise ValueError("invalid empty monster slot")
            previous = m["previous_intent"]
            if previous is not None and previous not in self.tokens["intents"]:
                raise ValueError("rich history icon")
            mc.append(
                [
                    self.tokens["monsters"][m["id"]],
                    self.tokens["intents"][m["intent"]],
                    self.tokens["intents"][previous] if previous is not None else 0,
                ]
            )
            values = self.numbers(dict(base, slot=i), MFIELDS)
            powers = base["powers"]
            if set(powers) - set(self.tokens["powers"]):
                raise ValueError("unknown rich enemy power")
            # Complete public enemy-power map: absence really means zero.
            values += self.numbers(
                {p: powers.get(p, 0) for p in self.spec["powers"]},
                [(p, 100) for p in self.spec["powers"]],
            )
            mn.append(values)
        for i, r in enumerate(rich["relics"]):
            require_keys(r, {"id", *(n for n, _ in RFIELDS)}, {"id", "order"})
            if type(r["order"]) is not int or r["order"] != i:
                raise ValueError("rich relic order")
            if r["id"] not in self.tokens["relics"]:
                raise ValueError("unknown rich relic")
            rc.append([self.tokens["relics"][r["id"]]])
            rn.append(self.numbers(r, RFIELDS))
        target = []
        for a in legal:
            t = a.get("target_index", -1)
            if type(t) is not int or t < -1 or t >= len(mc):
                raise ValueError("rich target alignment")
            if t >= 0 and rich["monsters"][t]["id"] == "invalid0":
                raise ValueError("empty monster target")
            target.append(t)
        result = dict(
            monster_categories=mc,
            monster_numbers=mn,
            relic_categories=rc,
            relic_numbers=rn,
            global_numbers=self.numbers(rich, GFIELDS),
            targets=target,
        )
        if include_old:
            result["old"] = old_encode(view, legal, self.old_spec)
        return result


def pack(encoded):
    return dict(
        old=prepare_public(encoded["old"]),
        mc=torch.tensor(encoded["monster_categories"], dtype=torch.long),
        mn=torch.tensor(encoded["monster_numbers"], dtype=torch.float32),
        rc=torch.tensor(encoded["relic_categories"], dtype=torch.long).reshape(-1, 1),
        rn=torch.tensor(encoded["relic_numbers"], dtype=torch.float32).reshape(
            -1, 2 * len(RFIELDS)
        ),
        glob=torch.tensor(encoded["global_numbers"], dtype=torch.float32),
        target=torch.tensor(encoded["targets"], dtype=torch.long),
    )


def collate(items):
    out = {}
    mc = []
    mn = []
    rc = []
    rn = []
    mi = []
    ri = []
    target = []
    states = []
    offset = 0
    for i, p in enumerate(items):
        mc.append(p["mc"])
        mn.append(p["mn"])
        rc.append(p["rc"])
        rn.append(p["rn"])
        mi.extend([i] * len(p["mc"]))
        ri.extend([i] * len(p["rc"]))
        target.extend((p["target"] + offset).where(p["target"] >= 0, -1).tolist())
        states.extend([i] * len(p["target"]))
        offset += len(p["mc"])
    out.update(
        mc=torch.cat(mc),
        mn=torch.cat(mn),
        rc=torch.cat(rc),
        rn=torch.cat(rn),
        mi=torch.tensor(mi, dtype=torch.long),
        ri=torch.tensor(ri, dtype=torch.long),
        mcounts=torch.tensor([len(p["mc"]) for p in items], dtype=torch.float32),
        rcounts=torch.tensor([len(p["rc"]) for p in items], dtype=torch.float32),
        glob=torch.stack([p["glob"] for p in items]),
        target=torch.tensor(target),
        states=torch.tensor(states),
        lengths=[len(p["target"]) for p in items],
    )
    out["old"] = old_collate([p["old"] for p in items])
    return out


class RichEncoder(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.monster_embedding = nn.Embedding(len(spec["monsters"]), 8)
        self.intent_embedding = nn.Embedding(len(spec["intents"]), 8, padding_idx=0)
        self.relic_embedding = nn.Embedding(len(spec["relics"]), 8)
        self.monster = nn.Linear(24 + 2 * len(MFIELDS) + 2 * len(spec["powers"]), 32)
        self.relic = nn.Linear(8 + 2 * len(RFIELDS), 32)
        self.context = nn.Linear(66 + 2 * len(GFIELDS), 32)

    def forward(self, b):
        mx = torch.cat(
            (
                self.monster_embedding(b["mc"][:, 0]),
                self.intent_embedding(b["mc"][:, 1:]).flatten(1),
                b["mn"],
            ),
            1,
        )
        enemies = self.monster(mx).tanh()
        relics = self.relic(
            torch.cat((self.relic_embedding(b["rc"][:, 0]), b["rn"]), 1)
        ).tanh()

        def pool(rows, assignment, counts):
            sums = rows.new_zeros((len(counts), 32)).index_add(0, assignment, rows)
            return torch.cat(
                (sums / counts.clamp_min(1)[:, None], counts.log1p()[:, None]), 1
            )

        context = self.context(
            torch.cat(
                (
                    pool(enemies, b["mi"], b["mcounts"]),
                    pool(relics, b["ri"], b["rcounts"]),
                    b["glob"],
                ),
                1,
            )
        ).tanh()
        targets = enemies[b["target"].clamp_min(0)] * (b["target"] >= 0)[:, None]
        return context, torch.cat((targets, (b["target"] >= 0).float()[:, None]), 1)


class Model(nn.Module):
    def __init__(self, parent, seed=94103, spec=None):
        super().__init__()
        self.spec = spec or schema()
        self.parent = copy.deepcopy(parent)
        self.parent_digest = digest(parent)
        self.base = load_base(parent)
        self.base.requires_grad_(False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            # Separate encoders intentionally prevent value fitting from changing
            # action preferences, including through newly introduced features.
            self.actor_extra = RichEncoder(self.spec)
            self.critic_extra = RichEncoder(self.spec)
            self.actor_hidden = nn.Linear(229, 32)
            self.actor_output = nn.Linear(32, 1)
            self.value_extra = nn.Linear(32, 64, bias=False)
            self.value_hidden = copy.deepcopy(self.base.value_hidden)
            self.outcome = copy.deepcopy(self.base.outcome)
            self.resource = copy.deepcopy(self.base.resource)
            for m in (self.value_hidden, self.outcome, self.resource):
                m.requires_grad_(True)
            nn.init.zeros_(self.actor_output.weight)
            nn.init.zeros_(self.actor_output.bias)
            nn.init.zeros_(self.value_extra.weight)

    def base_features(self, b):
        with torch.no_grad():
            contexts, candidates = features(self.base.policy, b["old"])
            candidates = candidates[b["old"]["inverse"]]
            x = torch.cat((contexts[b["states"]], candidates), 1)
            logits = self.base.policy.output(self.base.policy.head(x).tanh()).squeeze(
                -1
            )
        return contexts, candidates, logits

    def forward(self, b):
        context, candidates, base_logits = self.base_features(b)
        ac, target = self.actor_extra(b)
        vc, _ = self.critic_extra(b)
        x = torch.cat((context[b["states"]], candidates, ac[b["states"]], target), 1)
        logits = base_logits + self.actor_output(self.actor_hidden(x).tanh()).squeeze(
            -1
        )
        hidden = (self.value_hidden(context) + self.value_extra(vc)).tanh()
        outcomes = self.outcome(hidden)
        quantiles = F.softplus(self.resource(hidden).reshape(-1, 2, QUANTILES)).cumsum(
            -1
        )
        if not all(torch.isfinite(v).all() for v in (logits, outcomes, quantiles)):
            raise ValueError("nonfinite rich output")
        return logits.split(b["lengths"]), outcomes, quantiles

    def value_parameters(self):
        return [
            p
            for n, p in self.named_parameters()
            if n.startswith(
                (
                    "critic_extra.",
                    "value_extra.",
                    "value_hidden.",
                    "outcome.",
                    "resource.",
                )
            )
        ]

    def actor_parameters(self):
        return [
            p
            for n, p in self.named_parameters()
            if n.startswith(("actor_extra.", "actor_hidden.", "actor_output."))
        ]


def export(model):
    return dict(
        version=VERSION,
        schema=model.spec,
        parent=model.parent,
        tensors={
            n: dict(shape=list(p.shape), values=p.detach().flatten().tolist())
            for n, p in model.state_dict().items()
            if not n.startswith("base.")
        },
        semantics=dict(
            target="terminal-outcome-resource-quantiles-v1",
            value_to_actor_gradient=False,
            parent_frozen=True,
            full_belief=False,
            resource_scale=SCALE,
            quantile_levels=LEVELS,
        ),
    )
