"""Public deck startup descriptors, not a predicted opening hand or rule score.

Totals and densities are both supplied. Draw is potential per printed card,
separated into immediate, conditional, delayed and exhaust-triggered channels.
No sum assumes that the cards can be drawn/paid/played together. In particular,
draw_surplus subtracts the draw card itself, NOT other hand/HP/energy costs.
Relic effects and card identities remain in the shared entity representation.
"""
from dataclasses import dataclass
import math

from .public_encoding import SCALES, _key
from .run_afterstates import EncodedRunState, _digest
from .run_learning_encoding import BOTTLES


VERSION = 'public-deck-startup-descriptors-v1'
CAPS = dict(immediate_draw='draw', conditional_draw='conditional_draw',
    delayed_draw='delayed_draw', exhaust_draw='exhaust_draw_payoff',
    immediate_energy='energy', conditional_energy='conditional_energy',
    delayed_energy='delayed_energy', no_draw_cards='no_draw',
    power_cards='setup_cards', dead_cards='dead_draws',
    self_exhaust_cards='self_exhaust', exhaust_sources='exhaust_sources',
    skill_exhaust_engines='skill_exhaust_engine', innate_cards='innate')
COUNTS = ('deck_size', 'unknown_recipe_cards', 'cost_zero_cards', 'cost_one_cards',
    'cost_two_cards', 'cost_high_cards', 'x_cost_cards', 'unplayable_cards',
    'printed_energy', 'power_energy', 'immediate_draw_cards', 'draw_surplus',
    'draw_energy', 'free_draw_cards', 'conditional_draw_cards',
    'delayed_draw_cards', 'self_retain_cards', 'bottled_cards', *CAPS)
DENSITIES = tuple(k for k in COUNTS if k != 'deck_size')
COLUMNS = (*COUNTS, *(k+'_per_card' for k in DENSITIES))
# Scaling is an invertible numeric normalization, not a penalty or clipping.
SCALES_STARTUP = (20.,)*len(COUNTS) + (1.,)*len(DENSITIES)
SCHEMA = _digest(dict(version=VERSION, columns=COLUMNS, scales=SCALES_STARTUP,
    meaning='printed potential, not joint play or exact startup turns',
    draw_surplus='sum max(printed immediate draws minus one, zero) per draw card',
    unknown='any missing contributing descriptor masks that aggregate'))


@dataclass(frozen=True, slots=True)
class StartupProfile:
    values: tuple
    known: tuple

    def raw(self, name):
        i = COLUMNS.index(name)
        return self.values[i]*SCALES_STARTUP[i] if self.known[i] else None


def describe_startup(state, encoder):
    """Read ONLY encoded permanent deck rows, excluding offered cards/menus.

Unknown descriptors are masked rather than silently treated as an empty draw
engine. Actual opening order/turns would require a separate combat model; this
function neither samples an order nor consumes a simulator/audit state.
"""
    if not isinstance(state, EncodedRunState) or state.schema_id != encoder.schema_id:
        raise ValueError('versioned public run state required for startup profile')
    cards = state.entities.tables['cards']
    extra = state.entities.extra_tables['cards']
    token = encoder.base.vocabularies['pile'].index('deck')+2
    pile = cards.columns.index('pile_id')
    indices = [i for i,(v,k) in enumerate(zip(cards.values, cards.known, strict=True))
               if k[pile] and v[pile] == token]

    def read(i, field, supplemental=False):
        table = extra if supplemental else cards
        j = table.columns.index(field)
        return (table.values[i][j]*(1 if supplemental else SCALES.get(field, 1))
                if table.known[i][j] else None)

    def summed(values):
        return sum(values) if all(v is not None for v in values) else None

    costs = [read(i, 'base_cost') for i in indices]
    result = dict(deck_size=len(indices),
        unknown_recipe_cards=sum(read(i, 'recipe_reviewed', True) != 1 for i in indices))
    tests = dict(cost_zero_cards=lambda c:c == 0, cost_one_cards=lambda c:c == 1,
        cost_two_cards=lambda c:c == 2, cost_high_cards=lambda c:c >= 3,
        x_cost_cards=lambda c:c == -1, unplayable_cards=lambda c:c < -1,
        printed_energy=lambda c:max(0, c))
    for name, fn in tests.items():
        result[name] = summed([None if c is None else fn(c) for c in costs])
    for name, field in CAPS.items():
        result[name] = summed([read(i, field, True) for i in indices])
    draws = [read(i, 'draw', True) for i in indices]
    powers = [read(i, 'setup_cards', True) for i in indices]
    result['immediate_draw_cards'] = summed([None if d is None else int(d > 0) for d in draws])
    result['draw_surplus'] = summed([None if d is None else max(0, d-1) for d in draws])
    for field, target in (('conditional_draw', 'conditional_draw_cards'),
                          ('delayed_draw', 'delayed_draw_cards')):
        values = [read(i, field, True) for i in indices]
        result[target] = summed([None if v is None else int(v > 0) for v in values])
    for name, amounts, fn in (
        ('power_energy', powers, lambda c,a:max(0,c) if a > 0 else 0),
        ('draw_energy', draws, lambda c,a:max(0,c) if a > 0 else 0),
        ('free_draw_cards', draws, lambda c,a:int(c == 0 and a > 0))):
        result[name] = summed([None if c is None or a is None else fn(c,a)
                               for c,a in zip(costs, amounts, strict=True)])
    result['self_retain_cards'] = summed([read(i, 'self_retain', True) for i in indices])
    flags = [[read(i, field, True) for field in BOTTLES] for i in indices]
    result['bottled_cards'] = summed([int(any(row)) if all(v is not None for v in row) else None
                                     for row in flags])
    # No bottle relic means no bottle assignment, even in partial metrics data.
    relics = state.entities.tables['relics']
    ri = relics.columns.index('relic_id')
    names = encoder.base.vocabularies['relic']
    bottle_tokens = {names.index(_key(r))+2 for r in ('BOTTLED_FLAME','BOTTLED_LIGHTNING','BOTTLED_TORNADO')}
    if (all(k[ri] and v[ri] >= 2 for v,k in zip(relics.values,relics.known,strict=True))
            and not any(v[ri] in bottle_tokens for v in relics.values)):
        result['bottled_cards'] = 0
    n = max(1, len(indices))
    result.update({k+'_per_card':None if result[k] is None else result[k]/n for k in DENSITIES})
    values = tuple(0. if result[k] is None else float(result[k])/scale
                   for k,scale in zip(COLUMNS, SCALES_STARTUP, strict=True))
    if not all(math.isfinite(v) for v in values):
        raise ValueError('non-finite startup descriptor')
    return StartupProfile(values, tuple(result[k] is not None for k in COLUMNS))


def transition_features(before, after):
    """Before, after and explicit changes; every block retains missingness."""
    width = len(COLUMNS)
    for p in (before, after):
        if (not isinstance(p, StartupProfile) or len(p.values) != width or len(p.known) != width
                or any(type(k) is not bool for k in p.known)
                or any(not math.isfinite(v) or (not k and v != 0)
                       for v,k in zip(p.values,p.known,strict=True))):
            raise ValueError('invalid startup profile')
    known = tuple(a and b for a,b in zip(before.known, after.known, strict=True))
    delta = tuple(b-a if k else 0. for a,b,k in zip(before.values,after.values,known,strict=True))
    return (*before.values, *map(float,before.known), *after.values, *map(float,after.known),
            *delta, *map(float,known))
