"""A10--A20 card imitation with the publicly revealed, current-act boss.

Historical encounter summaries may recover a boss ID that was already on the
map, never its damage/turn count or the unrevealed second A20 boss. All other
missing partial fields retain the old masks. No change to sealed v1 policies.
"""
import copy
from dataclasses import replace

from . import expert_cards as legacy
from .expert_card_policy import project_reward as legacy_project, ExpertCardPolicy
from .metrics_cards import adapt_neow, LEGACY_BUILDS
from .metrics_a10 import cohort
from .policy_api import PolicyInput
from .actions import Action
from .rule_catalog import RuleCatalog

VERSION = 'expert-card-boss-partial-v2'
TRAIN_VERSION = 'metrics-a10-boss-card-bc-v1'
FLOORS = {1:16, 2:33, 3:50}
BOSSES = {1:frozenset(('The Guardian','Hexaghost','Slime Boss')),
          2:frozenset(('Champ','Automaton','Collector')),
          3:frozenset(('Awakened One','Time Eater','Donu And Deca'))}

# Reviewed vanilla AbstractEvent removal helpers serialize AbstractCard.cardID,
# NOT getMetricID(). A bare ID therefore does not select the unupgraded copy.
# Keep the old A20 importer/artifacts separate; this is the versioned A10+ path.
# Transform/upgrade logging has its own semantics and is not changed here.
LEGACY_BASE_ID_REMOVALS = frozenset((
    'Falling', 'Beggar', 'WeMeetAgain', 'Bonfire Elementals', 'The Cleric',
    'Golden Wing', 'Living Wall', 'Purifier', 'Designer', 'Back to Basics',
))


def canonical_boss(value, act, catalog):
    if value is None:
        return None
    entry = catalog.lookup('encounter', value) if isinstance(value, str) else None
    if not entry or entry['id'] not in BOSSES.get(act, ()):
        raise ValueError('boss does not belong to public act')
    return entry['id']


def historical_bosses(run, catalog=None):
    """Conservative identity-only reconstruction for the reviewed legacy build.

Require one matching B-room combat at the first boss floor of each act. Missing
or contradictory evidence remains UNKNOWN, not a guessed identity or dropped
card label. In particular, floor 51 can never supply Act 3's first boss.
"""
    catalog = catalog or RuleCatalog.bundled()
    result = {}
    for act, floor in FLOORS.items():
        boss, reason = None, 'unreviewed_boss_history_build'
        if run.get('build_version') in LEGACY_BUILDS:
            path, combats = run.get('path_per_floor'), run.get('damage_taken')
            reason = 'missing_boss_room_or_combat'
            if isinstance(path, list) and len(path) >= floor and path[floor-1] == 'B' and isinstance(combats, list):
                found = [c for c in combats if isinstance(c, dict) and
                         type(c.get('floor')) in (int, float) and c['floor'] == floor]
                reason = 'missing_or_ambiguous_first_boss'
                if len(found) == 1:
                    try:
                        boss = canonical_boss(found[0].get('enemies'), act, catalog)
                    except ValueError:
                        reason = 'unrecognized_first_boss'
                    else:
                        reason = 'first_boss_room_identity' if boss else 'missing_boss_identity'
        result[act] = {'boss':boss, 'reason':reason}
    return result


def with_boss(row, boss, *, provenance, catalog=None):
    catalog = catalog or RuleCatalog.bundled()
    result = copy.deepcopy(row)
    run = result['public']['observation']['run']
    # Floor 51's A20 reveal is not reconstructed by this version. Likewise do
    # not infer cross-act chest/transition timing from fixed floor alone.
    if run['floor'] > FLOORS.get(run['act'], -1):
        boss, provenance = None, 'outside_reviewed_boss_visibility_window'
    boss = canonical_boss(boss, run['act'], catalog)
    run.pop('boss', None)
    if boss is not None:
        run['boss'] = boss
    public = PolicyInput(result['public']['observation'],
        tuple(Action.from_dict(a) for a in result['public']['legal_actions']))
    result.update(version=VERSION, input_sha256=public.fingerprint(),
                  boss_context={'known':boss is not None, 'provenance':provenance})
    return result


def encode_row(row, encoder):
    if row.get('version') != VERSION:
        raise ValueError('boss-aware row contract mismatch')
    run = row['public']['observation']['run']
    boss = canonical_boss(run.get('boss'), run['act'], encoder.catalog)
    if (row.get('boss_context', {}).get('known') is not (boss is not None) or
            (boss is not None and run['floor'] > FLOORS[run['act']])):
        raise ValueError('boss visibility contract mismatch')
    # Exact existing partial mask/action machinery, with a separately versioned
    # public boss field. The neural architecture and tensor shapes do not change.
    return legacy.encode_row(dict(row, version=legacy.VERSION), encoder)


def reconstruct_win(run, *, catalog=None):
    category, reason = cohort(run)
    if category != 'win_candidate':
        raise legacy.HistoryGap('not_win_bc_cohort:'+reason)
    catalog = catalog or RuleCatalog.bundled()
    adapted, neow = adapt_neow(run, catalog=catalog)
    base_ids = LEGACY_BASE_ID_REMOVALS if 'neow_bonus_log' not in run else ()
    barriers = []
    combats = run.get('damage_taken')
    if combats is None:
        if any(legacy.integer(c['floor']) >= 35 for c in run['card_choices']):
            barriers.append({'floor':35,'reason':'missing_act3_combat_inventory_history'})
    elif not isinstance(combats, list):
        raise legacy.HistoryGap('invalid_combat_inventory_history')
    else:
        for combat in combats:
            if not isinstance(combat, dict) or not isinstance(combat.get('enemies'), str):
                raise legacy.HistoryGap('invalid_combat_inventory_history')
            floor = legacy.integer(combat['floor'])
            if combat['enemies'] == 'Writhing Mass':
                barriers.append({'floor':floor,'reason':'unlogged_combat_deck_effect:WRITHING_MASS'})
    result = legacy.reconstruct(adapted, catalog=catalog, base_id_removal_events=base_ids,
                                inventory_barriers=barriers)
    bosses = historical_bosses(run, catalog)
    result['rows'] = [with_boss(r, bosses[r['act']]['boss'],
        provenance=bosses[r['act']]['reason'], catalog=catalog) for r in result['rows']]
    return {**result, 'intake_version':VERSION, 'neow_mode':neow, 'boss_audit':bosses}


def project_reward(public, encoder):
    old = legacy_project(public, encoder)
    row = with_boss(old.row, public.observation['run'].get('boss'),
                    provenance='current_public_run', catalog=encoder.catalog)
    return replace(old, row=row, encoded=encode_row(row, encoder))


class BossExpertCardPolicy(ExpertCardPolicy):
    """Card-only opt-in; old weights must not silently acquire new features."""
    def __init__(self, checkpoint, expected_sha, encoder, *, width=32):
        import torch
        super().__init__(checkpoint, expected_sha, encoder, width=width)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['contract']['version'] != TRAIN_VERSION:
            raise ValueError('boss-aware checkpoint version required')

    def choose(self, public):
        import torch
        from .run_learning_model import tensorize
        projected = project_reward(public, self.encoder)
        with torch.inference_mode():
            distribution, _ = self.model.distribution(tensorize([projected.encoded], self.encoder))
        probabilities = distribution.probs[0]
        if not torch.isfinite(probabilities).all():
            raise ValueError('non-finite card probabilities')
        index = int(probabilities.argmax())
        action = projected.decode(index)
        if action not in public.legal_actions:
            raise ValueError('projected action is not currently legal')
        self.last_decision = {'version':VERSION, 'mode':'greedy_partial_card_bc_with_boss',
            'input_sha256':projected.row['input_sha256'], 'action':action.as_dict(),
            'meaning':projected.meanings[index], 'probabilities':probabilities.tolist(),
            'candidates':projected.meanings, 'fallback':False}
        return action
