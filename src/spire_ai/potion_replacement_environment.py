"""Execute a declared replacement as two audited, consecutive game actions."""
from dataclasses import replace
import copy

from .actions import ActionKind
from .policy_api import policy_input_from_state
from .environment import InvalidActionError
from .potion_replacement import VERSION, Commitment, combat_plan, plans, key


class ReplacementEnvironment:
    def __init__(self, game):
        self.game = game
        self.last_replacement = None

    def __getattr__(self, name):
        return getattr(self.game, name)

    def reset(self, **kwargs):
        self.last_replacement = None
        return self.game.reset(**kwargs)

    def clone_exact(self):
        clone = type(self)(self.game.clone_exact())
        clone.last_replacement = copy.deepcopy(self.last_replacement)
        return clone

    def _check_complete(self):
        if self.last_replacement and self.last_replacement['status'] != 'complete':
            raise InvalidActionError('previous potion replacement is incomplete; inspect its receipt')

    def replacement_choices(self):
        return plans(self.game._current_public_input())

    def step(self, action):
        self._check_complete()
        self.last_replacement = None
        if action.kind != ActionKind.DISCARD_POTION:
            return self.game.step(action)
        public = self.game._current_public_input()
        if public.observation.get('combat') is None:
            raise InvalidActionError('outside discard requires an explicitly selected replacement plan')
        return self.step_replacement(combat_plan(public,action))

    def step_replacement(self, plan):
        """Rewards/shops supply an explicit plan; combat uses its searched plan.

        The intermediate state is retained in the receipt, never returned as a
        decision where the caller could abandon the acquisition. No rollback or
        search of the live game's RNG is performed.
        """
        from .simulator_environment import _SimulatorOption, _reward_components
        self._check_complete()
        public = self.game._current_public_input()
        commit = Commitment.bind(public,plan)
        before = copy.deepcopy(public.as_dict())
        outside = public.observation.get('combat') is None
        if outside:
            # ConsoleSimulator supports this game command, but the historical
            # outside menu intentionally never exposed free-standing discards.
            # Add it only for this prevalidated transaction; the next state read
            # reconstructs the ordinary menu. No synthetic offer is introduced.
            option = _SimulatorOption(plan.discard,f'discard potion {plan.discard.potion_index}',
                                      'Discard for immediate replacement')
            self.game._options = (*self.game._options,option)
        first = self.game.step(plan.discard)
        after_discard = policy_input_from_state(first.state)
        self.last_replacement = dict(version=VERSION,status='discarded_pending_acquisition',plan=plan.as_dict(),
            before_fingerprint=public.fingerprint(),after_discard=after_discard.as_dict(),
            primitive_actions=[plan.discard.as_dict()],primitive_steps=1)
        # Failure here stops the caller with the partial receipt intact. It must
        # not quietly become a completed replacement or resume another policy.
        acquire = commit.followup(after_discard)
        second = self.game.step(acquire)
        self.last_replacement.update(status='acquired_pending_validation',primitive_steps=2,
                                     primitive_actions=[plan.discard.as_dict(),acquire.as_dict()])
        final = policy_input_from_state(second.state)
        inventory = final.observation['inventory']['potions']
        slot = plan.discard.potion_index
        if key(inventory[slot]['id']) == 'empty_potion_slot':
            raise InvalidActionError('replacement did not acquire a potion')
        if plan.source != 'refill' and key(inventory[slot]['id']) != key(plan.item_id):
            raise InvalidActionError('replacement acquired a different potion')
        if plan.source == 'refill' and any(key(p['id']) == 'empty_potion_slot' for p in inventory):
            raise InvalidActionError('refill left an unfilled replacement slot')
        changed_slots = {slot,plan.acquire.potion_index} if plan.source == 'refill' else {slot}
        old_inventory = before['observation']['inventory']['potions']
        if len(inventory) != len(old_inventory) or any(dict(p) != dict(inventory[i])
                for i,p in enumerate(old_inventory) if i not in changed_slots):
            raise InvalidActionError('replacement changed an unrelated potion slot')
        self.last_replacement.update(status='complete',primitive_steps=2,after_fingerprint=final.fingerprint(),
                                     acquire=acquire.as_dict())
        reward = _reward_components(before['observation'],final.observation,second.terminated)
        return replace(second,reward=reward,info=dict(second.info,potion_replacement=self.last_replacement))
