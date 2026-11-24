from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Mapping


class ActionKind(str, Enum):
    START = "start"
    PLAY_CARD = "play_card"
    USE_POTION = "use_potion"
    DISCARD_POTION = "discard_potion"
    END_TURN = "end_turn"
    CHOOSE = "choose"
    PROCEED = "proceed"
    CANCEL = "cancel"
    WAIT = "wait"
    REQUEST_STATE = "request_state"


@dataclass(frozen=True, slots=True)
class Action:
    """Backend-neutral game action.

    Internal indexes are zero-based. ``to_communication_command`` performs the one
    protocol-specific conversion: CommunicationMod's PLAY hand index is one-based.
    """

    kind: ActionKind
    card_index: int | None = None
    potion_index: int | None = None
    target_index: int | None = None
    choice_index: int | None = None
    character: str | None = None
    ascension: int | None = None
    seed: str | None = None
    wait_frames: int | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "Action":
        """Build an action from trace or policy data.

        Unknown fields are rejected instead of silently entering the policy space.
        This keeps serialized actions stable across the real-game and simulator
        backends.
        """

        raw_kind = value.get("kind")
        if not isinstance(raw_kind, str):
            raise ValueError("action kind must be a string")
        try:
            kind = ActionKind(raw_kind.lower())
        except ValueError as exc:
            raise ValueError(f"unknown action kind: {raw_kind}") from exc

        allowed = {field.name for field in fields(cls)}
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown action field(s): {names}")
        arguments = {key: item for key, item in value.items() if key != "kind"}
        return cls(kind=kind, **arguments)  # type: ignore[arg-type]

    def to_communication_command(self) -> str:
        if self.kind is ActionKind.START:
            if not self.character:
                raise ValueError("START requires a character")
            ascension = 0 if self.ascension is None else self.ascension
            if not 0 <= ascension <= 20:
                raise ValueError("ascension must be between 0 and 20")
            parts = ["start", self.character.upper(), str(ascension)]
            if self.seed:
                parts.append(self.seed)
            return " ".join(parts)

        if self.kind is ActionKind.PLAY_CARD:
            if self.card_index is None or self.card_index < 0:
                raise ValueError("PLAY_CARD requires a non-negative card_index")
            parts = ["play", str(self.card_index + 1)]
            if self.target_index is not None:
                if self.target_index < 0:
                    raise ValueError("target_index cannot be negative")
                parts.append(str(self.target_index))
            return " ".join(parts)

        if self.kind in (ActionKind.USE_POTION, ActionKind.DISCARD_POTION):
            if self.potion_index is None or self.potion_index < 0:
                raise ValueError("potion action requires a non-negative potion_index")
            operation = "use" if self.kind is ActionKind.USE_POTION else "discard"
            parts = ["potion", operation, str(self.potion_index)]
            if self.target_index is not None:
                if self.target_index < 0:
                    raise ValueError("target_index cannot be negative")
                parts.append(str(self.target_index))
            return " ".join(parts)

        if self.kind is ActionKind.END_TURN:
            return "end"

        if self.kind is ActionKind.CHOOSE:
            if self.choice_index is None or self.choice_index < 0:
                raise ValueError("CHOOSE requires a non-negative choice_index")
            return f"choose {self.choice_index}"

        if self.kind is ActionKind.PROCEED:
            return "proceed"

        if self.kind is ActionKind.CANCEL:
            return "cancel"

        if self.kind is ActionKind.WAIT:
            if self.wait_frames is None or self.wait_frames < 1:
                raise ValueError("WAIT requires a positive wait_frames value")
            return f"wait {self.wait_frames}"

        if self.kind is ActionKind.REQUEST_STATE:
            return "state"

        raise AssertionError(f"unhandled action kind: {self.kind}")

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"kind": self.kind.value}
        for field_name in (
            "card_index",
            "potion_index",
            "target_index",
            "choice_index",
            "character",
            "ascension",
            "seed",
            "wait_frames",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result
