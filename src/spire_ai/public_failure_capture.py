"""Bounded failure evidence from the policy boundary, never the native audit state."""
import json

from .policy_api import policy_input_from_state

VERSION = "public-failure-snapshot-v1"


def public_failure_snapshot(state, *, stage, error, limit_bytes=1_048_576):
    if type(limit_bytes) is not int or limit_bytes < 1024:
        raise ValueError("failure snapshot requires a positive bounded byte budget")
    result = {"version": VERSION, "stage": str(stage)[:160],
              "error_type": type(error).__name__, "message": str(error)[:1600]}
    if state is None:
        return {**result, "capture_status": "no_public_state"}
    try:
        public = policy_input_from_state(state)
        document = public.as_dict()
        result.update(fingerprint=public.fingerprint(), screen=document["observation"].get("screen", {}).get("type"),
                      selection_task=document["observation"].get("combat", {}).get("card_select_task"))
        payload = {**result, "capture_status": "complete", "public": document}
        size = len(json.dumps(payload, allow_nan=False).encode("utf-8"))
    except Exception as capture_error:
        # Do not mask the original failure or serialize the rejected raw state.
        return {**result, "capture_status": "capture_error", "capture_error_type": type(capture_error).__name__}
    if size > limit_bytes:
        return {**result, "capture_status": "size_limit", "required_bytes": size, "limit_bytes": limit_bytes}
    return payload
