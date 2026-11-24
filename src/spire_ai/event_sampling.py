"""Explicit pilot contract; this does not replace native-seed conditioning."""

STRICT_ROOT_MODEL = "native-seed-conditioned-v1"
EVENT_ROOT_MODEL = "public-event-offer-v1"
EVENT_REPLAY_ATTEMPTS = 64  # Per requested world, also subject to shared request caps.
RESOURCE_FALLBACK_VERSION = "event-resource-fallback-v1"
RESOURCE_REASONS = frozenset({
    "continuation_proposal_limit", "work_limit", "random_limit", "deadline",
    "step_budget", "sample_budget", "value_budget",
})


def validate_event_result(result):
    """Do not infer authorization from an exception's text or a generic limit."""
    from .environment import EnvironmentError
    if (result.get("root_model") != EVENT_ROOT_MODEL
            or result.get("root_sampling", {}).get("probability_model") != EVENT_ROOT_MODEL
            or result["root_sampling"].get("replay_attempts_per_world") != EVENT_REPLAY_ATTEMPTS
            or "resource_fallback" not in result):
        raise EnvironmentError("event probability result contract mismatch")
    resource = result["resource_fallback"]
    if resource is not None and (
        not isinstance(resource, dict)
        or resource.get("version") != RESOURCE_FALLBACK_VERSION
        or resource.get("reason") not in RESOURCE_REASONS
        or result.get("rounds") != [] or result.get("unavailable")
        or result.get("discarded") != result.get("samples")
    ):
        raise EnvironmentError("event resource fallback contract mismatch")
    return resource
