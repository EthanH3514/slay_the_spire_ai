"""Inference helpers from run_learning_combat_history_guard.py."""

from run_learning_combat_event_shuffle import ABI, ISOLATION_ABI
GUARD_ABI = "public-history-byte-work-guard-v1"


GUARD = dict(version=GUARD_ABI, minimum=8000000, maximum=128000000,
             per_world=4096, replay_allowance=4, max_history_bytes=1048576,
             formula="min(maximum,max(minimum,1024*(per_world+replay_allowance*history_bytes)))")


def allowance(byte_count):
    if type(byte_count) is not int or not 0 <= byte_count <= GUARD["max_history_bytes"]:
        raise ValueError("invalid public history byte count")
    return min(GUARD["maximum"], max(GUARD["minimum"],
        1024 * (GUARD["per_world"] + GUARD["replay_allowance"] * byte_count)))


def validate_receipt(sampling):
    if (sampling.get("history_work_guard") != GUARD_ABI or
            sampling.get("history_work_hard_limit") != GUARD["maximum"] or
            sampling.get("history_work_limit") != allowance(sampling.get("history_work_bytes")) or
            sampling.get("construction_work_limit") != sampling["history_work_limit"] or
            not 0 <= sampling["construction_work"] <= sampling["history_work_limit"] or
            sampling.get("random_word_limit") != 1000000 or sampling.get("replay_attempts_per_world") != 64 or
            sampling.get("shuffle_probability_model") != ABI or
            sampling.get("conditioning_evidence_isolation") != ISOLATION_ABI):
        raise ValueError("construction guard receipt mismatch")
