"""Validate the current-act PUBLIC graph without consulting any hidden channel."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import json


PUBLIC_MAP_SCHEMA = 1
ROOMS = frozenset({"MONSTER", "ELITE", "REST", "SHOP", "EVENT", "TREASURE", "BOSS"})


@dataclass(frozen=True, slots=True, eq=False)
class _ValidatedPublicMap(Mapping):
    """Owned, deeply immutable schema-checked graph; never a mutable proxy cache."""

    _data: Mapping

    def __init__(self, graph):
        from .policy_api import _freeze_public_value

        validate_public_map({"map": graph, "run": {"act": graph.get("act")}})
        object.__setattr__(self, "_data", _freeze_public_value(dict(graph)))

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __deepcopy__(self, memo):
        return self


@lru_cache(maxsize=8)
def _map_snapshot(serialized: str) -> _ValidatedPublicMap:
    return _ValidatedPublicMap(json.loads(serialized))


def freeze_public_map(observation: Mapping) -> _ValidatedPublicMap:
    """Bounded content-keyed reuse, including every dynamic and unexpected field.

    JSON distinguishes bool/int/float values unlike dict equality. No object-id,
    act-only, or caller-controlled 'already validated' flag can skip a changed
    graph. Context is checked anew even on an immutable cache hit.
    """
    graph = observation.get("map")
    if type(graph) is _ValidatedPublicMap:
        snapshot = graph
    else:
        if not isinstance(graph, dict):
            raise ValueError("missing_public_map")
        snapshot = _map_snapshot(json.dumps(graph, sort_keys=True, separators=(",", ":"), allow_nan=False))
    validate_public_map({"map": snapshot, "run": observation.get("run")})
    return snapshot


def _point(raw, *, start=False):
    if (not isinstance(raw, Mapping) or set(raw) != {"x", "y"}
            or type(raw["x"]) is not int or type(raw["y"]) is not int):
        raise ValueError("invalid_map_point")
    x, y = raw["x"], raw["y"]
    if start and (x, y) == (-1, -1):
        return x, y
    if not 0 <= x <= 6 or not 0 <= y <= 15:
        raise ValueError("map_point_out_of_bounds")
    return x, y


def validate_public_map(observation: Mapping) -> Mapping:
    """Fail explicitly on missing/unsupported fields; no reconstruction from seed.

    This validates structure, not original-game map fidelity or route quality.
    Extra fields are rejected so a future native extension cannot silently expose
    an unreviewed burning-elite modifier, second boss, or encounter identity.
    """
    graph = observation.get("map")
    if not isinstance(graph, Mapping):
        raise ValueError("missing_public_map")
    if type(graph.get("schema_version")) is not int or graph["schema_version"] != PUBLIC_MAP_SCHEMA:
        raise ValueError("unsupported_public_map_schema")
    if graph.get("available") is not True:
        raise ValueError("public_map_unavailable")
    if set(graph) != {"schema_version", "available", "act", "position", "boss_node",
                      "known_boss", "starts", "nodes"}:
        raise ValueError("unreviewed_public_map_fields")
    act = graph["act"]
    run = observation.get("run")
    if (not isinstance(run, Mapping) or type(run.get("act")) is not int
            or type(act) is not int or act not in {1, 2, 3, 4} or act != run["act"]):
        raise ValueError("public_map_act_mismatch")
    if not isinstance(graph["known_boss"], str) or not graph["known_boss"]:
        raise ValueError("missing_public_map_boss")
    if run.get("boss", graph["known_boss"]) != graph["known_boss"]:
        raise ValueError("public_map_boss_mismatch")
    if type(graph) is _ValidatedPublicMap:
        return graph  # Constructor checked graph structure; all descendants are immutable.
    boss = _point(graph["boss_node"])
    if boss != (3, 3 if act == 4 else 15):
        raise ValueError("invalid_public_map_boss_position")
    position = _point(graph["position"], start=True)
    if not isinstance(graph["nodes"], (list, tuple)) or not 1 <= len(graph["nodes"]) <= 106:
        raise ValueError("invalid_public_map_nodes")
    nodes = {}
    for node in graph["nodes"]:
        if not isinstance(node, Mapping) or set(node) != {"x", "y", "room", "burning_elite", "next"}:
            raise ValueError("unreviewed_public_map_node_fields")
        point = _point({"x": node["x"], "y": node["y"]})
        if (point in nodes or not isinstance(node["room"], str) or node["room"] not in ROOMS
                or type(node["burning_elite"]) is not bool):
            raise ValueError("invalid_public_map_node")
        if node["burning_elite"] and node["room"] != "ELITE":
            raise ValueError("invalid_public_map_burning_marker")
        if not isinstance(node["next"], (list, tuple)) or len(node["next"]) > 3:
            raise ValueError("invalid_public_map_edges")
        nodes[point] = node
    if boss not in nodes or nodes[boss]["room"] != "BOSS":
        raise ValueError("invalid_public_map_boss_node")
    if [point for point, node in nodes.items() if node["room"] == "BOSS"] != [boss]:
        raise ValueError("multiple_public_map_boss_nodes")
    if position != (-1, -1) and position not in nodes:
        raise ValueError("public_map_position_not_on_graph")
    for point, node in nodes.items():
        next_nodes = [_point(destination) for destination in node["next"]]
        if len(next_nodes) != len(set(next_nodes)) or any(
                destination not in nodes or destination[1] != point[1] + 1 for destination in next_nodes):
            raise ValueError("invalid_public_map_edge_target")
        if (point == boss) != (not next_nodes):
            raise ValueError("unexpected_public_map_dead_end")
    if not isinstance(graph["starts"], (list, tuple)):
        raise ValueError("invalid_public_map_starts")
    starts = [_point(point) for point in graph["starts"]]
    if (not starts or len(starts) != len(set(starts))
            or set(starts) != {point for point in nodes if point[1] == 0}):
        raise ValueError("invalid_public_map_starts")
    reached, pending = set(), list(starts)
    while pending:
        point = pending.pop()
        if point not in reached:
            reached.add(point)
            pending.extend(_point(p) for p in nodes[point]["next"])
    if reached != set(nodes):
        raise ValueError("unreachable_public_map_nodes")
    return graph


def public_map_status(observation: Mapping) -> str:
    try:
        validate_public_map(observation)
        return "valid"
    except ValueError as error:
        return str(error)


def link_route_options(observation: Mapping, rows: list[dict]) -> str:
    """Join public candidates to public nodes, never reinterpret action indexes."""
    try:
        graph = validate_public_map(observation)
        position = _point(graph["position"], start=True)
        nodes = {(node["x"], node["y"]): node for node in graph["nodes"]}
        reachable = (graph["starts"] if position == (-1, -1) else nodes[position]["next"])
        reachable = {_point(point) for point in reachable}
        destinations = []
        for row in rows:
            if row["kind"] != "route":
                raise ValueError("unparsed_route_action")
            target = (_point(graph["boss_node"]) if row["room"] == "boss"
                      else (row["x"], position[1] + 1))
            if target not in reachable or nodes[target]["room"].lower() != row["room"]:
                raise ValueError("route_action_map_mismatch")
            destinations.append(target)
        if set(destinations) != reachable or len(destinations) != len(set(destinations)):
            raise ValueError("route_actions_do_not_cover_reachable_nodes")
        for row, (x, y) in zip(rows, destinations):
            row["destination"] = {"x": x, "y": y}
        return "linked"
    except ValueError as error:
        return str(error)
