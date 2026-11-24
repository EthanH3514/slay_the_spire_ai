"""Receipt-gated, bounded win-only reconstruction from completed raw intake.

Raw outcomes/seeds stay in the intake audit. Model features are exclusively the
forward partial public prefix. No archive replay, loss labels or critic targets.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import random
import sqlite3
import time

from .expert_cards import HistoryGap, digest, encode_row, integer
from .metrics_cards import cohort, reconstruct_win
from .metrics_stream import ShardWriter, file_sha, split_group
from .run_learning_encoding import RunLearningEncoder
from .atomic_report import write_json_report as write_progress

ROOT = Path(__file__).resolve().parents[2]
VERSION = "metrics-sharded-card-dataset-v1"
SPLITS = ("train", "validation", "test")
LINE_LIMIT = 32 * 1024 * 1024
IMPLEMENTATION = (
    "src/spire_ai/metrics_dataset.py", "src/spire_ai/metrics_cards.py",
    "src/spire_ai/expert_cards.py", "src/spire_ai/metrics_stream.py",
    "src/spire_ai/rule_catalog.py", "src/spire_ai/rule-catalog-v1.json",
    "src/spire_ai/run_learning_encoding.py", "src/spire_ai/public_encoding.py",
    "src/spire_ai/policy_api.py", "src/spire_ai/deck_profile.py",
    "src/spire_ai/card_recipes.py", "src/spire_ai/rule_coverage.py",
    "src/spire_ai/atomic_report.py",
)


def implementation_hashes():
    return {p: file_sha(ROOT / p) for p in IMPLEMENTATION}


def write_document(path, value):
    """Atomic status/receipt publication; only inside a new experiment output."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def checked_path(root, name):
    candidate = Path(name)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("unsafe shard path")
    path = (root / candidate).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("shard path escapes dataset")
    return path


def verified_jsonl(root, entry):
    path = checked_path(root, entry["path"])
    if path.stat().st_size != entry["bytes"] or file_sha(path) != entry["sha256"]:
        raise ValueError("shard integrity mismatch: " + entry["path"])
    count = 0
    with gzip.open(path, "rb") as handle:
        while line := handle.readline(LINE_LIMIT + 1):
            if len(line) > LINE_LIMIT or not line.endswith(b"\n"):
                raise ValueError("unbounded or truncated JSONL record")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected object row")
            count += 1
            yield row
    if count != entry["rows"]:
        raise ValueError("shard row count mismatch")


class IntakeWins:
    """Read committed win files only; conflict status is always joined to index."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.receipt = json.loads((self.root / "receipt.json").read_text(encoding="utf-8"))
        self.source = json.loads((self.root / "source.json").read_text(encoding="utf-8"))
        if (self.receipt.get("complete") is not True or
                self.receipt.get("all_member_crcs_verified") is not True or
                self.receipt.get("decoder_exit") != 0 or
                self.receipt.get("contract_sha256") != digest(self.source) or
                self.receipt.get("source_sha256") != self.source.get("source_sha256")):
            raise ValueError("completed intake receipt required")
        self.identity = {name: file_sha(self.root / name) for name in ("receipt.json", "source.json")}
        self.stats = Counter()

    def rows(self):
        self.stats.clear()
        with closing(sqlite3.connect((self.root / "index.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            meta = dict(db.execute("SELECT key,value FROM meta"))
            last = db.execute("SELECT next_member,decoded_bytes,stats FROM batches ORDER BY seq DESC LIMIT 1").fetchone()
            if (meta.get("complete") != "true" or json.loads(meta["contract"]) != self.source or not last or
                    last[:2] != (self.receipt["committed_members"], self.receipt["decoded_bytes"]) or
                    last[:2] != (self.source["members"], self.source["decoded_bytes"]) or
                    json.loads(last[2]) != self.receipt["stats"]):
                raise ValueError("intake index/receipt mismatch")
            seen, paths = set(), set()
            for (encoded,) in db.execute("SELECT files FROM batches ORDER BY seq"):
                for entry in json.loads(encoded):
                    if entry["path"] in paths:
                        raise ValueError("duplicate manifest path")
                    paths.add(entry["path"])
                    if entry["kind"] != "audit-wins":
                        continue
                    for wrapped in verified_jsonl(self.root, entry):
                        self.stats["raw_win_rows"] += 1
                        run = wrapped["event"]
                        source_sha = digest(run)
                        group = digest({"game_uuid": run["play_id"]}) if run.get("play_id") else source_sha
                        expected_split = split_group(group)
                        if (wrapped["source_sha256"] != source_sha or wrapped["group"] != group or
                                wrapped["cohort"] != "win_candidate" or wrapped["usage"] != "raw_audit_only" or
                                cohort(run)[0] != "win_candidate" or wrapped["split"] != expected_split):
                            raise ValueError("raw win provenance/cohort mismatch")
                        indexed = db.execute("SELECT first_sha,cohort,split,conflicted FROM groups WHERE id=?",
                                             (group,)).fetchone()
                        if not indexed or indexed[:3] != (source_sha, "win_candidate", expected_split):
                            raise ValueError("win index identity mismatch")
                        if group in seen:
                            raise ValueError("duplicate committed win group")
                        seen.add(group)
                        if indexed[3]:
                            self.stats["conflicted_wins_excluded"] += 1
                            continue
                        if not db.execute("SELECT 1 FROM versions WHERE id=? AND sha=?", (group, source_sha)).fetchone():
                            raise ValueError("missing indexed version")
                        self.stats["eligible_win_rows"] += 1
                        yield wrapped
            if self.stats["raw_win_rows"] != self.receipt["stats"]["counts"].get("audit-wins", 0):
                raise ValueError("win manifest coverage mismatch")
        if any(file_sha(self.root / name) != expected for name, expected in self.identity.items()):
            raise ValueError("intake identity changed while reading")


def validate_partial_row(row, encoder):
    observation = row["public"]["observation"]
    if observation.get("player") != {}:
        raise ValueError("decision-time scalars must remain unknown")
    encoded = encode_row(row, encoder)
    global_table = encoded.tables["global"]
    if any(global_table.known[0][global_table.columns.index(name)] for name in ("hp", "max_hp", "gold")):
        raise ValueError("historical scalar mask leaked")
    relics = encoded.tables["relics"]
    if any(mask[relics.columns.index("order")] for mask in relics.known):
        raise ValueError("historical relic order leaked")
    return encoded


def reconstruct_admission(run, catalog):
    """Known ambiguity quarantines the run; unexplained mismatch also fails gate."""
    if "floor_reached" in run and integer(run["floor_reached"]) < 50:
        raise HistoryGap("victory_without_completed_run")
    result = reconstruct_win(run, catalog=catalog)
    first = result["barriers"][0]["reason"] if result["barriers"] else ""
    contradiction = first.startswith(("remove_missing_card:", "ambiguous_or_missing_upgrade:"))
    contradiction |= any("invalid_offer:singing_bowl_not_observed" in e["reasons"] for e in result["excluded"])
    if result["unexplained_final_mismatch"]:
        reason = "unexplained_final_mismatch"
    elif contradiction:
        reason = "inventory_contradiction_not_reliably_localized"
    elif result["final_deck_matches"] is None and not result["barriers"]:
        reason = "unverifiable_final_inventory"
    else:
        reason = None
    return result, reason


def verify_prior_review(review):
    """Bind v2 whole-run exclusions to an actual older unexplained mismatch.

An importer correction can add a late ambiguity barrier without explaining an
earlier inventory contradiction. Never let that barrier silently re-admit the
already reviewed run's prefixes. Final summaries remain audit-only.
"""
    if review.get("version") != "metrics-quarantine-review-v2":
        return {}
    parent = review.get("parent_dataset", {})
    directory = checked_path(ROOT, parent["path"])
    if not directory.is_relative_to(ROOT / "runtime") or directory == ROOT / "runtime":
        raise ValueError("review parent must be a named runtime dataset")
    receipt_path, report_path = directory / "receipt.json", directory / "report.json"
    receipt_sha, report_sha = file_sha(receipt_path), file_sha(report_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (receipt.get("complete") is not True or report_sha != parent.get("report_sha256") or
            report_sha != receipt.get("report_sha256") or report["intake_contract"] != review["intake_contract"]):
        raise ValueError("review parent receipt/contract mismatch")
    wanted = {e["group"]: e["source_sha256"] for e in review["entries"]}
    found, pins = set(), {receipt_path: receipt_sha, report_path: report_sha}
    for entry in report["files"]:
        if entry["kind"] != "run-audits":
            continue
        for audit in verified_jsonl(directory, entry):
            if audit["group"] not in wanted:
                continue
            if (audit["group"] in found or audit["source_sha256"] != wanted[audit["group"]] or
                    audit.get("unexplained_final_mismatch") is not True or audit.get("admitted_rows") != 0):
                raise ValueError("review not bound to an originally quarantined mismatch")
            found.add(audit["group"])
        pins[checked_path(directory, entry["path"])] = entry["sha256"]
    if found != set(wanted):
        raise ValueError("missing review parent audit")
    return pins


def build_dataset(intake, output, *, shard_rows=512, progress_every=200, quarantine_review=None):
    if shard_rows < 1 or progress_every < 1:
        raise ValueError("positive bounds required")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("refuse to overwrite a dataset")
    source = IntakeWins(intake)
    reviews, review_identity, review_parent_pins = {}, None, {}
    whole_run_review = False
    if quarantine_review is not None:
        quarantine_review = Path(quarantine_review)
        reviewed = json.loads(quarantine_review.read_text(encoding="utf-8"))
        if (reviewed.get("version") not in ("metrics-quarantine-review-v1", "metrics-quarantine-review-v2") or
                reviewed.get("intake_contract") != source.receipt["contract_sha256"]):
            raise ValueError("quarantine review contract mismatch")
        for entry in reviewed["entries"]:
            if (entry["group"] in reviews or len(entry["group"]) != 64 or len(entry["source_sha256"]) != 64 or
                    entry.get("disposition") != "exclude_entire_run" or not entry.get("evidence")):
                raise ValueError("invalid quarantine review")
            reviews[entry["group"]] = entry
        review_identity = {"sha256": file_sha(quarantine_review), "review": reviewed}
        whole_run_review = reviewed["version"] == "metrics-quarantine-review-v2"
        review_parent_pins = verify_prior_review(reviewed)
    applied_reviews = set()
    source_hashes = implementation_hashes()
    encoder = RunLearningEncoder()
    output.mkdir(parents=True)
    writers = {s: ShardWriter(output, s, row_limit=shard_rows) for s in (*SPLITS, "run-audits")}
    counters = {k: Counter() for k in ("totals", "builds", "neow", "refusals", "quarantines",
        "barriers", "excluded_choices", "acts", "labels", "splits", "run_splits", "final_audit",
        "build_admitted_runs", "act_split_rows", "act_split_runs", "detail_presence")}
    totals = counters["totals"]
    start = time.perf_counter()

    def status(phase):
        # Mutable status uses the reviewed, bounded Windows reader-lock retry.
        # Final receipts/documents keep their existing exclusive+fsync writer.
        write_progress(output / "progress.json", {"version": VERSION, "phase": phase,
            "updated_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
            "elapsed_seconds": time.perf_counter() - start, "totals": dict(totals),
            "expected_raw_wins": source.receipt["stats"]["counts"].get("audit-wins", 0)})

    status("reconstructing")
    try:
        for wrapped in source.rows():
            run = wrapped["event"]
            totals["wins_audited"] += 1
            build = str(run.get("build_version"))
            counters["builds"][build] += 1
            counters["neow"][str(run.get("neow_cost")) + "/" + str(run.get("neow_bonus"))] += 1
            counters["detail_presence"]["present" if "neow_bonus_log" in run else "absent"] += 1
            provenance = {k: wrapped[k] for k in ("group", "split", "source_sha256", "member_index",
                                                  "source_member", "record_index")}
            audit = {**provenance, "source_build": build}
            rows = []
            try:
                result, quarantine = reconstruct_admission(run, encoder.catalog)
            except (HistoryGap, KeyError, TypeError, ValueError) as error:
                counters["refusals"][str(error)] += 1
                totals["refused_runs"] += 1
                audit["refusal"] = str(error)
            else:
                rows = result.pop("rows")
                totals["reconstructed_runs"] += 1
                counters["barriers"].update(b["reason"] for b in result["barriers"])
                counters["excluded_choices"].update(reason for e in result["excluded"] for reason in e["reasons"])
                counters["final_audit"][str(result["final_deck_matches"])] += 1
                audit.update(result)
                reviewed_mismatch = (whole_run_review and wrapped["group"] in reviews and
                                     result["final_deck_matches"] is False)
                if quarantine or reviewed_mismatch:
                    if wrapped["group"] in reviews:
                        review = reviews[wrapped["group"]]
                        if (review["source_sha256"] != wrapped["source_sha256"] or
                                (quarantine != "unexplained_final_mismatch" and not reviewed_mismatch)):
                            raise ValueError("quarantine review does not match failure")
                        applied_reviews.add(wrapped["group"])
                        audit["pre_review_quarantine"] = quarantine
                        quarantine = "reviewed_unreconstructable_history"
                        audit["quarantine_review"] = review
                    counters["quarantines"][quarantine] += 1
                    totals["quarantined_runs"] += 1
                    audit.update(quarantine=quarantine, quarantined_rows=len(rows))
                    rows = []
                # Encode/check ALL of a run before writing any of its rows.
                for row in rows:
                    validate_partial_row(row, encoder)
            if rows:
                totals["admitted_runs"] += 1
                counters["run_splits"][wrapped["split"]] += 1
                counters["build_admitted_runs"][build] += 1
                counters["act_split_runs"].update(wrapped["split"] + ":" + str(a) for a in {r["act"] for r in rows})
            for row in rows:
                writers[wrapped["split"]].write({**row, **provenance, "source_build": build,
                    "dataset_version": VERSION, "usage": "receipt_gated_offline_card_bc",
                    "cohort": "a20_victory_true"})
                totals["admitted_rows"] += 1
                counters["acts"][str(row["act"])] += 1
                counters["labels"][row["label"]] += 1
                counters["splits"][wrapped["split"]] += 1
                counters["act_split_rows"][wrapped["split"] + ":" + str(row["act"])] += 1
            writers["run-audits"].write({**audit, "admitted_rows": len(rows)})
            if totals["wins_audited"] % progress_every == 0:
                status("reconstructing")
        if source_hashes != implementation_hashes():
            raise ValueError("reconstruction source changed during build")
        if applied_reviews != set(reviews):
            raise ValueError("unused quarantine review; re-review rather than silently exempt")
        if quarantine_review is not None and file_sha(quarantine_review) != review_identity["sha256"]:
            raise ValueError("quarantine review changed during build")
        if any(file_sha(path) != expected for path, expected in review_parent_pins.items()):
            raise ValueError("review parent changed during build")
    except BaseException as error:
        try:
            status("error:" + type(error).__name__)
        except OSError as status_error:
            # A persistent status lock must not obscure the original failure.
            error.add_note("Could not publish dataset failure status: " + repr(status_error))
        raise
    finally:
        for writer in writers.values():
            writer.close()
    gate = (totals["admitted_rows"] > 0 and
            not counters["quarantines"]["unexplained_final_mismatch"] and
            not counters["quarantines"]["unverifiable_final_inventory"] and
            all(counters["splits"][s] for s in SPLITS))
    report = {"version": VERSION, **{k: dict(v) for k, v in counters.items()},
        "source_intake": source.identity, "intake_contract": source.receipt["contract_sha256"],
        "intake_stats": dict(source.stats), "schema_id": encoder.schema_id, "implementation": source_hashes,
        "files": [entry for writer in writers.values() for entry in writer.files],
        "shard_rows": shard_rows, "elapsed_seconds": time.perf_counter() - start,
        "quarantine_review": review_identity,
        "offline_training_ready": gate, "production_ready": False, "critic_targets": 0,
        "contracts": {"losses": "audit-only, never loaded as labels",
            "split": "intake whole-run identity; unchanged",
            "features": "forward partial public state; no final-summary backfill",
            "rules": "legacy schema compatibility is not modern-game rules parity",
            "quarantine": "whole-run; unexplained final mismatch blocks corpus training gate"}}
    write_document(output / "report.json", report)
    write_document(output / "receipt.json", {"version": VERSION, "complete": True,
        "report_sha256": file_sha(output / "report.json"), "offline_training_ready": gate})
    status("complete")
    return report


class ShardedCardDataset:
    """Receipt/feature-gated loader; one shard plus a bounded shuffle buffer."""
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.receipt = json.loads((self.root / "receipt.json").read_text(encoding="utf-8"))
        if (self.receipt.get("version") != VERSION or self.receipt.get("complete") is not True or
                file_sha(self.root / "report.json") != self.receipt.get("report_sha256")):
            raise ValueError("dataset receipt mismatch")
        self.report = json.loads((self.root / "report.json").read_text(encoding="utf-8"))
        if not self.receipt.get("offline_training_ready") or not self.report.get("offline_training_ready"):
            raise ValueError("dataset failed offline training gate")
        if self.report["version"] != VERSION or self.report["implementation"] != implementation_hashes():
            raise ValueError("dataset implementation drift; re-audit required")
        self.files = {s: [] for s in SPLITS}
        paths = set()
        for entry in self.report["files"]:
            checked_path(self.root, entry["path"])
            if entry["path"] in paths:
                raise ValueError("duplicate dataset shard path")
            paths.add(entry["path"])
            if entry["kind"] in self.files:
                self.files[entry["kind"]].append(entry)
        for split in SPLITS:
            if sum(e["rows"] for e in self.files[split]) != self.report["splits"][split]:
                raise ValueError("split row manifest mismatch")

    def rows(self, split, *, shuffle_seed=None, buffer_rows=1024):
        if split not in SPLITS or buffer_rows < 1:
            raise ValueError("invalid split/shuffle bound")
        if split != "train" and shuffle_seed is not None:
            raise ValueError("held-out order is fixed")
        entries = list(self.files[split])
        rng = random.Random(shuffle_seed)
        if shuffle_seed is not None:
            rng.shuffle(entries)
        buffered = []
        for entry in entries:
            for row in verified_jsonl(self.root, entry):
                if (row.get("dataset_version") != VERSION or row.get("cohort") != "a20_victory_true" or
                        row.get("usage") != "receipt_gated_offline_card_bc" or row["split"] != split or
                        split_group(row["group"]) != split):
                    raise ValueError("row cohort/split contract mismatch")
                if shuffle_seed is None:
                    yield row
                else:
                    if len(buffered) >= buffer_rows:
                        yield buffered.pop(rng.randrange(len(buffered)))
                    buffered.append(row)
        rng.shuffle(buffered)
        yield from buffered

    def batches(self, split, encoder, *, batch_size=32, shuffle_seed=None, buffer_rows=1024):
        if batch_size < 1 or encoder.schema_id != self.report["schema_id"]:
            raise ValueError("invalid batch/encoder schema")
        batch = []
        for row in self.rows(split, shuffle_seed=shuffle_seed, buffer_rows=buffer_rows):
            batch.append({"row": row, "encoded": validate_partial_row(row, encoder)})
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch
