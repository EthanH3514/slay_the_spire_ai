"""Bounded metrics intake primitives. Raw records are audits, never observations.

A solid 7z stream is consumed in archive listing order. Exact byte boundaries
and every member CRC are checked, so mismatched extraction order fails closed.
Checkpoints refer to completed members; restart replays the compressed prefix.
SQLite is the authority for both checkpoints and committed immutable shards.
Uncommitted batch directories after a crash are ignored, not deleted/overwritten.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import uuid
import zlib

from .expert_cards import digest
from .metrics_cards import cohort

VERSION = "metrics-stream-intake-v1"


def split_group(group):
    bucket = int(hashlib.sha256(("expert-bc-split-v1/" + group).encode()).hexdigest(), 16) % 10
    return "train" if bucket < 7 else "validation" if bucket < 9 else "test"


@dataclass(frozen=True)
class Member:
    name: str
    size: int
    crc: int


def parse_listing(listing):
    """Parse 7z l -slt -ba, rejecting directories/encryption/non-JSON payloads."""
    members, entry = [], {}

    def finish():
        if not entry:
            return
        name = entry.get("Path", "")
        if not name.lower().endswith(".json") or entry.get("Encrypted") != "-":
            raise ValueError("unexpected archive member")
        if "D" in entry.get("Attributes", "") or entry.get("Folder") == "+":
            raise ValueError("archive directories are not supported")
        size = int(entry["Size"])
        crc = entry.get("CRC", "")
        if size < 2 or len(crc) != 8:
            raise ValueError("invalid member size/CRC")
        members.append(Member(name, size, int(crc, 16)))
        entry.clear()

    for line in listing.splitlines():
        if not line.strip():
            finish()
        else:
            key, sep, value = line.partition(" = ")
            if not sep or key in entry:
                raise ValueError("malformed archive listing")
            entry[key] = value
    finish()
    if not members or len({m.name for m in members}) != len(members):
        raise ValueError("empty/duplicate archive member listing")
    return members


class MemberStream(io.RawIOBase):
    def __init__(self, source, member, progress=lambda count: None):
        super().__init__()
        self.source, self.member, self.progress = source, member, progress
        self.remaining, self.crc = member.size, 0

    def readable(self):
        return True

    def readinto(self, target):
        count = min(len(target), self.remaining)
        if not count:
            return 0
        block = self.source.read(count)
        if not block:
            raise ValueError("truncated archive member: " + self.member.name)
        target[:len(block)] = block
        self.remaining -= len(block)
        self.crc = zlib.crc32(block, self.crc)
        self.progress(self.member.size - self.remaining)
        return len(block)

    def verify(self):
        if self.remaining or self.crc != self.member.crc:
            raise ValueError("member CRC/boundary mismatch: " + self.member.name)


def records(stream, chunk_size=65536, max_record_chars=2_000_000):
    """One bounded JSON array, including strict EOF and separator checking."""
    decoder = json.JSONDecoder(parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    buffer = ""

    def ready():
        nonlocal buffer
        buffer = buffer.lstrip()
        while not buffer:
            more = stream.read(chunk_size)
            if not more:
                raise ValueError("truncated JSON array")
            buffer = more.lstrip()

    ready()
    if buffer[0] != "[":
        raise ValueError("expected JSON array")
    buffer = buffer[1:]
    first = True
    while True:
        ready()
        if buffer[0] == "]":
            if buffer[1:].strip():
                raise ValueError("extra data after array")
            while tail := stream.read(chunk_size):
                if tail.strip():
                    raise ValueError("extra data after array")
            return
        if not first:
            if buffer[0] != ",":
                raise ValueError("missing array separator")
            buffer = buffer[1:]
            ready()
        while True:
            try:
                record, end = decoder.raw_decode(buffer)
                break
            except json.JSONDecodeError:
                if len(buffer) > max_record_chars:
                    raise ValueError("record exceeds intake bound") from None
                more = stream.read(chunk_size)
                if not more:
                    raise ValueError("invalid or truncated record") from None
                buffer += more
        if end > max_record_chars or not isinstance(record, dict):
            raise ValueError("expected bounded record object")
        buffer = buffer[end:]
        first = False
        yield record


def member_records(source, member, progress=lambda count: None):
    raw = MemberStream(source, member, progress)
    with io.TextIOWrapper(io.BufferedReader(raw, buffer_size=65536), encoding="utf-8", newline="") as text:
        yield from records(text)
        raw.verify()
    # Closing this limited stream never closes the shared decompressor pipe.


def replay_member(source, member, progress=lambda count: None):
    raw = MemberStream(source, member, progress)
    with io.BufferedReader(raw, buffer_size=1024 * 1024) as buffered:
        while buffered.read(1024 * 1024):
            pass
        raw.verify()


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class ShardWriter:
    def __init__(self, directory, kind, row_limit=1000, byte_limit=8 * 1024 * 1024):
        self.directory, self.kind = directory, kind
        self.row_limit, self.byte_limit = row_limit, byte_limit
        self.stream = self.handle = None
        self.files, self.rows, self.bytes = [], 0, 0

    def write(self, record):
        payload = (json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n").encode()
        if self.stream is not None and (self.rows >= self.row_limit or self.bytes + len(payload) > self.byte_limit):
            self.close()
        if self.stream is None:
            self.path = self.directory / f"{self.kind}-{len(self.files):04d}.jsonl.gz"
            self.handle = self.path.open("xb")
            self.stream = gzip.GzipFile(filename="", mode="wb", fileobj=self.handle, compresslevel=1, mtime=0)
            self.rows, self.bytes = 0, 0
        self.stream.write(payload)
        self.rows += 1
        self.bytes += len(payload)

    def close(self):
        if self.stream is None:
            return
        self.stream.close()
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        self.files.append({"path": self.path.name, "kind": self.kind, "rows": self.rows,
                           "bytes": self.path.stat().st_size, "sha256": file_sha(self.path)})
        self.stream = self.handle = None


class IntakeStore:
    """Transactional resume/dedup with bounded memory and raw outcome separation."""
    def __init__(self, output, contract, *, resume=False):
        self.output = Path(output)
        if self.output.exists() and not resume:
            raise FileExistsError("use --resume for an existing intake")
        if resume and not (self.output / "index.sqlite").is_file():
            raise ValueError("resume requires an existing intake index")
        self.output.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.output / "index.sqlite")
        self.active, self.writers = False, {}
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA cache_size=-16384")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS groups(id TEXT PRIMARY KEY, first_sha TEXT NOT NULL,
            cohort TEXT NOT NULL, split TEXT NOT NULL, conflicted INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS versions(id TEXT NOT NULL, sha TEXT NOT NULL, PRIMARY KEY(id,sha));
          CREATE TABLE IF NOT EXISTS batches(seq INTEGER PRIMARY KEY, next_member INTEGER NOT NULL,
            decoded_bytes INTEGER NOT NULL, stats TEXT NOT NULL, files TEXT NOT NULL);
        """)
        encoded = json.dumps(contract, sort_keys=True)
        old = self.db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
        if old is None:
            if resume:
                self.db.close()
                raise ValueError("incomplete intake contract")
            self.db.execute("INSERT INTO meta VALUES('contract',?)", (encoded,))
            self.db.commit()
        elif old[0] != encoded:
            self.db.close()
            raise ValueError("source/filter/implementation contract changed; resume refused")
        self.complete = self.db.execute("SELECT value FROM meta WHERE key='complete'").fetchone() is not None
        self._reload()

    def _reload(self):
        row = self.db.execute("SELECT seq,next_member,decoded_bytes,stats FROM batches ORDER BY seq DESC LIMIT 1").fetchone()
        self.sequence, self.next_member, self.decoded_bytes = (row[:3] if row else (-1, 0, 0))
        self.stats = {k: Counter(v) for k, v in json.loads(row[3]).items()} if row else {
            k: Counter() for k in ("counts", "excluded", "builds", "splits")}
        self.shard_bytes = sum(f["bytes"] for row in self.db.execute("SELECT files FROM batches") for f in json.loads(row[0]))

    def begin(self):
        if self.active or self.complete:
            raise ValueError("cannot begin this intake batch")
        self.directory = self.output / "batches" / f"{self.sequence + 1:06d}-{uuid.uuid4().hex}"
        self.directory.mkdir(parents=True)
        self.writers = {k: ShardWriter(self.directory, k) for k in ("audit-wins", "audit-losses", "audit-conflicts")}
        self.db.execute("BEGIN IMMEDIATE")
        self.active = True

    def accept(self, wrapped, member_index, member_name, record_index):
        if not self.active:
            raise ValueError("no active intake batch")
        counts = self.stats["counts"]
        counts["records"] += 1
        run = wrapped.get("event")
        if not isinstance(run, dict):
            # A schema change is not an innocuous exclusion of all records.
            raise ValueError("expected event wrapper")
        category, reason = cohort(run)
        counts[category] += 1
        if category == "exclude":
            self.stats["excluded"][reason] += 1
            return
        sha = digest(run)
        group = digest({"game_uuid": run["play_id"]}) if run.get("play_id") else sha
        if self.db.execute("SELECT 1 FROM versions WHERE id=? AND sha=?", (group, sha)).fetchone():
            counts["duplicate_identical"] += 1
            return
        prior = self.db.execute("SELECT first_sha,conflicted FROM groups WHERE id=?", (group,)).fetchone()
        split = split_group(group)
        if prior:
            # Never pick the winning version of a conflicted run. The index
            # invalidates the first version too; downstream loaders must join it.
            self.db.execute("UPDATE groups SET conflicted=1 WHERE id=?", (group,))
            kind = "audit-conflicts"
            counts["conflicting_versions"] += 1
            counts["conflicted_groups"] += not prior[1]
        else:
            self.db.execute("INSERT INTO groups(id,first_sha,cohort,split) VALUES(?,?,?,?)", (group, sha, category, split))
            kind = "audit-wins" if category == "win_candidate" else "audit-losses"
            counts[kind] += 1
            self.stats["splits"][category + ":" + split] += 1
            build = str(run.get("build_version"))
            if build not in self.stats["builds"] and len(self.stats["builds"]) >= 1000:
                build = "<other>"
            self.stats["builds"][build] += 1
            if category == "loss_retained_only":
                counts["loss_with_cause" if run.get("killed_by") else "loss_cause_unknown"] += 1
        self.db.execute("INSERT INTO versions VALUES(?,?)", (group, sha))
        self.writers[kind].write({"group": group, "source_sha256": sha, "split": split,
            "member_index": member_index, "source_member": member_name, "record_index": record_index,
            "cohort": category, "usage": "raw_audit_only", "event": run})

    def commit(self, next_member, decoded_bytes, before_db_commit=lambda: None):
        if not self.active or next_member <= self.next_member or decoded_bytes <= self.decoded_bytes:
            raise ValueError("checkpoint must advance completed members")
        files = []
        for writer in self.writers.values():
            writer.close()
            files.extend({**f, "path": (self.directory.relative_to(self.output) / f["path"]).as_posix()}
                         for f in writer.files)
        self.db.execute("INSERT INTO batches VALUES(?,?,?,?,?)", (self.sequence + 1, next_member, decoded_bytes,
            json.dumps(self.stats, sort_keys=True), json.dumps(files, sort_keys=True)))
        before_db_commit()  # testable crash window: durable files, uncommitted index
        self.db.commit()
        self.active = False
        self.sequence += 1
        self.next_member, self.decoded_bytes = next_member, decoded_bytes
        self.shard_bytes += sum(f["bytes"] for f in files)

    def verify_committed(self):
        """Bounded integrity pass on retained shards before resuming."""
        for (encoded,) in self.db.execute("SELECT files FROM batches ORDER BY seq"):
            for record in json.loads(encoded):
                path = (self.output / record["path"]).resolve()
                if not path.is_relative_to(self.output.resolve()):
                    raise ValueError("shard path escapes intake")
                if path.stat().st_size != record["bytes"] or file_sha(path) != record["sha256"]:
                    raise ValueError("committed shard integrity failure")

    def finish(self, expected_members, expected_bytes, decoder_exit):
        if self.active or self.next_member != expected_members or self.decoded_bytes != expected_bytes or decoder_exit != 0:
            raise ValueError("cannot certify incomplete archive")
        self.db.execute("INSERT INTO meta VALUES('complete','true')")
        self.db.commit()
        self.complete = True

    def close(self):
        try:
            for writer in self.writers.values():
                writer.close()
        finally:
            if self.active:
                self.db.rollback()
            self.db.close()
