"""Versioned, investor-specific case memory for the multimodal twin.

The index vector is built from public input text only. Historical responses are
stored as retrieval payloads, never as query or index text. The caller supplies
an encoder's vectors; this module enforces eligibility, vector integrity, and
the immutable release boundary independently of the encoder implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
from typing import Mapping, Protocol, Sequence
from uuid import uuid4


ARMS = frozenset({"T", "T+A", "T+V", "T+A+V"})
KINDS = frozenset({"turn", "outcome"})
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_RELEASE_SCHEMA = "investor_case_memory_v1"
_CASE_FIELDS = frozenset({
    "memory_id", "pitch_id", "investor_id", "broadcast_order", "arm",
    "split", "opening", "prior_public_turns", "source_sha256",
    "input_sha256", "reviewed_evidence", "kind", "situation", "concerns",
    "action", "response_excerpt", "deal_participation", "outcome_verified",
})


class MemoryContractError(ValueError):
    """A case, index, query, or release violates the retrieval contract."""


class TextEncoder(Protocol):
    """Local encoder interface; the revision must match the pinned index."""

    revision_sha256: str

    def encode(self, text: str) -> Sequence[float]: ...


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MemoryContractError("release content is not canonical JSON") from exc


def _digest(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: str, name: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise MemoryContractError(f"{name} must be a lowercase SHA-256 digest")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryContractError(f"{name} must be nonempty text")


def _broadcast_order(value: Sequence[int]) -> tuple[int, int, int]:
    if (not isinstance(value, (tuple, list)) or len(value) != 3 or
            any(isinstance(part, bool) or not isinstance(part, int) or part < 0
                for part in value)):
        raise MemoryContractError("broadcast_order must be (season, episode, segment)")
    return tuple(value)


def public_index_text(opening: str, prior_public_turns: Sequence[str]) -> str:
    """Construct the only permitted embedding text, before the response target."""
    _require_text(opening, "opening")
    if isinstance(prior_public_turns, (str, bytes)):
        raise MemoryContractError("prior_public_turns must be a sequence of turns")
    turns = tuple(prior_public_turns)
    for turn in turns:
        _require_text(turn, "prior public turn")
    return opening.strip() + "\n" + "\n".join(turn.strip() for turn in turns)


@dataclass(frozen=True)
class MemoryCase:
    memory_id: str
    pitch_id: str
    investor_id: str
    broadcast_order: tuple[int, int, int]
    arm: str
    split: str
    opening: str
    prior_public_turns: tuple[str, ...]
    source_sha256: str
    input_sha256: str
    reviewed_evidence: bool
    kind: str
    situation: str
    concerns: tuple[str, ...]
    action: str
    response_excerpt: str
    deal_participation: bool | None = None
    outcome_verified: bool = False

    def __post_init__(self) -> None:
        for name in ("memory_id", "pitch_id", "investor_id", "situation",
                     "action", "response_excerpt"):
            _require_text(getattr(self, name), name)
        _broadcast_order(self.broadcast_order)
        if self.arm not in ARMS or self.split != "train" or self.kind not in KINDS:
            raise MemoryContractError("case must use a known arm and training partition")
        if self.reviewed_evidence is not True:
            raise MemoryContractError("case evidence must be adjudicated before indexing")
        _require_digest(self.source_sha256, "source_sha256")
        _require_digest(self.input_sha256, "input_sha256")
        public_index_text(self.opening, self.prior_public_turns)
        if isinstance(self.concerns, (str, bytes)):
            raise MemoryContractError("concerns must be a sequence")
        for concern in self.concerns:
            _require_text(concern, "concern")
        if self.kind == "outcome":
            if self.outcome_verified is not True or type(self.deal_participation) is not bool:
                raise MemoryContractError("outcome memory requires a verified binary outcome")
        elif self.deal_participation is not None or self.outcome_verified is not False:
            raise MemoryContractError("turn memory cannot carry a pitch outcome")

    @property
    def index_text(self) -> str:
        return public_index_text(self.opening, self.prior_public_turns)


@dataclass(frozen=True)
class MemoryQuery:
    event_id: str
    pitch_id: str
    investor_id: str
    broadcast_order: tuple[int, int, int]
    arm: str
    opening: str
    prior_public_turns: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("event_id", "pitch_id", "investor_id"):
            _require_text(getattr(self, name), name)
        _broadcast_order(self.broadcast_order)
        if self.arm not in ARMS:
            raise MemoryContractError("unknown input arm")
        public_index_text(self.opening, self.prior_public_turns)

    @property
    def index_text(self) -> str:
        return public_index_text(self.opening, self.prior_public_turns)


def _unit_vector(values: Sequence[float], dimension: int | None = None) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise MemoryContractError("embedding must be a numeric vector")
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MemoryContractError("embedding must be a numeric vector") from exc
    if not vector or (dimension is not None and len(vector) != dimension):
        raise MemoryContractError("embedding dimension is missing or inconsistent")
    if any(not math.isfinite(value) for value in vector):
        raise MemoryContractError("embedding contains a non-finite value")
    norm = math.sqrt(math.fsum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0:
        raise MemoryContractError("embedding has zero or invalid norm")
    if abs(norm - 1.0) <= 1e-12:
        return vector
    return tuple(value / norm for value in vector)


def _publish_once(path: Path, value: object) -> None:
    """Link a fully flushed staged document into its final name exactly once."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.parent / f".{path.stem}.{uuid4().hex}.staged"
    try:
        with staged.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_bytes(value).decode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staged, path)
        except FileExistsError:
            pass
    finally:
        staged.unlink(missing_ok=True)


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    pitch_id: str
    investor_id: str
    broadcast_order: tuple[int, int, int]
    arm: str
    split: str
    kind: str
    similarity: float
    situation: str
    concerns: tuple[str, ...]
    action: str
    response_excerpt: str
    deal_participation: bool | None
    source_sha256: str
    input_sha256: str
    encoder_sha256: str
    snapshot_id: str

    @property
    def case_id(self) -> str:
        return self.memory_id


class MemorySnapshot:
    """Validated, content-addressed training index and historical payloads."""

    def __init__(self, cases: Sequence[MemoryCase], vectors: Sequence[Sequence[float]],
                 *, encoder_sha256: str, schema_sha256: str,
                 source_inventory_sha256: str, parent_snapshot_id: str | None = None):
        for name, value in (("encoder_sha256", encoder_sha256),
                            ("schema_sha256", schema_sha256),
                            ("source_inventory_sha256", source_inventory_sha256)):
            _require_digest(value, name)
        if parent_snapshot_id is not None:
            _require_digest(parent_snapshot_id, "parent_snapshot_id")
        if not cases or len(cases) != len(vectors):
            raise MemoryContractError("case and vector counts must match and be nonzero")
        pairs = list(zip(cases, vectors))
        if any(not isinstance(case, MemoryCase) for case, _ in pairs):
            raise MemoryContractError("every row must be a MemoryCase")
        pairs.sort(key=lambda pair: pair[0].memory_id)
        ids = [case.memory_id for case, _ in pairs]
        if len(set(ids)) != len(ids):
            raise MemoryContractError("duplicate memory_id in snapshot")
        self.cases = tuple(case for case, _ in pairs)
        try:
            dimension = len(pairs[0][1])
        except TypeError as exc:
            raise MemoryContractError("embedding must be a numeric vector") from exc
        self.vectors = tuple(_unit_vector(vector, dimension) for _, vector in pairs)
        self.encoder_sha256 = encoder_sha256
        self.schema_sha256 = schema_sha256
        self.source_inventory_sha256 = source_inventory_sha256
        self.parent_snapshot_id = parent_snapshot_id
        self.rows_sha256 = _digest([asdict(case) for case in self.cases])
        self.vectors_sha256 = _digest(self.vectors)
        self.index_sha256 = _digest({"rows": self.rows_sha256,
                                     "vectors": self.vectors_sha256,
                                     "encoder": encoder_sha256,
                                     "schema": schema_sha256,
                                     "source_inventory": source_inventory_sha256})
        self.snapshot_id = _digest(self._manifest_core())

    @classmethod
    def from_encoder(cls, cases: Sequence[MemoryCase], encoder: TextEncoder, *,
                     schema_sha256: str, source_inventory_sha256: str,
                     parent_snapshot_id: str | None = None) -> MemorySnapshot:
        """Embed only opening and prior public turns, excluding response payloads."""
        _require_digest(encoder.revision_sha256, "encoder revision")
        if not cases:
            raise MemoryContractError("cannot index an empty case collection")
        vectors = [encoder.encode(case.index_text) for case in cases]
        return cls(cases, vectors, encoder_sha256=encoder.revision_sha256,
                   schema_sha256=schema_sha256,
                   source_inventory_sha256=source_inventory_sha256,
                   parent_snapshot_id=parent_snapshot_id)

    def _manifest_core(self) -> dict:
        return {"release_schema": _RELEASE_SCHEMA, "row_count": len(self.cases),
                "vector_dimension": len(self.vectors[0]),
                "rows_sha256": self.rows_sha256,
                "vectors_sha256": self.vectors_sha256,
                "index_sha256": self.index_sha256,
                "encoder_sha256": self.encoder_sha256,
                "schema_sha256": self.schema_sha256,
                "source_inventory_sha256": self.source_inventory_sha256,
                "parent_snapshot_id": self.parent_snapshot_id}

    @property
    def manifest(self) -> dict:
        return {**self._manifest_core(), "snapshot_id": self.snapshot_id}

    def retrieve(self, query: MemoryQuery, query_vector: Sequence[float], *,
                 kind: str = "turn", top_k: int = 5) -> tuple[MemoryHit, ...]:
        if not isinstance(query, MemoryQuery) or kind not in KINDS:
            raise MemoryContractError("unknown query or memory kind")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise MemoryContractError("top_k must be a positive integer")
        unit_query = _unit_vector(query_vector, len(self.vectors[0]))
        ranked = []
        for case, vector in zip(self.cases, self.vectors):
            if (case.kind != kind or case.arm != query.arm or
                    case.investor_id != query.investor_id or
                    case.pitch_id == query.pitch_id or
                    case.broadcast_order >= query.broadcast_order):
                continue
            score = math.fsum(a * b for a, b in zip(unit_query, vector))
            ranked.append((score, case))
        ranked.sort(key=lambda item: (-item[0], item[1].memory_id))
        unique = []
        seen_pitches: set[str] = set()
        for score, case in ranked:
            if case.pitch_id not in seen_pitches:
                unique.append((score, case))
                seen_pitches.add(case.pitch_id)
        if kind == "outcome":
            positives = [item for item in unique if item[1].deal_participation]
            negatives = [item for item in unique if not item[1].deal_participation]
            half = top_k // 2
            selected = positives[:half] + negatives[:half]
            used = {case.memory_id for _, case in selected}
            selected.extend(item for item in unique if item[1].memory_id not in used)
            selected = sorted(selected[:top_k], key=lambda item: (-item[0], item[1].memory_id))
        else:
            selected = unique[:top_k]
        return tuple(MemoryHit(case.memory_id, case.pitch_id, case.investor_id,
                               case.broadcast_order, case.arm, case.split,
                               case.kind, score, case.situation, case.concerns,
                               case.action, case.response_excerpt,
                               case.deal_participation, case.source_sha256,
                               case.input_sha256, self.encoder_sha256,
                               self.snapshot_id) for score, case in selected)

    def retrieve_encoded(self, query: MemoryQuery, encoder: TextEncoder, *,
                         kind: str = "turn", top_k: int = 5) -> tuple[MemoryHit, ...]:
        """Encode the query's public text with the pinned encoder revision."""
        if encoder.revision_sha256 != self.encoder_sha256:
            raise MemoryContractError("query encoder differs from snapshot encoder")
        return self.retrieve(query, encoder.encode(query.index_text),
                             kind=kind, top_k=top_k)

    def to_document(self) -> dict:
        return {"manifest": self.manifest,
                "rows": [asdict(case) for case in self.cases],
                "vectors": self.vectors}

    @classmethod
    def from_document(cls, document: Mapping) -> MemorySnapshot:
        if not isinstance(document, Mapping) or set(document) != {"manifest", "rows", "vectors"}:
            raise MemoryContractError("invalid release document")
        manifest = document["manifest"]
        if (not isinstance(manifest, dict) or
                set(manifest) != {"release_schema", "row_count", "vector_dimension",
                                  "rows_sha256", "vectors_sha256", "index_sha256",
                                  "encoder_sha256", "schema_sha256",
                                  "source_inventory_sha256", "parent_snapshot_id",
                                  "snapshot_id"} or
                manifest.get("release_schema") != _RELEASE_SCHEMA):
            raise MemoryContractError("unknown release schema")
        rows = document["rows"]
        if not isinstance(rows, list):
            raise MemoryContractError("release rows must be a list")
        cases = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != _CASE_FIELDS:
                raise MemoryContractError("invalid case row in release")
            try:
                case = MemoryCase(**{**row,
                    "broadcast_order": tuple(row["broadcast_order"]),
                    "prior_public_turns": tuple(row["prior_public_turns"]),
                    "concerns": tuple(row["concerns"])})
            except (TypeError, ValueError) as exc:
                raise MemoryContractError("invalid case row in release") from exc
            cases.append(case)
        snapshot = cls(cases, document["vectors"],
                       encoder_sha256=manifest["encoder_sha256"],
                       schema_sha256=manifest["schema_sha256"],
                       source_inventory_sha256=manifest["source_inventory_sha256"],
                       parent_snapshot_id=manifest["parent_snapshot_id"])
        if snapshot.manifest != manifest:
            raise MemoryContractError("release digest or manifest mismatch")
        return snapshot


class SnapshotStore:
    """Publish immutable releases and bind a run to one release and checkpoint."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _release_path(self, snapshot_id: str) -> Path:
        _require_digest(snapshot_id, "snapshot_id")
        return self.root / "releases" / f"{snapshot_id}.json"

    def publish(self, snapshot: MemorySnapshot) -> str:
        if not isinstance(snapshot, MemorySnapshot):
            raise MemoryContractError("expected a validated MemorySnapshot")
        release_path = self._release_path(snapshot.snapshot_id)
        if snapshot.parent_snapshot_id is not None:
            self.load(snapshot.parent_snapshot_id)
        if not release_path.exists():
            _publish_once(release_path, snapshot.to_document())
        self.load(snapshot.snapshot_id)
        return snapshot.snapshot_id

    def load(self, snapshot_id: str) -> MemorySnapshot:
        source = self._release_path(snapshot_id)
        try:
            with source.open("r", encoding="utf-8") as stream:
                document = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryContractError("missing or unreadable knowledge release") from exc
        snapshot = MemorySnapshot.from_document(document)
        if snapshot.snapshot_id != snapshot_id:
            raise MemoryContractError("release filename differs from content digest")
        return snapshot

    def pin_run(self, run_id: str, snapshot_id: str, checkpoint_sha256: str) -> dict:
        _require_text(run_id, "run_id")
        _require_digest(checkpoint_sha256, "checkpoint_sha256")
        self.load(snapshot_id)
        binding = {"run_id": run_id, "snapshot_id": snapshot_id,
                   "checkpoint_sha256": checkpoint_sha256}
        binding["binding_sha256"] = _digest(binding)
        path = self.root / "runs" / f"{sha256(run_id.encode('utf-8')).hexdigest()}.json"
        if not path.exists():
            _publish_once(path, binding)
        try:
            with path.open("r", encoding="utf-8") as stream:
                existing = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise MemoryContractError("missing or unreadable run binding") from exc
        if existing != binding:
            raise MemoryContractError("run already pinned to another release or checkpoint")
        return binding
