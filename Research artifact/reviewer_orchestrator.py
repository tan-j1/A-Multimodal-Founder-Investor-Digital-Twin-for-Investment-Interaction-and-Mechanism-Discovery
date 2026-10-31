"""Version-pinned policy calls and causal founder--investor transitions.

The orchestration boundary admits reviewed founder evidence before a sealed
cutoff, queries an immutable investor-specific case index, passes a gated
multimodal prefix to a local policy service, and commits only validated turns.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Mapping, Protocol, Sequence

if __package__:
    from .reviewer_data_pipeline import validate_investor_events
    from .reviewer_dyad import (DyadContractError, FounderCue,
                                FounderTurn, InvestorTurn, MIN_RESPONSE_GAP_S,
                                PitchState, Terms, commitment)
    from .reviewer_memory import MemoryQuery, MemorySnapshot
else:
    from reviewer_data_pipeline import validate_investor_events
    from reviewer_dyad import (DyadContractError, FounderCue,
                               FounderTurn, InvestorTurn, MIN_RESPONSE_GAP_S,
                               PitchState, Terms, commitment)
    from reviewer_memory import MemoryQuery, MemorySnapshot


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_ARMS = frozenset({"T", "T+A", "T+V", "T+A+V"})
_MODALITIES = frozenset({"T", "A", "V"})
_MAX_RESPONSE_CHARS = 24_000


class OrchestrationError(ValueError):
    """A release, evidence packet, or model response violates the call contract."""


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OrchestrationError(f"{label} must be nonempty text")
    return value.strip()


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise OrchestrationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise OrchestrationError(f"{label} must be numeric seconds")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise OrchestrationError(f"{label} must be finite and nonnegative")
    return number


def _identifiers(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise OrchestrationError(f"{label} must be an array of identifiers")
    ids = tuple(value)
    if len(ids) != len(set(ids)):
        raise OrchestrationError(f"{label} contains duplicate identifiers")
    return ids


def _json_safe(value: Any, label: str) -> Any:
    try:
        commitment(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OrchestrationError(f"{label} must be finite JSON data") from exc
    return deepcopy(value)


def _strict_object(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OrchestrationError(f"{label} has an invalid field set")
    return dict(value)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OrchestrationError(f"duplicate generated field: {key}")
        result[key] = value
    return result


def _payload_object(payload: Mapping[str, Any] | str) -> dict[str, Any]:
    if isinstance(payload, str):
        if len(payload) > _MAX_RESPONSE_CHARS:
            raise OrchestrationError("generated response exceeds the size bound")
        try:
            parsed = json.loads(
                payload, object_pairs_hook=_unique_pairs,
                parse_constant=lambda _: (_ for _ in ()).throw(
                    OrchestrationError("nonfinite generated number")
                ),
            )
        except (json.JSONDecodeError, TypeError) as exc:
            raise OrchestrationError("generated response is not a JSON object") from exc
    else:
        parsed = _json_safe(dict(payload), "generated response")
    if not isinstance(parsed, dict):
        raise OrchestrationError("generated response is not a JSON object")
    return parsed


@dataclass(frozen=True)
class ObservationRef:
    """Source-linked channel record; missingness remains explicit."""

    observation_id: str
    pitch_id: str
    modality: str
    speaker: str
    start_episode_s: float
    end_episode_s: float
    source_sha256: str
    provenance: str
    quality: str
    available: bool
    investor_exposed: bool
    value: Any = None
    missing_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.observation_id, "observation_id")
        _text(self.pitch_id, "pitch_id")
        if self.modality not in _MODALITIES or self.speaker != "founder":
            raise OrchestrationError("only founder T/A/V observations are admitted")
        start = _seconds(self.start_episode_s, "observation start")
        end = _seconds(self.end_episode_s, "observation end")
        if end <= start:
            raise OrchestrationError("observation interval must have positive duration")
        _digest(self.source_sha256, "observation source")
        if self.provenance not in {"reviewed_founder_media", "model_generated_actor_state"}:
            raise OrchestrationError("observation provenance is unresolved")
        if self.quality not in {"clear", "usable", "generated"}:
            raise OrchestrationError("observation quality is not admitted")
        if type(self.available) is not bool:
            raise OrchestrationError("observation availability must be boolean")
        if type(self.investor_exposed) is not bool:
            raise OrchestrationError("investor exposure must be boolean")
        if self.available:
            if self.missing_reason is not None or self.value is None:
                raise OrchestrationError("available observation needs a value")
            if self.modality == "T":
                _text(self.value, "founder words")
            elif not isinstance(self.value, (Mapping, list)) or not self.value:
                raise OrchestrationError("available audio/video needs features")
            _json_safe(self.value, "observation value")
        elif self.value is not None or not isinstance(self.missing_reason, str):
            raise OrchestrationError("missing observation needs a reason and no value")
        elif not self.missing_reason.strip():
            raise OrchestrationError("missing observation needs a nonempty reason")

    def model_projection(self, cutoff_episode_s: float) -> dict[str, Any]:
        """The policy sees relative time and provenance, never episode position."""
        admitted = self.available and self.investor_exposed
        return {
            "observation_id": self.observation_id,
            "modality": self.modality,
            "window_seconds_before_cutoff": [
                round(cutoff_episode_s - self.start_episode_s, 6),
                round(cutoff_episode_s - self.end_episode_s, 6),
            ],
            "quality": self.quality,
            "available": admitted,
            "investor_exposed": self.investor_exposed,
            "missing_reason": (self.missing_reason if not self.available else
                               "not_investor_exposure" if not self.investor_exposed
                               else None),
            "value": _json_safe(self.value if admitted else None,
                                 "observation value"),
        }


@dataclass(frozen=True)
class PrefixPacket:
    """A learned prefix's opaque payload plus its auditable causal envelope."""

    payload: Any = field(repr=False)
    prefix_sha256: str
    observation_ids: tuple[str, ...]
    availability: Mapping[str, bool]
    gate_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        _digest(self.prefix_sha256, "prefix digest")
        _identifiers(self.observation_ids, "prefix observation IDs")
        if set(self.availability) != _MODALITIES or any(
            type(value) is not bool for value in self.availability.values()
        ):
            raise OrchestrationError("prefix availability must contain T, A, and V masks")
        if set(self.gate_weights) != _MODALITIES:
            raise OrchestrationError("prefix gate must contain T, A, and V weights")
        for modality, value in self.gate_weights.items():
            if (type(value) not in {int, float} or not math.isfinite(value) or
                value < 0 or value > 1 or
                (not self.availability[modality] and value > 1e-8)):
                raise OrchestrationError("prefix gate violates modality availability")
        if abs(sum(self.gate_weights.values()) - 1.0) > 1e-4:
            raise OrchestrationError("prefix gate weights must sum to one")


@dataclass(frozen=True)
class PolicyOutput:
    payload: Mapping[str, Any] | str
    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.payload, (str, Mapping)):
            raise OrchestrationError("policy output must be text or a JSON object")
        for label, count in (("input_tokens", self.input_tokens),
                             ("output_tokens", self.output_tokens)):
            if count is not None and (type(count) is not int or count < 0):
                raise OrchestrationError(f"{label} must be a nonnegative integer")


@dataclass(frozen=True)
class PolicyRequest:
    request_id: str
    run_id: str
    event_id: str
    snapshot_id: str
    role: str
    state_sha256: str
    checkpoint_sha256: str
    cutoff_episode_s: float
    admitted_observation_ids: tuple[str, ...]
    retrieved_memory_ids: tuple[str, ...]
    seed: int
    decoding: Mapping[str, Any]
    public_view: Mapping[str, Any]
    observations: tuple[Mapping[str, Any], ...]
    memories: tuple[Mapping[str, Any], ...]
    prefix: PrefixPacket | None = field(repr=False)
    repair_feedback: str | None = None

    def audit_envelope(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id, "run_id": self.run_id,
            "event_id": self.event_id, "snapshot_id": self.snapshot_id,
            "role": self.role, "state_sha256": self.state_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "cutoff_episode_s": self.cutoff_episode_s,
            "admitted_observation_ids": list(self.admitted_observation_ids),
            "retrieved_memory_ids": list(self.retrieved_memory_ids),
            "prefix_sha256": self.prefix.prefix_sha256 if self.prefix else None,
            "seed": self.seed, "decoding": deepcopy(dict(self.decoding)),
            "public_view": deepcopy(dict(self.public_view)),
        }


class QueryEncoder(Protocol):
    revision_sha256: str

    def encode(self, public_index_text: str) -> Sequence[float]: ...


class PrefixProjector(Protocol):
    def project(self, observations: Sequence[Mapping[str, Any]], *,
                condition: str) -> PrefixPacket: ...


class PolicyService(Protocol):
    def complete(self, request: PolicyRequest) -> PolicyOutput: ...


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    event_id: str
    pitch_id: str
    broadcast_order: tuple[int, int, int]
    condition: str
    seed: int
    investor_checkpoint_sha256: str
    founder_checkpoint_sha256: str
    decoding: Mapping[str, Any]
    max_investor_turns: int = 8
    max_repair_attempts: int = 2
    turn_memory_k: int = 5
    outcome_memory_k: int = 4

    def __post_init__(self) -> None:
        for label in ("run_id", "event_id", "pitch_id"):
            _text(getattr(self, label), label)
        if (not isinstance(self.broadcast_order, tuple) or
            len(self.broadcast_order) != 3 or any(
                type(part) is not int or part < 0 for part in self.broadcast_order
            )):
            raise OrchestrationError("broadcast_order must be (season, episode, segment)")
        if self.condition not in _ARMS:
            raise OrchestrationError("unknown modality condition")
        if type(self.seed) is not int or self.seed < 0:
            raise OrchestrationError("seed must be a nonnegative integer")
        _digest(self.investor_checkpoint_sha256, "investor checkpoint")
        _digest(self.founder_checkpoint_sha256, "founder checkpoint")
        for name, lower, upper in (("max_investor_turns", 1, 100),
                                   ("max_repair_attempts", 0, 5),
                                   ("turn_memory_k", 1, 20),
                                   ("outcome_memory_k", 1, 20)):
            value = getattr(self, name)
            if type(value) is not int or not lower <= value <= upper:
                raise OrchestrationError(f"{name} is outside its bounded range")
        if not isinstance(self.decoding, Mapping):
            raise OrchestrationError("decoding must be a mapping")
        _json_safe(dict(self.decoding), "decoding settings")


def _admit_observations(state: PitchState, refs: Sequence[ObservationRef],
                        *, cutoff_episode_s: float,
                        condition: str) -> tuple[ObservationRef, ...]:
    if not refs:
        raise OrchestrationError("investor call has no founder evidence")
    allowed = {"T"} | ({"A"} if "A" in condition else set()) | (
        {"V"} if "V" in condition else set()
    )
    seen: set[str] = set()
    admitted: list[ObservationRef] = []
    founder_windows = [turn["cue"] for turn in state.turns
                       if turn["party"] == "founder"]
    for ref in refs:
        if not isinstance(ref, ObservationRef) or ref.observation_id in seen:
            raise OrchestrationError("invalid or duplicate observation reference")
        seen.add(ref.observation_id)
        if ref.pitch_id != state.case_id:
            raise OrchestrationError("observation belongs to a different pitch")
        if ref.end_episode_s > cutoff_episode_s - MIN_RESPONSE_GAP_S + 1e-9:
            raise OrchestrationError("observation crosses the sealed response boundary")
        if not any(cue["start_episode_s"] <= ref.start_episode_s and
                   ref.end_episode_s <= cue["end_episode_s"]
                   for cue in founder_windows):
            raise OrchestrationError("observation is not inside a public founder cue")
        if ref.provenance == "reviewed_founder_media":
            if ref.quality not in {"clear", "usable"}:
                raise OrchestrationError("reviewed evidence has invalid quality")
        elif ref.quality != "generated":
            raise OrchestrationError("generated cue has invalid provenance quality")
        if ref.modality not in allowed:
            continue
        admitted.append(ref)
    if not any(ref.modality == "T" and ref.available and ref.investor_exposed
               for ref in admitted):
        raise OrchestrationError("the text anchor is unavailable")
    return tuple(admitted)


def generated_observations(state: PitchState) -> tuple[ObservationRef, ...]:
    """Bind a newly committed founder cue to source IDs for the next call."""
    if not state.turns or state.turns[-1]["party"] != "founder":
        raise OrchestrationError("generated observations require a founder turn")
    turn = state.turns[-1]
    cue = turn["cue"]
    source_sha256 = commitment({"pitch_id": state.case_id, "turn": turn})
    basis = f"{state.case_id}:turn:{turn['turn_number']}"
    channels = (("T", cue["speech_text"]),
                ("A", cue["audio_features"]),
                ("V", cue["video_features"]))
    observations = []
    for modality, value in channels:
        missing = None
        exposed = True
        if modality != "T":
            exposed = value.get("investor_exposed", True)
            missing = value.get("missing_reason")
            if value.get("available") is False and missing is None:
                missing = "unavailable_generated_channel"
            if value.get("status") in {"unknown", "not_assessed",
                                        "not_audible", "not_visible"}:
                missing = value["status"]
        available = missing is None
        observations.append(ObservationRef(
            observation_id=f"{basis}:{modality}", pitch_id=state.case_id,
            modality=modality, speaker="founder",
            start_episode_s=cue["start_episode_s"],
            end_episode_s=cue["end_episode_s"],
            source_sha256=source_sha256,
            provenance="model_generated_actor_state", quality="generated",
            available=available, investor_exposed=exposed,
            value=value if available else None, missing_reason=missing,
        ))
    return tuple(observations)


def reviewed_observations_from_event(
    event: Mapping[str, Any], *, exposure_by_observation_id: Mapping[str, bool]
) -> tuple[ObservationRef, ...]:
    """Bind normalized event channels to reviewed, exposure-aware source IDs."""
    row = validate_investor_events([event])[0]
    if not isinstance(exposure_by_observation_id, Mapping):
        raise OrchestrationError("reviewed channel exposure map is required")
    result = []
    for index, observation in enumerate(row["founder_observations"]):
        for modality, channel_name, value_key in (
            ("T", "speech", "text"),
            ("A", "audio", "features"),
            ("V", "video", "observations"),
        ):
            channel = observation[channel_name]
            observation_id = f"{row['event_id']}:O{index:04d}:{modality}"
            observed = channel["status"] == "observed"
            if modality == "T":
                exposed = True
            elif observed:
                exposed = exposure_by_observation_id.get(observation_id)
                if type(exposed) is not bool:
                    raise OrchestrationError(
                        f"missing exposure adjudication: {observation_id}"
                    )
            else:
                exposed = False
            result.append(ObservationRef(
                observation_id=observation_id, pitch_id=row["pitch_id"],
                modality=modality, speaker="founder",
                start_episode_s=observation["start_episode_s"],
                end_episode_s=observation["end_episode_s"],
                source_sha256=observation["media_sha256"],
                provenance="reviewed_founder_media",
                quality=observation["quality"], available=observed,
                investor_exposed=exposed,
                value=channel[value_key] if observed else None,
                missing_reason=None if observed else channel["status"],
            ))
    return tuple(result)


def conditioned_investor_view(state: PitchState, condition: str) -> dict[str, Any]:
    """Keep raw cues in the separately gated evidence packet."""
    if condition not in _ARMS:
        raise OrchestrationError("unknown modality condition")
    view = state.investor_view()
    for turn in view["turns"]:
        if turn["party"] != "founder":
            continue
        cue = turn["cue"]
        cue["audio_features"] = {"status": "gated_evidence_packet"}
        cue["video_features"] = {"status": "gated_evidence_packet"}
    return view


def _terms(value: Any) -> Terms | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise OrchestrationError("economic terms must be an object or null")
    if set(value) == {"cash_usd", "equity_pct", "conditions"}:
        return Terms.from_mapping(value)
    return Terms.from_training_mapping(value)


def _optional_text(value: Any, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _parse_investor(payload: Mapping[str, Any] | str,
                    *, state: PitchState, cutoff_episode_s: float,
                    observation_ids: set[str], memory_ids: set[str]) -> tuple[InvestorTurn, dict]:
    value = _strict_object(_payload_object(payload), {
        "actor", "action", "issue", "evidence_request", "response",
        "at_pitch_s", "end_pitch_s", "terms", "accepts_offer_id",
        "cited_observation_ids", "cited_memory_ids",
    }, "investor completion")
    if value["actor"] != "focal_investor":
        raise OrchestrationError("investor completion has the wrong actor")
    _optional_text(value["issue"], "investor issue")
    _optional_text(value["evidence_request"], "evidence request")
    cited_observations = _identifiers(value["cited_observation_ids"],
                                     "cited observations")
    cited_memories = _identifiers(value["cited_memory_ids"], "cited memories")
    if not set(cited_observations) <= observation_ids:
        raise OrchestrationError("investor cites evidence outside the sealed input")
    if not set(cited_memories) <= memory_ids:
        raise OrchestrationError("investor cites an unretrieved memory")
    onset = state.pitch_origin_s + _seconds(value["at_pitch_s"], "investor onset")
    end = state.pitch_origin_s + _seconds(value["end_pitch_s"], "investor end")
    if onset + 1e-9 < cutoff_episode_s:
        raise OrchestrationError("investor response begins before evidence cutoff")
    turn = InvestorTurn(
        value["action"], _text(value["response"], "investor response"),
        onset, end, _terms(value["terms"]),
        _optional_text(value["accepts_offer_id"], "accepted offer ID"),
    )
    return turn, {"issue": value["issue"],
                  "evidence_request": value["evidence_request"],
                  "cited_observation_ids": cited_observations,
                  "cited_memory_ids": cited_memories}


def _parse_founder(payload: Mapping[str, Any] | str,
                   *, state: PitchState) -> FounderTurn:
    value = _strict_object(_payload_object(payload), {
        "actor", "action", "response", "start_pitch_s", "end_pitch_s",
        "audio_features", "video_features", "disclosed_fact_ids",
        "terms", "accepts_offer_id",
    }, "founder completion")
    if value["actor"] != "founder":
        raise OrchestrationError("founder completion has the wrong actor")
    facts = _identifiers(value["disclosed_fact_ids"], "founder disclosed facts")
    if not isinstance(value["audio_features"], Mapping) or not isinstance(
        value["video_features"], Mapping
    ):
        raise OrchestrationError("founder audio/video features must be maps")
    cue = FounderCue(
        speech_text=_text(value["response"], "founder response"),
        audio_features=value["audio_features"],
        video_features=value["video_features"],
        start_episode_s=state.pitch_origin_s + _seconds(
            value["start_pitch_s"], "founder cue start"
        ),
        end_episode_s=state.pitch_origin_s + _seconds(
            value["end_pitch_s"], "founder cue end"
        ),
        disclosed_fact_ids=facts,
        provenance="model_generated_actor_state",
    )
    return FounderTurn(value["action"], cue, _terms(value["terms"]),
                       _optional_text(value["accepts_offer_id"],
                                      "accepted offer ID"))


class LocalOrchestrator:
    """One investor and one founder, with pinned memory and bounded repair."""

    def __init__(self, memory: MemorySnapshot, query_encoder: QueryEncoder,
                 prefix_projector: PrefixProjector,
                 policy_service: PolicyService):
        if not isinstance(memory, MemorySnapshot):
            raise OrchestrationError("memory must be a validated immutable snapshot")
        manifest = deepcopy(memory.manifest)
        if query_encoder.revision_sha256 != manifest["encoder_sha256"]:
            raise OrchestrationError("query encoder differs from memory release")
        self.memory = memory
        self.query_encoder = query_encoder
        self.prefix_projector = prefix_projector
        self.policy_service = policy_service
        self._manifest = manifest
        self._case_by_id = {case.memory_id: case for case in memory.cases}

    def _assert_release(self) -> None:
        if (self.memory.manifest != self._manifest or
            self.query_encoder.revision_sha256 != self._manifest["encoder_sha256"]):
            raise OrchestrationError("knowledge snapshot changed during the run")

    def _retrieve(self, state: PitchState, spec: RunSpec) -> tuple[dict, ...]:
        self._assert_release()
        turns = state.turns
        opening = turns[0]["cue"]["speech_text"]
        history = tuple(
            f"{turn['party']}: " +
            (turn["cue"]["speech_text"] if turn["party"] == "founder"
             else turn["response"])
            for turn in turns[1:]
        )
        query = MemoryQuery(spec.event_id, spec.pitch_id,
                            state.focal_investor_id, spec.broadcast_order,
                            spec.condition, opening, history)
        vector = self.query_encoder.encode(query.index_text)
        hits = (*self.memory.retrieve(query, vector, kind="turn",
                                      top_k=spec.turn_memory_k),
                *self.memory.retrieve(query, vector, kind="outcome",
                                      top_k=spec.outcome_memory_k))
        self._assert_release()
        result: list[dict] = []
        seen: set[str] = set()
        for hit in hits:
            case = self._case_by_id.get(hit.memory_id)
            if (case is None or hit.memory_id in seen or
                hit.snapshot_id != self._manifest["snapshot_id"] or
                case.pitch_id != hit.pitch_id or case.pitch_id == spec.pitch_id or
                case.broadcast_order >= spec.broadcast_order or
                case.investor_id != state.focal_investor_id or
                case.arm != spec.condition or case.split != "train" or
                case.reviewed_evidence is not True or case.kind != hit.kind or
                not math.isfinite(hit.similarity) or not -1.000001 <= hit.similarity <= 1.000001):
                raise OrchestrationError("retrieval violated release eligibility")
            seen.add(hit.memory_id)
            result.append({
                "memory_id": hit.memory_id, "source_pitch_id": hit.pitch_id,
                "kind": hit.kind, "similarity": float(hit.similarity),
                "source_sha256": case.source_sha256,
                "encoder_sha256": self._manifest["encoder_sha256"],
                "snapshot_id": hit.snapshot_id,
                "situation": hit.situation, "concerns": list(hit.concerns),
                "action": hit.action,
                "response_excerpt": hit.response_excerpt,
                "deal_participation": hit.deal_participation,
            })
        return tuple(result)

    def _prefix(self, evidence: tuple[ObservationRef, ...],
                cutoff_episode_s: float, condition: str) -> PrefixPacket:
        projected = tuple(ref.model_projection(cutoff_episode_s)
                          for ref in evidence)
        packet = self.prefix_projector.project(projected, condition=condition)
        if not isinstance(packet, PrefixPacket):
            raise OrchestrationError("prefix projector returned an invalid packet")
        ids = tuple(ref.observation_id for ref in evidence)
        masks = {modality: any(ref.modality == modality and ref.available and
                               ref.investor_exposed
                               for ref in evidence)
                 for modality in _MODALITIES}
        if set(packet.observation_ids) != set(ids) or dict(packet.availability) != masks:
            raise OrchestrationError("prefix uses evidence outside the causal gate")
        return packet

    def _request(self, *, spec: RunSpec, role: str, state: PitchState,
                 cutoff_episode_s: float, evidence: tuple[ObservationRef, ...],
                 memories: tuple[dict, ...], prefix: PrefixPacket | None,
                 attempt: int, investor_index: int,
                 repair_feedback: str | None) -> PolicyRequest:
        public_view = (conditioned_investor_view(state, spec.condition)
                       if role == "focal_investor"
                       else state.founder_view())
        state_sha256 = commitment(public_view)
        observation_ids = tuple(ref.observation_id for ref in evidence)
        memory_ids = tuple(hit["memory_id"] for hit in memories)
        seed = spec.seed + investor_index * 32 + (0 if role == "focal_investor"
                                                  else 16) + attempt
        checkpoint = (spec.investor_checkpoint_sha256
                      if role == "focal_investor"
                      else spec.founder_checkpoint_sha256)
        request_id = commitment({
            "run_id": spec.run_id, "event_id": spec.event_id,
            "snapshot_id": self._manifest["snapshot_id"], "role": role,
            "state_sha256": state_sha256, "cutoff": cutoff_episode_s,
            "observations": observation_ids, "memories": memory_ids,
            "prefix_sha256": prefix.prefix_sha256 if prefix else None,
            "checkpoint_sha256": checkpoint, "seed": seed,
            "decoding": dict(spec.decoding), "attempt": attempt,
        })
        return PolicyRequest(
            request_id=request_id, run_id=spec.run_id, event_id=spec.event_id,
            snapshot_id=self._manifest["snapshot_id"], role=role,
            state_sha256=state_sha256, checkpoint_sha256=checkpoint,
            cutoff_episode_s=cutoff_episode_s,
            admitted_observation_ids=observation_ids,
            retrieved_memory_ids=memory_ids, seed=seed,
            decoding=deepcopy(dict(spec.decoding)), public_view=public_view,
            observations=tuple(ref.model_projection(cutoff_episode_s)
                               for ref in evidence),
            memories=memories, prefix=prefix,
            repair_feedback=repair_feedback,
        )

    def _complete_turn(self, *, spec: RunSpec, role: str, state: PitchState,
                       cutoff_episode_s: float, evidence: tuple[ObservationRef, ...],
                       memories: tuple[dict, ...], prefix: PrefixPacket | None,
                       investor_index: int, trace: list[dict]) -> PitchState | None:
        feedback: str | None = None
        for attempt in range(spec.max_repair_attempts + 1):
            request = self._request(
                spec=spec, role=role, state=state,
                cutoff_episode_s=cutoff_episode_s, evidence=evidence,
                memories=memories, prefix=prefix, attempt=attempt,
                investor_index=investor_index, repair_feedback=feedback,
            )
            audit: dict[str, Any] = {"stage": "complete", **request.audit_envelope(),
                                     "attempt": attempt}
            try:
                output = self.policy_service.complete(request)
                if not isinstance(output, PolicyOutput):
                    raise OrchestrationError("policy service returned an invalid output")
                audit["output_sha256"] = commitment(output.payload)
                audit["input_tokens"] = output.input_tokens
                audit["output_tokens"] = output.output_tokens
                candidate = deepcopy(state)
                if role == "focal_investor":
                    turn, analysis = _parse_investor(
                        output.payload, state=state,
                        cutoff_episode_s=cutoff_episode_s,
                        observation_ids={ref.observation_id for ref in evidence
                                         if ref.available and ref.investor_exposed},
                        memory_ids=set(request.retrieved_memory_ids),
                    )
                    candidate.append_investor(turn,
                                              policy_input=request.audit_envelope())
                    audit["analysis_projection"] = analysis
                else:
                    turn = _parse_founder(output.payload, state=state)
                    candidate.append_founder(turn,
                                             policy_input=request.audit_envelope())
                audit["valid"] = True
                audit["state_after_sha256"] = commitment(candidate.public_view())
                audit["ledger_after_sha256"] = commitment(candidate.ledger.public())
                trace.append(audit)
                return candidate
            except (OrchestrationError, DyadContractError, TypeError,
                    ValueError, OverflowError) as exc:
                audit["valid"] = False
                audit["rejection_type"] = type(exc).__name__
                audit["rejection_reason"] = str(exc)[:500]
                trace.append(audit)
                feedback = ("Return a corrected object using only the frozen "
                            "public state, admitted evidence, available facts "
                            "and active offer: " + str(exc)[:300])
            except Exception as exc:
                audit["valid"] = False
                audit["rejection_type"] = type(exc).__name__
                audit["rejection_reason"] = "policy service error"
                trace.append(audit)
                return None
        return None

    def run(self, initial_state: PitchState,
            initial_observations: Sequence[ObservationRef], *,
            spec: RunSpec, initial_cutoff_episode_s: float,
            mode: str = "rollout", target_onset_episode_s: float | None = None
            ) -> dict[str, Any]:
        """Return a one-step response or a terminal/incomplete dyadic trace.

        ``initial_state`` must end with the observed or generated founder cue.
        Every policy call is based on a fresh frozen role view. Failed parses,
        invalid offers, and service errors leave the last committed state intact.
        """
        if mode not in {"one_step", "rollout"}:
            raise OrchestrationError("unknown simulation mode")
        if not isinstance(initial_state, PitchState) or initial_state.case_id != spec.pitch_id:
            raise OrchestrationError("run pitch does not match public state")
        if not initial_state.turns or initial_state.turns[-1]["party"] != "founder":
            raise OrchestrationError("run needs a preceding founder turn")
        cutoff = _seconds(initial_cutoff_episode_s, "initial evidence cutoff")
        if target_onset_episode_s is not None and cutoff > _seconds(
            target_onset_episode_s, "target onset"
        ):
            raise OrchestrationError("evidence cutoff follows the target onset")
        if cutoff < initial_state.turns[-1]["cue"]["end_episode_s"] + MIN_RESPONSE_GAP_S:
            raise OrchestrationError("initial cutoff does not guard founder evidence")
        state = deepcopy(initial_state)
        refs = tuple(initial_observations)
        trace: list[dict] = []

        def finish(status: str, reason: str | None = None) -> dict[str, Any]:
            return {"run_id": spec.run_id, "event_id": spec.event_id,
                    "mode": mode, "status": status, "reason": reason,
                    "snapshot_id": self._manifest["snapshot_id"],
                    "result": deepcopy(state.ledger.result),
                    "state": state.report(maximum_investor_turns=spec.max_investor_turns),
                    "trace": trace}

        for index in range(spec.max_investor_turns):
            try:
                self._assert_release()
                admitted = _admit_observations(
                    state, refs, cutoff_episode_s=cutoff,
                    condition=spec.condition,
                )
                frozen = {"stage": "freeze", "policy_role": "focal_investor",
                          "state_sha256": commitment(conditioned_investor_view(
                              state, spec.condition)),
                          "ledger_sha256": commitment(state.ledger.public()),
                          "snapshot_id": self._manifest["snapshot_id"],
                          "index_sha256": self._manifest["index_sha256"],
                          "source_inventory_sha256": self._manifest[
                              "source_inventory_sha256"],
                          "checkpoint_sha256": spec.investor_checkpoint_sha256,
                          "cutoff_episode_s": cutoff,
                          "admitted_observation_ids": [ref.observation_id
                                                       for ref in admitted],
                          "source_sha256": {ref.observation_id: ref.source_sha256
                                            for ref in admitted}}
                trace.append(frozen)
                memories = self._retrieve(state, spec)
                trace.append({"stage": "retrieve", "snapshot_id":
                              self._manifest["snapshot_id"],
                              "hits": [{key: hit[key] for key in (
                                  "memory_id", "source_pitch_id", "kind",
                                  "similarity", "source_sha256", "encoder_sha256"
                              )} for hit in memories]})
                prefix = self._prefix(admitted, cutoff, spec.condition)
                trace.append({"stage": "prefix", "prefix_sha256":
                              prefix.prefix_sha256,
                              "observation_ids": list(prefix.observation_ids),
                              "availability": dict(prefix.availability),
                              "gate_weights": dict(prefix.gate_weights)})
            except Exception as exc:
                trace.append({"stage": "prepare", "valid": False,
                              "rejection_type": type(exc).__name__,
                              "rejection_reason": "evidence or release preparation failed"})
                return finish("incomplete", "preparation_failed")

            next_state = self._complete_turn(
                spec=spec, role="focal_investor", state=state,
                cutoff_episode_s=cutoff, evidence=admitted,
                memories=memories, prefix=prefix,
                investor_index=index, trace=trace,
            )
            if next_state is None:
                return finish("incomplete", "invalid_investor_completion")
            state = next_state
            if mode == "one_step":
                return finish("completed")
            if state.ledger.result is not None:
                return finish(state.ledger.result["status"])

            trace.append({"stage": "freeze", "policy_role": "founder",
                          "state_sha256": commitment(state.founder_view()),
                          "ledger_sha256": commitment(state.ledger.public()),
                          "snapshot_id": self._manifest["snapshot_id"],
                          "checkpoint_sha256": spec.founder_checkpoint_sha256,
                          "cutoff_episode_s": state.turns[-1]["end_episode_s"],
                          "available_fact_sources": [
                              {"fact_id": fact["fact_id"],
                               "source_id": fact["source_id"],
                               "source_window_s": [fact["source_start_s"],
                                                   fact["source_end_s"]]}
                              for fact in state.factbook.available(
                                  state.turns[-1]["end_episode_s"])
                          ],
                          "admitted_observation_ids": []})
            next_state = self._complete_turn(
                spec=spec, role="founder", state=state,
                cutoff_episode_s=state.turns[-1]["end_episode_s"],
                evidence=(), memories=(), prefix=None,
                investor_index=index, trace=trace,
            )
            if next_state is None:
                return finish("incomplete", "invalid_founder_completion")
            state = next_state
            if state.ledger.result is not None:
                return finish(state.ledger.result["status"])
            try:
                refs = (*refs, *generated_observations(state))
                cutoff = state.turns[-1]["cue"]["end_episode_s"] + MIN_RESPONSE_GAP_S
            except Exception as exc:
                trace.append({"stage": "source_binding", "valid": False,
                              "rejection_type": type(exc).__name__})
                return finish("incomplete", "source_binding_failed")
        return finish("incomplete", "maximum_investor_turns_reached")
