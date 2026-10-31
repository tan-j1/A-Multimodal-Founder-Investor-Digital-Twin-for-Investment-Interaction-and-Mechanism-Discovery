"""Event-linked annotation contract for the single-investor multimodal twin.

Each measurement is one codebook facet on a source span. Two independent coders
record decisions before a third-person resolution when needed. Founder-input
codes and investor-response targets are validated in separate passes. The
projection below retains provenance while producing the normalized observation
shape consumed by ``reviewer_data_pipeline``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

try:
    from .reviewer_data_pipeline import (AUDIO_FEATURE_KEYS, DataContractError,
                                         MIN_RESPONSE_GAP_S, _validate_channel,
                                         canonical_sha256)
    from .reviewer_dyad import INVESTOR_ACTIONS
except ImportError:
    from reviewer_data_pipeline import (AUDIO_FEATURE_KEYS, DataContractError,
                                        MIN_RESPONSE_GAP_S, _validate_channel,
                                        canonical_sha256)
    from reviewer_dyad import INVESTOR_ACTIONS


@dataclass(frozen=True)
class Facet:
    kind: str  # number, integer, boolean, label, labels, text, vector
    choices: tuple[str, ...] = ()
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True)
class Code:
    domain: str  # input_text, input_audio, input_video, input_cross, target, metadata
    unit: str
    facets: Mapping[str, Facet]


def N(lower: float | None = None, upper: float | None = None) -> Facet:
    return Facet("number", lower=lower, upper=upper)


def I(lower: float | None = None, upper: float | None = None) -> Facet:
    return Facet("integer", lower=lower, upper=upper)


def L(*choices: str) -> Facet:
    return Facet("label", choices=choices)


def M(*choices: str) -> Facet:
    return Facet("labels", choices=choices)


B = Facet("boolean")
S = Facet("text")
VEC = Facet("vector")


# Table A4: fields are separated so a numerical claim and its time basis can
# carry distinct, inspectable source spans. Q codes are metadata on every row.
CODEBOOK: dict[str, Code] = {
    "T01": Code("input_text", "pitch", {
        "cash_ask_usd": N(0), "equity_offered_pct": N(0, 100),
        "implied_valuation_usd": N(0), "debt_usd": N(0),
        "royalty_rate_pct": N(0, 100), "condition": S}),
    "T02": Code("input_text", "claim", {
        "revenue_usd": N(0), "margin_pct": N(-100, 100),
        "sales_units": N(0), "customer_count": I(0), "growth_pct": N(-100),
        "time_basis": S, "market_size_usd": N(0), "competition": S,
        "novelty": S, "stage": S, "sector": S}),
    "T03": Code("input_text", "claim", {
        "patent": S, "contract": S, "order": S, "prior_funding_usd": N(0),
        "founder_experience": S, "team_role": S, "partner": S,
        "support": L("documentary", "verbal_only", "visible_demo", "none")}),
    "T04": Code("input_text", "clause", {
        "frame": M("founder_identity", "problem_solution", "analogy", "vision",
                   "scalability", "why", "how", "social_impact", "status")}),
    "T05": Code("input_text", "clause", {
        "concreteness": N(0, 1), "vagueness": N(0, 1), "hedge_count": I(0),
        "numerical_specificity": N(0, 1), "positive_count": I(0),
        "negative_count": I(0), "passionate_phrase": S}),
    "T06": Code("input_text", "turn_pair", {
        "prior_question_turn_id": I(0),
        "answer_quality": L("full", "partial", "nonanswer", "contradiction"),
        "new_evidence": B}),
    "A01": Code("input_audio", "speech_window", {
        "speaker_identity": L("founder", "cofounder"), "speaker_change": B,
        "overlap": B, "music": B, "edit": B,
        "channel_quality": L("clear", "usable", "poor")}),
    "A02": Code("input_audio", "speech_window", {
        "speech_rate_wpm": N(0), "pause_fraction": N(0, 1),
        "pause_duration_s": N(0), "response_latency_s": N(0),
        "speech_rate_change_wpm": N()}),
    "A03": Code("input_audio", "speech_window", {
        "f0_median_hz": N(0), "f0_range_hz": N(0),
        "f0_variation_hz": N(0), "voiced_fraction": N(0, 1),
        "contour": L("rising", "falling", "level", "mixed")}),
    "A04": Code("input_audio", "speech_window", {
        "global_rms_dbfs": N(upper=0), "frame_rms_dbfs_p10": N(upper=0),
        "frame_rms_dbfs_median": N(upper=0), "frame_rms_dbfs_p90": N(upper=0),
        "modulation_db": N(0), "emphasis_peak_db": N(0),
        "gain_change_db": N(), "broadcast_mix": B}),
    "A05": Code("input_audio", "speech_window", {
        "mfcc_mean": VEC, "mfcc_std": VEC,
        "prior_verified_voice_similarity": N(0, 1)}),
    "A06": Code("input_audio", "window", {
        "focus": I(1, 5), "stress": I(1, 5), "steadiness": I(1, 5),
        "warmth": I(1, 5), "clarity": I(1, 5)}),
    "V01": Code("input_video", "video_window", {
        "face_visible_share": N(0, 1), "body_visible_share": N(0, 1),
        "shot_continuity": L("continuous", "cut", "unknown"),
        "occlusion": L("none", "partial", "full", "unknown"),
        "cut_rate_per_min": N(0), "other_person_substitution": B,
        "investor_exposed": L("yes", "no", "unknown")}),
    "V02": Code("input_video", "gesture_span", {
        "gesture_type": L("depictive", "symbolic", "pointing", "beat", "absent"),
        "referent_claim": S, "gesture_present": B}),
    "V03": Code("input_video", "demo_span", {
        "object": S, "function_attempted": S,
        "visible_result": L("success", "failure", "uncertain"),
        "demo_present": B, "claim_offset_s": N()}),
    "V04": Code("input_video", "face_span", {
        "smile_candidate": B, "peak_intensity": N(0, 1),
        "duration_s": N(0), "phase": L("opening", "explanation", "demo", "negotiation"),
        "transition": L("onset", "sustain", "offset", "mixed")}),
    "V05": Code("input_video", "visible_frame", {
        "first_visible_time_s": N(0),
        "face_view": L("frontal", "three_quarter", "profile", "obscured"),
        "posture": S, "orientation": S,
        "image_quality": L("clear", "usable", "poor"),
        "perceived_confidence": I(1, 5), "perceived_trustworthiness": I(1, 5)}),
    "V06": Code("input_video", "video_window", {
        "gaze_direction": L("investor", "product", "other", "unknown"),
        "body_orientation": L("investor", "product", "other", "unknown"),
        "stance": S, "lean": L("forward", "neutral", "back", "unknown"),
        "movement": S}),
    "X01": Code("input_cross", "aligned_span", {
        "relation": L("corroborates", "contradicts", "adds_information", "unavailable"),
        "temporal_overlap_s": N(0)}),
    "Y01": Code("target", "investor_turn", {
        "action": L("question", "assessment", "offer", "counteroffer", "accept",
                    "exit", "reenter", "negotiate", "other"),
        "response_text": S}),
    "Y02": Code("target", "investor_turn", {
        "issue": S, "proof_requested": S,
        "focus": L("promotion", "prevention", "both", "none"),
        "valence": L("positive", "negative", "neutral", "mixed"),
        "linked_founder_evidence": S}),
    "Y03": Code("target", "offer_state", {
        "offer_owner": S, "cash_usd": N(0), "equity_pct": N(0, 100),
        "royalty": N(0), "contingency": S, "counter": B,
        "withdrawal": B, "acceptance_reference": S}),
    "Y04": Code("target", "pitch_end", {
        "outcome": L("accepted_focal_participation", "no_deal", "incomplete"),
        "accepted_terms": S}),
    "Q01": Code("metadata", "every_record", {
        "provenance_and_cutoff": S}),
    "Q02": Code("metadata", "every_window", {
        "missingness_and_uncertainty": S}),
}

INPUT_CODES = frozenset(k for k, spec in CODEBOOK.items() if spec.domain.startswith("input_"))
TARGET_CODES = frozenset(k for k, spec in CODEBOOK.items() if spec.domain == "target")
MISSING_REASONS = frozenset({"off_camera", "soundtrack_mix", "unidentified_speaker",
                             "edit", "weak_asr", "disputed_boundary", "not_audible",
                             "not_visible", "not_assessed", "unknown"})
SOURCE_TYPES = frozenset({"transcript", "audio", "video", "frame", "ledger"})
ABSENCE_FACETS = frozenset({("V02", "gesture_type"), ("V02", "gesture_present"),
                            ("V03", "demo_present"), ("V04", "smile_candidate")})
INPUT_KEYS = frozenset({"annotation_id", "event_id", "pitch_id", "code_id", "facet",
                        "source", "decisions", "adjudication", "links"})
SOURCE_KEYS = frozenset({"source_id", "media_sha256", "parent_media_sha256",
                         "source_type", "speaker",
                         "turn_id", "start_episode_s", "end_episode_s",
                         "start_pitch_s", "end_pitch_s", "first_available_episode_s",
                         "evidence_text", "coverage", "founder_visible", "investor_exposed"})
DECISION_KEYS = frozenset({"coder_id", "value", "availability", "missing_reason",
                           "confidence", "blind_to"})
RESOLUTION_KEYS = frozenset(DECISION_KEYS | {"resolution", "adjudicator_id", "notes"})


def _exact(value: Any, keys: frozenset[str], label: str) -> Mapping:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise DataContractError(f"{label}: expected keys {sorted(keys)}")
    return value


def _text(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if (not isinstance(value, str) or not value.strip() or len(value) > 4000 or
            any(ord(ch) < 32 and ch not in "\t\n\r" for ch in value)):
        raise DataContractError(f"{label}: nonempty printable text required")
    return value.strip()


def _number(value: Any, label: str, *, lower: float | None = None,
            upper: float | None = None, integer: bool = False) -> float | int:
    if type(value) not in (int, float) or (integer and type(value) is not int):
        raise DataContractError(f"{label}: numeric value required")
    try:
        finite = isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite or (lower is not None and value < lower) or (upper is not None and value > upper):
        raise DataContractError(f"{label}: outside finite admissible range")
    return value


def _facet_value(value: Any, facet: Facet, label: str) -> None:
    if facet.kind == "number":
        _number(value, label, lower=facet.lower, upper=facet.upper)
    elif facet.kind == "integer":
        _number(value, label, lower=facet.lower, upper=facet.upper, integer=True)
    elif facet.kind == "boolean":
        if type(value) is not bool:
            raise DataContractError(f"{label}: boolean required")
    elif facet.kind == "label":
        if value not in facet.choices:
            raise DataContractError(f"{label}: label outside codebook")
    elif facet.kind == "labels":
        if (not isinstance(value, list) or not value or
                any(not isinstance(item, str) or item not in facet.choices for item in value) or
                len(set(value)) != len(value)):
            raise DataContractError(f"{label}: nonempty unique codebook labels required")
    elif facet.kind == "text":
        _text(value, label)
    elif facet.kind == "vector":
        if (not isinstance(value, list) or not 1 <= len(value) <= 64
                or any(type(item) not in (float, int) or not isfinite(item) for item in value)):
            raise DataContractError(f"{label}: finite numeric vector required")
    else:
        raise DataContractError(f"{label}: unrecognized codebook type")


def _source(source: Any, label: str, *, domain: str, cutoff_s: float,
            target_onset_s: float | None, pitch_offset_s: float | None) -> None:
    row = _exact(source, SOURCE_KEYS, label)
    _text(row["source_id"], f"{label}.source_id")
    for digest_field in ("media_sha256", "parent_media_sha256"):
        digest = row[digest_field]
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(ch not in "0123456789abcdef" for ch in digest)):
            raise DataContractError(f"{label}: lowercase {digest_field} required")
    kind = row["source_type"]
    if not isinstance(kind, str) or kind not in SOURCE_TYPES:
        raise DataContractError(f"{label}: unknown source type")
    start = _number(row["start_episode_s"], f"{label}.start_episode_s", lower=0)
    end = _number(row["end_episode_s"], f"{label}.end_episode_s", lower=0)
    p_start = _number(row["start_pitch_s"], f"{label}.start_pitch_s", lower=0)
    p_end = _number(row["end_pitch_s"], f"{label}.end_pitch_s", lower=0)
    first = _number(row["first_available_episode_s"],
                    f"{label}.first_available_episode_s", lower=0)
    if end <= start or p_end <= p_start or first < end:
        raise DataContractError(f"{label}: invalid span or first-available time")
    if pitch_offset_s is not None and (abs(start - pitch_offset_s - p_start) > 0.05 or
                                       abs(end - pitch_offset_s - p_end) > 0.05):
        raise DataContractError(f"{label}: episode and pitch clocks disagree")
    if type(row["turn_id"]) is not int or row["turn_id"] < 0:
        raise DataContractError(f"{label}: nonnegative turn ID required")
    _text(row["speaker"], f"{label}.speaker")
    _text(row["evidence_text"], f"{label}.evidence_text")
    if not isinstance(row["coverage"], str) or row["coverage"] not in {"complete", "partial", "unknown"}:
        raise DataContractError(f"{label}: unknown coverage state")
    for field in ("founder_visible", "investor_exposed"):
        if row[field] is not None and type(row[field]) is not bool:
            raise DataContractError(f"{label}.{field}: boolean or null required")
    if domain.startswith("input_"):
        if end > cutoff_s - MIN_RESPONSE_GAP_S or first > cutoff_s:
            raise DataContractError(f"{label}: founder evidence crosses sealed cutoff")
        allowed = {"input_text": {"transcript"}, "input_audio": {"audio"},
                   "input_video": {"video", "frame"},
                   "input_cross": {"transcript", "audio", "video", "frame"}}[domain]
        if kind not in allowed:
            raise DataContractError(f"{label}: wrong source modality")
    elif domain == "target":
        if target_onset_s is None or start < target_onset_s or first < target_onset_s:
            raise DataContractError(f"{label}: target annotation precedes response onset")
        if kind not in {"transcript", "ledger"}:
            raise DataContractError(f"{label}: investor target needs transcript or ledger")


def _decision(decision: Any, label: str, *, facet: Facet, blind_to: str,
              code_id: str, facet_name: str, source: Mapping) -> None:
    row = _exact(decision, DECISION_KEYS, label)
    _text(row["coder_id"], f"{label}.coder_id")
    if row["blind_to"] != blind_to:
        raise DataContractError(f"{label}: coder blinding does not match phase")
    _number(row["confidence"], f"{label}.confidence", lower=0, upper=1)
    state = row["availability"]
    if state == "observed":
        if row["missing_reason"] is not None:
            raise DataContractError(f"{label}: observed behavior has missingness reason")
        _facet_value(row["value"], facet, f"{label}.value")
        absent = (row["value"] is False or row["value"] == "absent")
        if absent and (code_id, facet_name) in ABSENCE_FACETS and source["coverage"] != "complete":
            raise DataContractError(f"{label}: absence requires complete relevant coverage")
    elif state == "unavailable":
        if (row["value"] is not None or not isinstance(row["missing_reason"], str)
                or row["missing_reason"] not in MISSING_REASONS):
            raise DataContractError(f"{label}: unavailable evidence must have a Q02 reason")
    else:
        raise DataContractError(f"{label}: unknown availability state")


def _resolution(resolution: Any, label: str, *, facet: Facet, blind_to: str,
                code_id: str, facet_name: str, source: Mapping,
                decisions: Sequence[Mapping]) -> None:
    row = _exact(resolution, RESOLUTION_KEYS, label)
    _decision({key: row[key] for key in DECISION_KEYS}, label, facet=facet,
              blind_to=blind_to, code_id=code_id, facet_name=facet_name, source=source)
    _text(row["notes"], f"{label}.notes", nullable=True)
    pair = [(d["availability"], d["missing_reason"], d["value"]) for d in decisions]
    final = (row["availability"], row["missing_reason"], row["value"])
    if row["resolution"] == "agreement":
        if pair[0] != pair[1] or final != pair[0] or row["adjudicator_id"] is not None:
            raise DataContractError(f"{label}: agreement must preserve identical coder decisions")
        if row["coder_id"] not in {d["coder_id"] for d in decisions}:
            raise DataContractError(f"{label}: agreement resolver must be a source coder")
    elif row["resolution"] == "adjudicated":
        third = _text(row["adjudicator_id"], f"{label}.adjudicator_id")
        if (pair[0] == pair[1] or third in {d["coder_id"] for d in decisions}
                or row["coder_id"] != third or not row["notes"]):
            raise DataContractError(f"{label}: disagreement needs independent documented adjudication")
    else:
        raise DataContractError(f"{label}: unknown resolution")


def validate_annotations(records: Sequence[Mapping], *, phase: str,
                         cutoff_s: float, target_onset_s: float | None = None,
                         pitch_offset_s: float | None = None) -> list[dict]:
    """Validate a sealed input batch or separately coded investor-target batch.

    ``phase='input'`` rejects Y targets, enforces the 0.25 s guard, and requires
    both input coders to be blind to the investor reply. ``phase='target'``
    accepts Y codes only and checks that coders are blind to model predictions.
    """
    if phase not in {"input", "target"}:
        raise DataContractError("phase must be input or target")
    _number(cutoff_s, "cutoff_s", lower=0)
    if target_onset_s is not None:
        _number(target_onset_s, "target_onset_s", lower=cutoff_s)
    if phase == "target" and target_onset_s is None:
        raise DataContractError("target phase requires investor response onset")
    if pitch_offset_s is not None:
        _number(pitch_offset_s, "pitch_offset_s", lower=0)
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise DataContractError("annotations must be a sequence")
    allowed = INPUT_CODES if phase == "input" else TARGET_CODES
    blind_to = "investor_reply" if phase == "input" else "model_prediction"
    seen: set[str] = set()
    event_pitch: dict[str, str] = {}
    source_media: dict[tuple[str, str], str] = {}
    event_parent_media: dict[str, str] = {}
    validated = []
    for index, raw in enumerate(records):
        row = _exact(raw, INPUT_KEYS, f"annotation {index}")
        annotation_id = _text(row["annotation_id"], f"annotation {index}.annotation_id")
        event_id = _text(row["event_id"], f"{annotation_id}.event_id")
        pitch_id = _text(row["pitch_id"], f"{annotation_id}.pitch_id")
        if annotation_id in seen:
            raise DataContractError(f"{annotation_id}: duplicate annotation")
        seen.add(annotation_id)
        if event_id in event_pitch and event_pitch[event_id] != pitch_id:
            raise DataContractError(f"{event_id}: inconsistent pitch ID")
        event_pitch[event_id] = pitch_id
        code_id = row["code_id"]
        if not isinstance(code_id, str) or code_id not in allowed:
            raise DataContractError(f"{annotation_id}: code is outside {phase} phase")
        code = CODEBOOK[code_id]
        facet_name = row["facet"]
        if not isinstance(facet_name, str) or facet_name not in code.facets:
            raise DataContractError(f"{annotation_id}: facet outside {code_id}")
        source = row["source"]
        _source(source, f"{annotation_id}.source", domain=code.domain,
                cutoff_s=cutoff_s, target_onset_s=target_onset_s,
                pitch_offset_s=pitch_offset_s)
        digest = source["media_sha256"]
        source_key = (event_id, source["source_id"])
        if source_key in source_media and source_media[source_key] != digest:
            raise DataContractError(f"{event_id}: media digest changes for source_id")
        source_media[source_key] = digest
        parent_digest = source["parent_media_sha256"]
        if event_id in event_parent_media and event_parent_media[event_id] != parent_digest:
            raise DataContractError(f"{event_id}: parent media digest changes within event")
        event_parent_media[event_id] = parent_digest
        if (not isinstance(row["decisions"], list) or len(row["decisions"]) != 2):
            raise DataContractError(f"{annotation_id}: exactly two independent decisions required")
        decisions = row["decisions"]
        facet = code.facets[facet_name]
        for j, decision in enumerate(decisions):
            _decision(decision, f"{annotation_id}.decision{j+1}", facet=facet,
                      blind_to=blind_to, code_id=code_id,
                      facet_name=facet_name, source=source)
        if decisions[0]["coder_id"] == decisions[1]["coder_id"]:
            raise DataContractError(f"{annotation_id}: coder identities must differ")
        _resolution(row["adjudication"], f"{annotation_id}.adjudication",
                    facet=facet, blind_to=blind_to, code_id=code_id,
                    facet_name=facet_name, source=source, decisions=decisions)
        final = row["adjudication"]
        if code.domain.startswith("input_") and final["availability"] == "observed":
            if source["speaker"] not in {"founder", "cofounder"}:
                raise DataContractError(f"{annotation_id}: observed input has wrong speaker")
            if code.domain == "input_video" and source["founder_visible"] is not True:
                raise DataContractError(f"{annotation_id}: observed video has no identified founder")
            if code_id == "V01" and facet_name == "investor_exposed":
                expected = {True: "yes", False: "no", None: "unknown"}[source["investor_exposed"]]
                if final["value"] != expected:
                    raise DataContractError(f"{annotation_id}: exposure label contradicts source metadata")
        if code_id in {"Y01", "Y02"} and source["speaker"] != "focal_investor":
            raise DataContractError(f"{annotation_id}: target turn is not the focal investor")
        if code_id == "Y03" and source["speaker"] not in {"focal_investor", "public_ledger"}:
            raise DataContractError(f"{annotation_id}: terms must be a focal-investor offer or public ledger")
        if code_id == "Y04" and source["speaker"] != "public_ledger":
            raise DataContractError(f"{annotation_id}: outcome must be derived from public ledger")
        if code_id == "Y04" and source["source_type"] != "ledger":
            raise DataContractError(f"{annotation_id}: outcome requires ledger source")
        if (not isinstance(row["links"], list) or
                any(not isinstance(link, str) or not link.strip() for link in row["links"]) or
                len(row["links"]) != len(set(row["links"]))):
            raise DataContractError(f"{annotation_id}: links must be unique annotation IDs")
        validated.append(dict(row))
    by_id = {row["annotation_id"]: row for row in validated}
    for row in validated:
        if row["code_id"] != "X01":
            continue
        links = row["links"]
        if len(links) != 2 or any(link not in by_id for link in links):
            raise DataContractError(f"{row['annotation_id']}: cross-channel links must resolve")
        linked = [by_id[link] for link in links]
        if (any(item["event_id"] != row["event_id"] for item in linked) or
                {item["code_id"][0] for item in linked} not in ({"T", "V"}, {"T", "A"})):
            raise DataContractError(f"{row['annotation_id']}: link one claim to audio or video")
        source_start = row["source"]["start_episode_s"]
        source_end = row["source"]["end_episode_s"]
        if any(item["source"]["end_episode_s"] < source_start or
               item["source"]["start_episode_s"] > source_end for item in linked):
            raise DataContractError(f"{row['annotation_id']}: linked spans do not overlap")
    return validated


def policy_action_from_annotation(action_annotation: Mapping, *, cutoff_s: float,
                                  target_onset_s: float,
                                  policy_resolution: Mapping | None = None) -> str:
    """Cross the rich Y01 taxonomy to the simulator's six public actions.

    Directly supported Y01 actions retain their exact identity. For a coded
    ``reenter``, ``negotiate``, or ``other`` action, a distinct two-coder policy
    resolution must explicitly assign one of the six simulator classes. No
    default or silent rare-class collapse is permitted.
    """
    rows = validate_annotations([action_annotation], phase="target",
                                cutoff_s=cutoff_s, target_onset_s=target_onset_s)
    row = rows[0]
    if row["code_id"] != "Y01" or row["facet"] != "action":
        raise DataContractError("policy action requires a Y01.action annotation")
    final = row["adjudication"]
    if final["availability"] != "observed":
        raise DataContractError("unavailable target action cannot enter simulator")
    coded_action = final["value"]
    if policy_resolution is None:
        if coded_action not in INVESTOR_ACTIONS:
            raise DataContractError("rare Y01 action needs separately adjudicated policy class")
        return coded_action
    mapping = _exact(policy_resolution,
                     frozenset({"source_annotation_id", "decisions", "adjudication"}),
                     "policy action resolution")
    if mapping["source_annotation_id"] != row["annotation_id"]:
        raise DataContractError("policy resolution points to another Y01 annotation")
    decisions = mapping["decisions"]
    if not isinstance(decisions, list) or len(decisions) != 2:
        raise DataContractError("policy resolution requires two independent decisions")
    facet = L(*sorted(INVESTOR_ACTIONS))
    for index, decision in enumerate(decisions):
        _decision(decision, f"policy decision {index+1}", facet=facet,
                  blind_to="model_prediction", code_id="Y01",
                  facet_name="policy_action", source=row["source"])
    if decisions[0]["coder_id"] == decisions[1]["coder_id"]:
        raise DataContractError("policy resolution coders must differ")
    _resolution(mapping["adjudication"], "policy adjudication", facet=facet,
                blind_to="model_prediction", code_id="Y01",
                facet_name="policy_action", source=row["source"],
                decisions=decisions)
    resolved = mapping["adjudication"]
    if resolved["availability"] != "observed":
        raise DataContractError("policy class cannot be unavailable")
    if coded_action in INVESTOR_ACTIONS and resolved["value"] != coded_action:
        raise DataContractError("supported Y01 action must preserve its identity")
    return resolved["value"]


def _icc_absolute(pairs: Sequence[tuple[float, float]]) -> float | None:
    """Two-way absolute-agreement single-measure ICC (ICC(A,1))."""
    n, k = len(pairs), 2
    if n < 2:
        return None
    grand = sum(sum(pair) for pair in pairs) / (n * k)
    row_means = [sum(pair) / k for pair in pairs]
    col_means = [sum(pair[j] for pair in pairs) / n for j in range(k)]
    ms_row = k * sum((mean - grand) ** 2 for mean in row_means) / (n - 1)
    ms_col = n * sum((mean - grand) ** 2 for mean in col_means) / (k - 1)
    ms_error = sum((pairs[i][j] - row_means[i] - col_means[j] + grand) ** 2
                   for i in range(n) for j in range(k)) / ((n - 1) * (k - 1))
    denominator = ms_row + (k - 1) * ms_error + k * (ms_col - ms_error) / n
    return None if abs(denominator) < 1e-12 else (ms_row - ms_error) / denominator


def reliability_audit(records: Sequence[Mapping], *, phase: str, cutoff_s: float,
                      target_onset_s: float | None = None) -> dict:
    """Pairwise pre-adjudication agreement, with missingness reported separately."""
    rows = validate_annotations(records, phase=phase, cutoff_s=cutoff_s,
                                target_onset_s=target_onset_s)
    grouped: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
    missing_by_domain: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        domain = CODEBOOK[row["code_id"]].domain
        left, right = row["decisions"]
        missing_by_domain[domain].append((left["availability"], right["availability"]))
        if left["availability"] == right["availability"] == "observed":
            grouped[f"{row['code_id']}.{row['facet']}"].append((left["value"], right["value"]))
    output: dict[str, dict] = {}
    for key, pairs in sorted(grouped.items()):
        code_id, facet_name = key.split(".", 1)
        facet = CODEBOOK[code_id].facets[facet_name]
        if facet.kind in {"number", "integer"}:
            numbers = [(float(a), float(b)) for a, b in pairs]
            output[key] = {"paired_n": len(numbers),
                           "mae": sum(abs(a - b) for a, b in numbers) / len(numbers),
                           "icc_a1": _icc_absolute(numbers)}
        elif facet.kind in {"label", "boolean", "labels"}:
            classes = list(facet.choices) if facet.choices else [False, True]
            if facet.kind == "labels":
                classes = list(facet.choices)
            f1 = {}
            for label in classes:
                tp = sum(label in a and label in b if facet.kind == "labels"
                         else a == label and b == label for a, b in pairs)
                fp = sum(label not in a and label in b if facet.kind == "labels"
                         else a != label and b == label for a, b in pairs)
                fn = sum(label in a and label not in b if facet.kind == "labels"
                         else a == label and b != label for a, b in pairs)
                f1[str(label)] = None if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
            output[key] = {"paired_n": len(pairs),
                           "exact_agreement": sum(a == b for a, b in pairs) / len(pairs),
                           "per_class_f1": f1}
        else:
            output[key] = {"paired_n": len(pairs),
                           "exact_agreement": sum(a == b for a, b in pairs) / len(pairs)}
    return {"phase": phase, "records": len(rows), "facets": output,
            "availability": {domain: {
                "paired_n": len(pairs),
                "agreement": sum(a == b for a, b in pairs) / len(pairs),
                "coder_1": dict(Counter(a for a, _ in pairs)),
                "coder_2": dict(Counter(b for _, b in pairs)),
            } for domain, pairs in sorted(missing_by_domain.items())}}


_AUDIO_MAP = {
    ("A02", "pause_fraction"): "pause_fraction",
    ("A02", "speech_rate_wpm"): "speech_rate_wpm",
    ("A03", "f0_median_hz"): "f0_median_hz",
    ("A03", "voiced_fraction"): "voiced_fraction",
    ("A04", "global_rms_dbfs"): "global_rms_dbfs",
    ("A04", "frame_rms_dbfs_p10"): "frame_rms_dbfs_p10",
    ("A04", "frame_rms_dbfs_median"): "frame_rms_dbfs_median",
    ("A04", "frame_rms_dbfs_p90"): "frame_rms_dbfs_p90",
}
_REQUIRED_AUDIO = AUDIO_FEATURE_KEYS - {"pause_fraction", "f0_median_hz", "speech_rate_wpm"}


def project_adjudicated_window(records: Sequence[Mapping], *, event_id: str,
                               pitch_id: str, turn_id: int, media_sha256: str,
                               start_episode_s: float, end_episode_s: float,
                               cutoff_s: float, speech_text: str,
                               pitch_offset_s: float | None = None,
                               quality: str = "usable") -> dict:
    """Project reviewed T/A/V/X facets to a pipeline observation and sidecar.

    No unavailable audio feature is imputed. A partially measured acoustic
    channel remains ``not_assessed``; observed video is admitted only when
    founder identity and investor exposure are confirmed. The returned
    ``observation`` has exactly ``OBSERVATION_KEYS`` for the data pipeline.
    """
    rows = validate_annotations(records, phase="input", cutoff_s=cutoff_s,
                                pitch_offset_s=pitch_offset_s)
    _text(event_id, "event_id")
    _text(pitch_id, "pitch_id")
    _text(speech_text, "speech_text")
    if type(turn_id) is not int or turn_id < 0:
        raise DataContractError("turn_id must be nonnegative integer")
    _number(start_episode_s, "window start", lower=0)
    _number(end_episode_s, "window end", lower=0)
    if end_episode_s <= start_episode_s or end_episode_s > cutoff_s - MIN_RESPONSE_GAP_S:
        raise DataContractError("projected window crosses sealed cutoff")
    if quality not in {"clear", "usable"}:
        raise DataContractError("projected window has unusable quality")
    selected = [row for row in rows if row["event_id"] == event_id and
                row["pitch_id"] == pitch_id and row["source"]["turn_id"] == turn_id and
                row["source"]["start_episode_s"] >= start_episode_s and
                row["source"]["end_episode_s"] <= end_episode_s]
    selected.sort(key=lambda row: (row["source"]["start_episode_s"],
                                   row["source"]["end_episode_s"],
                                   row["code_id"], row["facet"], row["annotation_id"]))
    if not selected or not any(row["code_id"].startswith("T") for row in selected):
        raise DataContractError("window needs source-linked founder text codes")
    if (not isinstance(media_sha256, str) or len(media_sha256) != 64 or
            any(ch not in "0123456789abcdef" for ch in media_sha256)):
        raise DataContractError("projected primary media SHA-256 is malformed")
    if any(row["source"]["parent_media_sha256"] != media_sha256 for row in selected):
        raise DataContractError("projected primary media digest differs from source parent")
    coder_pairs = {tuple(d["coder_id"] for d in row["decisions"]) for row in selected}
    if len(coder_pairs) != 1:
        raise DataContractError("window uses different input coder pairs")
    third_ids = {row["adjudication"]["adjudicator_id"] for row in selected
                 if row["adjudication"]["resolution"] == "adjudicated"}
    if len(third_ids) > 1:
        raise DataContractError("window uses different adjudicators")
    for row in selected:
        if row["code_id"].startswith("T") and row["adjudication"]["availability"] == "observed":
            quote = " ".join(row["source"]["evidence_text"].casefold().split())
            transcript = " ".join(speech_text.casefold().split())
            if quote not in transcript:
                raise DataContractError(f"{row['annotation_id']}: source quote absent from founder speech")
    feature_values: dict[str, float] = {}
    feature_missing: dict[str, str] = {}
    video_descriptions: list[str] = []
    video_exposures = []
    selected_by_id = {row["annotation_id"]: row for row in selected}
    for row in selected:
        code_id, facet = row["code_id"], row["facet"]
        decision = row["adjudication"]
        key = _AUDIO_MAP.get((code_id, facet))
        if key:
            if decision["availability"] == "observed":
                numeric = float(decision["value"])
                if key in feature_values and feature_values[key] != numeric:
                    raise DataContractError(f"{event_id}: conflicting {key} measurements")
                feature_values[key] = numeric
            elif key in {"pause_fraction", "f0_median_hz", "speech_rate_wpm"}:
                feature_missing[key] = decision["missing_reason"]
        if code_id.startswith("V") or code_id == "X01":
            exposed = row["source"]["investor_exposed"]
            if code_id.startswith("V"):
                video_exposures.append(exposed)
            elif code_id == "X01":
                linked_video = [selected_by_id[link] for link in row["links"]
                                if link in selected_by_id and selected_by_id[link]["code_id"].startswith("V")]
                exposed = bool(linked_video and linked_video[0]["source"]["investor_exposed"] is True)
            if decision["availability"] == "observed" and exposed is True:
                value = decision["value"]
                rendered = ", ".join(value) if isinstance(value, list) else str(value).lower() if isinstance(value, bool) else str(value)
                video_descriptions.append(f"{code_id}.{facet}: {rendered}")
    if any(exposure is True for exposure in video_exposures) and video_descriptions:
        video = {"status": "observed", "observations": video_descriptions}
    else:
        state = "not_visible" if video_exposures and all(exposure is False for exposure in video_exposures) else "unknown" if video_exposures else "not_assessed"
        video = {"status": state, "observations": None}
    verified_speaker = any(row["code_id"] == "A01" and
                           row["facet"] == "speaker_identity" and
                           row["adjudication"]["availability"] == "observed" and
                           row["adjudication"]["value"] in {"founder", "cofounder"}
                           for row in selected)
    poor_audio = any(row["code_id"] == "A01" and
                     row["facet"] == "channel_quality" and
                     row["adjudication"]["availability"] == "observed" and
                     row["adjudication"]["value"] == "poor" for row in selected)
    if _REQUIRED_AUDIO <= feature_values.keys() and verified_speaker and not poor_audio:
        for optional in ("pause_fraction", "f0_median_hz", "speech_rate_wpm"):
            feature_values.setdefault(optional, None)
            if feature_values[optional] is None:
                feature_missing.setdefault(optional, "not_assessed")
        audio = {"status": "observed", "features": feature_values,
                 "missing": feature_missing}
    else:
        unavailable = [row["adjudication"]["missing_reason"] for row in selected
                       if row["code_id"].startswith("A") and
                       row["adjudication"]["availability"] == "unavailable"]
        status = "not_audible" if unavailable and all(reason in {
            "not_audible", "soundtrack_mix", "unidentified_speaker"} for reason in unavailable) else "not_assessed"
        audio = {"status": status, "features": None, "missing": None}
    _validate_channel(audio, "audio", "projected audio")
    _validate_channel(video, "video", "projected video")
    review = {"reviewer_ids": list(next(iter(coder_pairs))),
              "adjudicator_id": next(iter(third_ids)) if third_ids else None,
              "blind_to_target": True,
              "resolution": "adjudicated" if third_ids else "agreement",
              "speaker_confirmed": True,
              "founder_visible_confirmed": video["status"] == "observed"}
    observation = {"turn_id": turn_id, "start_episode_s": start_episode_s,
                   "end_episode_s": end_episode_s, "media_sha256": media_sha256,
                   "quality": quality, "review": review,
                   "speech": {"status": "observed", "text": speech_text},
                   "audio": audio, "video": video}
    source_links = [{"annotation_id": row["annotation_id"],
                     "code_id": row["code_id"], "facet": row["facet"],
                     "source_id": row["source"]["source_id"],
                     "span_episode_s": [row["source"]["start_episode_s"],
                                        row["source"]["end_episode_s"]],
                     "media_sha256": row["source"]["media_sha256"],
                     "parent_media_sha256": row["source"]["parent_media_sha256"],
                     "availability": row["adjudication"]["availability"],
                     "missing_reason": row["adjudication"]["missing_reason"]}
                    for row in selected]
    return {"observation": observation,
            "source_links": source_links,
            "source_links_sha256": canonical_sha256(source_links),
            "code_coverage": dict(sorted(Counter(row["code_id"] for row in selected).items()))}
