"""Data contracts and supervised rows for a founder--investor digital twin.

This module implements the data path: reviewed founder observations, time-safe
four-condition projections, separate investor and founder supervision,
completion-only labels, and dataset provenance. It accepts normalized records.

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .reviewer_dyad import FOUNDER_ACTIONS, INVESTOR_ACTIONS
else:
    from reviewer_dyad import FOUNDER_ACTIONS, INVESTOR_ACTIONS


SPLITS = ("train", "validation", "test")
CONDITIONS = ("T", "T+A", "T+V", "T+A+V")
HISTORY_SPEAKERS = frozenset({"founder", "focal_investor", "other_investor", "other"})
SPEECH_MISSING = frozenset({"unknown", "not_assessed", "not_audible"})
AUDIO_MISSING = frozenset({"unknown", "not_assessed", "not_audible"})
VIDEO_MISSING = frozenset({"unknown", "not_assessed", "not_visible"})
MIN_RESPONSE_GAP_S = 0.25
MAX_CUTOFF_LEAD_S = 30.0
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EVENT_KEYS = frozenset({"event_id", "pitch_id", "split", "target_onset_episode_s",
                        "cutoff_episode_s", "history", "founder_observations", "target"})
HISTORY_KEYS = frozenset({"turn_id", "speaker", "text", "start_episode_s", "end_episode_s"})
OBSERVATION_KEYS = frozenset({"turn_id", "start_episode_s", "end_episode_s", "media_sha256",
                              "quality", "review", "speech", "audio", "video"})
REVIEW_KEYS = frozenset({"reviewer_ids", "adjudicator_id", "blind_to_target",
                         "resolution", "speaker_confirmed", "founder_visible_confirmed"})
AUDIO_FEATURE_KEYS = frozenset({"global_rms_dbfs", "frame_rms_dbfs_p10",
                                "frame_rms_dbfs_median", "frame_rms_dbfs_p90",
                                "pause_fraction", "f0_median_hz", "voiced_fraction",
                                "speech_rate_wpm"})
FOUNDER_STEP_KEYS = frozenset({"sample_id", "pitch_id", "split", "target_start_episode_s",
                              "fact_book", "prior_dialogue", "standing_offer", "target"})
FOUNDER_FACT_KEYS = frozenset({"fact_id", "statement", "source_id", "source_start_s",
                              "source_end_s", "available_from_s", "status"})
FOUNDER_TARGET_KEYS = frozenset({"action", "response", "fact_ids", "terms",
                                 "audio_cue", "video_cue"})
TERMS_KEYS = frozenset({"cash_usd", "equity_pct", "loan_usd", "royalty",
                        "conditions", "participants", "status"})


class DataContractError(ValueError):
    """A normalized record would violate a supervision or temporal boundary."""


def _exact(value: object, keys: frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        actual = set(value) if isinstance(value, dict) else set()
        raise DataContractError(
            f"{label}: missing={sorted(keys - actual)}, unexpected={sorted(actual - keys)}"
        )
    return value


def _string(value: object, label: str, *, max_length: int = 2000) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > max_length
            or any(ord(char) < 32 and char not in "\t\n\r" for char in value)):
        raise DataContractError(f"{label}: expected nonempty printable text")
    return value.strip()


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label}: expected numeric seconds")
    try:
        result = float(value)
    except OverflowError as exc:
        raise DataContractError(f"{label}: number is too large") from exc
    if not math.isfinite(result) or result < 0:
        raise DataContractError(f"{label}: expected finite nonnegative seconds")
    return result


def _number(value: object, label: str, *, minimum: float | None = None,
            maximum: float | None = None, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataContractError(f"{label}: expected numeric value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise DataContractError(f"{label}: number is too large") from exc
    if (not math.isfinite(result) or (minimum is not None and result < minimum)
            or (maximum is not None and result > maximum)):
        raise DataContractError(f"{label}: outside admissible finite range")
    return result


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not HEX_SHA256.fullmatch(value):
        raise DataContractError(f"{label}: expected lowercase SHA-256")
    return value


def canonical_sha256(value: object) -> str:
    """Stable content identity for exported rows and model-facing projections."""
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _json_object_unique(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise DataContractError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_jsonl(path: str | Path) -> list[dict]:
    """Read normalized JSONL, rejecting duplicate keys and non-finite numbers."""
    source = Path(path)
    if not source.is_file():
        raise DataContractError(f"missing JSONL file: {source}")
    rows = []
    with source.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(
                    line, object_pairs_hook=_json_object_unique,
                    parse_constant=lambda token: (_ for _ in ()).throw(
                        DataContractError(f"non-finite JSON number: {token}")),
                )
            except (ValueError, TypeError) as exc:
                raise DataContractError(f"{source.name}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise DataContractError(f"{source.name}:{line_number}: expected object")
            rows.append(value)
    if not rows:
        raise DataContractError(f"empty JSONL file: {source}")
    return rows


def _validate_terms(value: object, label: str) -> dict | None:
    if value is None:
        return None
    terms = _exact(value, TERMS_KEYS, label)
    economic = {}
    for key in ("cash_usd", "equity_pct", "loan_usd", "royalty"):
        economic[key] = _number(terms[key], f"{label}.{key}", minimum=0,
                                maximum=100 if key == "equity_pct" else None,
                                nullable=True)
    if not any(amount is not None and amount > 0 for amount in economic.values()):
        raise DataContractError(f"{label}: offer has no positive economic component")
    if not isinstance(terms["conditions"], list) or not isinstance(terms["participants"], list):
        raise DataContractError(f"{label}: conditions and participants must be lists")
    for item in terms["conditions"]:
        _string(item, f"{label}.condition", max_length=500)
    if terms["participants"] != ["focal_investor"]:
        raise DataContractError(f"{label}: only the focal investor may offer terms")
    if terms["status"] not in {"proposed", "accepted", "withdrawn"}:
        raise DataContractError(f"{label}: unknown terms status")
    return terms


def _validate_channel(channel: object, kind: str, label: str) -> dict:
    value_key = {"speech": "text", "audio": "features", "video": "observations"}[kind]
    channel_keys = frozenset({"status", value_key, "missing"}) if kind == "audio" else frozenset({"status", value_key})
    channel = _exact(channel, channel_keys, label)
    status = channel["status"]
    if status == "observed":
        value = channel[value_key]
        if kind == "speech":
            _string(value, f"{label}.text", max_length=4000)
        elif kind == "video":
            if not isinstance(value, list) or not value:
                raise DataContractError(f"{label}: observed video needs visible descriptions")
            for item in value:
                _string(item, f"{label}.observation", max_length=500)
        else:
            features = _exact(value, AUDIO_FEATURE_KEYS, f"{label}.features")
            levels = [_number(features[key], f"{label}.{key}", maximum=0)
                      for key in ("frame_rms_dbfs_p10", "frame_rms_dbfs_median", "frame_rms_dbfs_p90")]
            if levels != sorted(levels):
                raise DataContractError(f"{label}: frame-level quantiles are unordered")
            _number(features["global_rms_dbfs"], f"{label}.global_rms_dbfs", maximum=0)
            _number(features["pause_fraction"], f"{label}.pause_fraction",
                    minimum=0, maximum=1, nullable=True)
            _number(features["f0_median_hz"], f"{label}.f0_median_hz",
                    minimum=0, nullable=True)
            _number(features["voiced_fraction"], f"{label}.voiced_fraction",
                    minimum=0, maximum=1)
            _number(features["speech_rate_wpm"], f"{label}.speech_rate_wpm",
                    minimum=0, nullable=True)
            missing = channel["missing"]
            nullable = {"pause_fraction", "f0_median_hz", "speech_rate_wpm"}
            if (not isinstance(missing, dict) or set(missing) !=
                    {key for key in nullable if features[key] is None}):
                raise DataContractError(f"{label}: feature-level missing reasons disagree")
            for reason in missing.values():
                _string(reason, f"{label}.missing reason", max_length=100)
    else:
        permitted = {"speech": SPEECH_MISSING, "audio": AUDIO_MISSING,
                     "video": VIDEO_MISSING}[kind]
        if (status not in permitted or channel[value_key] is not None or
                (kind == "audio" and channel["missing"] is not None)):
            raise DataContractError(f"{label}: missing channel needs a valid reason and null value")
    return channel


def audio_channel_from_feature_record(record: Mapping, *, event_id: str,
                                      media_sha256: str,
                                      start_episode_s: float,
                                      end_episode_s: float) -> dict:
    """Join a reviewed PCM extractor result without sending media identity to a model."""
    expected = frozenset({"schema_version", "event_id", "source_media_sha256", "pcm_sha256",
                          "window_episode_s", "sample_rate_hz", "duration_s", "features",
                          "measurement", "quality_flags", "missing"})
    result = _exact(record, expected, "audio feature record")
    if (result["schema_version"] != "reviewed_founder_pcm_features_v1" or
            result["event_id"] != event_id or result["source_media_sha256"] != media_sha256):
        raise DataContractError("audio feature identity or media digest mismatch")
    _sha256(result["pcm_sha256"], "audio PCM digest")
    window = result["window_episode_s"]
    if (not isinstance(window, list) or len(window) != 2 or
            abs(_seconds(window[0], "audio window start") - start_episode_s) > 1e-3 or
            abs(_seconds(window[1], "audio window end") - end_episode_s) > 1e-3):
        raise DataContractError("audio feature window differs from reviewed founder window")
    if (type(result["sample_rate_hz"]) is not int or
            not 8_000 <= result["sample_rate_hz"] <= 48_000 or
            abs(_seconds(result["duration_s"], "audio feature duration") -
                (end_episode_s - start_episode_s)) > 0.05):
        raise DataContractError("audio feature sample rate or duration mismatch")
    if not isinstance(result["measurement"], dict) or not isinstance(result["quality_flags"], list):
        raise DataContractError("audio feature method metadata is malformed")
    channel = {"status": "observed", "features": deepcopy(result["features"]),
               "missing": deepcopy(result["missing"])}
    _validate_channel(channel, "audio", "joined audio channel")
    return channel


def _validate_review(value: object, label: str) -> dict:
    review = _exact(value, REVIEW_KEYS, label)
    ids = review["reviewer_ids"]
    if (not isinstance(ids, list) or len(ids) != 2 or
            any(not isinstance(item, str) or not item.strip() for item in ids) or
            ids[0] == ids[1]):
        raise DataContractError(f"{label}: two independent reviewers required")
    if review["blind_to_target"] is not True:
        raise DataContractError(f"{label}: reviewers must be blind to the target")
    if review["resolution"] == "agreement":
        if review["adjudicator_id"] is not None:
            raise DataContractError(f"{label}: agreement has no adjudicator")
    elif review["resolution"] == "adjudicated":
        adjudicator = _string(review["adjudicator_id"], f"{label}.adjudicator_id",
                              max_length=80)
        if adjudicator in ids:
            raise DataContractError(f"{label}: adjudicator must be independent")
    else:
        raise DataContractError(f"{label}: unknown review resolution")
    for field in ("speaker_confirmed", "founder_visible_confirmed"):
        if type(review[field]) is not bool:
            raise DataContractError(f"{label}.{field}: expected boolean")
    return review


def validate_investor_events(events: Sequence[Mapping]) -> list[dict]:
    """Validate frozen, target-blind Season 5 event exports in episode seconds.

    Every cue window is assigned to a preceding founder turn, and each usable
    observation ends at least 0.25 seconds before the reviewed cutoff. The
    approximate transcript onset can be up to 30 seconds later than the cutoff.
    """
    if not isinstance(events, (list, tuple)) or not events:
        raise DataContractError("investor events must be a nonempty sequence")
    seen_events: set[str] = set()
    pitch_splits: dict[str, str] = {}
    pitch_media: dict[str, str] = {}
    validated = []
    for index, raw in enumerate(events):
        row = _exact(raw, EVENT_KEYS, f"event {index}")
        event_id = _string(row["event_id"], "event_id", max_length=120)
        pitch_id = _string(row["pitch_id"], "pitch_id", max_length=100)
        split = row["split"]
        if split not in SPLITS or event_id in seen_events:
            raise DataContractError(f"{event_id}: invalid split, identity, or duplicate event")
        seen_events.add(event_id)
        if pitch_id in pitch_splits and pitch_splits[pitch_id] != split:
            raise DataContractError(f"{pitch_id}: pitch crosses data splits")
        pitch_splits[pitch_id] = split
        onset = _seconds(row["target_onset_episode_s"], f"{event_id}.onset")
        cutoff = _seconds(row["cutoff_episode_s"], f"{event_id}.cutoff")
        if not 0 <= onset - cutoff <= MAX_CUTOFF_LEAD_S:
            raise DataContractError(f"{event_id}: cutoff differs from target onset")
        if not isinstance(row["history"], list) or not row["history"]:
            raise DataContractError(f"{event_id}: empty prior dialogue")
        prior_ids: set[int] = set()
        founder_turns: dict[int, tuple[float, float]] = {}
        previous_id = -1
        previous_start = -1.0
        for turn in row["history"]:
            turn = _exact(turn, HISTORY_KEYS, f"{event_id}.history turn")
            turn_id = turn["turn_id"]
            if type(turn_id) is not int or turn_id <= previous_id or turn_id in prior_ids:
                raise DataContractError(f"{event_id}: history turn IDs must increase")
            previous_id = turn_id
            prior_ids.add(turn_id)
            if turn["speaker"] not in HISTORY_SPEAKERS:
                raise DataContractError(f"{event_id}: unknown history speaker")
            _string(turn["text"], f"{event_id}.history text", max_length=4000)
            start = _seconds(turn["start_episode_s"], f"{event_id}.history start")
            end = _seconds(turn["end_episode_s"], f"{event_id}.history end")
            if end <= start or end > cutoff:
                raise DataContractError(f"{event_id}: history reaches target cutoff")
            if start < previous_start:
                raise DataContractError(f"{event_id}: history timestamps contradict turn order")
            previous_start = start
            if turn["speaker"] == "founder":
                founder_turns[turn_id] = (start, end)
        if not isinstance(row["founder_observations"], list) or not row["founder_observations"]:
            raise DataContractError(f"{event_id}: no founder observations")
        for obs_index, observation in enumerate(row["founder_observations"]):
            label = f"{event_id}.observation {obs_index}"
            obs = _exact(observation, OBSERVATION_KEYS, label)
            turn_id = obs["turn_id"]
            if type(turn_id) is not int or turn_id not in founder_turns:
                raise DataContractError(f"{label}: observation is not tied to a founder turn")
            start = _seconds(obs["start_episode_s"], f"{label}.start")
            end = _seconds(obs["end_episode_s"], f"{label}.end")
            founder_start, founder_end = founder_turns[turn_id]
            if (end <= start or start < founder_start or end > founder_end or
                    end > cutoff - MIN_RESPONSE_GAP_S):
                raise DataContractError(f"{label}: unsafe founder/response time boundary")
            digest = _sha256(obs["media_sha256"], f"{label}.media_sha256")
            if pitch_id in pitch_media and pitch_media[pitch_id] != digest:
                raise DataContractError(f"{pitch_id}: media hash changes within pitch")
            pitch_media[pitch_id] = digest
            if obs["quality"] not in {"clear", "usable"}:
                raise DataContractError(f"{label}: unusable observation")
            review = _validate_review(obs["review"], f"{label}.review")
            speech = _validate_channel(obs["speech"], "speech", f"{label}.speech")
            audio = _validate_channel(obs["audio"], "audio", f"{label}.audio")
            video = _validate_channel(obs["video"], "video", f"{label}.video")
            if (speech["status"] == "observed" or audio["status"] == "observed") and not review["speaker_confirmed"]:
                raise DataContractError(f"{label}: founder speaker not confirmed")
            if video["status"] == "observed" and not review["founder_visible_confirmed"]:
                raise DataContractError(f"{label}: visible founder not confirmed")
        target = row["target"]
        if not isinstance(target, dict) or set(target) != {"action", "concerns", "response", "terms"}:
            raise DataContractError(f"{event_id}: invalid investor target schema")
        action = _string(target["action"], f"{event_id}.target.action", max_length=40)
        if action not in INVESTOR_ACTIONS:
            raise DataContractError(f"{event_id}: unsupported investor target action")
        _string(target["response"], f"{event_id}.target.response", max_length=4000)
        if not isinstance(target["concerns"], list):
            raise DataContractError(f"{event_id}: target concerns must be a list")
        for concern in target["concerns"]:
            _string(concern, f"{event_id}.target.concern", max_length=500)
        terms = _validate_terms(target["terms"], f"{event_id}.target.terms")
        if action in {"offer", "counteroffer", "accept"}:
            expected_status = "accepted" if action == "accept" else "proposed"
            if terms is None or terms["status"] != expected_status:
                raise DataContractError(f"{event_id}: investor action/terms mismatch")
        elif terms is not None:
            raise DataContractError(f"{event_id}: non-economic investor action carries terms")
        validated.append(deepcopy(row))
    return validated


def _project_channel(channel: dict, kind: str, enabled: bool) -> dict:
    value_key = {"speech": "text", "audio": "features", "video": "observations"}[kind]
    if not enabled:
        result = {"available": False, "missing_reason": "withheld_by_condition", value_key: None}
        if kind == "audio":
            result["feature_missing"] = None
        return result
    if channel["status"] != "observed":
        result = {"available": False, "missing_reason": channel["status"], value_key: None}
        if kind == "audio":
            result["feature_missing"] = None
        return result
    result = {"available": True, "missing_reason": None,
              value_key: deepcopy(channel[value_key])}
    if kind == "audio":
        result["feature_missing"] = deepcopy(channel["missing"])
    return result


def project_investor_input(event: Mapping, condition: str) -> dict:
    """Construct a model-facing input with explicit modality availability masks."""
    if condition not in CONDITIONS:
        raise DataContractError(f"unknown modality condition: {condition}")
    row = validate_investor_events([event])[0]
    cutoff = row["cutoff_episode_s"]
    return {
        "task": "focal_investor_next_public_response",
        "conversation": [
            {"turn_id": turn["turn_id"], "speaker": turn["speaker"], "text": turn["text"]}
            for turn in row["history"]
        ],
        "founder_observations": [
            {
                "turn_id": obs["turn_id"],
                "window_seconds_before_cutoff": [
                    round(cutoff - obs["start_episode_s"], 3),
                    round(cutoff - obs["end_episode_s"], 3),
                ],
                "speech": _project_channel(obs["speech"], "speech", True),
                "audio": _project_channel(obs["audio"], "audio", "A" in condition),
                "video": _project_channel(obs["video"], "video", "V" in condition),
            }
            for obs in row["founder_observations"]
        ],
    }


def build_investor_examples(events: Sequence[Mapping]) -> dict[str, list[dict]]:
    """Create matched four-arm supervised rows on the same events and targets."""
    rows = validate_investor_events(events)
    examples: dict[str, list[dict]] = {condition: [] for condition in CONDITIONS}
    for row in rows:
        for condition in CONDITIONS:
            model_input = project_investor_input(row, condition)
            examples[condition].append({
                "sample_id": row["event_id"], "pitch_id": row["pitch_id"],
                "split": row["split"], "condition": condition,
                "input": model_input, "target": deepcopy(row["target"]),
                "input_sha256": canonical_sha256(model_input),
                "target_sha256": canonical_sha256(row["target"]),
            })
    verify_matched_conditions(examples)
    return examples


def verify_matched_conditions(examples: Mapping[str, Sequence[Mapping]]) -> dict:
    """Require common event order, target bytes, and spoken context in all arms."""
    if set(examples) != set(CONDITIONS) or not examples["T"]:
        raise DataContractError("all four nonempty modality conditions are required")
    reference = list(examples["T"])
    maximal = list(examples["T+A+V"])
    for condition in CONDITIONS:
        arm = list(examples[condition])
        if len(arm) != len(reference):
            raise DataContractError("four conditions have different event counts")
        for left, right, full in zip(reference, arm, maximal):
            if (right["condition"] != condition or
                    right["input_sha256"] != canonical_sha256(right["input"]) or
                    right["target_sha256"] != canonical_sha256(right["target"])):
                raise DataContractError("condition identity or content hash differs")
            if ((left["sample_id"], left["pitch_id"], left["split"], left["target_sha256"])
                    != (right["sample_id"], right["pitch_id"], right["split"], right["target_sha256"])):
                raise DataContractError("conditions differ in event order or supervision")
            l_input, r_input = left["input"], right["input"]
            allowed_input_keys = {"task", "conversation", "founder_observations"}
            if (set(l_input) != allowed_input_keys or set(r_input) != allowed_input_keys or
                    l_input["task"] != r_input["task"]):
                raise DataContractError("condition-specific prompt metadata or text is forbidden")
            if l_input["conversation"] != r_input["conversation"]:
                raise DataContractError("conditions have different public dialogue")
            l_obs, r_obs = l_input["founder_observations"], r_input["founder_observations"]
            if len(l_obs) != len(r_obs):
                raise DataContractError("conditions have different observation windows")
            full_obs = full["input"]["founder_observations"]
            for l_window, r_window, full_window in zip(l_obs, r_obs, full_obs):
                if (set(l_window) != {"turn_id", "window_seconds_before_cutoff", "speech", "audio", "video"}
                        or set(r_window) != set(l_window)):
                    raise DataContractError("condition-specific observation metadata is forbidden")
                if (l_window["turn_id"] != r_window["turn_id"] or
                        l_window["window_seconds_before_cutoff"] != r_window["window_seconds_before_cutoff"] or
                        l_window["speech"] != r_window["speech"]):
                    raise DataContractError("conditions differ beyond audio/video availability")
                for kind, token in (("audio", "A"), ("video", "V")):
                    expected = (full_window[kind] if token in condition else {
                        "available": False, "missing_reason": "withheld_by_condition",
                        "features" if kind == "audio" else "observations": None,
                        **({"feature_missing": None} if kind == "audio" else {}),
                    })
                    if r_window[kind] != expected:
                        raise DataContractError("modality arm differs beyond its availability mask")
    return {
        "events_per_condition": len(reference),
        "ordered_targets_sha256": canonical_sha256(
            [(row["sample_id"], row["target_sha256"]) for row in reference]
        ),
    }


def verify_target_independence(event: Mapping, condition: str) -> None:
    """Changing the label alone must leave every input byte unchanged."""
    original = project_investor_input(event, condition)
    altered = deepcopy(event)
    altered["target"]["response"] += " [changed label]"
    if original != project_investor_input(altered, condition):
        raise DataContractError("investor target influences the model-facing input")


def validate_founder_steps(steps: Sequence[Mapping], *, pitch_splits: Mapping[str, str] | None = None) -> list[dict]:
    """Validate independent founder-response rows and their fact availability."""
    if not isinstance(steps, (list, tuple)) or not steps:
        raise DataContractError("founder steps must be a nonempty sequence")
    assignments = dict(pitch_splits or {})
    seen: set[str] = set()
    validated = []
    for index, raw in enumerate(steps):
        row = _exact(raw, FOUNDER_STEP_KEYS, f"founder step {index}")
        sample_id = _string(row["sample_id"], "founder sample_id", max_length=120)
        pitch_id = _string(row["pitch_id"], "founder pitch_id", max_length=100)
        split = row["split"]
        if split not in SPLITS or sample_id in seen:
            raise DataContractError(f"{sample_id}: duplicate or invalid founder step")
        seen.add(sample_id)
        if pitch_id in assignments and assignments[pitch_id] != split:
            raise DataContractError(f"{pitch_id}: founder/investor pitch crosses data splits")
        assignments[pitch_id] = split
        start = _seconds(row["target_start_episode_s"], f"{sample_id}.target_start")
        if not isinstance(row["fact_book"], list):
            raise DataContractError(f"{sample_id}: fact_book must be a list")
        fact_ids: set[str] = set()
        for fact in row["fact_book"]:
            fact = _exact(fact, FOUNDER_FACT_KEYS, f"{sample_id}.fact")
            fact_id = _string(fact["fact_id"], f"{sample_id}.fact_id", max_length=80)
            _string(fact["statement"], f"{sample_id}.statement", max_length=1000)
            _string(fact["source_id"], f"{sample_id}.source_id", max_length=120)
            source_start = _seconds(fact["source_start_s"], f"{sample_id}.source_start_s")
            source_end = _seconds(fact["source_end_s"], f"{sample_id}.source_end_s")
            available = _seconds(fact["available_from_s"], f"{sample_id}.available_from_s")
            if source_end <= source_start or source_end > available or available > start:
                raise DataContractError(f"{sample_id}: fact source or availability crosses target")
            if fact["status"] not in {"founder_claim", "verified", "planned", "unknown"}:
                raise DataContractError(f"{sample_id}: invalid fact status")
            if fact_id in fact_ids:
                raise DataContractError(f"{sample_id}: duplicate fact_id")
            fact_ids.add(fact_id)
        if not isinstance(row["prior_dialogue"], list) or not row["prior_dialogue"]:
            raise DataContractError(f"{sample_id}: prior_dialogue must be nonempty")
        last_turn = -1
        previous_start = -1.0
        for turn in row["prior_dialogue"]:
            turn = _exact(turn, HISTORY_KEYS, f"{sample_id}.prior turn")
            turn_id = turn["turn_id"]
            if type(turn_id) is not int or turn_id <= last_turn:
                raise DataContractError(f"{sample_id}: prior turn IDs must increase")
            last_turn = turn_id
            if turn["speaker"] not in HISTORY_SPEAKERS:
                raise DataContractError(f"{sample_id}: unknown prior speaker")
            _string(turn["text"], f"{sample_id}.prior text", max_length=4000)
            first = _seconds(turn["start_episode_s"], f"{sample_id}.prior start")
            end = _seconds(turn["end_episode_s"], f"{sample_id}.prior end")
            if end <= first or end > start - MIN_RESPONSE_GAP_S:
                raise DataContractError(f"{sample_id}: prior dialogue reaches founder target")
            if first < previous_start:
                raise DataContractError(f"{sample_id}: prior timestamps contradict turn order")
            previous_start = first
        offer = _validate_terms(row["standing_offer"], f"{sample_id}.standing_offer")
        if offer is not None and offer["status"] != "proposed":
            raise DataContractError(f"{sample_id}: standing offer must be open")
        target = _exact(row["target"], FOUNDER_TARGET_KEYS, f"{sample_id}.target")
        action = _string(target["action"], f"{sample_id}.target.action", max_length=40)
        if action not in FOUNDER_ACTIONS - {"opening"}:
            raise DataContractError(f"{sample_id}: unsupported founder response action")
        _string(target["response"], f"{sample_id}.target.response", max_length=4000)
        ids = target["fact_ids"]
        if (not isinstance(ids, list) or any(not isinstance(item, str) for item in ids)
                or len(ids) != len(set(ids)) or not set(ids) <= fact_ids):
            raise DataContractError(f"{sample_id}: target cites unavailable founder facts")
        terms = _validate_terms(target["terms"], f"{sample_id}.target.terms")
        _validate_channel(target["audio_cue"], "audio", f"{sample_id}.target.audio_cue")
        _validate_channel(target["video_cue"], "video", f"{sample_id}.target.video_cue")
        if action in {"accept", "counteroffer", "decline"} and offer is None:
            raise DataContractError(f"{sample_id}: negotiation action requires an open offer")
        if action in {"accept", "counteroffer"}:
            expected_status = "accepted" if action == "accept" else "proposed"
            if terms is None or terms["status"] != expected_status:
                raise DataContractError(f"{sample_id}: founder economic action/terms mismatch")
        elif terms is not None:
            raise DataContractError(f"{sample_id}: non-economic founder action carries terms")
        if action == "accept":
            if _economics(offer) != _economics(terms):
                raise DataContractError(f"{sample_id}: acceptance must match a standing offer")
        validated.append(deepcopy(row))
    return validated


def _economics(terms: Mapping) -> tuple:
    return tuple(terms[key] for key in ("cash_usd", "equity_pct", "loan_usd", "royalty", "conditions"))


def build_founder_examples(steps: Sequence[Mapping], *, investor_events: Sequence[Mapping] = ()) -> list[dict]:
    """Build founder-actor supervision independently of investor-response rows."""
    pitch_splits = {event["pitch_id"]: event["split"] for event in validate_investor_events(investor_events)} if investor_events else {}
    rows = validate_founder_steps(steps, pitch_splits=pitch_splits)
    examples = []
    for row in rows:
        actor_input = {
            "task": "founder_next_public_response",
            "fact_book": deepcopy(row["fact_book"]),
            "conversation": [
                {"turn_id": turn["turn_id"], "speaker": turn["speaker"], "text": turn["text"]}
                for turn in row["prior_dialogue"]
            ],
            "standing_offer": deepcopy(row["standing_offer"]),
        }
        examples.append({
            "sample_id": row["sample_id"], "pitch_id": row["pitch_id"],
            "split": row["split"], "input": actor_input,
            "target": deepcopy(row["target"]),
            "input_sha256": canonical_sha256(actor_input),
            "target_sha256": canonical_sha256(row["target"]),
        })
    return examples


def assert_disjoint_studies(multimodal_pitches: Sequence[str],
                            multi_investor_pitches: Sequence[str]) -> dict:
    """Keep confirmatory pitch IDs separate across the two research programs."""
    if (isinstance(multimodal_pitches, (str, bytes)) or
            isinstance(multi_investor_pitches, (str, bytes))):
        raise DataContractError("study pitch IDs must be sequences, not strings")
    first = {_string(value, "multimodal pitch ID", max_length=100) for value in multimodal_pitches}
    second = {_string(value, "multi-investor pitch ID", max_length=100) for value in multi_investor_pitches}
    overlap = first & second
    if overlap:
        raise DataContractError(f"confirmatory studies share pitch IDs: {sorted(overlap)}")
    return {"multimodal_pitches": len(first), "multi_investor_pitches": len(second),
            "shared_pitches": 0}


def completion_only_token_row(prompt_ids: Sequence[int], completion_ids: Sequence[int],
                              *, max_length: int) -> dict:
    """Append a completion and mask all prompt tokens from causal LM loss."""
    if type(max_length) is not int or max_length < 1:
        raise DataContractError("max_length must be a positive integer")
    if (not prompt_ids or not completion_ids or
            any(type(token) is not int or token < 0 for token in (*prompt_ids, *completion_ids))):
        raise DataContractError("token IDs must be nonempty nonnegative integer sequences")
    if len(prompt_ids) + len(completion_ids) > max_length:
        raise DataContractError("context overflow; silent truncation is forbidden")
    tokens = list(prompt_ids) + list(completion_ids)
    labels = [-100] * len(prompt_ids) + list(completion_ids)
    return {"input_ids": tokens, "attention_mask": [1] * len(tokens), "labels": labels}


def coverage_audit(investor_events: Sequence[Mapping],
                   founder_steps: Sequence[Mapping] = ()) -> dict:
    """Summarize pitch partitions, channel availability, and content identities."""
    events = validate_investor_events(investor_events)
    pitch_splits = {row["pitch_id"]: row["split"] for row in events}
    founders = validate_founder_steps(founder_steps, pitch_splits=pitch_splits) if founder_steps else []
    by_split: dict[str, dict] = {}
    for split in SPLITS:
        selected = [row for row in events if row["split"] == split]
        counts = {channel: Counter(obs[channel]["status"]
                                   for row in selected for obs in row["founder_observations"])
                  for channel in ("speech", "audio", "video")}
        by_split[split] = {
            "investor_events": len(selected),
            "investor_pitches": len({row["pitch_id"] for row in selected}),
            "founder_examples": sum(row["split"] == split for row in founders),
            "founder_pitches": len({row["pitch_id"] for row in founders if row["split"] == split}),
            "channels": {name: dict(sorted(count.items())) for name, count in counts.items()},
        }
    examples = build_investor_examples(events)
    return {
        "schema": "reviewed_founder_turns_v1",
        "splits": by_split,
        "source_records_sha256": canonical_sha256({"investor_events": events, "founder_steps": founders}),
        "condition_input_sha256": {
            name: canonical_sha256([row["input_sha256"] for row in examples[name]])
            for name in CONDITIONS
        },
        "matched_targets": verify_matched_conditions(examples),
    }
