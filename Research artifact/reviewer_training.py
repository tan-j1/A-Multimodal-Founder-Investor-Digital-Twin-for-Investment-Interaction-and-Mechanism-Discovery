"""Matched supervision and inference bindings for one focal-investor twin.

The model-facing rows are normalized, time-sealed records. Model-native
token batches, LoRA optimization, and multimodal fusion are specified in reviewer_model.
Fitted policies are supplied through an inference interface and bound to
immutable source, split, initialization, and model commitments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from typing import Protocol

if __package__:
    from . import reviewer_data_pipeline as data
    from . import reviewer_dyad as dyad
    from . import reviewer_evaluation as evaluation
else:
    import reviewer_data_pipeline as data
    import reviewer_dyad as dyad
    import reviewer_evaluation as evaluation


SCHEMA_VERSION = "reviewer_policy_binding_v2"
BOUND_FIELDS = (
    "source_records_sha256", "matched_targets_sha256", "initialization_sha256",
    "train_ids", "validation_ids", "train_pitches", "validation_pitches",
)


class TrainingContractError(ValueError):
    """A supervision row, fitted-policy binding, or prediction is invalid."""


def _hash(value: object) -> str:
    packed = json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(packed).hexdigest()


def _digest(value: object, label: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(char not in "0123456789abcdef" for char in value)):
        raise TrainingContractError(f"{label} must be a SHA-256 digest")
    return value


def prepare_matched_investor_rows(investor_events: Sequence[Mapping]) -> dict:
    """Seal identical event order and targets across T, T+A, T+V, and T+A+V."""
    events = data.validate_investor_events(investor_events)
    arms = data.build_investor_examples(events)
    matched = data.verify_matched_conditions(arms)
    by_split: dict[str, set[str]] = {split: set() for split in data.SPLITS}
    for row in events:
        by_split[row["split"]].add(row["pitch_id"])
    for left in data.SPLITS:
        for right in data.SPLITS:
            if left != right and by_split[left] & by_split[right]:
                raise TrainingContractError("pitch crosses frozen data partitions")
    return {
        "conditions": arms,
        "manifest": {
            "schema_version": SCHEMA_VERSION,
            "source_records_sha256": data.coverage_audit(events)["source_records_sha256"],
            "matched_targets_sha256": matched["ordered_targets_sha256"],
            "events_per_condition": matched["events_per_condition"],
            "pitch_ids_by_split": {key: sorted(value) for key, value in by_split.items()},
        },
    }


def prepare_founder_rows(founder_steps: Sequence[Mapping], *,
                         investor_events: Sequence[Mapping] = ()) -> dict:
    """Keep actor-response supervision distinct from investor-response rows."""
    steps = data.validate_founder_steps(founder_steps)
    rows = data.build_founder_examples(steps, investor_events=investor_events)
    return {"rows": rows, "source_records_sha256": _hash(steps),
            "ordered_targets_sha256": _hash(
                [(row["sample_id"], row["target_sha256"]) for row in rows])}


class StructuredPredictor(Protocol):
    """A fitted local model with its native tokenizer and generation policy."""

    def predict(self, model_input: Mapping) -> Mapping: ...


@dataclass(frozen=True)
class FittedArm:
    condition: str
    backend: StructuredPredictor
    manifest: Mapping

    def __post_init__(self) -> None:
        if self.condition not in (*data.CONDITIONS, "founder_actor"):
            raise TrainingContractError("unknown policy condition")
        if not callable(getattr(self.backend, "predict", None)):
            raise TrainingContractError("fitted backend needs predict(model_input)")
        if not isinstance(self.manifest, Mapping) or self.manifest.get("condition") != self.condition:
            raise TrainingContractError("policy manifest has a different condition")
        _digest(self.manifest.get("model_state_sha256"), "model_state_sha256")
        _digest(self.manifest.get("source_records_sha256"), "source_records_sha256")

    def predict(self, model_input: Mapping) -> dict:
        if not isinstance(model_input, Mapping):
            raise TrainingContractError("model input must be a mapping")
        if set(model_input) & {"target", "gold", "label", "condition", "event_id", "pitch_id"}:
            raise TrainingContractError("supervision metadata entered model input")
        input_sha256 = _hash(model_input)
        result = self.backend.predict(deepcopy(model_input))
        if not isinstance(result, Mapping) or not isinstance(result.get("target"), Mapping):
            raise TrainingContractError("model did not return a structured target")
        if result.get("input_sha256", input_sha256) != input_sha256:
            raise TrainingContractError("model input commitment changed")
        response = dict(result)
        response["target"] = deepcopy(dict(result["target"]))
        response["input_sha256"] = input_sha256
        response["checkpoint_model_sha256"] = self.manifest["model_state_sha256"]
        for key in ("selection_score", "mean_completion_nll"):
            if key in response and (isinstance(response[key], bool) or
                                    not isinstance(response[key], (int, float)) or
                                    not math.isfinite(response[key])):
                raise TrainingContractError(f"{key} must be finite when reported")
        return response


def verify_fitted_arms(arms: Mapping[str, FittedArm]) -> dict:
    """Check matched rows, frozen splits, and shared base initialization."""
    if set(arms) != set(data.CONDITIONS):
        raise TrainingContractError("the four modality conditions are required")
    reference = arms["T"].manifest
    for field in BOUND_FIELDS[:3]:
        _digest(reference.get(field), field)
    for field in BOUND_FIELDS[3:]:
        value = reference.get(field)
        if (not isinstance(value, (list, tuple)) or not value or
                any(not isinstance(item, str) or not item for item in value) or
                len(value) != len(set(value))):
            raise TrainingContractError(f"{field} must contain unique identifiers")
    for condition in data.CONDITIONS:
        arm = arms[condition]
        if arm.condition != condition:
            raise TrainingContractError("policy condition and arm key differ")
        for field in BOUND_FIELDS:
            if arm.manifest.get(field) != reference.get(field):
                raise TrainingContractError(f"four arms differ in {field}")
    if (set(reference["train_pitches"]) & set(reference["validation_pitches"]) or
            set(reference["train_ids"]) & set(reference["validation_ids"])):
        raise TrainingContractError("training and validation partitions overlap")
    return {field: deepcopy(reference[field]) for field in BOUND_FIELDS}


def export_heldout_predictions(arms: Mapping[str, FittedArm],
                               investor_events: Sequence[Mapping],
                               *, split: str = "test") -> tuple[list[dict], list[dict]]:
    """Export matched structured turns for the evaluation module."""
    if split != "test" or set(arms) != set(data.CONDITIONS):
        raise TrainingContractError("held-out export requires four arms and the test split")
    verify_fitted_arms(arms)
    events = data.validate_investor_events(investor_events)
    selected = [row for row in events if row["split"] == split]
    fit_events = [row for row in events if row["split"] in {"train", "validation"}]
    if not selected:
        raise TrainingContractError("no held-out investor events")
    expected_source = data.coverage_audit(fit_events)["source_records_sha256"]
    selected_ids = {row["event_id"] for row in selected}
    for condition, arm in arms.items():
        manifest = arm.manifest
        if (arm.condition != condition or manifest.get("source_records_sha256") != expected_source or
                selected_ids & set(manifest.get("train_ids", [])) or
                selected_ids & set(manifest.get("validation_ids", []))):
            raise TrainingContractError("held-out event or source entered fitted arm")
    examples = data.build_investor_examples(selected)
    gold = [
        {"event_id": row["event_id"], "pitch_id": row["pitch_id"],
         "action": row["target"]["action"], "concerns": deepcopy(row["target"]["concerns"]),
         "response": row["target"]["response"], "terms": deepcopy(row["target"]["terms"])}
        for row in selected
    ]
    predictions = []
    for condition in data.CONDITIONS:
        for row in examples[condition]:
            result = arms[condition].predict(row["input"])
            if result["input_sha256"] != row["input_sha256"]:
                raise TrainingContractError("held-out input commitment changed at inference")
            target = result["target"]
            if not isinstance(target, dict) or set(target) != {"action", "concerns", "terms", "response"}:
                raise TrainingContractError("investor backend returned an invalid turn target")
            predictions.append({
                "event_id": row["sample_id"], "condition": condition,
                "action": target["action"], "concerns": deepcopy(target["concerns"]),
                "terms": deepcopy(target["terms"]), "response": target["response"],
                "input_sha256": result["input_sha256"],
            })
    if {(row["event_id"], row["condition"]) for row in predictions} != {
            (row["event_id"], condition) for row in selected for condition in data.CONDITIONS}:
        raise TrainingContractError("held-out event/condition pairs are incomplete")
    return gold, predictions


def score_heldout_predictions(arms: Mapping[str, FittedArm],
                              investor_events: Sequence[Mapping]) -> dict:
    gold, predictions = export_heldout_predictions(arms, investor_events)
    return evaluation.score_matched_turns(gold, predictions)


def _dyad_terms(value: object) -> dyad.Terms | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TrainingContractError("policy terms must be an object or null")
    try:
        return dyad.Terms.from_training_mapping(value)
    except dyad.DyadContractError as exc:
        raise TrainingContractError(str(exc)) from exc


def project_dyad_investor_view(view: Mapping, *, condition: str,
                               response_delay_s: float = 0.5) -> dict:
    """Project a live dyad view to the frozen four-arm investor input schema."""
    if condition not in data.CONDITIONS or not isinstance(view, Mapping):
        raise TrainingContractError("invalid dyad investor projection request")
    turns = view.get("turns")
    if not isinstance(turns, list) or not turns or turns[-1].get("party") != "founder":
        raise TrainingContractError("investor projection requires a latest founder turn")
    if (isinstance(response_delay_s, bool) or not isinstance(response_delay_s, (int, float))
            or not math.isfinite(response_delay_s) or response_delay_s < data.MIN_RESPONSE_GAP_S):
        raise TrainingContractError("response delay violates the pre-response gap")
    cutoff = float(turns[-1]["cue"]["end_pitch_s"]) + response_delay_s
    conversation = []
    observations = []
    for turn in turns:
        number = turn["turn_number"]
        party = turn["party"]
        if party == "founder":
            cue = turn["cue"]
            conversation.append({"turn_id": number, "speaker": "founder",
                                 "text": cue["speech_text"]})
            audio_cue = cue["audio_features"]
            if isinstance(audio_cue, Mapping) and isinstance(audio_cue.get("features"), Mapping):
                audio = {"available": True, "missing_reason": None,
                         "features": deepcopy(audio_cue["features"]),
                         "feature_missing": deepcopy(audio_cue.get("feature_missing", {}))}
            else:
                audio = {"available": False,
                         "missing_reason": str(audio_cue.get("missing_reason", "unknown"))
                         if isinstance(audio_cue, Mapping) else "unknown",
                         "features": None, "feature_missing": None}
            video_cue = cue["video_features"]
            if isinstance(video_cue, Mapping) and isinstance(video_cue.get("observations"), list):
                video = {"available": True, "missing_reason": None,
                         "observations": deepcopy(video_cue["observations"])}
            else:
                video = {"available": False,
                         "missing_reason": str(video_cue.get("missing_reason", "unknown"))
                         if isinstance(video_cue, Mapping) else "unknown",
                         "observations": None}
            if "A" not in condition:
                audio = {"available": False, "missing_reason": "withheld_by_condition",
                         "features": None, "feature_missing": None}
            if "V" not in condition:
                video = {"available": False, "missing_reason": "withheld_by_condition",
                         "observations": None}
            observations.append({
                "turn_id": number,
                "window_seconds_before_cutoff": [
                    round(cutoff - float(cue["start_pitch_s"]), 3),
                    round(cutoff - float(cue["end_pitch_s"]), 3),
                ],
                "speech": {"available": True, "missing_reason": None,
                           "text": cue["speech_text"]},
                "audio": audio, "video": video,
            })
        elif party == "focal_investor":
            conversation.append({"turn_id": number, "speaker": "focal_investor",
                                 "text": turn["response"]})
        else:
            raise TrainingContractError("dyad view has a party outside the one-investor system")
    return {"task": "focal_investor_next_public_response",
            "conversation": conversation, "founder_observations": observations}


def project_dyad_founder_view(view: Mapping) -> dict:
    """Project prior dialogue, available facts, and standing offer for actor inference."""
    if not isinstance(view, Mapping) or not isinstance(view.get("turns"), list):
        raise TrainingContractError("invalid dyad founder view")
    turns = view["turns"]
    if not turns or turns[-1].get("party") != "focal_investor":
        raise TrainingContractError("founder projection requires a latest investor turn")
    facts = view.get("available_founder_facts")
    if not isinstance(facts, list):
        raise TrainingContractError("founder view has no available fact book")
    conversation = []
    for turn in turns:
        party = turn["party"]
        if party not in {"founder", "focal_investor"}:
            raise TrainingContractError("founder view has an unknown party")
        text = turn["cue"]["speech_text"] if party == "founder" else turn["response"]
        conversation.append({"turn_id": turn["turn_number"], "speaker": party,
                             "text": text})
    active = view.get("offer_ledger", {}).get("active_offer_id")
    proposals = view.get("offer_ledger", {}).get("proposals", [])
    matching = [item for item in proposals if item.get("offer_id") == active] if active else []
    if active and len(matching) != 1:
        raise TrainingContractError("active offer is absent from public proposal ledger")
    standing = deepcopy(matching[0]["terms"]) if matching else None
    if standing is not None:
        # The dyad publishes lossless decimal strings; the supervised founder
        # input contract uses JSON numeric fields. Restore that exact shape.
        for key in ("cash_usd", "equity_pct", "loan_usd", "royalty"):
            raw = standing.get(key)
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise TrainingContractError("public dyad terms must use decimal strings")
            try:
                number = Decimal(raw)
                numeric = int(number) if number == number.to_integral_value() else float(number)
            except (InvalidOperation, ValueError, OverflowError) as exc:
                raise TrainingContractError("public dyad terms contain invalid decimal") from exc
            if not number.is_finite() or (isinstance(numeric, float) and not math.isfinite(numeric)):
                raise TrainingContractError("public dyad terms contain nonfinite decimal")
            standing[key] = numeric
        try:
            data._validate_terms(standing, "projected standing_offer")
        except data.DataContractError as exc:
            raise TrainingContractError(str(exc)) from exc
    return {"task": "founder_next_public_response",
            "fact_book": [{key: fact[key] for key in data.FOUNDER_FACT_KEYS}
                          for fact in facts],
            "conversation": conversation,
            "standing_offer": standing}


class InvestorInferencePolicy:
    """Bridge a fitted arm to the dyad's public investor-turn contract.

    ``view_projector`` is supplied by the deployment pipeline so the rollout
    view is encoded with the same frozen input schema as training. Timing is
    explicit in the dyad view; this bridge adds a configurable response delay
    and spoken-turn duration.
    """

    def __init__(self, arm: FittedArm, view_projector: Callable[[Mapping], Mapping] | None = None,
                 *, response_delay_s: float = 0.5, response_duration_s: float = 2.0):
        if arm.condition not in data.CONDITIONS or (view_projector is not None and not callable(view_projector)):
            raise TrainingContractError("investor policy requires a fitted modality arm and projector")
        for name, value in (("response_delay_s", response_delay_s),
                            ("response_duration_s", response_duration_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise TrainingContractError(f"{name} must be finite and positive")
        self.arm = arm
        self.view_projector = (view_projector if view_projector is not None else
                               lambda view: project_dyad_investor_view(
                                   view, condition=arm.condition,
                                   response_delay_s=response_delay_s))
        self.delay = float(response_delay_s)
        self.duration = float(response_duration_s)

    def next_turn(self, view: Mapping) -> dyad.InvestorTurn:
        if (not isinstance(view, Mapping) or not isinstance(view.get("turns"), list) or
                not view["turns"] or view["turns"][-1].get("party") != "founder"):
            raise TrainingContractError("investor policy requires a founder turn")
        model_input = self.view_projector(view)
        prediction = self.arm.predict(model_input)["target"]
        if prediction.get("action") not in dyad.INVESTOR_ACTIONS:
            raise TrainingContractError("predicted investor action is outside dyad vocabulary")
        cue = view["turns"][-1]["cue"]
        onset = float(cue["end_pitch_s"]) + self.delay
        end = onset + self.duration
        offer_id = (view.get("offer_ledger", {}).get("active_offer_id")
                    if prediction["action"] == "accept" else None)
        return dyad.InvestorTurn(prediction["action"], prediction["response"],
                                 at_episode_s=onset, end_episode_s=end,
                                 terms=_dyad_terms(prediction.get("terms")),
                                 accepts_offer_id=offer_id)


class FounderInferencePolicy:
    """Bridge a fitted founder actor to the dyad's structured cue-state turn."""

    def __init__(self, arm: FittedArm, view_projector: Callable[[Mapping], Mapping] | None = None,
                 *, reply_delay_s: float = 0.5, cue_duration_s: float = 3.0):
        if arm.condition != "founder_actor" or (view_projector is not None and not callable(view_projector)):
            raise TrainingContractError("founder policy requires founder arm and projector")
        for name, value in (("reply_delay_s", reply_delay_s), ("cue_duration_s", cue_duration_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise TrainingContractError(f"{name} must be finite and positive")
        self.arm = arm
        self.view_projector = view_projector or project_dyad_founder_view
        self.delay = float(reply_delay_s)
        self.duration = float(cue_duration_s)

    def next_turn(self, view: Mapping) -> dyad.FounderTurn:
        if (not isinstance(view, Mapping) or not isinstance(view.get("turns"), list) or
                not view["turns"] or view["turns"][-1].get("party") != "focal_investor"):
            raise TrainingContractError("founder policy requires a focal investor turn")
        prediction = self.arm.predict(self.view_projector(view))["target"]
        if prediction.get("action") not in dyad.FOUNDER_ACTIONS:
            raise TrainingContractError("predicted founder action is outside dyad vocabulary")
        audio = prediction.get("audio_cue")
        video = prediction.get("video_cue")
        if not isinstance(audio, Mapping) or not isinstance(video, Mapping):
            raise TrainingContractError("founder target needs structured audio/video cue states")
        audio_features = ({"features": deepcopy(audio["features"]),
                           "feature_missing": deepcopy(audio.get("missing", {}))}
                          if audio.get("status") == "observed" else
                          {"missing_reason": audio.get("status")})
        video_features = (deepcopy(video["observations"]) if video.get("status") == "observed" else
                          {"missing_reason": video.get("status")})
        if isinstance(video_features, list):
            video_features = {"observations": video_features}
        onset = float(view["turns"][-1]["end_pitch_s"]) + self.delay
        cue = dyad.FounderCue(prediction["response"], audio_features, video_features,
                             start_episode_s=onset, end_episode_s=onset + self.duration,
                             disclosed_fact_ids=tuple(prediction.get("fact_ids", [])),
                             provenance="model_generated_actor_state")
        offer_id = (view.get("offer_ledger", {}).get("active_offer_id")
                    if prediction["action"] == "accept" else None)
        return dyad.FounderTurn(prediction["action"], cue,
                                terms=_dyad_terms(prediction.get("terms")),
                                accepts_offer_id=offer_id)
