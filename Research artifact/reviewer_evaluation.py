"""Matched turn and full-trajectory evaluation for a focal-investor simulator.

Pure-Python methods are independent of model architecture and operate on
normalized predictions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
import math
import random
import re

if __package__:
    from .reviewer_dyad import INVESTOR_ACTIONS
else:
    from reviewer_dyad import INVESTOR_ACTIONS


ARMS = ("T", "T+A", "T+V", "T+A+V")
TERMINALS = {"deal", "no_deal", "incomplete"}
NUMBER = re.compile(r"(?<![\w.])([$€£])?\s*(\d+(?:,\d{3})*(?:\.\d+)?)(%)?(?![\w.])")


class EvaluationError(ValueError):
    """Rows cannot support a matched evaluation."""


def _money(value: object, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise EvaluationError(f"{label} must be a finite decimal")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise EvaluationError(f"{label} must be a finite decimal") from exc
    if not number.is_finite() or number < 0:
        raise EvaluationError(f"{label} must be nonnegative and finite")
    return format(number.normalize(), "f")


def _terms(value: object) -> tuple | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise EvaluationError("terms must be an object")
    dyad_keys = {"cash_usd", "equity_pct", "conditions"}
    turn_keys = dyad_keys | {"loan_usd", "royalty", "participants", "status"}
    if set(value) not in (dyad_keys, turn_keys):
        raise EvaluationError("terms require the dyadic or full turn schema")
    cash = _money(value["cash_usd"], "cash_usd", nullable=True)
    equity = _money(value["equity_pct"], "equity_pct", nullable=True)
    if equity is not None and Decimal(equity) > 100:
        raise EvaluationError("equity_pct exceeds 100")
    conditions = value["conditions"]
    if (not isinstance(conditions, list) or
        any(not isinstance(item, str) or not item.strip() for item in conditions)):
        raise EvaluationError("conditions must be a list of text")
    if set(value) == dyad_keys:
        if not any(item is not None and Decimal(item) > 0 for item in (cash, equity)):
            raise EvaluationError("offer lacks a positive economic component")
        return (cash, equity, tuple(conditions))
    loan = _money(value["loan_usd"], "loan_usd", nullable=True)
    royalty = _money(value["royalty"], "royalty", nullable=True)
    if not any(item is not None and Decimal(item) > 0
               for item in (cash, equity, loan, royalty)):
        raise EvaluationError("offer lacks a positive economic component")
    participants = value["participants"]
    if participants != ["focal_investor"] or value["status"] not in {
        "proposed", "accepted", "withdrawn"
    }:
        raise EvaluationError("invalid participants or offer status")
    return (cash, equity, loan, royalty, tuple(conditions), tuple(participants), value["status"])


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f"{label} must be nonempty text")
    return value.strip()


def _concerns(value: object) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise EvaluationError("concerns must be a list of nonempty strings")
    normalized = {item.strip().casefold() for item in value}
    if len(normalized) != len(value):
        raise EvaluationError("duplicate concern label")
    return normalized


def _investor_action_terms(action: object, terms: object) -> None:
    if not isinstance(action, str) or action not in INVESTOR_ACTIONS:
        raise EvaluationError("action is outside the focal-investor vocabulary")
    economic = action in {"offer", "counteroffer", "accept"}
    if economic != (terms is not None):
        raise EvaluationError("economic action and terms disagree")
    _terms(terms)


def _numbers(value: str) -> set[str]:
    return {"".join(part or "" for part in match.groups()).replace(",", "")
            for match in NUMBER.finditer(value)}


def _macro_f1(gold: list[str], predicted: list[str]) -> float:
    if len(gold) != len(predicted) or not gold:
        raise EvaluationError("F1 requires nonempty matched labels")
    values = []
    for label in sorted(set(gold) | set(predicted)):
        tp = sum(a == b == label for a, b in zip(gold, predicted))
        fp = sum(a != label and b == label for a, b in zip(gold, predicted))
        fn = sum(a == label and b != label for a, b in zip(gold, predicted))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(values) / len(values)


def _interval(values: list[float], *, seed: int = 3407, draws: int = 2000) -> list[float] | None:
    """Pitch-cluster bootstrap of a paired per-pitch difference."""
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(draws))
    return [round(samples[int(0.025 * (draws - 1))], 4),
            round(samples[int(0.975 * (draws - 1))], 4)]


def _coded_labels(value: object, label: str) -> set[str]:
    """Compare adjudicated taxonomy codes, not raw wording."""
    if (not isinstance(value, list) or
            any(not isinstance(item, str) or not item.strip() for item in value)):
        raise EvaluationError(f"{label} must be a list of nonempty codes")
    codes = {item.strip().casefold() for item in value}
    if len(codes) != len(value):
        raise EvaluationError(f"{label} contains duplicate codes")
    return codes


def _coded_id(value: object, label: str) -> str:
    identifier = _text(value, label)
    if identifier != value:
        raise EvaluationError(f"{label} must not have surrounding whitespace")
    return identifier


def _source_ids(value: object, label: str) -> set[str]:
    if (not isinstance(value, list) or
            any(not isinstance(item, str) or not item.strip() for item in value)):
        raise EvaluationError(f"{label} must be a list of source IDs")
    identifiers = {item.strip() for item in value}
    if len(identifiers) != len(value):
        raise EvaluationError(f"{label} contains duplicate source IDs")
    return identifiers


def _set_jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def _pitch_metric(values: list[tuple[str, float]], *, seed: int,
                  draws: int) -> dict:
    """Equal-weight pitches; bootstrap resamples whole pitches."""
    by_pitch: dict[str, list[float]] = defaultdict(list)
    for pitch_id, value in values:
        if not math.isfinite(value):
            raise EvaluationError("metric value must be finite")
        by_pitch[pitch_id].append(value)
    pitch_means = [sum(entries) / len(entries) for _, entries in sorted(by_pitch.items())]
    return {
        "pitch_equal_mean": round(sum(pitch_means) / len(pitch_means), 4)
        if pitch_means else None,
        "pitch_bootstrap_95_interval": _interval(pitch_means, seed=seed, draws=draws),
        "events": len(values),
        "pitches": len(by_pitch),
    }


def _numeric_fields(value: object) -> dict[str, Decimal | None]:
    _terms(value)
    keys = ("cash_usd", "equity_pct", "loan_usd", "royalty")
    if value is None:
        return {key: None for key in keys}
    return {
        key: Decimal(_money(value[key], key)) if key in value and value[key] is not None
        else None for key in keys
    }


def _validated_ratings(ratings: list[dict] | None,
                       required: set[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """Two condition-blinded raters independently score each response."""
    if ratings is None:
        return {}
    if not isinstance(ratings, list):
        raise EvaluationError("ratings must be a list")
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    fields = {"event_id", "condition", "rater_id", "response_fidelity",
              "specificity", "public_evidence_grounding", "blinded_to_condition"}
    for row in ratings:
        if not isinstance(row, dict) or set(row) != fields:
            raise EvaluationError("invalid blinded-rating schema")
        if not isinstance(row["condition"], str) or row["condition"] not in ARMS:
            raise EvaluationError("rating condition is outside the matched arms")
        key = (_coded_id(row["event_id"], "rating event_id"), row["condition"])
        if key not in required or row["blinded_to_condition"] is not True:
            raise EvaluationError("rating is unmatched or was not condition-blinded")
        _coded_id(row["rater_id"], "rater_id")
        if any(other["rater_id"] == row["rater_id"] for other in grouped[key]):
            raise EvaluationError(f"duplicate rater for {key}")
        for field in ("response_fidelity", "specificity", "public_evidence_grounding"):
            score = row[field]
            if (isinstance(score, bool) or not isinstance(score, (int, float)) or
                    not math.isfinite(score) or not 1 <= score <= 5):
                raise EvaluationError(f"{field} must be a finite score on the 1–5 scale")
        grouped[key].append(row)
    if set(grouped) != required or any(len(rows) != 2 for rows in grouped.values()):
        raise EvaluationError("each matched response requires two independent ratings")
    return grouped


def score_coded_responses(gold_rows: list[dict], predictions: list[dict], *,
                          ratings: list[dict] | None = None,
                          seed: int = 3407, draws: int = 2000) -> dict:
    """Evaluate separately coded Y02/Y03 response features across matched arms.

    Gold fields: event_id, pitch_id, issue_codes, evidence_request_codes,
    terms, public_source_ids, linked_source_ids. The public IDs are taken from
    the sealed, pre-response evidence ledger; gold linked IDs are target-coded.
    Prediction fields: event_id, condition, issue_codes,
    evidence_request_codes, terms, cited_source_ids. Codes use a fixed
    adjudicated taxonomy. Source-ID checks measure provenance eligibility and
    target-link agreement; they cannot establish semantic entailment. Optional
    1–5 ratings provide blinded judgments of fidelity, specificity, and
    grounding. Each event must appear in all four modality conditions.
    """
    if type(seed) is not int or type(draws) is not int or draws < 100:
        raise EvaluationError("integer seed and at least 100 bootstrap draws are required")
    gold_fields = {"event_id", "pitch_id", "issue_codes", "evidence_request_codes",
                   "terms", "public_source_ids", "linked_source_ids"}
    prediction_fields = {"event_id", "condition", "issue_codes",
                         "evidence_request_codes", "terms", "cited_source_ids"}
    gold: dict[str, dict] = {}
    for row in gold_rows:
        if not isinstance(row, dict) or set(row) != gold_fields:
            raise EvaluationError("invalid coded gold-response schema")
        event_id = _coded_id(row["event_id"], "gold event_id")
        if event_id in gold:
            raise EvaluationError(f"duplicate coded gold event: {event_id}")
        _coded_id(row["pitch_id"], "gold pitch_id")
        _coded_labels(row["issue_codes"], "gold issue_codes")
        _coded_labels(row["evidence_request_codes"], "gold evidence_request_codes")
        _terms(row["terms"])
        public = _source_ids(row["public_source_ids"], "public_source_ids")
        linked = _source_ids(row["linked_source_ids"], "gold linked_source_ids")
        if not linked <= public:
            raise EvaluationError("target-linked source was not in sealed public evidence")
        gold[event_id] = row
    if not gold:
        raise EvaluationError("empty coded gold-response table")

    required = {(event_id, arm) for event_id in gold for arm in ARMS}
    indexed: dict[tuple[str, str], dict] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) != prediction_fields:
            raise EvaluationError("invalid coded prediction schema")
        if not isinstance(row["condition"], str) or row["condition"] not in ARMS:
            raise EvaluationError("prediction condition is outside the matched arms")
        key = (_coded_id(row["event_id"], "prediction event_id"), row["condition"])
        if key not in required or key in indexed:
            raise EvaluationError(f"unknown or duplicate coded prediction: {key}")
        _coded_labels(row["issue_codes"], "predicted issue_codes")
        _coded_labels(row["evidence_request_codes"], "predicted evidence_request_codes")
        _terms(row["terms"])
        _source_ids(row["cited_source_ids"], "predicted cited_source_ids")
        indexed[key] = row
    if set(indexed) != required:
        raise EvaluationError(f"missing {len(required - set(indexed))} coded predictions")
    rating_index = _validated_ratings(ratings, required)

    ordered = [gold[event_id] for event_id in sorted(gold)]
    report = {"events": len(ordered),
              "pitches": len({row["pitch_id"] for row in ordered}),
              "conditions": {}, "paired_ablations": {}}
    event_values: dict[str, dict[tuple[str, str], float]] = defaultdict(dict)
    for arm in ARMS:
        measures: dict[str, list[tuple[str, float]]] = defaultdict(list)
        field_errors: dict[str, list[tuple[str, float]]] = defaultdict(list)
        field_gold_counts = defaultdict(int)
        field_missing_counts = defaultdict(int)
        unsupported_source_ids = 0
        cited_source_ids = 0
        gold_offers = 0
        false_positive_offers = 0
        for truth in ordered:
            event_id, pitch = truth["event_id"], truth["pitch_id"]
            estimate = indexed[(event_id, arm)]
            for label, field in (("issue", "issue_codes"),
                                 ("evidence_request", "evidence_request_codes")):
                actual = _coded_labels(truth[field], f"gold {field}")
                predicted = _coded_labels(estimate[field], f"predicted {field}")
                exact = float(actual == predicted)
                agreement = _set_jaccard(actual, predicted)
                measures[f"{label}_exact_agreement"].append((pitch, exact))
                measures[f"{label}_jaccard"].append((pitch, agreement))
                event_values[f"{label}_jaccard"][(event_id, arm)] = agreement

            public = _source_ids(truth["public_source_ids"], "public_source_ids")
            linked = _source_ids(truth["linked_source_ids"], "gold linked_source_ids")
            cited = _source_ids(estimate["cited_source_ids"], "predicted cited_source_ids")
            cited_source_ids += len(cited)
            unsupported_source_ids += len(cited - public)
            measures["source_citation_coverage"].append((pitch, float(bool(cited))))
            measures["target_linked_source_jaccard"].append(
                (pitch, _set_jaccard(linked, cited)))
            if cited:
                measures["public_source_admissibility"].append(
                    (pitch, len(cited & public) / len(cited)))

            actual_terms = _numeric_fields(truth["terms"])
            estimated_terms = _numeric_fields(estimate["terms"])
            measures["offer_presence_accuracy"].append(
                (pitch, float((truth["terms"] is None) == (estimate["terms"] is None))))
            if truth["terms"] is None and estimate["terms"] is not None:
                false_positive_offers += 1
            if truth["terms"] is not None:
                gold_offers += 1
                numeric_exact = float(actual_terms == estimated_terms)
                measures["numeric_terms_exact_on_gold_offers"].append((pitch, numeric_exact))
                event_values["numeric_terms_exact_on_gold_offers"][(event_id, arm)] = numeric_exact
                for field, gold_number in actual_terms.items():
                    if gold_number is None:
                        continue
                    field_gold_counts[field] += 1
                    predicted_number = estimated_terms[field]
                    if predicted_number is None:
                        field_missing_counts[field] += 1
                    else:
                        field_errors[field].append(
                            (pitch, float(abs(predicted_number - gold_number))))

            if rating_index:
                pair = rating_index[(event_id, arm)]
                for field in ("response_fidelity", "specificity", "public_evidence_grounding"):
                    average = sum(row[field] for row in pair) / 2
                    measures[f"blinded_{field}"].append((pitch, average))
                    measures[f"{field}_rater_absolute_difference"].append(
                        (pitch, abs(pair[0][field] - pair[1][field])))
                    if field == "response_fidelity":
                        event_values["blinded_response_fidelity"][(event_id, arm)] = average

        condition = {name: _pitch_metric(rows, seed=seed, draws=draws)
                     for name, rows in sorted(measures.items())}
        for name in ("public_source_admissibility", "numeric_terms_exact_on_gold_offers"):
            condition.setdefault(name, _pitch_metric([], seed=seed, draws=draws))
        condition["numerical_term_error"] = {
            field: {**_pitch_metric(field_errors[field], seed=seed, draws=draws),
                    "unit": "USD" if field in {"cash_usd", "loan_usd"}
                    else "percentage_points" if field == "equity_pct" else "coded_royalty_units",
                    "gold_values": field_gold_counts[field],
                    "missing_predictions": field_missing_counts[field]}
            for field in ("cash_usd", "equity_pct", "loan_usd", "royalty")
        }
        condition["gold_offers"] = gold_offers
        condition["false_positive_offers"] = false_positive_offers
        condition["cited_source_ids"] = cited_source_ids
        condition["unsupported_source_ids"] = unsupported_source_ids
        report["conditions"][arm] = condition

    for name, high, low in (("audio_given_video", "T+A+V", "T+V"),
                            ("video_given_audio", "T+A+V", "T+A")):
        differences = {}
        for metric, event_map in sorted(event_values.items()):
            rows = [(truth["pitch_id"], event_map[(truth["event_id"], high)] -
                     event_map[(truth["event_id"], low)])
                    for truth in ordered if (truth["event_id"], high) in event_map and
                    (truth["event_id"], low) in event_map]
            differences[metric] = _pitch_metric(rows, seed=seed, draws=draws)
        report["paired_ablations"][name] = differences
    return report


def score_matched_turns(gold_rows: list[dict], predictions: list[dict], *,
                        expected_input_sha256: Mapping[tuple[str, str], str] | None = None) -> dict:
    """Require four predictions per event, then report event and pitch diagnostics.

    Gold fields: event_id, pitch_id, action, concerns, response, terms.
    Prediction fields: the same excluding pitch_id, plus condition. Prediction
    records may carry an input commitment, which is checked by the data module.
    """
    gold: dict[str, dict] = {}
    for row in gold_rows:
        if not isinstance(row, dict) or set(row) != {
            "event_id", "pitch_id", "action", "concerns", "response", "terms"
        }:
            raise EvaluationError("invalid gold-turn schema")
        event_id = _text(row["event_id"], "event_id")
        if event_id in gold:
            raise EvaluationError(f"duplicate gold event: {event_id}")
        _text(row["pitch_id"], "pitch_id")
        _investor_action_terms(row["action"], row["terms"])
        _text(row["response"], "response")
        _concerns(row["concerns"])
        gold[event_id] = row
    if not gold:
        raise EvaluationError("empty gold-turn table")
    indexed: dict[tuple[str, str], dict] = {}
    for row in predictions:
        if not isinstance(row, dict) or set(row) - {
            "event_id", "condition", "action", "concerns", "response", "terms", "input_sha256"
        } or not {"event_id", "condition", "action", "concerns", "response", "terms"}.issubset(row):
            raise EvaluationError("invalid prediction schema")
        event_id = _text(row["event_id"], "prediction event_id")
        condition = row["condition"]
        key = (event_id, condition)
        if event_id not in gold or condition not in ARMS or key in indexed:
            raise EvaluationError(f"unknown or duplicate matched prediction: {key}")
        _investor_action_terms(row["action"], row["terms"])
        _text(row["response"], "predicted response")
        _concerns(row["concerns"])
        indexed[key] = row
    required = {(event_id, arm) for event_id in gold for arm in ARMS}
    if set(indexed) != required:
        raise EvaluationError(f"missing {len(required - set(indexed))} matched turn predictions")
    if expected_input_sha256 is not None:
        if set(expected_input_sha256) != required:
            raise EvaluationError("expected input commitments do not match evaluation events")
        for key, row in indexed.items():
            if row.get("input_sha256") != expected_input_sha256[key]:
                raise EvaluationError(f"prediction input commitment differs: {key}")

    ordered = [gold[event_id] for event_id in sorted(gold)]
    report = {"events": len(ordered), "pitches": len({row["pitch_id"] for row in ordered}), "conditions": {}}
    for arm in ARMS:
        predictions_for_arm = [indexed[(row["event_id"], arm)] for row in ordered]
        action_true = [row["action"] for row in ordered]
        action_pred = [row["action"] for row in predictions_for_arm]
        concern_scores = []
        for truth, estimate in zip(ordered, predictions_for_arm):
            left, right = _concerns(truth["concerns"]), _concerns(estimate["concerns"])
            concern_scores.append(len(left & right) / len(left | right) if left | right else 1.0)
        numeric = [(truth, estimate) for truth, estimate in zip(ordered, predictions_for_arm)
                   if _numbers(truth["response"])]
        offered = [(truth, estimate) for truth, estimate in zip(ordered, predictions_for_arm)
                   if truth["terms"] is not None]
        report["conditions"][arm] = {
            "action_accuracy": round(sum(a == b for a, b in zip(action_true, action_pred)) / len(ordered), 4),
            "action_macro_f1": round(_macro_f1(action_true, action_pred), 4),
            "mean_concern_jaccard": round(sum(concern_scores) / len(concern_scores), 4),
            "terms_exact_on_gold_offers": sum(_terms(a["terms"]) == _terms(b["terms"])
                                               for a, b in offered),
            "gold_offers": len(offered),
            "written_numbers_exact": sum(_numbers(a["response"]) == _numbers(b["response"])
                                         for a, b in numeric),
            "gold_responses_with_numbers": len(numeric),
        }
    by_pitch: dict[str, list[float]] = defaultdict(list)
    for row in ordered:
        event_id = row["event_id"]
        correct_full = float(indexed[(event_id, "T+A+V")]["action"] == row["action"])
        correct_text = float(indexed[(event_id, "T")]["action"] == row["action"])
        by_pitch[row["pitch_id"]].append(correct_full - correct_text)
    pitch_deltas = [sum(values) / len(values) for _, values in sorted(by_pitch.items())]
    report["paired_all_channels_minus_text_action"] = {
        "pitch_equal_difference": round(sum(pitch_deltas) / len(pitch_deltas), 4),
        "pitch_bootstrap_95_interval": _interval(pitch_deltas),
    }
    return report


def score_rollouts(gold_rows: list[dict], simulated_rows: list[dict]) -> dict:
    """Score one or more seeds per held-out pitch; keep incomplete explicit."""
    gold = {}
    for row in gold_rows:
        if not isinstance(row, dict) or set(row) != {"pitch_id", "terminal", "accepted_terms"}:
            raise EvaluationError("invalid gold outcome schema")
        pitch = _text(row["pitch_id"], "pitch_id")
        if pitch in gold or row["terminal"] not in {"deal", "no_deal"}:
            raise EvaluationError("duplicate pitch or invalid observed terminal")
        if (row["terminal"] == "deal") != (row["accepted_terms"] is not None):
            raise EvaluationError("observed deal/terms mismatch")
        _terms(row["accepted_terms"])
        gold[pitch] = row
    if not gold:
        raise EvaluationError("empty outcome table")
    by_pitch: dict[str, list[dict]] = defaultdict(list)
    seen: set[tuple[str, int]] = set()
    for row in simulated_rows:
        if not isinstance(row, dict) or set(row) != {"pitch_id", "seed", "terminal", "accepted_terms"}:
            raise EvaluationError("invalid simulation outcome schema")
        pitch = _text(row["pitch_id"], "simulated pitch_id")
        seed = row["seed"]
        if pitch not in gold or type(seed) is not int or (pitch, seed) in seen:
            raise EvaluationError("unknown pitch, noninteger seed, or duplicate rollout")
        if row["terminal"] not in TERMINALS or (row["terminal"] == "deal") != (
            row["accepted_terms"] is not None
        ):
            raise EvaluationError("simulated terminal/terms mismatch")
        _terms(row["accepted_terms"])
        seen.add((pitch, seed))
        by_pitch[pitch].append(row)
    if set(by_pitch) != set(gold):
        raise EvaluationError("a gold pitch has no rollout")
    seeds_per_pitch = {pitch: {row["seed"] for row in rows} for pitch, rows in by_pitch.items()}
    if len({tuple(sorted(seeds)) for seeds in seeds_per_pitch.values()}) != 1:
        raise EvaluationError("rollout seeds differ across pitches")
    all_rows = [row for rows in by_pitch.values() for row in rows]
    completed = [row for row in all_rows if row["terminal"] != "incomplete"]
    matched = sum(row["terminal"] == gold[row["pitch_id"]]["terminal"] for row in completed)
    deal_terms = [row for row in completed if row["terminal"] == "deal" and
                  gold[row["pitch_id"]]["terminal"] == "deal"]
    brier = []
    for pitch, rows in sorted(by_pitch.items()):
        completed_pitch = [row for row in rows if row["terminal"] != "incomplete"]
        if not completed_pitch:
            continue
        probability = sum(row["terminal"] == "deal" for row in completed_pitch) / len(completed_pitch)
        observed = float(gold[pitch]["terminal"] == "deal")
        brier.append((probability - observed) ** 2)
    return {
        "pitches": len(gold), "seeds_per_pitch": len(next(iter(seeds_per_pitch.values()))),
        "rollouts": len(all_rows), "completed_rollouts": len(completed),
        "completion_rate": round(len(completed) / len(all_rows), 4),
        "terminal_accuracy_on_completed": round(matched / len(completed), 4) if completed else None,
        "deal_terms_exact_on_matching_deals": sum(
            _terms(row["accepted_terms"]) == _terms(gold[row["pitch_id"]]["accepted_terms"])
            for row in deal_terms),
        "matching_deals": len(deal_terms),
        "completed_pitches": len(brier),
        "mean_deal_brier_on_completed": round(sum(brier) / len(brier), 4) if brier else None,
    }


def normalize_rollout_report(report: dict, *, seed: int) -> dict:
    """Convert the dyad audit report to the held-out outcome scorer's row.

    The accepted terms are checked against the public offer ledger rather than
    trusted from a model-generated result alone.
    """
    if type(seed) is not int or not isinstance(report, dict):
        raise EvaluationError("rollout report and integer seed are required")
    pitch = _text(report.get("case_id"), "report case_id")
    result = report.get("result")
    ledger = report.get("offer_ledger")
    if not isinstance(result, dict) or not isinstance(ledger, dict):
        raise EvaluationError("rollout result and public offer ledger are required")
    status = result.get("status")
    if status not in TERMINALS:
        raise EvaluationError("invalid rollout terminal")
    proposals = ledger.get("proposals")
    if not isinstance(proposals, list) or ledger.get("result") != (
        None if status == "incomplete" else result
    ):
        raise EvaluationError("rollout result disagrees with offer ledger")
    accepted = None
    if status == "deal":
        offer_id = _text(result.get("accepted_offer_id"), "accepted offer ID")
        candidates = [proposal for proposal in proposals if isinstance(proposal, dict)
                      and proposal.get("offer_id") == offer_id
                      and proposal.get("status") == "accepted"]
        if len(candidates) != 1 or candidates[0].get("terms") != result.get("terms"):
            raise EvaluationError("accepted result lacks a matching ledger offer")
        accepted = result["terms"]
        _terms(accepted)
    elif result.get("terms") is not None:
        raise EvaluationError("a non-deal result cannot contain accepted terms")
    return {"pitch_id": pitch, "seed": seed, "terminal": status,
            "accepted_terms": accepted}
