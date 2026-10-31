"""Auditable founder--single-investor rollout interface.

This standard-library module accepts injected founder and investor policies.
It defines the public turn contract and deterministic negotiation transitions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Protocol


MIN_RESPONSE_GAP_S = 0.25
FOUNDER_ACTIONS = frozenset({"opening", "answer", "counteroffer", "accept",
                             "decline", "withdraw"})
INVESTOR_ACTIONS = frozenset({"question", "assessment", "offer",
                              "counteroffer", "accept", "exit"})
PARTIES = frozenset({"founder", "focal_investor"})


class DyadContractError(ValueError):
    """A turn, evidence state, or economic transition is invalid."""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DyadContractError(f"{label} must be nonempty text")
    if any(ord(char) < 32 and char not in "\t\n\r" for char in value):
        raise DyadContractError(f"{label} contains control characters")
    return value.strip()


def _seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DyadContractError(f"{label} must be numeric seconds")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise DyadContractError(f"{label} must be finite and nonnegative")
    return number


def _decimal(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise DyadContractError(f"{label} must be a finite decimal number")
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise DyadContractError(f"{label} must be a finite decimal number") from exc
    if not number.is_finite():
        raise DyadContractError(f"{label} must be finite")
    return number


def _optional_decimal(value: Any, label: str) -> Decimal | None:
    return None if value is None else _decimal(value, label)


def _canonical_number(value: Decimal) -> str:
    return format(value.normalize(), "f")


def _safe_json_value(value: Any, label: str) -> Any:
    """Allow exportable cue values, rejecting opaque media and nonfinite values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DyadContractError(f"{label} contains a nonfinite number")
        return value
    if isinstance(value, list):
        return [_safe_json_value(item, label) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise DyadContractError(f"{label} has an invalid feature name")
            result[key] = _safe_json_value(item, label)
        return result
    raise DyadContractError(f"{label} contains a non-JSON feature value")


def _feature_map(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise DyadContractError(f"{label} must be a nonempty feature map")
    return _safe_json_value(dict(value), label)


def commitment(value: Any) -> str:
    """Stable digest of the exact policy-facing view, without model internals."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Fact:
    fact_id: str
    statement: str
    source_id: str
    source_start_s: float
    source_end_s: float
    available_from_s: float
    status: str = "founder_claim"

    def __post_init__(self) -> None:
        _nonempty(self.fact_id, "fact_id")
        _nonempty(self.statement, "fact statement")
        _nonempty(self.source_id, "fact source_id")
        source_start = _seconds(self.source_start_s, "fact source start")
        source_end = _seconds(self.source_end_s, "fact source end")
        available = _seconds(self.available_from_s, "fact availability")
        if source_end <= source_start or source_end > available:
            raise DyadContractError("fact source span must precede availability")
        if self.status not in {"founder_claim", "verified", "planned", "unknown"}:
            raise DyadContractError("unknown fact status")

    def to_dict(self) -> dict[str, Any]:
        return {"fact_id": self.fact_id, "statement": self.statement,
                "source_id": self.source_id, "source_start_s": self.source_start_s,
                "source_end_s": self.source_end_s,
                "available_from_s": self.available_from_s, "status": self.status}


class FactBook:
    """Private founder facts with a mechanical availability boundary."""

    def __init__(self, facts: list[Fact]):
        if not facts or not all(isinstance(fact, Fact) for fact in facts):
            raise DyadContractError("fact book requires Fact entries")
        self._facts = {fact.fact_id: fact for fact in facts}
        if len(self._facts) != len(facts):
            raise DyadContractError("duplicate fact_id")

    def available(self, at_s: float) -> list[dict[str, Any]]:
        timestamp = _seconds(at_s, "fact query time")
        return [fact.to_dict() for fact in self._facts.values()
                if fact.available_from_s <= timestamp]

    def validate_disclosure(self, fact_ids: tuple[str, ...], at_s: float) -> None:
        timestamp = _seconds(at_s, "disclosure time")
        if len(set(fact_ids)) != len(fact_ids):
            raise DyadContractError("duplicate disclosed fact_id")
        for fact_id in fact_ids:
            fact = self._facts.get(fact_id)
            if fact is None:
                raise DyadContractError(f"unknown disclosed fact_id: {fact_id}")
            if fact.available_from_s > timestamp:
                raise DyadContractError(f"fact not yet available: {fact_id}")


@dataclass(frozen=True)
class FounderCue:
    speech_text: str
    audio_features: Mapping[str, Any]
    video_features: Mapping[str, Any]
    start_episode_s: float
    end_episode_s: float
    disclosed_fact_ids: tuple[str, ...] = ()
    provenance: str = "model_generated_actor_state"

    def __post_init__(self) -> None:
        _nonempty(self.speech_text, "founder speech")
        start = _seconds(self.start_episode_s, "founder cue start")
        end = _seconds(self.end_episode_s, "founder cue end")
        if end <= start:
            raise DyadContractError("founder cue must have positive duration")
        audio = _feature_map(self.audio_features, "audio features")
        video = _feature_map(self.video_features, "video features")
        if (not isinstance(self.disclosed_fact_ids, tuple) or
            any(not isinstance(item, str) or not item for item in self.disclosed_fact_ids)):
            raise DyadContractError("disclosed_fact_ids must be a tuple of identifiers")
        if self.provenance not in {"reviewed_founder_media", "model_generated_actor_state"}:
            raise DyadContractError("unresolved founder cue provenance")
        object.__setattr__(self, "audio_features", audio)
        object.__setattr__(self, "video_features", video)

    def to_public(self) -> dict[str, Any]:
        return {"speech_text": self.speech_text,
                "audio_features": _feature_map(self.audio_features, "audio features"),
                "video_features": _feature_map(self.video_features, "video features"),
                "start_episode_s": self.start_episode_s,
                "end_episode_s": self.end_episode_s,
                "provenance": self.provenance}


@dataclass(frozen=True)
class Terms:
    """One focal investor's economic terms; all components are retained."""

    cash_usd: Decimal | None
    equity_pct: Decimal | None
    conditions: tuple[str, ...] = ()
    loan_usd: Decimal | None = None
    royalty: Decimal | None = None
    full_schema: bool = field(default=False, compare=False, repr=False)
    declared_status: str = field(default="proposed", compare=False, repr=False)

    def __post_init__(self) -> None:
        cash = _optional_decimal(self.cash_usd, "cash_usd")
        equity = _optional_decimal(self.equity_pct, "equity_pct")
        loan = _optional_decimal(self.loan_usd, "loan_usd")
        royalty = _optional_decimal(self.royalty, "royalty")
        components = (cash, equity, loan, royalty)
        if any(value is not None and value < 0 for value in components):
            raise DyadContractError("economic components must be nonnegative")
        if equity is not None and equity > 100:
            raise DyadContractError("equity_pct cannot exceed 100")
        if not any(value is not None and value > 0 for value in components):
            raise DyadContractError("at least one economic component must be positive")
        if (not isinstance(self.conditions, tuple) or
            any(not isinstance(item, str) or not item.strip() for item in self.conditions)):
            raise DyadContractError("conditions must be a tuple of text")
        if type(self.full_schema) is not bool:
            raise DyadContractError("full_schema must be boolean")
        if self.declared_status not in {"proposed", "accepted", "withdrawn"}:
            raise DyadContractError("invalid declared offer status")
        object.__setattr__(self, "cash_usd", cash)
        object.__setattr__(self, "equity_pct", equity)
        object.__setattr__(self, "loan_usd", loan)
        object.__setattr__(self, "royalty", royalty)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Terms:
        if not isinstance(value, Mapping) or set(value) != {"cash_usd", "equity_pct", "conditions"}:
            raise DyadContractError("terms require cash_usd, equity_pct, conditions")
        conditions = value["conditions"]
        if not isinstance(conditions, (list, tuple)):
            raise DyadContractError("conditions must be a list or tuple")
        return cls(_optional_decimal(value["cash_usd"], "cash_usd"),
                   _optional_decimal(value["equity_pct"], "equity_pct"), tuple(conditions))

    @classmethod
    def from_training_mapping(cls, value: Mapping[str, Any]) -> Terms:
        expected = {"cash_usd", "equity_pct", "loan_usd", "royalty",
                    "conditions", "participants", "status"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise DyadContractError("training terms require the seven-field schema")
        if value["participants"] != ["focal_investor"]:
            raise DyadContractError("training terms require one focal investor")
        if value["status"] not in {"open", "proposed", "accepted", "withdrawn"}:
            raise DyadContractError("invalid training offer status")
        conditions = value["conditions"]
        if not isinstance(conditions, list):
            raise DyadContractError("training conditions must be a list")
        status = "proposed" if value["status"] == "open" else value["status"]
        return cls(_optional_decimal(value["cash_usd"], "cash_usd"),
                   _optional_decimal(value["equity_pct"], "equity_pct"),
                   tuple(conditions),
                   _optional_decimal(value["loan_usd"], "loan_usd"),
                   _optional_decimal(value["royalty"], "royalty"),
                   full_schema=True, declared_status=status)

    def to_public(self, *, status: str | None = None) -> dict[str, Any]:
        chosen_status = self.declared_status if status is None else status
        if chosen_status not in {"proposed", "accepted", "withdrawn"}:
            raise DyadContractError("invalid public offer status")
        result = {"cash_usd": _canonical_number(self.cash_usd) if self.cash_usd is not None else None,
                  "equity_pct": _canonical_number(self.equity_pct) if self.equity_pct is not None else None,
                  "conditions": list(self.conditions)}
        if self.full_schema or self.loan_usd is not None or self.royalty is not None:
            result.update({"loan_usd": _canonical_number(self.loan_usd) if self.loan_usd is not None else None,
                           "royalty": _canonical_number(self.royalty) if self.royalty is not None else None,
                           "participants": ["focal_investor"], "status": chosen_status})
        return result


@dataclass(frozen=True)
class FounderTurn:
    action: str
    cue: FounderCue
    terms: Terms | None = None
    accepts_offer_id: str | None = None

    def __post_init__(self) -> None:
        if self.action not in FOUNDER_ACTIONS or not isinstance(self.cue, FounderCue):
            raise DyadContractError("invalid founder action or cue")
        if self.action in {"counteroffer", "accept"} and not isinstance(self.terms, Terms):
            raise DyadContractError("founder economic action requires complete terms")
        if self.action not in {"counteroffer", "accept"} and self.terms is not None:
            raise DyadContractError("founder non-economic action cannot carry terms")
        if self.action == "accept":
            _nonempty(self.accepts_offer_id, "accepted offer_id")
        elif self.accepts_offer_id is not None:
            raise DyadContractError("only accept may reference an offer_id")


@dataclass(frozen=True)
class InvestorTurn:
    action: str
    response: str
    at_episode_s: float
    end_episode_s: float
    terms: Terms | None = None
    accepts_offer_id: str | None = None

    def __post_init__(self) -> None:
        if self.action not in INVESTOR_ACTIONS:
            raise DyadContractError("invalid focal-investor action")
        _nonempty(self.response, "investor response")
        onset = _seconds(self.at_episode_s, "investor response onset")
        end = _seconds(self.end_episode_s, "investor response end")
        if end <= onset:
            raise DyadContractError("investor response must have positive duration")
        if self.action in {"offer", "counteroffer", "accept"} and not isinstance(self.terms, Terms):
            raise DyadContractError("investor economic action requires complete terms")
        if self.action not in {"offer", "counteroffer", "accept"} and self.terms is not None:
            raise DyadContractError("investor non-economic action cannot carry terms")
        if self.action == "accept":
            _nonempty(self.accepts_offer_id, "accepted offer_id")
        elif self.accepts_offer_id is not None:
            raise DyadContractError("only accept may reference an offer_id")


@dataclass
class Proposal:
    offer_id: str
    proposer: str
    terms: Terms
    introduced_turn: int
    status: str = "open"
    closed_turn: int | None = None

    def to_public(self) -> dict[str, Any]:
        term_status = ("accepted" if self.status == "accepted" else
                       "proposed" if self.status == "open" else "withdrawn")
        return {"offer_id": self.offer_id, "proposer": self.proposer,
                "terms": self.terms.to_public(status=term_status),
                "introduced_turn": self.introduced_turn,
                "status": self.status, "closed_turn": self.closed_turn}


class OfferLedger:
    """A transactionally validated one-founder, one-investor state machine."""

    def __init__(self) -> None:
        self.proposals: list[Proposal] = []
        self.result: dict[str, Any] | None = None

    @property
    def active(self) -> Proposal | None:
        return next((proposal for proposal in reversed(self.proposals)
                     if proposal.status == "open"), None)

    def _check(self, party: str, action: str, terms: Terms | None,
               accepts_offer_id: str | None) -> None:
        if party not in PARTIES:
            raise DyadContractError("only the founder and focal investor may act")
        if ((party == "founder" and action not in FOUNDER_ACTIONS) or
            (party == "focal_investor" and action not in INVESTOR_ACTIONS)):
            raise DyadContractError("action is invalid for this party")
        if self.result is not None:
            raise DyadContractError("pitch already has a terminal result")
        if action in {"offer", "counteroffer", "accept"}:
            if not isinstance(terms, Terms):
                raise DyadContractError("economic action requires complete terms")
        elif terms is not None:
            raise DyadContractError("non-economic action cannot carry terms")
        if action != "accept" and accepts_offer_id is not None:
            raise DyadContractError("only accept may reference an offer_id")
        active = self.active
        if action in {"offer", "counteroffer"}:
            if terms is None or accepts_offer_id is not None:
                raise DyadContractError("proposal requires terms without an accepted ID")
            if action == "offer" and (party != "focal_investor" or active is not None):
                raise DyadContractError("only the focal investor may open a proposal")
            if action == "counteroffer" and (active is None or active.proposer == party):
                raise DyadContractError("counteroffer requires an opposing active proposal")
        elif action == "accept":
            if (active is None or active.proposer == party or
                accepts_offer_id != active.offer_id or terms != active.terms):
                raise DyadContractError("acceptance must match the opposing active offer ID and terms")
        elif action == "decline":
            if party != "founder" or active is None or active.proposer != "focal_investor":
                raise DyadContractError("decline requires an active focal-investor offer")
        elif action == "withdraw":
            if party != "founder":
                raise DyadContractError("only the founder may withdraw the pitch")
        elif action == "exit":
            if party != "focal_investor":
                raise DyadContractError("only the focal investor may exit")
        elif action not in {"opening", "answer", "question", "assessment"}:
            raise DyadContractError("unsupported negotiation action")

    def advance(self, party: str, action: str, terms: Terms | None = None,
                accepts_offer_id: str | None = None, *, turn_number: int) -> None:
        """Apply only after all validation; a rejected transition changes nothing."""
        self._check(party, action, terms, accepts_offer_id)
        active = self.active
        if action in {"offer", "counteroffer"}:
            if active is not None:
                active.status, active.closed_turn = "superseded", turn_number
            self.proposals.append(Proposal(f"P{len(self.proposals) + 1:03d}",
                                           party, terms, turn_number))  # type: ignore[arg-type]
        elif action == "accept":
            assert active is not None
            active.status, active.closed_turn = "accepted", turn_number
            self.result = {"status": "deal", "accepted_offer_id": active.offer_id,
                           "terms": active.terms.to_public(status="accepted"),
                           "accepted_by": party}
        elif action == "decline":
            assert active is not None
            active.status, active.closed_turn = "declined", turn_number
        elif action in {"withdraw", "exit"}:
            if active is not None:
                active.status, active.closed_turn = "withdrawn", turn_number
            self.result = {"status": "no_deal",
                           "reason": "founder_withdrawal" if action == "withdraw" else "investor_exit"}

    def public(self) -> dict[str, Any]:
        active = self.active
        return {"active_offer_id": active.offer_id if active else None,
                "proposals": [proposal.to_public() for proposal in self.proposals],
                "result": deepcopy(self.result)}


class FounderPolicy(Protocol):
    def next_turn(self, view: Mapping[str, Any]) -> FounderTurn: ...


class InvestorPolicy(Protocol):
    def next_turn(self, view: Mapping[str, Any]) -> InvestorTurn: ...


@dataclass
class PitchState:
    case_id: str
    factbook: FactBook
    focal_investor_id: str = "focal_investor"
    ledger: OfferLedger = field(default_factory=OfferLedger)
    turns: list[dict[str, Any]] = field(default_factory=list)
    input_commitments: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        _nonempty(self.case_id, "case_id")
        _nonempty(self.focal_investor_id, "focal_investor_id")
        if not isinstance(self.factbook, FactBook):
            raise DyadContractError("PitchState requires a FactBook")

    def _record_commitment(self, role: str, value: Mapping[str, Any], turn_number: int) -> None:
        self.input_commitments.append({"role": role, "turn_number": turn_number,
                                       "input_sha256": commitment(value)})

    @property
    def pitch_origin_s(self) -> float:
        if not self.turns:
            raise DyadContractError("pitch origin requires an opening turn")
        return float(self.turns[0]["cue"]["start_episode_s"])

    def public_view(self) -> dict[str, Any]:
        """Project policy input without case identity or episode-position metadata."""
        origin = self.pitch_origin_s
        projected = []
        for turn in self.turns:
            item = {key: deepcopy(value) for key, value in turn.items()
                    if key not in {"disclosed_fact_ids", "at_episode_s", "end_episode_s"}}
            if turn["party"] == "founder":
                cue = item["cue"]
                cue["start_pitch_s"] = round(cue.pop("start_episode_s") - origin, 6)
                cue["end_pitch_s"] = round(cue.pop("end_episode_s") - origin, 6)
            else:
                item["at_pitch_s"] = round(turn["at_episode_s"] - origin, 6)
                item["end_pitch_s"] = round(turn["end_episode_s"] - origin, 6)
            projected.append(item)
        latest = projected[-1]
        clock = latest["cue"]["end_pitch_s"] if latest["party"] == "founder" else latest["end_pitch_s"]
        return {"focal_investor_id": self.focal_investor_id,
                "pitch_clock_s": clock,
                "turns": projected, "offer_ledger": self.ledger.public()}

    def investor_view(self) -> dict[str, Any]:
        if not self.turns or self.turns[-1]["party"] != "founder":
            raise DyadContractError("investor requires a preceding founder turn")
        return self.public_view()

    def founder_view(self) -> dict[str, Any]:
        if not self.turns or self.turns[-1]["party"] != "focal_investor":
            raise DyadContractError("founder requires a preceding investor turn")
        view = self.public_view()
        view["available_founder_facts"] = [
            {key: value for key, value in fact.items() if key != "available_from_s"}
            for fact in self.factbook.available(self.turns[-1]["end_episode_s"])]
        return view

    def append_founder(self, turn: FounderTurn, *, policy_input: Mapping[str, Any] | None = None) -> None:
        if not isinstance(turn, FounderTurn):
            raise DyadContractError("founder policy must return FounderTurn")
        if self.ledger.result is not None:
            raise DyadContractError("cannot append after terminal result")
        if not self.turns:
            if turn.action != "opening":
                raise DyadContractError("first founder turn must be opening")
        else:
            previous = self.turns[-1]
            if previous["party"] != "focal_investor":
                raise DyadContractError("founder turns cannot occur consecutively")
            if turn.cue.start_episode_s < previous["end_episode_s"]:
                raise DyadContractError("founder cue overlaps the investor response")
            if turn.action == "opening":
                raise DyadContractError("opening occurs only once")
        self.factbook.validate_disclosure(turn.cue.disclosed_fact_ids,
                                          turn.cue.end_episode_s)
        turn_number = len(self.turns)
        self.ledger.advance("founder", turn.action, turn.terms,
                            turn.accepts_offer_id, turn_number=turn_number)
        record = {"turn_number": turn_number, "party": "founder",
                  "action": turn.action, "cue": turn.cue.to_public(),
                  "disclosed_fact_ids": list(turn.cue.disclosed_fact_ids),
                  "terms": turn.terms.to_public(status="accepted" if turn.action == "accept" else "proposed")
                  if turn.terms else None,
                  "accepts_offer_id": turn.accepts_offer_id}
        self.turns.append(record)
        if policy_input is not None:
            self._record_commitment("founder", policy_input, turn_number)

    def append_investor(self, turn: InvestorTurn, *, policy_input: Mapping[str, Any] | None = None) -> None:
        if not isinstance(turn, InvestorTurn):
            raise DyadContractError("investor policy must return InvestorTurn")
        if self.ledger.result is not None or not self.turns or self.turns[-1]["party"] != "founder":
            raise DyadContractError("investor requires a nonterminal founder turn")
        founder_end = self.turns[-1]["cue"]["end_episode_s"]
        if turn.at_episode_s - founder_end < MIN_RESPONSE_GAP_S:
            raise DyadContractError("investor response crosses founder cue cutoff")
        turn_number = len(self.turns)
        self.ledger.advance("focal_investor", turn.action, turn.terms,
                            turn.accepts_offer_id, turn_number=turn_number)
        self.turns.append({"turn_number": turn_number, "party": "focal_investor",
                           "action": turn.action, "response": turn.response,
                           "at_episode_s": turn.at_episode_s,
                           "end_episode_s": turn.end_episode_s,
                           "terms": turn.terms.to_public(status="accepted" if turn.action == "accept" else "proposed")
                           if turn.terms else None,
                           "accepts_offer_id": turn.accepts_offer_id})
        if policy_input is not None:
            self._record_commitment("focal_investor", policy_input, turn_number)

    def report(self, *, maximum_investor_turns: int) -> dict[str, Any]:
        result = self.ledger.result or {"status": "incomplete",
                                         "reason": "maximum_investor_turns_reached"}
        return {"case_id": self.case_id, "focal_investor_id": self.focal_investor_id,
                "result": deepcopy(result), "maximum_investor_turns": maximum_investor_turns,
                "turns": deepcopy(self.turns), "offer_ledger": self.ledger.public(),
                "input_commitments": deepcopy(self.input_commitments)}


def rollout(initial_founder: FounderTurn, founder_policy: FounderPolicy,
            investor_policy: InvestorPolicy, *, case_id: str, factbook: FactBook,
            max_investor_turns: int = 8,
            focal_investor_id: str = "focal_investor") -> dict[str, Any]:
    """Alternate policies until terminal state or limit.

    Policy inputs and returned turn times use seconds from the pitch opening.
    The initial founder turn and audit report retain episode-relative timestamps.
    """
    if type(max_investor_turns) is not int or max_investor_turns < 1:
        raise DyadContractError("max_investor_turns must be a positive integer")
    state = PitchState(case_id, factbook, focal_investor_id)
    state.append_founder(initial_founder)
    for index in range(max_investor_turns):
        investor_input = state.investor_view()
        investor = investor_policy.next_turn(deepcopy(investor_input))
        if not isinstance(investor, InvestorTurn):
            raise DyadContractError("investor policy must return InvestorTurn")
        absolute_investor = replace(investor,
                                    at_episode_s=state.pitch_origin_s + investor.at_episode_s,
                                    end_episode_s=state.pitch_origin_s + investor.end_episode_s)
        state.append_investor(absolute_investor, policy_input=investor_input)
        if state.ledger.result is not None:
            break
        founder_input = state.founder_view()
        founder = founder_policy.next_turn(deepcopy(founder_input))
        if not isinstance(founder, FounderTurn):
            raise DyadContractError("founder policy must return FounderTurn")
        absolute_cue = replace(founder.cue,
                               start_episode_s=state.pitch_origin_s + founder.cue.start_episode_s,
                               end_episode_s=state.pitch_origin_s + founder.cue.end_episode_s)
        state.append_founder(replace(founder, cue=absolute_cue),
                             policy_input=founder_input)
        if state.ledger.result is not None:
            break
    return state.report(maximum_investor_turns=max_investor_turns)
