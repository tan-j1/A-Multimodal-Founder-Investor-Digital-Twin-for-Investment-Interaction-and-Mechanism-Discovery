"""Investor-conditioned multimodal completion policy.

This module specifies the model-side algorithm. Reviewed, time-aligned founder
features arrive through ``FeatureBatch``; tokenization and source admission are
separate contracts. The base causal language model is fixed, a low-rank investor
adapter is trained, and a learned reliability gate maps T/A/V features to a
bounded soft prefix. The loss is evaluated only on target completion tokens.

The functions accept model and tokenizer objects supplied by the caller.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    torch = None
    nn = None
    F = None


CONDITIONS = ("T", "T+A", "T+V", "T+A+V")
MODALITIES = ("T", "A", "V")
_CONDITION_MASK = {
    "T": (True, False, False),
    "T+A": (True, True, False),
    "T+V": (True, False, True),
    "T+A+V": (True, True, True),
}


class ModelContractError(ValueError):
    """The causal input, evidence mask, or supervised target is invalid."""


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to construct the completion policy")


@dataclass(frozen=True)
class LoraSpec:
    rank: int
    alpha: float
    dropout: float
    target_modules: str | tuple[str, ...] = "all-linear"

    def validate(self) -> None:
        if (type(self.rank) is not int or self.rank < 1 or
                isinstance(self.alpha, bool) or
                not isinstance(self.alpha, (int, float)) or
                not math.isfinite(self.alpha) or self.alpha <= 0 or
                isinstance(self.dropout, bool) or
                not isinstance(self.dropout, (int, float)) or
                not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ModelContractError("invalid low-rank adapter specification")
        if (not isinstance(self.target_modules, (str, tuple)) or
                not self.target_modules or
                isinstance(self.target_modules, tuple) and
                (not self.target_modules or
                 any(not isinstance(name, str) or not name for name in self.target_modules))):
            raise ModelContractError("target_modules must name linear layers")


def attach_investor_lora(base_lm: Any, spec: LoraSpec, peft_api: Any | None = None) -> Any:
    """Freeze the base and attach W' = W + (alpha / rank) BA to linear layers.

    The caller supplies the causal base model. ``peft_api`` may be an injected
    PEFT-compatible module, making the adapter construction auditable without
    binding this file to a model installation.
    """
    spec.validate()
    if peft_api is None:
        try:
            import peft as peft_api
        except ImportError as exc:
            raise RuntimeError("PEFT is required to attach the investor adapter") from exc
    base_lm.requires_grad_(False)
    config = peft_api.LoraConfig(
        r=spec.rank,
        lora_alpha=spec.alpha,
        lora_dropout=spec.dropout,
        target_modules=(list(spec.target_modules) if isinstance(spec.target_modules, tuple)
                        else spec.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapted = peft_api.get_peft_model(base_lm, config)
    trainable = [name for name, parameter in adapted.named_parameters()
                 if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise ModelContractError("only investor low-rank weights may be trainable")
    return adapted


def freeze_feature_encoders(encoders: Mapping[str, Any]) -> None:
    """Use fixed upstream text, audio, and video encoders in fusion stage one."""
    if set(encoders) != set(MODALITIES):
        raise ModelContractError("T, A, and V feature encoders are required")
    for encoder in encoders.values():
        encoder.requires_grad_(False)
        encoder.eval()


@dataclass(frozen=True)
class FusionSpec:
    text_dim: int
    audio_dim: int
    video_dim: int
    hidden_dim: int
    gate_dim: int
    prefix_tokens: int
    lm_embedding_dim: int
    prefix_bound: float = 1.0
    min_response_gap_s: float = 0.25

    def validate(self) -> None:
        dimensions = (self.text_dim, self.audio_dim, self.video_dim,
                      self.hidden_dim, self.gate_dim, self.prefix_tokens,
                      self.lm_embedding_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ModelContractError("fusion and prefix dimensions must be positive")
        if (isinstance(self.prefix_bound, bool) or
                not isinstance(self.prefix_bound, (int, float)) or
                not math.isfinite(self.prefix_bound) or self.prefix_bound <= 0 or
                isinstance(self.min_response_gap_s, bool) or
                not isinstance(self.min_response_gap_s, (int, float)) or
                not math.isfinite(self.min_response_gap_s) or
                self.min_response_gap_s < 0):
            raise ModelContractError("invalid prefix bound or temporal guard")


@dataclass(frozen=True)
class FeatureBatch:
    """Fixed encoder outputs plus explicit provenance and timing fields.

    Each [batch, 3] field is in T/A/V order. The audio and video branches are
    admitted only when observed, reviewed, founder-origin, investor-exposed,
    and wholly before the target response cutoff. A missing branch may contain
    a zero feature vector, but it never receives nonzero gate weight.
    """

    text: Any
    audio: Any
    video: Any
    observed: Any
    reviewed: Any
    founder_origin: Any
    investor_exposed: Any
    quality: Any
    end_before_cutoff_s: Any
    clause_lag_s: Any

    def validate(self, spec: FusionSpec) -> None:
        _require_torch()
        spec.validate()
        features = ((self.text, spec.text_dim),
                    (self.audio, spec.audio_dim),
                    (self.video, spec.video_dim))
        if not all(isinstance(value, torch.Tensor) and value.ndim == 2 and
                   value.shape[1] == width and value.is_floating_point()
                   for value, width in features):
            raise ModelContractError("T/A/V feature dimensions or types differ")
        batch = self.text.shape[0]
        device = self.text.device
        if batch < 1 or any(value.shape[0] != batch or value.device != device or
                            not bool(torch.isfinite(value).all())
                            for value, _ in features):
            raise ModelContractError("feature batch, device, or finiteness differs")
        for name in ("observed", "reviewed", "founder_origin", "investor_exposed"):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.bool or
                    value.shape != (batch, 3) or value.device != device):
                raise ModelContractError(f"{name} needs a [batch, T/A/V] Boolean mask")
        for name in ("quality", "end_before_cutoff_s", "clause_lag_s"):
            value = getattr(self, name)
            if (not isinstance(value, torch.Tensor) or not value.is_floating_point() or
                    value.shape != (batch, 3) or value.device != device or
                    not bool(torch.isfinite(value).all())):
                raise ModelContractError(f"{name} needs finite [batch, T/A/V] values")
        if (not bool(((self.quality >= 0) & (self.quality <= 1)).all()) or
                not bool((self.clause_lag_s >= 0).all())):
            raise ModelContractError("quality or clause lag is invalid")
        text_admitted = (self.observed[:, 0] & self.reviewed[:, 0] &
                         self.founder_origin[:, 0] & self.investor_exposed[:, 0] &
                         (self.end_before_cutoff_s[:, 0] >=
                          spec.min_response_gap_s))
        if not bool(text_admitted.all()):
            raise ModelContractError("text anchor lacks reviewed pre-response provenance")

    def admitted(self, min_response_gap_s: float) -> Any:
        """Compute the causal evidence mask; text anchors every event."""
        mask = (self.observed & self.reviewed & self.founder_origin &
                self.investor_exposed &
                (self.end_before_cutoff_s >= min_response_gap_s))
        return mask


@dataclass(frozen=True)
class FusionOutput:
    prefix: Any
    gate_weights: Any
    availability: Any


class ReliabilityFusion(nn.Module if nn is not None else object):
    """Learn P_m, g_phi(z_m, q_m, Delta t_m), and bounded reshape(U z)."""

    def __init__(self, spec: FusionSpec):
        _require_torch()
        super().__init__()
        spec.validate()
        self.spec = spec
        dims = dict(zip(MODALITIES, (spec.text_dim, spec.audio_dim,
                                     spec.video_dim)))
        self.projections = nn.ModuleDict({
            name: nn.Linear(width, spec.hidden_dim) for name, width in dims.items()
        })
        self.gates = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width + 2, spec.gate_dim),
                                nn.Tanh(), nn.Linear(spec.gate_dim, 1))
            for name, width in dims.items()
        })
        self.prefix_map = nn.Linear(spec.hidden_dim,
                                    spec.prefix_tokens * spec.lm_embedding_dim)

    def forward(self, batch: FeatureBatch, *, condition: str,
                audio_mask_probability: float = 0.0,
                video_mask_probability: float = 0.0,
                generator: Any | None = None) -> FusionOutput:
        batch.validate(self.spec)
        if condition not in CONDITIONS:
            raise ModelContractError("unknown matched modality condition")
        for probability in (audio_mask_probability, video_mask_probability):
            if (isinstance(probability, bool) or
                    not isinstance(probability, (int, float)) or
                    not math.isfinite(probability) or not 0 <= probability < 1):
                raise ModelContractError("stochastic modality mask must be in [0, 1)")
        if not self.training and (audio_mask_probability or video_mask_probability):
            raise ModelContractError("validation and inference retain measured masks")

        available = batch.admitted(self.spec.min_response_gap_s)
        allowed = torch.tensor(_CONDITION_MASK[condition], dtype=torch.bool,
                               device=batch.text.device).unsqueeze(0)
        available = available & allowed
        for position, probability in ((1, audio_mask_probability),
                                      (2, video_mask_probability)):
            if probability:
                keep = torch.rand(available.shape[0], device=available.device,
                                  generator=generator) >= probability
                available[:, position] &= keep
        if not bool(available[:, 0].all()):
            raise ModelContractError("text is required for each founder event")

        logits = []
        projections = []
        for position, (name, raw) in enumerate(zip(MODALITIES,
                                                    (batch.text, batch.audio, batch.video))):
            usable = available[:, position:position + 1]
            safe = torch.where(usable, raw.detach(), torch.zeros_like(raw))
            safe = safe.to(dtype=self.projections[name].weight.dtype)
            quality = torch.where(usable, batch.quality[:, position:position + 1],
                                  torch.zeros_like(batch.quality[:, position:position + 1]))
            quality = quality.to(dtype=safe.dtype)
            lag = torch.where(usable, batch.clause_lag_s[:, position:position + 1],
                              torch.zeros_like(batch.clause_lag_s[:, position:position + 1]))
            lag = lag.to(dtype=safe.dtype)
            gate_input = torch.cat((safe, quality, torch.log1p(lag)), dim=-1)
            logits.append(self.gates[name](gate_input))
            projections.append(self.projections[name](safe))
        gate_logits = torch.cat(logits, dim=1)
        gate_logits = gate_logits.masked_fill(~available,
                                               torch.finfo(gate_logits.dtype).min)
        beta = torch.softmax(gate_logits, dim=1)
        stacked = torch.stack(projections, dim=1)
        fused = torch.sum(beta.unsqueeze(-1) * stacked, dim=1)
        prefix = torch.tanh(self.prefix_map(fused)) * self.spec.prefix_bound
        prefix = prefix.reshape(batch.text.shape[0], self.spec.prefix_tokens,
                                self.spec.lm_embedding_dim)
        return FusionOutput(prefix=prefix, gate_weights=beta,
                            availability=available)


@dataclass(frozen=True)
class ModelTrainingBatch:
    """Model-tokenized prompt and structured completion, right padded if needed."""

    input_ids: Any
    attention_mask: Any
    labels: Any
    prompt_lengths: Any
    features: FeatureBatch

    def validate(self, spec: FusionSpec) -> None:
        _require_torch()
        self.features.validate(spec)
        batch = self.features.text.shape[0]
        if (not isinstance(self.input_ids, torch.Tensor) or
                not all(isinstance(value, torch.Tensor) and value.shape ==
                    self.input_ids.shape and value.device == self.input_ids.device
                    for value in (self.input_ids, self.attention_mask, self.labels)) or
                self.input_ids.ndim != 2 or self.input_ids.shape[0] != batch or
                self.input_ids.device != self.features.text.device or
                not isinstance(self.prompt_lengths, torch.Tensor) or
                self.prompt_lengths.shape != (batch,) or
                self.prompt_lengths.device != self.input_ids.device):
            raise ModelContractError("model tokens and founder evidence do not align")
        if (self.input_ids.dtype not in (torch.int32, torch.int64) or
                self.labels.dtype not in (torch.int32, torch.int64) or
                self.prompt_lengths.dtype not in (torch.int32, torch.int64) or
                self.attention_mask.dtype not in (torch.bool, torch.int32, torch.int64)):
            raise ModelContractError("causal token rows require integer IDs and masks")
        length = self.input_ids.shape[1]
        attended = self.attention_mask.bool()
        if (length < 2 or not bool(((self.attention_mask == 0) |
                                    (self.attention_mask == 1)).all()) or
                not bool(attended[:, 0].all()) or
                bool((attended[:, 1:] & ~attended[:, :-1]).any()) or
                not bool((self.prompt_lengths >= 1).all()) or
                not bool((self.prompt_lengths < attended.sum(dim=1)).all())):
            raise ModelContractError("empty, left-padded, or truncated completion")
        if not bool((self.input_ids[attended] >= 0).all()):
            raise ModelContractError("attended model token IDs must be nonnegative")
        positions = torch.arange(length, device=self.input_ids.device).unsqueeze(0)
        target_positions = (positions >= self.prompt_lengths.unsqueeze(1)) & attended
        if (not bool((self.labels[~target_positions] == -100).all()) or
                not bool((self.labels[target_positions] ==
                          self.input_ids[target_positions]).all())):
            raise ModelContractError("only complete target tokens may receive loss")


@dataclass(frozen=True)
class PolicyOutput:
    loss: Any
    target_tokens: int
    gate_weights: Any
    availability: Any


class MultimodalPolicy(nn.Module if nn is not None else object):
    """Causal LM with an investor LoRA adapter and founder-evidence soft prefix."""

    def __init__(self, adapted_lm: Any, fusion: ReliabilityFusion):
        _require_torch()
        super().__init__()
        if not hasattr(adapted_lm, "get_input_embeddings"):
            raise ModelContractError("causal LM must expose input embeddings")
        embedding = adapted_lm.get_input_embeddings()
        if embedding.embedding_dim != fusion.spec.lm_embedding_dim:
            raise ModelContractError("soft prefix width differs from LM embeddings")
        self.lm = adapted_lm
        self.fusion = fusion

    def forward(self, batch: ModelTrainingBatch, *, condition: str,
                audio_mask_probability: float = 0.0,
                video_mask_probability: float = 0.0,
                generator: Any | None = None) -> PolicyOutput:
        batch.validate(self.fusion.spec)
        fused = self.fusion(batch.features, condition=condition,
                            audio_mask_probability=audio_mask_probability,
                            video_mask_probability=video_mask_probability,
                            generator=generator)
        embedded = self.lm.get_input_embeddings()(batch.input_ids)
        if fused.prefix.dtype != embedded.dtype:
            prefix = fused.prefix.to(dtype=embedded.dtype)
        else:
            prefix = fused.prefix
        inputs_embeds = torch.cat((prefix, embedded), dim=1)
        prefix_attention = torch.ones((embedded.shape[0], prefix.shape[1]),
                                      dtype=batch.attention_mask.dtype,
                                      device=batch.attention_mask.device)
        attention = torch.cat((prefix_attention, batch.attention_mask), dim=1)
        prefix_labels = torch.full((embedded.shape[0], prefix.shape[1]), -100,
                                   dtype=batch.labels.dtype, device=batch.labels.device)
        labels = torch.cat((prefix_labels, batch.labels), dim=1)
        outputs = self.lm(inputs_embeds=inputs_embeds, attention_mask=attention,
                          use_cache=False, return_dict=True)
        logits = outputs.logits
        target_tokens = int((labels[:, 1:] != -100).sum().item())
        if target_tokens < 1:
            raise ModelContractError("completion contains no supervised tokens")
        loss_sum = F.cross_entropy(logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100,
                                   reduction="sum")
        return PolicyOutput(loss=loss_sum / target_tokens,
                            target_tokens=target_tokens,
                            gate_weights=fused.gate_weights,
                            availability=fused.availability)


@dataclass(frozen=True)
class FitSpec:
    condition: str
    epochs: int
    audio_mask_probability: float = 0.0
    video_mask_probability: float = 0.0
    max_gradient_norm: float = 1.0

    def validate(self) -> None:
        if (self.condition not in CONDITIONS or type(self.epochs) is not int or
                self.epochs < 1 or
                isinstance(self.max_gradient_norm, bool) or
                not isinstance(self.max_gradient_norm, (int, float)) or
                not math.isfinite(self.max_gradient_norm) or
                self.max_gradient_norm <= 0):
            raise ModelContractError("invalid matched-arm fitting specification")
        for probability in (self.audio_mask_probability,
                            self.video_mask_probability):
            if (isinstance(probability, bool) or
                    not isinstance(probability, (int, float)) or
                    not math.isfinite(probability) or not 0 <= probability < 1):
                raise ModelContractError("stochastic mask probability must be in [0, 1)")


def train_step(policy: MultimodalPolicy, batch: ModelTrainingBatch, optimizer: Any,
               fit: FitSpec, *, generator: Any | None = None) -> tuple[float, int]:
    """One optimizer update on completion tokens; base weights stay frozen."""
    _require_torch()
    fit.validate()
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    result = policy(batch, condition=fit.condition,
                    audio_mask_probability=fit.audio_mask_probability,
                    video_mask_probability=fit.video_mask_probability,
                    generator=generator)
    if not bool(torch.isfinite(result.loss)):
        raise ModelContractError("nonfinite completion-only training loss")
    result.loss.backward()
    nn.utils.clip_grad_norm_(policy.parameters(), fit.max_gradient_norm,
                             error_if_nonfinite=True)
    optimizer.step()
    return float(result.loss.detach().item()), result.target_tokens


def fit_multimodal_policy(policy: MultimodalPolicy,
                          training_batches: Sequence[ModelTrainingBatch],
                          validation_batches: Sequence[ModelTrainingBatch],
                          optimizer: Any, fit: FitSpec,
                          *, generator: Any | None = None) -> list[dict[str, float | int]]:
    """Fit one of four matched arms with identical externally fixed batches.

    Validation uses observed availability only. The caller supplies the same
    event IDs, target completions, base initialization, optimizer budget, and
    decoder settings for T, T+A, T+V, and T+A+V; the packet contract checks
    those equalities before this model-side loop.
    """
    _require_torch()
    fit.validate()
    if not training_batches or not validation_batches:
        raise ModelContractError("training and validation batches are required")
    history: list[dict[str, float | int]] = []
    for epoch in range(fit.epochs):
        train_loss_sum = 0.0
        train_tokens = 0
        for batch in training_batches:
            loss, count = train_step(policy, batch, optimizer, fit,
                                     generator=generator)
            train_loss_sum += loss * count
            train_tokens += count
        policy.eval()
        validation_loss_sum = 0.0
        validation_tokens = 0
        with torch.no_grad():
            for batch in validation_batches:
                output = policy(batch, condition=fit.condition)
                validation_loss_sum += float(output.loss.item()) * output.target_tokens
                validation_tokens += output.target_tokens
        history.append({
            "epoch": epoch + 1,
            "training_target_tokens": train_tokens,
            "training_completion_nll": train_loss_sum / train_tokens,
            "validation_target_tokens": validation_tokens,
            "validation_completion_nll": validation_loss_sum / validation_tokens,
        })
    return history


@dataclass(frozen=True)
class StructuredCompletion:
    value: dict
    token_ids: tuple[int, ...]
    log_probability: float
    gate_weights: tuple[float, float, float]


def generate_structured(policy: MultimodalPolicy, tokenizer: Any,
                        prompt_ids: Any, features: FeatureBatch, *,
                        condition: str, max_new_tokens: int,
                        parse_completion: Callable[[str], Mapping] = json.loads,
                        validate_completion: Callable[[Mapping], None] | None = None
                        ) -> StructuredCompletion:
    """Greedy causal completion with the same prefix and a JSON schema hook.

    The returned log probability is a sequence score, not a calibrated
    probability of the investor's real-world action.
    """
    _require_torch()
    if (type(max_new_tokens) is not int or max_new_tokens < 1 or
            not isinstance(prompt_ids, torch.Tensor) or prompt_ids.ndim != 2 or
            prompt_ids.shape[0] != 1 or prompt_ids.shape[1] < 1 or
            prompt_ids.dtype not in (torch.int32, torch.int64) or
            prompt_ids.device != features.text.device or
            features.text.shape[0] != 1 or
            not isinstance(tokenizer.eos_token_id, int)):
        raise ModelContractError("structured generation needs one model-native prompt")
    features.validate(policy.fusion.spec)
    policy.eval()
    generated: list[int] = []
    log_probability = 0.0
    with torch.no_grad():
        fused = policy.fusion(features, condition=condition)
        current = prompt_ids
        for _ in range(max_new_tokens):
            embedded = policy.lm.get_input_embeddings()(current)
            prefix = fused.prefix.to(dtype=embedded.dtype)
            combined = torch.cat((prefix, embedded), dim=1)
            attention = torch.ones(combined.shape[:2], dtype=torch.long,
                                   device=combined.device)
            logits = policy.lm(inputs_embeds=combined,
                               attention_mask=attention, use_cache=False,
                               return_dict=True).logits[:, -1, :].float()
            token = int(torch.argmax(logits, dim=-1).item())
            log_probability += float(F.log_softmax(logits, dim=-1)[0, token].item())
            if token == tokenizer.eos_token_id:
                break
            generated.append(token)
            continuation = torch.tensor([[token]], dtype=current.dtype,
                                        device=current.device)
            current = torch.cat((current, continuation), dim=1)
        else:
            raise ModelContractError("structured completion exceeded token boundary")
    try:
        parsed = parse_completion(tokenizer.decode(generated,
                                                    skip_special_tokens=True))
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ModelContractError("investor completion is not parseable") from exc
    if not isinstance(parsed, Mapping):
        raise ModelContractError("investor completion must be a JSON object")
    if validate_completion is not None:
        validate_completion(parsed)
    return StructuredCompletion(value=dict(parsed), token_ids=tuple(generated),
                                log_probability=log_probability,
                                gate_weights=tuple(float(value) for value in
                                                   fused.gate_weights[0].tolist()))
