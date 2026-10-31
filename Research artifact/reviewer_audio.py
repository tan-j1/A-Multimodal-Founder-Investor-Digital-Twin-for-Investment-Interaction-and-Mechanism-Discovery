"""Auditable acoustic summaries for reviewed, founder-only PCM windows.

This module accepts mono 16-bit PCM, not a mixed television soundtrack.  Its
window record must refer to a reviewed founder segment ending before the focal
investor's next response.  The numbers describe the waveform; no emotion or
persuasion label is inferred from them.

The optional word alignment is clip-relative and independently reviewed.  It
supplies a word-rate estimate; waveform energy alone cannot count words.
"""

from __future__ import annotations

from array import array
import hashlib
import math
from pathlib import Path
import re
import statistics
import sys
import wave


WINDOW_FIELDS = frozenset({
    "event_id", "reviewed", "speaker_role", "start_episode_s",
    "end_episode_s", "cutoff_episode_s", "media_sha256",
})
WORD_FIELDS = frozenset({"word", "start_s", "end_s"})
MIN_PRE_RESPONSE_GAP_S = 0.25
MIN_WINDOW_S = 0.25
MAX_WINDOW_S = 60.0
FRAME_S = 0.040
HOP_S = 0.010
F0_MIN_HZ = 70.0
F0_MAX_HZ = 400.0
MAX_PITCH_FRAMES = 160
DBFS_FLOOR = -120.0


class AudioContractError(ValueError):
    """Audio, timing, or source evidence violates the reviewer interface."""


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioContractError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise AudioContractError(f"{label} must be finite and nonnegative")
    return number


def validate_window(window: dict, *, duration_s: float) -> dict:
    """Check identity, reviewed founder role, source digest, and temporal gap."""
    if not isinstance(window, dict) or set(window) != WINDOW_FIELDS:
        raise AudioContractError(f"window requires exactly {sorted(WINDOW_FIELDS)}")
    event_id = window["event_id"]
    if (not isinstance(event_id, str) or not event_id.strip() or
        len(event_id) > 120 or any(ord(c) < 32 for c in event_id)):
        raise AudioContractError("event_id must be short printable text")
    if window["reviewed"] is not True or window["speaker_role"] != "founder":
        raise AudioContractError("audio must have reviewed founder-only attribution")
    start = _finite_nonnegative(window["start_episode_s"], "start_episode_s")
    end = _finite_nonnegative(window["end_episode_s"], "end_episode_s")
    cutoff = _finite_nonnegative(window["cutoff_episode_s"], "cutoff_episode_s")
    if not MIN_WINDOW_S <= end - start <= MAX_WINDOW_S:
        raise AudioContractError("window duration is outside the 0.25–60 s range")
    if end > cutoff - MIN_PRE_RESPONSE_GAP_S + 1e-9:
        raise AudioContractError("founder audio crosses the response cutoff margin")
    if abs((end - start) - duration_s) > max(0.02, 2.0 / 8_000):
        raise AudioContractError("PCM duration differs from the reviewed window")
    digest = window["media_sha256"]
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise AudioContractError("media_sha256 must be a lowercase SHA-256 digest")
    return {
        "event_id": event_id.strip(),
        "start_episode_s": start,
        "end_episode_s": end,
        "cutoff_episode_s": cutoff,
        "source_media_sha256": digest,
    }


def validate_aligned_words(words: list[dict] | None, *, duration_s: float,
                           alignment_reviewed: bool) -> list[dict] | None:
    """Keep absent alignment distinct from a reviewed zero-word window."""
    if words is None:
        return None
    if alignment_reviewed is not True:
        raise AudioContractError("word timings require independent alignment review")
    if not isinstance(words, list):
        raise AudioContractError("aligned_words must be a list or null")
    validated = []
    previous_end = 0.0
    for index, word in enumerate(words):
        if not isinstance(word, dict) or set(word) != WORD_FIELDS:
            raise AudioContractError(f"aligned word {index} requires {sorted(WORD_FIELDS)}")
        token = word["word"]
        if (not isinstance(token, str) or not token.strip() or
            any(ord(char) < 32 for char in token)):
            raise AudioContractError(f"aligned word {index} has invalid text")
        start = _finite_nonnegative(word["start_s"], f"word {index} start_s")
        end = _finite_nonnegative(word["end_s"], f"word {index} end_s")
        if end <= start or end > duration_s + 2e-3 or start < previous_end - 2e-3:
            raise AudioContractError(f"aligned word {index} has invalid or overlapping timing")
        validated.append({"word": token.strip(), "start_s": start, "end_s": end})
        previous_end = end
    return validated


def _decode_pcm16le(pcm: bytes) -> array:
    if not isinstance(pcm, bytes) or not pcm or len(pcm) % 2:
        raise AudioContractError("PCM must be nonempty little-endian signed 16-bit bytes")
    values = array("h")
    values.frombytes(pcm)
    if values.itemsize != 2:
        raise AudioContractError("this Python runtime has no 16-bit signed array type")
    if sys.byteorder != "little":
        values.byteswap()
    return values


def _dbfs(rms_pcm: float) -> float:
    if rms_pcm <= 0:
        return DBFS_FLOOR
    return max(DBFS_FLOOR, 20.0 * math.log10(rms_pcm / 32768.0))


def _frame_rms_dbfs(samples: array, sample_rate: int) -> list[float]:
    width = max(1, round(FRAME_S * sample_rate))
    hop = max(1, round(HOP_S * sample_rate))
    if len(samples) < width:
        raise AudioContractError("PCM is shorter than one 40 ms frame")
    energy = sum(int(value) * int(value) for value in samples[:width])
    result = [_dbfs(math.sqrt(energy / width))]
    previous = 0
    for start in range(hop, len(samples) - width + 1, hop):
        removed = samples[previous:start]
        added = samples[previous + width:start + width]
        energy += sum(int(value) * int(value) for value in added)
        energy -= sum(int(value) * int(value) for value in removed)
        # Integer arithmetic should keep energy nonnegative; clamp tiny defects
        # from unusual array implementations before the square root.
        result.append(_dbfs(math.sqrt(max(0, energy) / width)))
        previous = start
    return result


def _quantile(values: list[float], proportion: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * proportion
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _pitch_hz(frame: array, sample_rate: int) -> float | None:
    """Estimate F0 from normalized, zero-mean autocorrelation of one frame."""
    stride = max(1, math.ceil(sample_rate / 8_000))
    reduced = [float(value) for value in frame[::stride]]
    effective_rate = sample_rate / stride
    mean = sum(reduced) / len(reduced)
    centered = [value - mean for value in reduced]
    min_lag = max(2, math.floor(effective_rate / F0_MAX_HZ))
    max_lag = min(len(centered) // 2, math.ceil(effective_rate / F0_MIN_HZ))
    if min_lag >= max_lag:
        return None
    candidates = []
    for lag in range(min_lag, max_lag + 1):
        left, right = centered[:-lag], centered[lag:]
        norm_left = sum(value * value for value in left)
        norm_right = sum(value * value for value in right)
        if norm_left <= 1 or norm_right <= 1:
            continue
        correlation = sum(a * b for a, b in zip(left, right)) / math.sqrt(
            norm_left * norm_right)
        candidates.append((lag, correlation))
    if not candidates:
        return None
    best = max(score for _, score in candidates)
    if best < 0.70:
        return None
    # Sines have strong peaks at multiples of the period.  Prefer the earliest
    # near-maximal peak to avoid a systematic octave-down estimate.
    lag = next(lag for lag, score in candidates if score >= max(0.70, best * 0.98))
    return effective_rate / lag


def _pitch_summary(samples: array, sample_rate: int, frame_db: list[float],
                   active_threshold_dbfs: float) -> tuple[float | None, float]:
    width = max(1, round(FRAME_S * sample_rate))
    hop = max(1, round(HOP_S * sample_rate))
    frame_count = len(frame_db)
    if frame_count <= MAX_PITCH_FRAMES:
        indices = list(range(frame_count))
    else:
        indices = sorted({round(i * (frame_count - 1) / (MAX_PITCH_FRAMES - 1))
                          for i in range(MAX_PITCH_FRAMES)})
    estimates = []
    for index in indices:
        if frame_db[index] < active_threshold_dbfs:
            continue
        start = index * hop
        estimate = _pitch_hz(samples[start:start + width], sample_rate)
        if estimate is not None:
            estimates.append(estimate)
    return ((statistics.median(estimates) if estimates else None),
            len(estimates) / len(indices))


def extract_pcm16le(pcm: bytes, sample_rate: int, window: dict, *,
                    aligned_words: list[dict] | None = None,
                    alignment_reviewed: bool = False) -> dict:
    """Return waveform and aligned-word measures with explicit missingness.

    `window` is a reviewed founder-only event segment. `aligned_words`, when
    present, contains clip-relative word intervals and requires independent
    confirmation via `alignment_reviewed=True`.
    """
    if type(sample_rate) is not int or not 8_000 <= sample_rate <= 48_000:
        raise AudioContractError("sample_rate must be an integer from 8–48 kHz")
    samples = _decode_pcm16le(pcm)
    duration_s = len(samples) / sample_rate
    provenance = validate_window(window, duration_s=duration_s)
    words = validate_aligned_words(aligned_words, duration_s=duration_s,
                                   alignment_reviewed=alignment_reviewed)
    frame_db = _frame_rms_dbfs(samples, sample_rate)
    global_rms = _dbfs(math.sqrt(sum(int(value) * int(value) for value in samples) /
                                  len(samples)))
    frame_p10 = _quantile(frame_db, 0.10)
    frame_median = statistics.median(frame_db)
    frame_p90 = _quantile(frame_db, 0.90)
    threshold = max(-65.0, frame_p90 - 25.0)
    active = [value >= threshold for value in frame_db]
    active_fraction = sum(active) / len(active)
    f0, voiced_fraction = _pitch_summary(samples, sample_rate, frame_db, threshold)
    flags = []
    missing = {}
    if active_fraction == 0:
        flags.append("no_audible_speech_energy")
        missing["pause_fraction"] = "no_audible_speech_energy"
    if f0 is None:
        missing["f0_median_hz"] = "no_reliable_periodicity"
    if voiced_fraction < 0.10:
        flags.append("low_voiced_coverage")
    if sum(abs(value) >= 32760 for value in samples) / len(samples) >= 0.01:
        flags.append("possible_clipping")
    if words is None:
        speech_rate = None
        missing["speech_rate_wpm"] = "reviewed_word_alignment_absent"
    elif not words:
        speech_rate = None
        missing["speech_rate_wpm"] = "no_aligned_words"
    else:
        aligned_span_s = words[-1]["end_s"] - words[0]["start_s"]
        if aligned_span_s < 0.05:
            speech_rate = None
            missing["speech_rate_wpm"] = "aligned_span_too_short"
        else:
            speech_rate = 60.0 * len(words) / aligned_span_s
    return {
        "schema_version": "reviewed_founder_pcm_features_v1",
        "event_id": provenance["event_id"],
        "source_media_sha256": provenance["source_media_sha256"],
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        "window_episode_s": [provenance["start_episode_s"],
                             provenance["end_episode_s"]],
        "sample_rate_hz": sample_rate,
        "duration_s": round(duration_s, 3),
        "features": {
            "global_rms_dbfs": round(global_rms, 3),
            "frame_rms_dbfs_p10": round(frame_p10, 3),
            "frame_rms_dbfs_median": round(frame_median, 3),
            "frame_rms_dbfs_p90": round(frame_p90, 3),
            "pause_fraction": round(1.0 - active_fraction, 4) if active_fraction else None,
            "f0_median_hz": round(f0, 2) if f0 is not None else None,
            "voiced_fraction": round(voiced_fraction, 4),
            "speech_rate_wpm": round(speech_rate, 2) if speech_rate is not None else None,
        },
        "measurement": {
            "frame_ms": 40,
            "hop_ms": 10,
            "pause_threshold_dbfs": round(threshold, 3),
            "f0_search_hz": [F0_MIN_HZ, F0_MAX_HZ],
            "pitch_frames_sampled": min(len(frame_db), MAX_PITCH_FRAMES),
            "aligned_word_count": len(words) if words is not None else None,
        },
        "quality_flags": flags,
        "missing": missing,
    }


def extract_wav(path: str | Path, window: dict, *,
                aligned_words: list[dict] | None = None,
                alignment_reviewed: bool = False) -> dict:
    """Read an uncompressed mono PCM16 WAV without writing decoded media."""
    try:
        with wave.open(str(path), "rb") as source:
            if (source.getnchannels() != 1 or source.getsampwidth() != 2 or
                source.getcomptype() != "NONE"):
                raise AudioContractError("WAV must be uncompressed mono PCM16")
            sample_rate = source.getframerate()
            frames = source.readframes(source.getnframes())
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioContractError(f"cannot read PCM WAV: {exc}") from exc
    return extract_pcm16le(frames, sample_rate, window,
                           aligned_words=aligned_words,
                           alignment_reviewed=alignment_reviewed)
