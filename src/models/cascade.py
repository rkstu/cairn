"""Model cascade — confidence-based escalation from screen → reason model."""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import ollama

from src.config import CairnConfig


class ModelNotAvailableError(Exception):
    """Raised when a model is not available in Ollama."""
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model '{model}' not available. Run: ollama pull {model}")


def _extract_confidence(text: str) -> float:
    """Extract a verbalized confidence score from model output."""
    patterns = [
        # "Confidence: 0.95" or "**Confidence:** 0.95" or "**4) Confidence: 1.0**"
        r"\*?\*?\s*(?:\d+[\.\)]\s*)?confidence[:\s\*]*\s*([0-9]*\.?[0-9]+)",
        # "Confidence: 95%"
        r"confidence[:\s\*]*\s*([0-9]+)\s*%",
        # "0.85 confidence"
        r"([0-9]*\.[0-9]+)\s*confidence",
        # "high confidence (0.9)" or "(confidence: 0.85)"
        r"confidence[^0-9]*([0-9]*\.?[0-9]+)",
        # Standalone "0.85" or "0.9" on a line with "confidence" nearby
        r"confidence.*?([0-9]\.[0-9]+)",
    ]
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            val = float(match.group(1))
            if val > 1.0:
                return min(val / 100, 1.0)
            return val
    return 0.5  # Default uncertainty when no confidence extracted


def _extract_triage_level(text: str) -> str:
    """Extract triage level (RED/YELLOW/GREEN) from model output.

    Handles negation patterns like "RED - NO, GREEN - YES" where the
    fine-tuned model lists all levels with YES/NO markers. Uses [^,]*?
    to restrict matching within comma-separated clauses.
    """
    upper = text.upper()

    for pattern in [
        r"TRIAGE\s*(?:LEVEL)?[:\s]*\*{0,2}\s*(RED|YELLOW|GREEN)",
        r"CLASSIFICATION[:\s]*\*{0,2}\s*(RED|YELLOW|GREEN)",
        r"LEVEL[:\s]*\*{0,2}\s*(RED|YELLOW|GREEN)",
    ]:
        m = re.search(pattern, upper)
        if m:
            return m.group(1)

    confirmed = []
    negated = set()
    for level in ["RED", "YELLOW", "GREEN"]:
        if re.search(rf"\b{level}\b[^,]*?-\s*YES", upper):
            confirmed.append(level)
        if re.search(rf"\b{level}\b[^,]*?-\s*NO", upper):
            negated.add(level)

    if confirmed:
        return confirmed[0]

    for level in ["RED", "YELLOW", "GREEN"]:
        if level not in negated and re.search(rf"\b{level}\b", upper):
            return level

    return "YELLOW"


class ModelCascade:
    """Confidence-based model cascade using Ollama.

    E4B screens every case. If confidence is below threshold or case is
    high-acuity, escalates to 26B-A4B with thinking mode.
    """

    def __init__(self, config: CairnConfig | None = None):
        self.config = config or CairnConfig()
        self.client = ollama.Client(host=self.config.ollama_base_url)
        self._available_models: set[str] | None = None

    def _is_model_available(self, model: str) -> bool:
        """Check if a model is loaded in Ollama (cached on first call)."""
        if self._available_models is None:
            try:
                self._available_models = {
                    m.model for m in self.client.list()["models"]
                }
            except Exception:
                self._available_models = set()
        # Match exact name, or name:latest against name
        if model in self._available_models:
            return True
        if f"{model}:latest" in self._available_models:
            return True
        return False

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
    ) -> str:
        """Call an Ollama model and return the response text."""
        if not self._is_model_available(model):
            raise ModelNotAvailableError(model)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        try:
            response = self.client.chat(**kwargs)
        except Exception as e:
            if "not found" in str(e).lower():
                raise ModelNotAvailableError(model) from e
            raise
        return response["message"]["content"]

    def _consistency_check(
        self,
        model: str,
        messages: list[dict[str, Any]],
        n_samples: int = 3,
    ) -> tuple[float, str]:
        """Run N samples and measure agreement on triage level."""
        levels: list[str] = []
        responses: list[str] = []
        for _ in range(n_samples):
            resp = self._call_model(model, messages)
            levels.append(_extract_triage_level(resp))
            responses.append(resp)
        most_common = Counter(levels).most_common(1)[0]
        consistency = most_common[1] / len(levels)
        # Return the response that matches the most common triage level
        for resp, level in zip(responses, levels):
            if level == most_common[0]:
                return consistency, resp
        return consistency, responses[0]

    def triage(
        self,
        patient_description: str,
        image_path: str | None = None,
        vitals: dict | None = None,
        tools: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Run the triage cascade. Returns assessment with confidence and model used."""

        # Build the prompt
        context_parts = [patient_description]
        if vitals:
            context_parts.append(f"\nVital signs: {json.dumps(vitals)}")

        system_msg = {
            "role": "system",
            "content": (
                "You are Cairn, an offline medical triage assistant for community health workers. "
                "Assess the patient and provide:\n"
                "1. Triage level: RED (immediate), YELLOW (urgent), or GREEN (non-urgent)\n"
                "2. Assessment: brief clinical reasoning\n"
                "3. Recommendations: actionable next steps\n"
                "4. Confidence: 0.0 to 1.0 — how confident you are in this assessment\n"
                "5. If uncertain, say so explicitly. Never guess on medication dosages.\n"
                "Format your response as structured text with clear labels."
            ),
        }

        user_msg: dict[str, Any] = {
            "role": "user",
            "content": "\n".join(context_parts),
        }
        if image_path:
            user_msg["images"] = [image_path]

        messages = [system_msg, user_msg]

        # Stage 1: Screen with E4B
        screen_response = self._call_model(self.config.models.screen, messages)
        screen_confidence = _extract_confidence(screen_response)
        triage_level = _extract_triage_level(screen_response)

        is_high_acuity = triage_level == "RED"
        needs_escalation = screen_confidence < self.config.thresholds.escalate_to_reason

        if not needs_escalation and not is_high_acuity:
            return {
                "model_used": self.config.models.screen,
                "stage": "screen",
                "triage_level": triage_level,
                "confidence": screen_confidence,
                "assessment": screen_response,
                "escalated": False,
                "flagged_for_human": screen_confidence < self.config.thresholds.flag_for_human,
            }

        # Stage 2: Escalate to 26B-A4B with prior context
        reason_messages = [
            system_msg,
            {
                "role": "user",
                "content": (
                    f"{user_msg['content']}\n\n"
                    f"Prior screening assessment (confidence {screen_confidence:.2f}):\n"
                    f"{screen_response}\n\n"
                    "Please provide a thorough second assessment. Use step-by-step reasoning. "
                    "If you disagree with the screening, explain why."
                ),
            },
        ]
        if image_path:
            reason_messages[1]["images"] = [image_path]

        try:
            # Use consistency sampling for high-stakes cases
            if is_high_acuity:
                consistency, reason_response = self._consistency_check(
                    self.config.models.reason,
                    reason_messages,
                    n_samples=self.config.thresholds.consistency_samples,
                )
                reason_confidence = (
                    0.6 * _extract_confidence(reason_response) + 0.4 * consistency
                )
            else:
                reason_response = self._call_model(
                    self.config.models.reason, reason_messages, tools=tools
                )
                reason_confidence = _extract_confidence(reason_response)
        except ModelNotAvailableError:
            # Reason model not available — return screen result with a note
            return {
                "model_used": self.config.models.screen,
                "stage": "screen_only",
                "triage_level": triage_level,
                "confidence": screen_confidence,
                "assessment": screen_response,
                "escalated": False,
                "escalation_attempted": True,
                "escalation_reason": "high_acuity" if is_high_acuity else "low_confidence",
                "note": f"Escalation model '{self.config.models.reason}' not available. Using screen result.",
                "flagged_for_human": True,  # Always flag when escalation fails
            }

        return {
            "model_used": self.config.models.reason,
            "stage": "reason",
            "triage_level": _extract_triage_level(reason_response),
            "confidence": reason_confidence,
            "assessment": reason_response,
            "screen_assessment": screen_response,
            "screen_confidence": screen_confidence,
            "escalated": True,
            "escalation_reason": "high_acuity" if is_high_acuity else "low_confidence",
            "flagged_for_human": reason_confidence < self.config.thresholds.flag_for_human,
        }
