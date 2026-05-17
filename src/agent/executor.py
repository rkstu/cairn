"""Tool executor — handles Gemma 4 function calls and feeds results back."""

from __future__ import annotations

import json
from typing import Any

import ollama

from src.agent.tools import TRIAGE_TOOLS
from src.config import CairnConfig
from src.safety.drug_check import DrugSafetyChecker
from src.sync.store import LocalStore


class ToolExecutor:
    """Execute tool calls from Gemma 4 and return results.

    Implements the full agentic loop:
    1. Model receives patient info + tool definitions
    2. Model emits tool_calls
    3. We execute tools locally (BLE devices, drug DB, store)
    4. We feed results back to the model
    5. Model generates final assessment with tool results
    """

    def __init__(self, config: CairnConfig | None = None):
        self.config = config or CairnConfig()
        self.client = ollama.Client(host=self.config.ollama_base_url)
        self.drug_checker = DrugSafetyChecker(self.config.onsides_db_path)
        self.store = LocalStore(self.config.db_path)

        # Mock device readings for demo (replace with real BLE in production)
        self._mock_devices: dict[str, dict] = {}

    def register_mock_device(self, address: str, device_type: str, values: dict):
        """Register a mock BLE device for demo purposes."""
        self._mock_devices[address] = {"type": device_type, "values": values}

    def execute_tool(self, name: str, arguments: dict) -> dict[str, Any]:
        """Execute a single tool call and return the result."""
        if name == "read_pulse_oximeter":
            return self._read_device(arguments.get("device_address", ""), "pulse_oximeter")
        elif name == "read_blood_pressure":
            return self._read_device(arguments.get("device_address", ""), "blood_pressure")
        elif name == "read_glucometer":
            return self._read_device(arguments.get("device_address", ""), "glucometer")
        elif name == "check_drug_interactions":
            return self._check_drugs(arguments.get("medications", []))
        elif name == "queue_escalation_report":
            return self._queue_report(arguments)
        elif name == "scan_medical_devices":
            return {"devices": list(self._mock_devices.values())}
        else:
            return {"error": f"Unknown tool: {name}"}

    def run_agentic_loop(
        self,
        model: str,
        messages: list[dict],
        max_iterations: int = 3,
    ) -> dict[str, Any]:
        """Run the full agentic loop: model → tool calls → execution → model."""
        tool_calls_made = []

        for i in range(max_iterations):
            resp = self.client.chat(
                model=model,
                messages=messages,
                tools=TRIAGE_TOOLS,
            )
            msg = resp["message"]

            # No tool calls — model is done, return final response
            if not msg.get("tool_calls"):
                return {
                    "response": msg.get("content", ""),
                    "tool_calls": tool_calls_made,
                    "iterations": i + 1,
                }

            # Execute each tool call
            messages.append(msg)  # Add assistant message with tool calls
            for tc in msg["tool_calls"]:
                fn_name = tc["function"]["name"]
                fn_args = tc["function"]["arguments"]
                result = self.execute_tool(fn_name, fn_args)
                tool_calls_made.append({
                    "tool": fn_name,
                    "arguments": fn_args,
                    "result": result,
                })
                # Feed result back as tool response
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

        # Max iterations reached
        return {
            "response": "Max tool iterations reached. Please review manually.",
            "tool_calls": tool_calls_made,
            "iterations": max_iterations,
        }

    def _read_device(self, address: str, expected_type: str) -> dict:
        """Read from a BLE device (mock or real)."""
        if address in self._mock_devices:
            device = self._mock_devices[address]
            return {
                "status": "success",
                "device_type": device["type"],
                "address": address,
                "readings": device["values"],
            }
        return {
            "status": "error",
            "message": f"Device {address} not found. Ensure it is powered on and in range.",
        }

    def _check_drugs(self, medications: list[str]) -> dict:
        """Check drug interactions via OnSIDES."""
        if not medications:
            return {"status": "no_medications", "interactions": []}
        flags = self.drug_checker.check_all(medications)
        return {
            "status": "checked",
            "medication_count": len(medications),
            "interactions": [
                {"severity": f.severity.value, "detail": f.detail}
                for f in flags
            ],
        }

    def _queue_report(self, arguments: dict) -> dict:
        """Queue a triage report for remote sync."""
        case_id = arguments.get("patient_id", "unknown")
        self.store.save_case({
            "case_id": case_id,
            "timestamp": "",
            "patient_description": arguments.get("assessment", ""),
            "triage_level": arguments.get("triage_level", "YELLOW"),
            "confidence": arguments.get("confidence", 0.0),
            "flagged_for_human": True,
        })
        return {
            "status": "queued",
            "case_id": case_id,
            "message": "Report queued for physician review when connectivity returns.",
        }
