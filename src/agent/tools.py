"""Gemma 4 native function calling tool definitions for triage workflows."""

TRIAGE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_pulse_oximeter",
            "description": "Read SpO2 percentage and pulse rate from a connected BLE pulse oximeter",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_address": {
                        "type": "string",
                        "description": "BLE MAC address of the pulse oximeter",
                    }
                },
                "required": ["device_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_blood_pressure",
            "description": "Read systolic/diastolic blood pressure from a connected BLE monitor",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_address": {
                        "type": "string",
                        "description": "BLE MAC address of the blood pressure monitor",
                    }
                },
                "required": ["device_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_glucometer",
            "description": "Read blood glucose level from a connected BLE glucometer",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_address": {
                        "type": "string",
                        "description": "BLE MAC address of the glucometer",
                    }
                },
                "required": ["device_address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_drug_interactions",
            "description": (
                "Check for known adverse drug interactions between medications. "
                "Uses offline OnSIDES database — no internet required."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "medications": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of medication names to check",
                    }
                },
                "required": ["medications"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_escalation_report",
            "description": (
                "Queue a structured FHIR triage report for sync to remote server "
                "when connectivity is restored. Used for cases requiring human review."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "triage_level": {
                        "type": "string",
                        "enum": ["RED", "YELLOW", "GREEN"],
                    },
                    "assessment": {"type": "string"},
                    "confidence": {"type": "number"},
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["patient_id", "triage_level", "assessment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_medical_devices",
            "description": "Scan for nearby BLE medical devices (pulse oximeters, BP monitors, glucometers)",
            "parameters": {
                "type": "object",
                "properties": {
                    "timeout": {
                        "type": "number",
                        "description": "Scan duration in seconds (default 10)",
                    }
                },
            },
        },
    },
]
