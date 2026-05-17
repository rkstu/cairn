"""Cairn configuration — model names, thresholds, device UUIDs."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Ollama model identifiers for the triage cascade."""

    screen: str = "cairn-e4b-triage"  # Fine-tuned Gemma 4 E4B — triage-calibrated (50%→85%)
    reason: str = "gemma4:26b"  # Gemma 4 26B-A4B MoE — deep reasoning
    whisper: str = "base"  # Whisper model size for audio transcription


@dataclass(frozen=True)
class TriageThresholds:
    """Confidence thresholds for cascade escalation."""

    escalate_to_reason: float = 0.7  # Below this → escalate from screen to reason model
    flag_for_human: float = 0.5  # Below this on ANY model → flag for human review
    consistency_samples: int = 3  # Number of samples for consistency-based calibration


@dataclass(frozen=True)
class VitalBounds:
    """Deterministic vital sign safety thresholds (clinical standards)."""

    spo2_critical: int = 90  # SpO2 below this → EMERGENCY
    spo2_warning: int = 94
    hr_high_critical: int = 150
    hr_low_critical: int = 40
    systolic_bp_critical: int = 180
    diastolic_bp_critical: int = 120
    temp_high_critical: float = 40.0  # Celsius
    temp_low_critical: float = 35.0
    glucose_high_critical: int = 400  # mg/dL
    glucose_low_critical: int = 54


@dataclass(frozen=True)
class BLEServices:
    """Standard Bluetooth GATT service UUIDs for medical devices."""

    pulse_oximeter: str = "00001822-0000-1000-8000-00805f9b34fb"
    blood_pressure: str = "00001810-0000-1000-8000-00805f9b34fb"
    glucometer: str = "00001808-0000-1000-8000-00805f9b34fb"
    thermometer: str = "00001809-0000-1000-8000-00805f9b34fb"
    heart_rate: str = "0000180d-0000-1000-8000-00805f9b34fb"


@dataclass
class CairnConfig:
    """Top-level Cairn configuration."""

    models: ModelConfig = field(default_factory=ModelConfig)
    thresholds: TriageThresholds = field(default_factory=TriageThresholds)
    vitals: VitalBounds = field(default_factory=VitalBounds)
    ble: BLEServices = field(default_factory=BLEServices)
    ollama_base_url: str = "http://localhost:11434"
    db_path: str = "./data/cairn.db"
    onsides_db_path: str = "./data/onsides.db"
    sync_remote_url: str = ""  # CouchDB URL, empty = offline only
    enable_thinking: bool = True
