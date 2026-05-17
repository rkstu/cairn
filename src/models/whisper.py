"""Local Whisper speech-to-text for hands-free medical dictation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class WhisperTranscriber:
    """Offline speech-to-text using OpenAI Whisper running locally."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: str | Path) -> dict[str, Any]:
        """Transcribe an audio file to text.

        Returns dict with 'text' and 'language' keys.
        """
        result = self.model.transcribe(str(audio_path))
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "en"),
            "segments": [
                {"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in result.get("segments", [])
            ],
        }

    def transcribe_text(self, audio_path: str | Path) -> str:
        """Transcribe and return just the text string."""
        return self.transcribe(audio_path)["text"]
