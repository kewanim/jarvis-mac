"""
Jarvis voice module — Piper TTS with jgkawell/jarvis ONNX model.
Downloads the model on first run (~60MB), cached locally after that.
"""

import io
import wave
import struct
import sounddevice as sd
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download

MODEL_REPO = "jgkawell/jarvis"
MODEL_FILE = "en/en_GB/jarvis/medium/en_GB-jarvis-medium.onnx"
CONFIG_FILE = "en/en_GB/jarvis/medium/en_GB-jarvis-medium.onnx.json"
CACHE_DIR = Path.home() / ".cache" / "jarvis-mac" / "voice"


class JarvisVoice:
    def __init__(self):
        self._ensure_model()
        self._load()

    def _ensure_model(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.model_path = CACHE_DIR / "jarvis-medium.onnx"
        self.config_path = CACHE_DIR / "jarvis-medium.onnx.json"

        if not self.model_path.exists():
            print("  [cyan]Downloading JARVIS voice model (~60MB, first run only)...[/cyan]")
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=MODEL_FILE,
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            hf_hub_download(
                repo_id=MODEL_REPO,
                filename=CONFIG_FILE,
                local_dir=CACHE_DIR,
                local_dir_use_symlinks=False,
            )
            # Move to flat cache location
            downloaded_model = CACHE_DIR / MODEL_FILE
            downloaded_config = CACHE_DIR / CONFIG_FILE
            downloaded_model.rename(self.model_path)
            downloaded_config.rename(self.config_path)

    def _load(self):
        from piper import PiperVoice
        self.voice = PiperVoice.load(str(self.model_path), config_path=str(self.config_path))

    def speak(self, text: str):
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            self.voice.synthesize(text, wf)

        buf.seek(0)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)

        sd.play(audio, samplerate=sample_rate, blocking=True)
