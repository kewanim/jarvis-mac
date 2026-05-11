import time
import threading
import sounddevice as sd
from queue import Queue
from pydantic import BaseModel
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

from stt.VoiceActivityDetection import VADDetector
from mlx_lm import load, generate
from tts.jarvis_voice import JarvisVoice

# Keep Whisper import at bottom to avoid init errors (upstream note)
from stt.whisper.transcribe import FastTranscriber

console = Console()

MASTER_PROMPT = (
    "You are J.A.R.V.I.S. — Just A Rather Very Intelligent System. "
    "You are a sophisticated AI assistant running offline on Apple Silicon. "
    "Respond in no more than three sentences. "
    "You are precise, calm, and slightly formal. Address the user as 'Sir' at all times. "
    "Only respond with dialogue — no stage directions, no asterisks, no formatting."
)

BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝

        Just A Rather Very Intelligent System
              Running Offline  ·  MLX
"""


class ChatMessage(BaseModel):
    role: str
    content: str


class JarvisClient:
    def __init__(self):
        self.listening = False
        self.history: list[ChatMessage] = []
        self.vad_data: Queue = Queue()
        self._boot()

    # ------------------------------------------------------------------ boot

    def _boot(self):
        console.clear()
        console.print(
            Panel(
                Align.center(Text(BANNER, style="bright_cyan")),
                border_style="bright_cyan",
                box=box.DOUBLE,
                padding=(0, 2),
            )
        )
        console.print()

        self._log_status("Loading speech recognition model...")
        self.stt = FastTranscriber("mlx-community/whisper-large-v3-mlx-4bit")

        self._log_status("Loading language model  (Llama 3 8B)...")
        self.model, self.tokenizer = load("mlx-community/Meta-Llama-3-8B-Instruct-4bit")

        self._log_status("Loading JARVIS voice...")
        self.tts = JarvisVoice()

        self._log_status("Calibrating microphone...")
        self.vad = VADDetector(lambda: None, self._on_speech_end, sensitivity=0.3)

        console.print()
        console.print(
            Panel(
                Align.center(
                    Text("SYSTEM ONLINE — GOOD DAY, SIR.", style="bold bright_cyan")
                ),
                border_style="bright_cyan",
                box=box.HEAVY,
                padding=(1, 4),
            )
        )
        console.print()
        self.tts.speak("Good day, Sir. J.A.R.V.I.S. is online and ready.")

    def _log_status(self, message: str):
        console.print(f"  [dim cyan]›[/dim cyan]  [cyan]{message}[/cyan]")

    # --------------------------------------------------------------- events

    def _on_speech_end(self, data):
        if data.any():
            self.vad_data.put(data)

    def _toggle_listening(self):
        self.listening = not self.listening

    # ------------------------------------------------------------ rendering

    def _divider(self, label: str, style: str):
        width = console.width or 80
        pad = (width - len(label) - 6) // 2
        line = "─" * pad
        console.print(f"  [{style}]{line}  {label}  {line}[/{style}]")

    def _print_exchange(self, content: str, role: str):
        if role == "user":
            console.print(
                Panel(
                    Text(content, style="bright_white"),
                    title="[bright_white]  YOU  [/bright_white]",
                    border_style="white",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )
        else:
            console.print(
                Panel(
                    Text(content, style="bright_cyan"),
                    title="[bright_cyan]  J.A.R.V.I.S.  [/bright_cyan]",
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                    padding=(0, 2),
                )
            )
        console.print()

    # ------------------------------------------------------------ history

    def _add_to_history(self, content: str, role: str):
        self._print_exchange(content, role)
        prompt_content = f"{MASTER_PROMPT}\n\n{content}" if role == "user" else content
        self.history.append(ChatMessage(content=prompt_content, role=role))

    def _build_prompt(self) -> str:
        result = ""
        for msg in self.history:
            result += f"<|{msg.role}|>{msg.content}<|end|>\n"
        return result

    # ------------------------------------------------------------ main loop

    def start(self):
        t_vad = threading.Thread(target=self.vad.startListening, daemon=True)
        t_vad.start()
        self._toggle_listening()
        self._divider("LISTENING", "bright_cyan")

        while True:
            if not self.vad_data.empty():
                data = self.vad_data.get()

                if self.listening and len(data) > 12000:
                    self._toggle_listening()
                    self._divider("PROCESSING SPEECH", "yellow")

                    transcribed = self.stt.transcribe(data, language="en")
                    user_text = transcribed["text"].strip()

                    if not user_text:
                        self._toggle_listening()
                        self._divider("LISTENING", "bright_cyan")
                        continue

                    self._add_to_history(user_text, "user")
                    self._divider("THINKING", "yellow")

                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=self._build_prompt() + "\n<|assistant|>",
                        verbose=False,
                        max_tokens=150,
                    )
                    response = (
                        response.split("<|assistant|>")[0]
                        .split("<|end|>")[0]
                        .strip()
                    )

                    self._add_to_history(response, "assistant")
                    self._divider("SPEAKING", "bright_green")
                    self.tts.speak(response)

                    self._toggle_listening()
                    self._divider("LISTENING", "bright_cyan")

            time.sleep(0.05)


if __name__ == "__main__":
    JarvisClient().start()
