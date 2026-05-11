# jarvis-mac

A personal AI voice assistant running fully offline on Apple Silicon Mac.

Inspired by [HuW Prosser's jarvis-mlx](https://github.com/huwprosser/jarvis-mlx), built and customised for my own use.

## Features
- 🎙️ Offline speech-to-text via Whisper
- 🧠 Local LLM inference via Apple MLX (Phi-3 / Llama 3)
- 🔊 Local text-to-speech via MeloTTS
- 🔒 100% private — no cloud, no subscriptions

## Hardware
- Apple Silicon Mac (M1+)
- 16GB RAM recommended

## Setup
See [docs/setup.md](docs/setup.md) for full installation instructions.

## Branch Structure
- `main` — stable releases only
- `dev` — active development
- `feature/stt` — speech-to-text
- `feature/llm` — language model
- `feature/tts` — text-to-speech
