# Luduan

Luduan is now a **Rust command-line audio-to-text tool for macOS** built on:

- `cpal` for audio device discovery and recording
- `whisper-rs` / `whisper.cpp` for local speech-to-text
- Apple Metal acceleration on Apple Silicon

## Build

```bash
make check
make build
```

## Install

```bash
make release
```

That builds the release binary and installs it to:

```bash
~/.cargo/bin/luduan
```

## Usage

List audio devices:

```bash
luduan list-dev
```

List supported languages:

```bash
luduan list-lang
```

Record from the default input device with automatic language detection:

```bash
luduan record
```

Record with explicit language, append output to a file, and provide Whisper prompt context:

```bash
luduan record \
  -l en \
  -o transcript.txt \
  -a \
  -p "OpenAI, Luduan, whisper-rs" \
  -f ./terms.txt
```

## Record command features

- Defaults to the system default input device
- Defaults to `auto` language detection
- Accepts the full Whisper language set by code, name, or alias
- Streams transcript chunks to stdout
- Optionally writes or appends transcripts to a text file
- Supports inline prompt text and prompt context files

## Model behavior

If no local GGML Whisper model is present yet, Luduan tries to download:

```text
ggml-base.bin
```

You can also provide your own model explicitly:

```bash
luduan record --model /path/to/ggml-*.bin
```
