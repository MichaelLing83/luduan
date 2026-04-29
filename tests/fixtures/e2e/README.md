# End-to-End Transcription Fixtures

Record one audio file for each text file in `expected/`, using the same base name and placing it in `audio/`.

Example:

```text
expected/zh_01_daily_status.txt
audio/zh_01_daily_status.wav
```

Suggested recording settings:

- WAV or M4A format
- 16 kHz or 48 kHz sample rate
- mono or stereo
- 10-30 seconds per file
- read the text naturally, without adding extra words

These fixtures are intended for repeatable `benchmark` accuracy checks across languages, mixed-language content, proper nouns, numbers, and punctuation.

Run the full evaluation:

```bash
cargo run -- benchmark \
  --model-name large-v3 \
  --chunk-seconds 10
```

Or through `make`:

```bash
make benchmark ARGS="--model-name large-v3 --chunk-seconds 10"
```

Enable Ollama correction by passing a model:

```bash
cargo run -- benchmark \
  --model-name large-v3 \
  --ollama-model qwen2.5:7b \
  --ollama-context-file tests/fixtures/e2e/context.txt
```

`--model-name` uses Luduan's model cache and downloads the model when needed. You can still pass an explicit path with `--model /path/to/ggml-*.bin`.

The benchmark looks for audio with the same base name as the manifest entry. For example, `audio/zh_01_daily_status.m4a` is accepted even though the manifest lists `audio/zh_01_daily_status.wav`. Non-WAV audio requires `ffmpeg`.

Run a grid search across parameter candidates by passing comma-separated values:

```bash
cargo run -- benchmark \
  --model-name large-v3,large-v3-turbo \
  --chunk-seconds 5,10 \
  --ollama-model none,qwen2.5:7b \
  --ollama-context-file-last-lines none,40
```

Or:

```bash
make benchmark ARGS="--model-name large-v3,large-v3-turbo --chunk-seconds 5,10 --ollama-model none,qwen2.5:7b"
```

Use `none` or `null` to omit optional values for one candidate. Each combination is written under `target/benchmark/<timestamp>/`, and `summary.tsv` ranks every run by overall accuracy.
