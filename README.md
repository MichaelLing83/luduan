# Luduan

<img src="assets/luduan-logo-128.png" alt="Luduan logo" width="120" />

- **English:** [docs/README.en.md](docs/README.en.md)
- **简体中文：** [docs/README.zh-CN.md](docs/README.zh-CN.md)

## Name origin / 名称由来

**English:** The name **Luduan** comes from **甪端**, an auspicious creature in Chinese mythology. In Mandarin Chinese, **甪端** is pronounced **lù duān**. It is often described as a one-horned beast associated with discernment, justice, peaceful rule, and understanding human language. That makes it a fitting name for a speech-to-text tool focused on listening accurately and turning speech into clear text.

**简体中文：** **Luduan / 甪端** 这个名字来自中国古代神话中的瑞兽 **甪端**。“甪端”读作 **lù duān**。甪端通常被描述为独角、通晓人言、能辨是非，并象征太平与明辨，因此很适合作为一款强调准确聆听与清晰转写的语音转文字工具名称。

## Rust CLI

Luduan is now a **Rust command-line audio-to-text tool for macOS** built on local Whisper via `whisper-rs` / `whisper.cpp`.

```bash
make check
make build
make release
```

After installation:

```bash
~/.cargo/bin/luduan --help
```

Model management:

```bash
luduan model list
luduan model download base
luduan model local
```

Context-file template:

```bash
examples/context-file-template.txt
```

Use it like this:

```bash
luduan record -f ./examples/context-file-template.txt
```
