# Luduan

<img src="../assets/luduan-logo-128.png" alt="Luduan logo" width="120" />

[English](README.en.md) | [简体中文](README.zh-CN.md)

Luduan 现在是一个 **基于 Rust 的 macOS 命令行语音转文字工具**，底层通过 `whisper-rs` / `whisper.cpp` 使用本地 Whisper。

## 名称由来

**Luduan / 甪端** 这个名字来自中国古代神话中的瑞兽 **甪端**。
“甪端”读作 **lù duān**。
甪端通常被描述为独角、通晓人言、能辨是非，并象征太平与明辨。
因此，这个名字很适合作为一款强调准确聆听与清晰转写的语音转文字工具名称。

## 构建

```bash
make check
make build
```

## 安装

```bash
make release
```

它会构建 release 二进制并安装到：

```bash
~/.cargo/bin/luduan
```

## 模型管理

```bash
luduan model list
luduan model download base
luduan model local
```

## Context-file 模板

仓库里已经附带了一个可填写的模板：

```bash
examples/context-file-template.txt
```

可以这样使用：

```bash
luduan record -f ./examples/context-file-template.txt
```

## 用法

列出音频设备：

```bash
luduan list-dev
```

列出支持的语言：

```bash
luduan list-lang
```

使用默认输入设备并自动检测语言开始录音转写：

```bash
luduan record
```

指定语言、追加输出文件，并向 Whisper 提供上下文提示：

```bash
luduan record \
  -l en \
  -o transcript.txt \
  -a \
  -p "OpenAI, Luduan, whisper-rs" \
  -f ./terms.txt
```

## `record` 特性

- 默认使用系统默认音频输入设备
- 默认使用 `auto` 语言检测
- 支持完整 Whisper 语言集合，可用代码、名称或别名指定
- 会把转写结果流式输出到 stdout
- 可选写入或追加到文本文件
- 支持直接传入 prompt 文本或从上下文文件加载 prompt

## 模型行为

如果本地还没有 GGML Whisper 模型，Luduan 会尝试自动下载：

```text
ggml-base.bin
```

你也可以显式指定自己的模型：

```bash
luduan record --model /path/to/ggml-*.bin
```
