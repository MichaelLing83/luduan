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

使用本地 Ollama 模型在输出前修正转写结果：

```bash
luduan record \
  --ollama-model qwen2.5:7b \
  --ollama-context-file ./correction-context.txt \
  --ollama-context-file-last-lines 80 \
  --ollama-prompt "只纠正人名、地名、专有名词和明确拼写错误，不要添加任何内容。"
```

转写 WAV 文件，方便用固定音频样本做准确率调试：

```bash
luduan transcribe-file ./tests/fixtures/audio/case01.wav \
  -l zh \
  --model-name large-v3 \
  -f ./terms.txt
```

遍历参数组合，找出测试集上的最佳转写配置：

```bash
luduan benchmark \
  --fixtures tests/fixtures/e2e \
  --model-name large-v3,large-v3-turbo \
  --chunk-seconds 5,10 \
  --whisper-no-context true,false \
  --whisper-single-segment true,false \
  --whisper-best-of 1,3 \
  --silence-threshold 0.001,0.003 \
  --ollama-model none,qwen2.5:7b
```

## `record` 特性

- 默认使用系统默认音频输入设备
- 默认使用 `auto` 语言检测
- 支持完整 Whisper 语言集合，可用代码、名称或别名指定
- 会把转写结果流式输出到 stdout
- 可选写入或追加到文本文件
- 支持直接传入 prompt 文本或从上下文文件加载 prompt
- 可选调用本地 Ollama 模型，根据额外上下文修正每段转写后再输出
- Ollama 上下文文件较大时，可只读取最后 N 个非空行
- 可通过 `--ollama-prompt` 覆盖默认修正指令
- `transcribe-file` 支持同一组转写、输出和 Ollama 修正参数，用于可复现地调试 WAV 音频样本
- `record` 和 `transcribe-file` 可调整 Whisper 的 `--whisper-no-context`、`--whisper-single-segment`、`--whisper-best-of` 和 `--silence-threshold`
- `benchmark` 可以读取 fixture 音频，遍历模型、chunk 时长、Whisper 参数、Ollama 等候选参数并输出最佳组合

## 模型行为

如果本地还没有 GGML Whisper 模型，Luduan 会尝试自动下载：

```text
ggml-base.bin
```

你也可以显式指定自己的模型：

```bash
luduan record --model /path/to/ggml-*.bin
luduan record --model-name large-v3
```
