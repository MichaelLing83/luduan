# Luduan

[English](README.en.md) | [简体中文](README.zh-CN.md) | [Svenska](README.sv.md) | [Français](README.fr.md)

Luduan 是一个 macOS 语音转文字工具，目标是比 macOS 自带听写有更高的准确率。
它使用本地 [Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper)（通过 MLX 针对 Apple Silicon 优化）进行语音识别，并使用本地 [Ollama](https://ollama.com) LLM 做后处理。

## 名称由来

**Luduan / 甪端** 这个名字来自中国古代神话中的瑞兽 **甪端**。
“甪端”读作 **lù duān**（拼音）。
甪端通常被描述为独角、通晓人言、能辨是非，并象征太平与明辨的神兽。
因此，这个名字很适合一款强调多语言听写、准确理解和清晰转写的应用。

## 功能
- **全局快捷键**（默认：`Cmd+Shift+Space`）—— 按一次开始，再按一次结束
- **流式转写** —— Whisper 会在你说话时按音频块处理
- **LLM 后处理** —— Ollama 负责修正语法和标点
- **可选的应用上下文** —— 可读取光标前的文本，帮助识别人名、缩写和术语
- **自动粘贴** —— 转写结果会直接粘贴到当前聚焦的应用中
- **菜单栏应用** —— 常驻 macOS 菜单栏并显示状态图标
- **完全本地运行** —— 不依赖云端

## 运行要求
- macOS（推荐 Apple Silicon，M1+）
- 已安装并运行 [Ollama](https://ollama.com)
- Python 3.11+

## 安装

### 构建 macOS App（推荐）

构建成独立 `.app` 后，你授予权限的是 **Luduan** 本身，而不是 Terminal。

```bash
make app          # 构建 dist/Luduan.app
make install      # 复制到 /Applications/Luduan.app
```

安装后：
1. 打开 **系统设置 → 隐私与安全性 → 输入监控**
2. 添加 **Luduan** 并启用（全局快捷键需要）
3. 从 Spotlight 或 `/Applications` 中启动 Luduan

### 直接在终端运行（开发模式）

```bash
make dev          # 准备 .venv
make run          # 不打包，直接运行
```

> 注意：如果直接在终端运行，需要给 **Terminal** 而不是 Luduan 授予输入监控权限。

### 快捷键
- **Cmd+Shift+Space** —— 开始/停止录音
- 菜单栏图标会显示当前状态：空闲 🎙 / 录音中 🔴 / 处理中 ⏳
- 菜单栏还可以选择 **语言**、切换 **Use App Context**，以及执行 **Prepare Offline Use**

## 配置

配置文件路径：`~/.config/luduan/config.toml`

```toml
[hotkey]
keys = ["cmd", "shift", "space"]

[whisper]
model = "base"
language = ""

[ollama]
host = "http://localhost:11434"
model = "gemma3:4b"
enabled = true

[audio]
sample_rate = 16000
channels = 1

[context]
enabled = false
max_chars = 500
```

## 离线准备

你可以通过菜单栏应用提前下载 Luduan 离线运行所需的资源：

**Luduan 菜单栏图标 → Prepare Offline Use**

它会：
- 将当前配置的 Whisper 模型缓存到本地
- 确保当前配置的 Ollama 模型已经存在于本地

进度和结果会显示在菜单中的 **Offline:** 状态行中，完成后还会弹出系统通知。

完成后，只要这些模型仍保存在本地，并且你的 Mac 已安装 Ollama，Luduan 就可以离线使用。
