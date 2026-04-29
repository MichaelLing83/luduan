#![allow(unexpected_cfgs)]

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample, Stream, StreamConfig, SupportedStreamConfig};
use directories::ProjectDirs;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CHUNK_SECONDS: f32 = 5.0;
const MIN_FLUSH_SECONDS: f32 = 0.5;
const SILENCE_RMS_THRESHOLD: f32 = 0.003;
const DEFAULT_MODEL_NAME: &str = "ggml-base.bin";
const MODEL_BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";
const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";
const DEFAULT_OLLAMA_CORRECTION_PROMPT: &str = "Correct the following speech-to-text transcript using the provided context. \
Do not add, remove, complete, summarize, or rewrite any content. \
Only correct clear recognition or spelling errors, especially people names, place names, and proper nouns. \
Preserve the original wording, meaning, language, and punctuation unless a spelling correction requires a minimal change. \
Output only the corrected transcript, with no explanation.";
const AVAILABLE_MODELS: &[ModelSpec] = &[
    ModelSpec::new("tiny", "ggml-tiny.bin", "75 MiB"),
    ModelSpec::new("tiny.en", "ggml-tiny.en.bin", "75 MiB"),
    ModelSpec::new("tiny-q5_1", "ggml-tiny-q5_1.bin", "31 MiB"),
    ModelSpec::new("tiny.en-q5_1", "ggml-tiny.en-q5_1.bin", "31 MiB"),
    ModelSpec::new("tiny-q8_0", "ggml-tiny-q8_0.bin", "42 MiB"),
    ModelSpec::new("base", "ggml-base.bin", "142 MiB"),
    ModelSpec::new("base.en", "ggml-base.en.bin", "142 MiB"),
    ModelSpec::new("base-q5_1", "ggml-base-q5_1.bin", "57 MiB"),
    ModelSpec::new("base.en-q5_1", "ggml-base.en-q5_1.bin", "57 MiB"),
    ModelSpec::new("base-q8_0", "ggml-base-q8_0.bin", "78 MiB"),
    ModelSpec::new("small", "ggml-small.bin", "466 MiB"),
    ModelSpec::new("small.en", "ggml-small.en.bin", "466 MiB"),
    ModelSpec::new("small.en-tdrz", "ggml-small.en-tdrz.bin", "465 MiB"),
    ModelSpec::new("small-q5_1", "ggml-small-q5_1.bin", "182 MiB"),
    ModelSpec::new("small.en-q5_1", "ggml-small.en-q5_1.bin", "182 MiB"),
    ModelSpec::new("small-q8_0", "ggml-small-q8_0.bin", "252 MiB"),
    ModelSpec::new("medium", "ggml-medium.bin", "1.5 GiB"),
    ModelSpec::new("medium.en", "ggml-medium.en.bin", "1.5 GiB"),
    ModelSpec::new("medium-q5_0", "ggml-medium-q5_0.bin", "514 MiB"),
    ModelSpec::new("medium.en-q5_0", "ggml-medium.en-q5_0.bin", "514 MiB"),
    ModelSpec::new("medium-q8_0", "ggml-medium-q8_0.bin", "785 MiB"),
    ModelSpec::new("large-v1", "ggml-large-v1.bin", "2.9 GiB"),
    ModelSpec::new("large-v2", "ggml-large-v2.bin", "2.9 GiB"),
    ModelSpec::new("large-v2-q5_0", "ggml-large-v2-q5_0.bin", "1.1 GiB"),
    ModelSpec::new("large-v2-q8_0", "ggml-large-v2-q8_0.bin", "1.5 GiB"),
    ModelSpec::new("large-v3", "ggml-large-v3.bin", "2.9 GiB"),
    ModelSpec::new("large-v3-q5_0", "ggml-large-v3-q5_0.bin", "1.1 GiB"),
    ModelSpec::new("large-v3-turbo", "ggml-large-v3-turbo.bin", "1.5 GiB"),
    ModelSpec::new(
        "large-v3-turbo-q5_0",
        "ggml-large-v3-turbo-q5_0.bin",
        "547 MiB",
    ),
    ModelSpec::new(
        "large-v3-turbo-q8_0",
        "ggml-large-v3-turbo-q8_0.bin",
        "834 MiB",
    ),
];
const SUPPORTED_LANGUAGES: &[LanguageSpec] = &[
    LanguageSpec::auto(),
    LanguageSpec::new("English", "en", &["english", "en"]),
    LanguageSpec::new("Chinese", "zh", &["chinese", "zh", "cn", "mandarin"]),
    LanguageSpec::new("German", "de", &["german", "de"]),
    LanguageSpec::new("Spanish", "es", &["spanish", "es", "castilian"]),
    LanguageSpec::new("Russian", "ru", &["russian", "ru"]),
    LanguageSpec::new("Korean", "ko", &["korean", "ko"]),
    LanguageSpec::new("French", "fr", &["french", "fr"]),
    LanguageSpec::new("Japanese", "ja", &["japanese", "ja"]),
    LanguageSpec::new("Portuguese", "pt", &["portuguese", "pt"]),
    LanguageSpec::new("Turkish", "tr", &["turkish", "tr"]),
    LanguageSpec::new("Polish", "pl", &["polish", "pl"]),
    LanguageSpec::new("Catalan", "ca", &["catalan", "ca", "valencian"]),
    LanguageSpec::new("Dutch", "nl", &["dutch", "nl", "flemish"]),
    LanguageSpec::new("Arabic", "ar", &["arabic", "ar"]),
    LanguageSpec::new("Swedish", "sv", &["swedish", "sv", "se"]),
    LanguageSpec::new("Italian", "it", &["italian", "it"]),
    LanguageSpec::new("Indonesian", "id", &["indonesian", "id"]),
    LanguageSpec::new("Hindi", "hi", &["hindi", "hi"]),
    LanguageSpec::new("Finnish", "fi", &["finnish", "fi"]),
    LanguageSpec::new("Vietnamese", "vi", &["vietnamese", "vi"]),
    LanguageSpec::new("Hebrew", "he", &["hebrew", "he"]),
    LanguageSpec::new("Ukrainian", "uk", &["ukrainian", "uk"]),
    LanguageSpec::new("Greek", "el", &["greek", "el"]),
    LanguageSpec::new("Malay", "ms", &["malay", "ms"]),
    LanguageSpec::new("Czech", "cs", &["czech", "cs"]),
    LanguageSpec::new(
        "Romanian",
        "ro",
        &["romanian", "ro", "moldavian", "moldovan"],
    ),
    LanguageSpec::new("Danish", "da", &["danish", "da"]),
    LanguageSpec::new("Hungarian", "hu", &["hungarian", "hu"]),
    LanguageSpec::new("Tamil", "ta", &["tamil", "ta"]),
    LanguageSpec::new("Norwegian", "no", &["norwegian", "no"]),
    LanguageSpec::new("Thai", "th", &["thai", "th"]),
    LanguageSpec::new("Urdu", "ur", &["urdu", "ur"]),
    LanguageSpec::new("Croatian", "hr", &["croatian", "hr"]),
    LanguageSpec::new("Bulgarian", "bg", &["bulgarian", "bg"]),
    LanguageSpec::new("Lithuanian", "lt", &["lithuanian", "lt"]),
    LanguageSpec::new("Latin", "la", &["latin", "la"]),
    LanguageSpec::new("Maori", "mi", &["maori", "mi"]),
    LanguageSpec::new("Malayalam", "ml", &["malayalam", "ml"]),
    LanguageSpec::new("Welsh", "cy", &["welsh", "cy"]),
    LanguageSpec::new("Slovak", "sk", &["slovak", "sk"]),
    LanguageSpec::new("Telugu", "te", &["telugu", "te"]),
    LanguageSpec::new("Persian", "fa", &["persian", "fa"]),
    LanguageSpec::new("Latvian", "lv", &["latvian", "lv"]),
    LanguageSpec::new("Bengali", "bn", &["bengali", "bn"]),
    LanguageSpec::new("Serbian", "sr", &["serbian", "sr"]),
    LanguageSpec::new("Azerbaijani", "az", &["azerbaijani", "az"]),
    LanguageSpec::new("Slovenian", "sl", &["slovenian", "sl"]),
    LanguageSpec::new("Kannada", "kn", &["kannada", "kn"]),
    LanguageSpec::new("Estonian", "et", &["estonian", "et"]),
    LanguageSpec::new("Macedonian", "mk", &["macedonian", "mk"]),
    LanguageSpec::new("Breton", "br", &["breton", "br"]),
    LanguageSpec::new("Basque", "eu", &["basque", "eu"]),
    LanguageSpec::new("Icelandic", "is", &["icelandic", "is"]),
    LanguageSpec::new("Armenian", "hy", &["armenian", "hy"]),
    LanguageSpec::new("Nepali", "ne", &["nepali", "ne"]),
    LanguageSpec::new("Mongolian", "mn", &["mongolian", "mn"]),
    LanguageSpec::new("Bosnian", "bs", &["bosnian", "bs"]),
    LanguageSpec::new("Kazakh", "kk", &["kazakh", "kk"]),
    LanguageSpec::new("Albanian", "sq", &["albanian", "sq"]),
    LanguageSpec::new("Swahili", "sw", &["swahili", "sw"]),
    LanguageSpec::new("Galician", "gl", &["galician", "gl"]),
    LanguageSpec::new("Marathi", "mr", &["marathi", "mr"]),
    LanguageSpec::new("Punjabi", "pa", &["punjabi", "pa", "panjabi"]),
    LanguageSpec::new("Sinhala", "si", &["sinhala", "si", "sinhalese"]),
    LanguageSpec::new("Khmer", "km", &["khmer", "km"]),
    LanguageSpec::new("Shona", "sn", &["shona", "sn"]),
    LanguageSpec::new("Yoruba", "yo", &["yoruba", "yo"]),
    LanguageSpec::new("Somali", "so", &["somali", "so"]),
    LanguageSpec::new("Afrikaans", "af", &["afrikaans", "af"]),
    LanguageSpec::new("Occitan", "oc", &["occitan", "oc"]),
    LanguageSpec::new("Georgian", "ka", &["georgian", "ka"]),
    LanguageSpec::new("Belarusian", "be", &["belarusian", "be"]),
    LanguageSpec::new("Tajik", "tg", &["tajik", "tg"]),
    LanguageSpec::new("Sindhi", "sd", &["sindhi", "sd"]),
    LanguageSpec::new("Gujarati", "gu", &["gujarati", "gu"]),
    LanguageSpec::new("Amharic", "am", &["amharic", "am"]),
    LanguageSpec::new("Yiddish", "yi", &["yiddish", "yi"]),
    LanguageSpec::new("Lao", "lo", &["lao", "lo"]),
    LanguageSpec::new("Uzbek", "uz", &["uzbek", "uz"]),
    LanguageSpec::new("Faroese", "fo", &["faroese", "fo"]),
    LanguageSpec::new("Haitian Creole", "ht", &["haitian creole", "ht", "haitian"]),
    LanguageSpec::new("Pashto", "ps", &["pashto", "ps", "pushto"]),
    LanguageSpec::new("Turkmen", "tk", &["turkmen", "tk"]),
    LanguageSpec::new("Nynorsk", "nn", &["nynorsk", "nn", "norwegian nynorsk"]),
    LanguageSpec::new("Maltese", "mt", &["maltese", "mt"]),
    LanguageSpec::new("Sanskrit", "sa", &["sanskrit", "sa"]),
    LanguageSpec::new(
        "Luxembourgish",
        "lb",
        &["luxembourgish", "lb", "letzeburgesch"],
    ),
    LanguageSpec::new("Myanmar", "my", &["myanmar", "my", "burmese"]),
    LanguageSpec::new("Tibetan", "bo", &["tibetan", "bo"]),
    LanguageSpec::new("Tagalog", "tl", &["tagalog", "tl"]),
    LanguageSpec::new("Malagasy", "mg", &["malagasy", "mg"]),
    LanguageSpec::new("Assamese", "as", &["assamese", "as"]),
    LanguageSpec::new("Tatar", "tt", &["tatar", "tt"]),
    LanguageSpec::new("Hawaiian", "haw", &["hawaiian", "haw"]),
    LanguageSpec::new("Lingala", "ln", &["lingala", "ln"]),
    LanguageSpec::new("Hausa", "ha", &["hausa", "ha"]),
    LanguageSpec::new("Bashkir", "ba", &["bashkir", "ba"]),
    LanguageSpec::new("Javanese", "jw", &["javanese", "jw"]),
    LanguageSpec::new("Sundanese", "su", &["sundanese", "su"]),
    LanguageSpec::new("Cantonese", "yue", &["cantonese", "yue"]),
];

#[derive(Parser, Debug)]
#[command(name = "luduan")]
#[command(about = "Local audio device listing and streaming speech-to-text")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List available audio input and output devices
    ListDev,
    /// List supported transcription languages and short codes
    ListLang,
    /// Manage downloadable Whisper models
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    /// Record from an input device and stream transcripts to stdout
    Record(RecordArgs),
    /// Transcribe an audio file and stream transcripts to stdout
    TranscribeFile(TranscribeFileArgs),
    /// Benchmark transcription accuracy across fixture audio and parameter grids
    Benchmark(BenchmarkArgs),
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    /// List models that can be downloaded
    List,
    /// Download a model into Luduan's local cache
    Download(DownloadModelArgs),
    /// List models already downloaded into Luduan's local cache
    Local,
}

#[derive(Args, Debug)]
struct DownloadModelArgs {
    /// Model name from `luduan model list`, e.g. `base`, `large-v3`, `base.en`
    model: String,
}

#[derive(Args, Debug)]
struct RecordArgs {
    /// Input device index from `list-dev`, `default`, or part/all of the device name
    #[arg(short = 'i', long, default_value = "default")]
    input: String,

    /// Optional output text file for transcribed chunks
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Append to the output file instead of overwriting it
    #[arg(short = 'a', long)]
    append: bool,

    /// Language: auto, or any supported Whisper language name/code (see `list-lang`)
    #[arg(short = 'l', long, default_value = "auto")]
    language: String,

    /// Inline prompt text to guide Whisper with domain terms or names
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Read additional prompt context from a text file
    #[arg(short = 'f', long)]
    context_file: Option<PathBuf>,

    /// Ollama model used to correct each transcribed chunk before output
    #[arg(long)]
    ollama_model: Option<String>,

    /// Ollama server base URL
    #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
    ollama_url: String,

    /// Inline context text to guide Ollama transcript correction
    #[arg(long)]
    ollama_context: Option<String>,

    /// Read additional Ollama correction context from a text file
    #[arg(long)]
    ollama_context_file: Option<PathBuf>,

    /// Only read the last N non-empty lines from --ollama-context-file
    #[arg(long)]
    ollama_context_file_last_lines: Option<usize>,

    /// Prompt that instructs Ollama how to correct transcript chunks.
    ///
    /// Defaults to: "Correct the following speech-to-text transcript using the provided context. Do not add, remove, complete, summarize, or rewrite any content. Only correct clear recognition or spelling errors, especially people names, place names, and proper nouns. Preserve the original wording, meaning, language, and punctuation unless a spelling correction requires a minimal change. Output only the corrected transcript, with no explanation."
    #[arg(long)]
    ollama_prompt: Option<String>,

    /// Path to a whisper.cpp GGML model file
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Model name from `luduan model list`, e.g. `base`, `large-v3`, `base.en`
    #[arg(long)]
    model_name: Option<String>,

    /// Chunk size in seconds for streaming transcription
    #[arg(short = 'c', long, default_value_t = DEFAULT_CHUNK_SECONDS)]
    chunk_seconds: f32,

    /// Whether each Whisper chunk should ignore previous context
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    whisper_no_context: bool,

    /// Whether Whisper should force a single output segment per chunk
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    whisper_single_segment: bool,

    /// Whisper greedy decoder best_of value
    #[arg(long, default_value_t = 1)]
    whisper_best_of: i32,

    /// RMS threshold below which chunks are treated as silence
    #[arg(long, default_value_t = SILENCE_RMS_THRESHOLD)]
    silence_threshold: f32,
}

#[derive(Args, Debug)]
struct TranscribeFileArgs {
    /// WAV audio file to transcribe
    input: PathBuf,

    /// Optional output text file for transcribed chunks
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Append to the output file instead of overwriting it
    #[arg(short = 'a', long)]
    append: bool,

    /// Language: auto, or any supported Whisper language name/code (see `list-lang`)
    #[arg(short = 'l', long, default_value = "auto")]
    language: String,

    /// Inline prompt text to guide Whisper with domain terms or names
    #[arg(short = 'p', long)]
    prompt: Option<String>,

    /// Read additional prompt context from a text file
    #[arg(short = 'f', long)]
    context_file: Option<PathBuf>,

    /// Ollama model used to correct each transcribed chunk before output
    #[arg(long)]
    ollama_model: Option<String>,

    /// Ollama server base URL
    #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
    ollama_url: String,

    /// Inline context text to guide Ollama transcript correction
    #[arg(long)]
    ollama_context: Option<String>,

    /// Read additional Ollama correction context from a text file
    #[arg(long)]
    ollama_context_file: Option<PathBuf>,

    /// Only read the last N non-empty lines from --ollama-context-file
    #[arg(long)]
    ollama_context_file_last_lines: Option<usize>,

    /// Prompt that instructs Ollama how to correct transcript chunks.
    ///
    /// Defaults to: "Correct the following speech-to-text transcript using the provided context. Do not add, remove, complete, summarize, or rewrite any content. Only correct clear recognition or spelling errors, especially people names, place names, and proper nouns. Preserve the original wording, meaning, language, and punctuation unless a spelling correction requires a minimal change. Output only the corrected transcript, with no explanation."
    #[arg(long)]
    ollama_prompt: Option<String>,

    /// Path to a whisper.cpp GGML model file
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Model name from `luduan model list`, e.g. `base`, `large-v3`, `base.en`
    #[arg(long)]
    model_name: Option<String>,

    /// Chunk size in seconds for streaming transcription
    #[arg(short = 'c', long, default_value_t = DEFAULT_CHUNK_SECONDS)]
    chunk_seconds: f32,

    /// Whether each Whisper chunk should ignore previous context
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    whisper_no_context: bool,

    /// Whether Whisper should force a single output segment per chunk
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    whisper_single_segment: bool,

    /// Whisper greedy decoder best_of value
    #[arg(long, default_value_t = 1)]
    whisper_best_of: i32,

    /// RMS threshold below which chunks are treated as silence
    #[arg(long, default_value_t = SILENCE_RMS_THRESHOLD)]
    silence_threshold: f32,
}

#[derive(Args, Debug)]
struct BenchmarkArgs {
    /// Fixture directory containing manifest.tsv, expected/, audio/, and optional context.txt
    #[arg(long, default_value = "tests/fixtures/e2e")]
    fixtures: PathBuf,

    /// Run only this case id. Can be passed multiple times
    #[arg(long = "case")]
    cases: Vec<String>,

    /// Path to a whisper.cpp GGML model file
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Comma-separated model names from `luduan model list`, e.g. large-v3,large-v3-turbo
    #[arg(long)]
    model_name: Option<String>,

    /// Comma-separated chunk sizes in seconds, e.g. 5,10,15
    #[arg(long, default_value = "5")]
    chunk_seconds: String,

    /// Comma-separated Whisper no-context values, true or false
    #[arg(long, default_value = "true")]
    whisper_no_context: String,

    /// Comma-separated Whisper single-segment values, true or false
    #[arg(long, default_value = "true")]
    whisper_single_segment: String,

    /// Comma-separated Whisper greedy decoder best_of values, e.g. 1,3,5
    #[arg(long, default_value = "1")]
    whisper_best_of: String,

    /// Comma-separated RMS silence thresholds, e.g. 0,0.001,0.003
    #[arg(long, default_value = "0.003")]
    silence_threshold: String,

    /// Comma-separated languages, or fixture to use each manifest language
    #[arg(long, default_value = "fixture")]
    language: String,

    /// Whisper context file. Defaults to fixtures/context.txt when present
    #[arg(short = 'f', long)]
    context_file: Option<PathBuf>,

    /// Do not pass a Whisper context file
    #[arg(long)]
    no_context: bool,

    /// Comma-separated Ollama models. Use none to disable Ollama for a run
    #[arg(long, default_value = "none")]
    ollama_model: String,

    /// Ollama server base URL
    #[arg(long, default_value = DEFAULT_OLLAMA_URL)]
    ollama_url: String,

    /// Ollama correction context file. Defaults to fixtures/context.txt when Ollama is enabled
    #[arg(long)]
    ollama_context_file: Option<PathBuf>,

    /// Comma-separated last-line limits for Ollama context. Use none for full file
    #[arg(long, default_value = "none")]
    ollama_context_file_last_lines: String,

    /// Prompt that instructs Ollama how to correct transcript chunks
    #[arg(long)]
    ollama_prompt: Option<String>,

    /// Output directory for benchmark runs
    #[arg(long, default_value = "target/benchmark")]
    output_dir: PathBuf,

    /// Keep converted WAV files in the output directory
    #[arg(long)]
    keep_wav: bool,

    /// Fail if a manifest case has no matching audio file
    #[arg(long)]
    require_audio: bool,

    /// Keep letter case when computing accuracy
    #[arg(long)]
    preserve_case: bool,

    /// Keep punctuation when computing accuracy
    #[arg(long)]
    keep_punctuation: bool,

    /// Keep whitespace when computing accuracy
    #[arg(long)]
    keep_whitespace: bool,

    /// Exit with failure if best overall accuracy is below this percentage
    #[arg(long)]
    fail_under: Option<f32>,
}

#[derive(Clone)]
struct InputDeviceInfo {
    index: usize,
    name: String,
    device: cpal::Device,
    config: SupportedStreamConfig,
    is_default: bool,
}

#[derive(Default)]
struct CaptureBuffer {
    pending: Vec<f32>,
    chunk_frames: usize,
}

struct LanguageSpec {
    name: &'static str,
    code: Option<&'static str>,
    aliases: &'static [&'static str],
}

struct ModelSpec {
    name: &'static str,
    file_name: &'static str,
    size: &'static str,
}

struct OllamaCorrection {
    endpoint: String,
    model: String,
    correction_prompt: String,
    context: Option<String>,
    client: Client,
}

#[derive(Clone, Copy)]
struct WhisperOptions {
    no_context: bool,
    single_segment: bool,
    best_of: i32,
    silence_threshold: f32,
}

#[derive(Clone)]
struct BenchmarkCase {
    id: String,
    language: String,
    expected_text: PathBuf,
    audio_file: PathBuf,
}

#[derive(Clone)]
struct BenchmarkConfig {
    model_label: String,
    model_path: PathBuf,
    chunk_seconds: f32,
    language: Option<String>,
    ollama_model: Option<String>,
    ollama_context_file_last_lines: Option<usize>,
    whisper: WhisperOptions,
}

struct BenchmarkCaseResult {
    case_id: String,
    language: String,
    status: String,
    expected_chars: usize,
    distance: usize,
    accuracy: f32,
    message: Option<String>,
}

struct BenchmarkRunResult {
    index: usize,
    config: BenchmarkConfig,
    output_dir: PathBuf,
    cases: Vec<BenchmarkCaseResult>,
    accuracy: f32,
}

#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaGenerateResponse {
    response: String,
}

impl LanguageSpec {
    const fn auto() -> Self {
        Self {
            name: "Auto Detect",
            code: None,
            aliases: &["auto"],
        }
    }

    const fn new(name: &'static str, code: &'static str, aliases: &'static [&'static str]) -> Self {
        Self {
            name,
            code: Some(code),
            aliases,
        }
    }
}

impl ModelSpec {
    const fn new(name: &'static str, file_name: &'static str, size: &'static str) -> Self {
        Self {
            name,
            file_name,
            size,
        }
    }

    fn url(&self) -> String {
        format!("{MODEL_BASE_URL}/{}", self.file_name)
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::ListDev => list_devices(),
        Commands::ListLang => list_languages(),
        Commands::Model { command } => model_command(command),
        Commands::Record(args) => record(args),
        Commands::TranscribeFile(args) => transcribe_file(args),
        Commands::Benchmark(args) => benchmark(args),
    }
}

fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default_input_name = host.default_input_device().map(|d| device_name(&d));
    let default_output_name = host.default_output_device().map(|d| device_name(&d));

    let devices = host
        .devices()
        .context("failed to enumerate audio devices")?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    for device in devices {
        let name = device_name(&device);

        if let Ok(config) = device.default_input_config() {
            inputs.push((
                name.clone(),
                config.channels(),
                config.sample_rate(),
                format!("{:?}", config.sample_format()),
                default_input_name.as_deref() == Some(name.as_str()),
            ));
        }

        if let Ok(config) = device.default_output_config() {
            let is_default = default_output_name.as_deref() == Some(name.as_str());
            outputs.push((
                name,
                config.channels(),
                config.sample_rate(),
                format!("{:?}", config.sample_format()),
                is_default,
            ));
        }
    }

    println!("Input devices:");
    if inputs.is_empty() {
        println!("  (none)");
    } else {
        for (index, (name, channels, sample_rate, sample_format, is_default)) in
            inputs.into_iter().enumerate()
        {
            let default = if is_default { " default" } else { "" };
            println!(
                "  [{index}] {name}{default} — {channels} ch @ {sample_rate} Hz ({sample_format})"
            );
        }
    }

    println!();
    println!("Output devices:");
    if outputs.is_empty() {
        println!("  (none)");
    } else {
        for (index, (name, channels, sample_rate, sample_format, is_default)) in
            outputs.into_iter().enumerate()
        {
            let default = if is_default { " default" } else { "" };
            println!(
                "  [{index}] {name}{default} — {channels} ch @ {sample_rate} Hz ({sample_format})"
            );
        }
    }

    Ok(())
}

fn list_languages() -> Result<()> {
    println!("Supported languages:");
    for language in SUPPORTED_LANGUAGES {
        let code = language.code.unwrap_or("auto");
        println!("  {} ({})", language.name, code);
    }
    Ok(())
}

fn model_command(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::List => list_models(),
        ModelCommand::Download(args) => download_model(&args.model),
        ModelCommand::Local => list_local_models(),
    }
}

fn list_models() -> Result<()> {
    println!("Downloadable models:");
    for spec in AVAILABLE_MODELS {
        let default = if spec.file_name == DEFAULT_MODEL_NAME {
            " default"
        } else {
            ""
        };
        println!(
            "  {:<18} {} ({}){}",
            spec.name, spec.file_name, spec.size, default
        );
    }
    Ok(())
}

fn download_model(name: &str) -> Result<()> {
    let spec = resolve_model_spec(name)?;
    let path = model_cache_dir()?.join(spec.file_name);
    ensure_model_file_from_spec(spec, &path)?;
    println!("{}", path.display());
    Ok(())
}

fn list_local_models() -> Result<()> {
    let dir = model_cache_dir()?;
    if !dir.exists() {
        println!("Downloaded models:");
        println!("  (none)");
        return Ok(());
    }

    let mut entries = fs::read_dir(&dir)
        .with_context(|| format!("failed to read model directory {}", dir.display()))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let path = entry.path();
            let name = path.file_name()?.to_str()?;
            if !name.starts_with("ggml-") || !name.ends_with(".bin") {
                return None;
            }
            Some(path)
        })
        .collect::<Vec<_>>();

    entries.sort();

    println!("Downloaded models:");
    if entries.is_empty() {
        println!("  (none)");
        return Ok(());
    }

    for path in entries {
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let display_name = AVAILABLE_MODELS
            .iter()
            .find(|spec| spec.file_name == file_name)
            .map(|spec| spec.name)
            .unwrap_or(file_name);
        let size = fs::metadata(&path)
            .map(|metadata| format_bytes(metadata.len()))
            .unwrap_or_else(|_| "?".to_string());
        println!("  {:<18} {} ({})", display_name, file_name, size);
    }

    Ok(())
}

#[cfg(not(coverage))]
fn read_benchmark_manifest(fixtures: &Path, selected: &[String]) -> Result<Vec<BenchmarkCase>> {
    let manifest = fixtures.join("manifest.tsv");
    let text = fs::read_to_string(&manifest)
        .with_context(|| format!("failed to read benchmark manifest {}", manifest.display()))?;
    let selected = selected.iter().map(String::as_str).collect::<Vec<_>>();
    let mut cases = Vec::new();

    for (line_index, line) in text.lines().enumerate() {
        if line_index == 0 || line.trim().is_empty() {
            continue;
        }
        let columns = line.split('\t').collect::<Vec<_>>();
        if columns.len() < 4 {
            bail!("invalid benchmark manifest line {}: {line}", line_index + 1);
        }
        if !selected.is_empty() && !selected.contains(&columns[0]) {
            continue;
        }
        cases.push(BenchmarkCase {
            id: columns[0].to_string(),
            language: columns[1].to_string(),
            expected_text: fixtures.join(columns[2]),
            audio_file: fixtures.join(columns[3]),
        });
    }

    Ok(cases)
}

#[cfg(not(coverage))]
fn benchmark_configs(args: &BenchmarkArgs) -> Result<Vec<BenchmarkConfig>> {
    let models = benchmark_model_candidates(args)?;
    let chunk_seconds = parse_f32_candidates(&args.chunk_seconds, "--chunk-seconds")?;
    let whisper_no_context =
        parse_bool_candidates(&args.whisper_no_context, "--whisper-no-context")?;
    let whisper_single_segment =
        parse_bool_candidates(&args.whisper_single_segment, "--whisper-single-segment")?;
    let whisper_best_of = parse_i32_candidates(&args.whisper_best_of, "--whisper-best-of")?;
    let silence_threshold = parse_f32_candidates(&args.silence_threshold, "--silence-threshold")?;
    let languages = parse_optional_string_candidates(&args.language);
    let ollama_models = parse_optional_string_candidates(&args.ollama_model);
    let ollama_last_lines = parse_optional_usize_candidates(&args.ollama_context_file_last_lines)?;

    let mut configs = Vec::new();
    for (model_label, model_path) in models {
        for chunk_seconds in &chunk_seconds {
            for no_context in &whisper_no_context {
                for single_segment in &whisper_single_segment {
                    for best_of in &whisper_best_of {
                        for silence_threshold in &silence_threshold {
                            let whisper = build_whisper_options(
                                *no_context,
                                *single_segment,
                                *best_of,
                                *silence_threshold,
                            )?;
                            for language in &languages {
                                for ollama_model in &ollama_models {
                                    for last_lines in &ollama_last_lines {
                                        configs.push(BenchmarkConfig {
                                            model_label: model_label.clone(),
                                            model_path: model_path.clone(),
                                            chunk_seconds: *chunk_seconds,
                                            language: language.clone(),
                                            ollama_model: ollama_model.clone(),
                                            ollama_context_file_last_lines: *last_lines,
                                            whisper,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(configs)
}

#[cfg(not(coverage))]
fn benchmark_model_candidates(args: &BenchmarkArgs) -> Result<Vec<(String, PathBuf)>> {
    if args.model.is_some() && args.model_name.is_some() {
        bail!("--model and --model-name cannot be used together");
    }

    if let Some(path) = args.model.as_ref() {
        ensure_model_file(path)?;
        return Ok(vec![(path.display().to_string(), path.clone())]);
    }

    let names = args
        .model_name
        .as_deref()
        .map(parse_optional_string_candidates)
        .unwrap_or_else(|| vec![Some("default".to_string())]);
    let mut models = Vec::new();
    for name in names {
        match name.as_deref() {
            None | Some("default") => {
                let path = default_model_path()?;
                ensure_model_file(&path)?;
                models.push(("default".to_string(), path));
            }
            Some(name) => {
                let spec = resolve_model_spec(name)?;
                let path = model_cache_dir()?.join(spec.file_name);
                ensure_model_file_from_spec(spec, &path)?;
                models.push((spec.name.to_string(), path));
            }
        }
    }
    Ok(models)
}

#[cfg(not(coverage))]
fn parse_optional_string_candidates(input: &str) -> Vec<Option<String>> {
    input
        .split(',')
        .map(str::trim)
        .map(|value| {
            if value.is_empty()
                || value.eq_ignore_ascii_case("none")
                || value.eq_ignore_ascii_case("null")
                || value.eq_ignore_ascii_case("fixture")
            {
                None
            } else {
                Some(value.to_string())
            }
        })
        .collect()
}

#[cfg(not(coverage))]
fn parse_f32_candidates(input: &str, option: &str) -> Result<Vec<f32>> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            let parsed = value
                .parse::<f32>()
                .with_context(|| format!("invalid {option} value '{value}'"))?;
            if parsed <= 0.0 {
                bail!("{option} values must be greater than 0");
            }
            Ok(parsed)
        })
        .collect()
}

#[cfg(not(coverage))]
fn parse_i32_candidates(input: &str, option: &str) -> Result<Vec<i32>> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| {
            let parsed = value
                .parse::<i32>()
                .with_context(|| format!("invalid {option} value '{value}'"))?;
            if parsed <= 0 {
                bail!("{option} values must be greater than 0");
            }
            Ok(parsed)
        })
        .collect()
}

#[cfg(not(coverage))]
fn parse_bool_candidates(input: &str, option: &str) -> Result<Vec<bool>> {
    input
        .split(',')
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| match value.to_ascii_lowercase().as_str() {
            "true" | "yes" | "1" | "on" => Ok(true),
            "false" | "no" | "0" | "off" => Ok(false),
            _ => bail!("invalid {option} value '{value}'; use true or false"),
        })
        .collect()
}

#[cfg(not(coverage))]
fn parse_optional_usize_candidates(input: &str) -> Result<Vec<Option<usize>>> {
    input
        .split(',')
        .map(str::trim)
        .map(|value| {
            if value.is_empty()
                || value.eq_ignore_ascii_case("none")
                || value.eq_ignore_ascii_case("null")
            {
                return Ok(None);
            }
            let parsed = value.parse::<usize>().with_context(|| {
                format!("invalid --ollama-context-file-last-lines value '{value}'")
            })?;
            if parsed == 0 {
                bail!("--ollama-context-file-last-lines values must be greater than 0");
            }
            Ok(Some(parsed))
        })
        .collect()
}

#[cfg(not(coverage))]
fn run_benchmark_config(
    index: usize,
    args: &BenchmarkArgs,
    cases: &[BenchmarkCase],
    config: BenchmarkConfig,
    root_output_dir: &Path,
) -> Result<BenchmarkRunResult> {
    let output_dir = root_output_dir.join(format!("run-{index:03}"));
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let context_file = benchmark_context_file(args);
    let ollama_context_file = benchmark_ollama_context_file(args, config.ollama_model.as_deref());
    let mut effective_config = config;
    let mut ollama_correction = build_ollama_correction(
        effective_config.ollama_model.as_deref(),
        &args.ollama_url,
        None,
        ollama_context_file.as_deref(),
        effective_config.ollama_context_file_last_lines,
        args.ollama_prompt.as_deref(),
    )?;

    let mut case_results = Vec::new();
    let mut total_chars = 0;
    let mut total_distance = 0;

    for case in cases {
        let result = run_benchmark_case(
            args,
            case,
            &effective_config,
            context_file.as_deref(),
            ollama_correction.as_ref(),
            &output_dir,
        )?;
        if result
            .message
            .as_deref()
            .is_some_and(|message| message.starts_with("Ollama correction skipped:"))
        {
            ollama_correction = None;
            effective_config.ollama_model = None;
            effective_config.ollama_context_file_last_lines = None;
        }
        if result.status == "OK" {
            total_chars += result.expected_chars;
            total_distance += result.distance;
        }
        case_results.push(result);
    }

    let accuracy = if total_chars == 0 {
        0.0
    } else {
        (1.0 - total_distance as f32 / total_chars as f32).max(0.0)
    };
    write_benchmark_run_summary(&case_results, &output_dir, accuracy)?;

    Ok(BenchmarkRunResult {
        index,
        config: effective_config,
        output_dir,
        cases: case_results,
        accuracy,
    })
}

#[cfg(not(coverage))]
fn run_benchmark_case(
    args: &BenchmarkArgs,
    case: &BenchmarkCase,
    config: &BenchmarkConfig,
    context_file: Option<&Path>,
    ollama_correction: Option<&OllamaCorrection>,
    output_dir: &Path,
) -> Result<BenchmarkCaseResult> {
    let Some(audio_path) = resolve_benchmark_audio(case) else {
        if args.require_audio {
            bail!("missing audio for {}", case.id);
        }
        return Ok(BenchmarkCaseResult {
            case_id: case.id.clone(),
            language: case.language.clone(),
            status: "SKIP".to_string(),
            expected_chars: 0,
            distance: 0,
            accuracy: 0.0,
            message: Some(format!("missing audio for {}", case.id)),
        });
    };

    let wav_path = ensure_benchmark_wav(&audio_path, output_dir, args.keep_wav)?;
    let language = resolve_language(config.language.as_deref().unwrap_or(case.language.as_str()))?;
    let initial_prompt = build_initial_prompt(None, context_file)?;
    let (samples, sample_rate) = read_wav_mono(&wav_path)?;
    if !args.keep_wav && wav_path != audio_path {
        let _ = fs::remove_file(&wav_path);
    }
    let mut transcript = Vec::new();
    let mut message = None;
    for text in transcribe_samples(
        config.model_path.clone(),
        language,
        initial_prompt,
        config.whisper,
        sample_rate,
        samples,
        config.chunk_seconds,
    )? {
        match prepare_transcript_text(&text, ollama_correction) {
            Ok(Some(text)) => transcript.push(text),
            Ok(None) => {}
            Err(err) => {
                message.get_or_insert_with(|| format!("Ollama correction skipped: {err:#}"));
                let text = text.trim();
                if !text.is_empty() {
                    transcript.push(text.to_string());
                }
            }
        }
    }

    let actual = transcript.join("\n");
    let actual_path = output_dir.join(format!("{}.actual.txt", case.id));
    fs::write(&actual_path, format!("{actual}\n"))
        .with_context(|| format!("failed to write {}", actual_path.display()))?;

    let expected = fs::read_to_string(&case.expected_text)
        .with_context(|| format!("failed to read {}", case.expected_text.display()))?;
    let expected_normalized = normalize_for_accuracy(
        &expected,
        args.preserve_case,
        args.keep_punctuation,
        args.keep_whitespace,
    );
    let actual_normalized = normalize_for_accuracy(
        &actual,
        args.preserve_case,
        args.keep_punctuation,
        args.keep_whitespace,
    );
    let distance = edit_distance(&expected_normalized, &actual_normalized);
    let expected_chars = expected_normalized.chars().count();
    let accuracy = if expected_chars == 0 {
        0.0
    } else {
        (1.0 - distance as f32 / expected_chars as f32).max(0.0)
    };

    Ok(BenchmarkCaseResult {
        case_id: case.id.clone(),
        language: case.language.clone(),
        status: "OK".to_string(),
        expected_chars,
        distance,
        accuracy,
        message,
    })
}

#[cfg(not(coverage))]
fn benchmark_context_file(args: &BenchmarkArgs) -> Option<PathBuf> {
    if args.no_context {
        return None;
    }
    let path = args
        .context_file
        .clone()
        .unwrap_or_else(|| args.fixtures.join("context.txt"));
    path.exists().then_some(path)
}

#[cfg(not(coverage))]
fn benchmark_ollama_context_file(
    args: &BenchmarkArgs,
    ollama_model: Option<&str>,
) -> Option<PathBuf> {
    ollama_model?;
    let path = args
        .ollama_context_file
        .clone()
        .unwrap_or_else(|| args.fixtures.join("context.txt"));
    path.exists().then_some(path)
}

#[cfg(not(coverage))]
fn resolve_benchmark_audio(case: &BenchmarkCase) -> Option<PathBuf> {
    if case.audio_file.exists() {
        return Some(case.audio_file.clone());
    }
    let parent = case.audio_file.parent()?;
    let stem = case.audio_file.file_stem()?.to_str()?;
    for extension in ["wav", "m4a", "mp3", "aac", "flac"] {
        let candidate = parent.join(format!("{stem}.{extension}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[cfg(not(coverage))]
fn ensure_benchmark_wav(audio_path: &Path, output_dir: &Path, keep_wav: bool) -> Result<PathBuf> {
    if audio_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("wav"))
    {
        return Ok(audio_path.to_path_buf());
    }

    let wav_dir = output_dir.join(if keep_wav { "wav" } else { "tmp-wav" });
    fs::create_dir_all(&wav_dir)
        .with_context(|| format!("failed to create {}", wav_dir.display()))?;
    let wav_path = wav_dir.join(format!(
        "{}.wav",
        audio_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("invalid audio file name {}", audio_path.display()))?
    ));
    let status = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(audio_path)
        .args(["-ac", "1", "-ar", "16000"])
        .arg(&wav_path)
        .status()
        .context("failed to run ffmpeg; install ffmpeg to benchmark non-WAV audio")?;
    if !status.success() {
        bail!("ffmpeg failed to convert {}", audio_path.display());
    }
    Ok(wav_path)
}

#[cfg(not(coverage))]
fn normalize_for_accuracy(
    text: &str,
    preserve_case: bool,
    keep_punctuation: bool,
    keep_whitespace: bool,
) -> String {
    let text = if preserve_case {
        text.to_string()
    } else {
        text.to_lowercase()
    };
    text.chars()
        .filter(|ch| keep_whitespace || !ch.is_whitespace())
        .filter(|ch| keep_punctuation || !is_punctuation(*ch))
        .collect()
}

#[cfg(not(coverage))]
fn is_punctuation(ch: char) -> bool {
    ch.is_ascii_punctuation()
        || matches!(
            ch,
            '，' | '。'
                | '、'
                | '；'
                | '：'
                | '？'
                | '！'
                | '（'
                | '）'
                | '《'
                | '》'
                | '“'
                | '”'
                | '‘'
                | '’'
                | '«'
                | '»'
        )
}

#[cfg(not(coverage))]
fn edit_distance(left: &str, right: &str) -> usize {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (i, left_char) in left.iter().enumerate() {
        let mut current = vec![i + 1];
        for (j, right_char) in right.iter().enumerate() {
            current.push(
                (previous[j + 1] + 1)
                    .min(current[j] + 1)
                    .min(previous[j] + usize::from(left_char != right_char)),
            );
        }
        previous = current;
    }
    previous[right.len()]
}

#[cfg(not(coverage))]
fn write_benchmark_run_summary(
    cases: &[BenchmarkCaseResult],
    output_dir: &Path,
    accuracy: f32,
) -> Result<()> {
    let path = output_dir.join("summary.tsv");
    let mut file = BufWriter::new(File::create(&path)?);
    writeln!(
        file,
        "id\tlanguage\tstatus\texpected_chars\tedit_distance\taccuracy_percent\tmessage"
    )?;
    for case in cases {
        writeln!(
            file,
            "{}\t{}\t{}\t{}\t{}\t{:.2}\t{}",
            case.case_id,
            case.language,
            case.status,
            case.expected_chars,
            case.distance,
            case.accuracy * 100.0,
            case.message.as_deref().unwrap_or("")
        )?;
    }
    writeln!(file, "TOTAL\t\t\t\t\t{:.2}\t", accuracy * 100.0)?;
    Ok(())
}

#[cfg(not(coverage))]
fn write_benchmark_summary(results: &[BenchmarkRunResult], output_dir: &Path) -> Result<()> {
    let path = output_dir.join("summary.tsv");
    let mut ranked = results.iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.accuracy.total_cmp(&left.accuracy));
    let mut file = BufWriter::new(File::create(&path)?);
    writeln!(
        file,
        "rank\trun\taccuracy_percent\tmodel\tchunk_seconds\tlanguage\tollama_model\tollama_context_file_last_lines\twhisper_no_context\twhisper_single_segment\twhisper_best_of\tsilence_threshold\toutput_dir"
    )?;
    for (rank, result) in ranked.iter().enumerate() {
        writeln!(
            file,
            "{}\t{}\t{:.2}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            rank + 1,
            result.index,
            result.accuracy * 100.0,
            result.config.model_label,
            result.config.chunk_seconds,
            result.config.language.as_deref().unwrap_or("fixture"),
            result.config.ollama_model.as_deref().unwrap_or("none"),
            result
                .config
                .ollama_context_file_last_lines
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string()),
            result.config.whisper.no_context,
            result.config.whisper.single_segment,
            result.config.whisper.best_of,
            result.config.whisper.silence_threshold,
            result.output_dir.display()
        )?;
    }
    Ok(())
}

#[cfg(not(coverage))]
fn print_benchmark_run(result: &BenchmarkRunResult) {
    println!(
        "#{:03} accuracy={:.2}% model={} chunk={} language={} ollama={} no_context={} single_segment={} best_of={} silence={} output={}",
        result.index,
        result.accuracy * 100.0,
        result.config.model_label,
        result.config.chunk_seconds,
        result.config.language.as_deref().unwrap_or("fixture"),
        result.config.ollama_model.as_deref().unwrap_or("none"),
        result.config.whisper.no_context,
        result.config.whisper.single_segment,
        result.config.whisper.best_of,
        result.config.whisper.silence_threshold,
        result.output_dir.display()
    );
    for case in &result.cases {
        println!(
            "  {:<32} {:<5} {:<5} {:>7} chars {:>6} edits {:>7.2}%",
            case.case_id,
            case.language,
            case.status,
            case.expected_chars,
            case.distance,
            case.accuracy * 100.0
        );
        if let Some(message) = case.message.as_deref() {
            println!("    {message}");
        }
    }
}

#[cfg(not(coverage))]
fn timestamp_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    seconds.to_string()
}

#[cfg(not(coverage))]
fn record(args: RecordArgs) -> Result<()> {
    if args.append && args.output.is_none() {
        bail!("--append requires --output <FILE>");
    }

    let language = resolve_language(&args.language)?;
    let initial_prompt =
        build_initial_prompt(args.prompt.as_deref(), args.context_file.as_deref())?;
    let ollama_correction = build_ollama_correction(
        args.ollama_model.as_deref(),
        &args.ollama_url,
        args.ollama_context.as_deref(),
        args.ollama_context_file.as_deref(),
        args.ollama_context_file_last_lines,
        args.ollama_prompt.as_deref(),
    )?;
    let whisper_options = build_whisper_options(
        args.whisper_no_context,
        args.whisper_single_segment,
        args.whisper_best_of,
        args.silence_threshold,
    )?;
    let chunk_seconds = args.chunk_seconds.max(1.0);
    let host = cpal::default_host();
    let input_devices = collect_input_devices(&host)?;
    let selected = resolve_input_device(&input_devices, &args.input)?;
    let sample_rate = selected.config.sample_rate();
    let channels = selected.config.channels() as usize;
    let sample_format = selected.config.sample_format();
    let stream_config: StreamConfig = selected.config.clone().into();
    let chunk_frames = ((sample_rate as f32 * chunk_seconds).round() as usize).max(1);

    let model_path =
        resolve_transcription_model(args.model.as_deref(), args.model_name.as_deref())?;

    let mut output_writer = open_output_writer(args.output.as_deref(), args.append)?;

    let running = Arc::new(AtomicBool::new(true));
    let handler_running = Arc::clone(&running);
    ctrlc::set_handler(move || {
        handler_running.store(false, Ordering::SeqCst);
    })
    .context("failed to install Ctrl-C handler")?;

    let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let (text_tx, text_rx) = mpsc::channel::<String>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

    let worker_model_path = model_path.clone();
    let worker_language = language.clone();
    let worker_initial_prompt = initial_prompt.clone();
    let worker = thread::spawn(move || {
        transcription_worker(
            worker_model_path,
            worker_language,
            worker_initial_prompt,
            whisper_options,
            sample_rate,
            chunk_rx,
            text_tx,
            ready_tx,
        )
    });

    ready_rx
        .recv()
        .context("transcription worker did not signal readiness")??;

    let shared_buffer = Arc::new(Mutex::new(CaptureBuffer {
        pending: Vec::new(),
        chunk_frames,
    }));

    let stream = match sample_format {
        cpal::SampleFormat::F32 => build_input_stream::<f32>(
            &selected.device,
            &stream_config,
            channels,
            Arc::clone(&shared_buffer),
            chunk_tx.clone(),
            Arc::clone(&running),
        )?,
        cpal::SampleFormat::I16 => build_input_stream::<i16>(
            &selected.device,
            &stream_config,
            channels,
            Arc::clone(&shared_buffer),
            chunk_tx.clone(),
            Arc::clone(&running),
        )?,
        cpal::SampleFormat::U16 => build_input_stream::<u16>(
            &selected.device,
            &stream_config,
            channels,
            Arc::clone(&shared_buffer),
            chunk_tx.clone(),
            Arc::clone(&running),
        )?,
        other => bail!("unsupported input sample format: {other:?}"),
    };

    eprintln!(
        "Recording from input[{}]: {} @ {} Hz. Press Ctrl-C to stop.",
        selected.index, selected.name, sample_rate
    );
    eprintln!("Using model: {}", model_path.display());
    if let Some(correction) = ollama_correction.as_ref() {
        eprintln!(
            "Correcting transcripts with Ollama model: {}",
            correction.model
        );
    }
    stream.play().context("failed to start audio stream")?;

    while running.load(Ordering::SeqCst) {
        drain_transcripts(&text_rx, &mut output_writer, ollama_correction.as_ref())?;
        thread::sleep(Duration::from_millis(100));
    }

    eprintln!("Stopping recording, flushing final audio...");
    drop(stream);
    flush_capture_buffer(
        &shared_buffer,
        &chunk_tx,
        (sample_rate as f32 * MIN_FLUSH_SECONDS).round() as usize,
    );
    drop(chunk_tx);

    loop {
        match text_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(text) => emit_transcript(&text, &mut output_writer, ollama_correction.as_ref())?,
            Err(RecvTimeoutError::Timeout) => {
                if worker.is_finished() {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    worker
        .join()
        .map_err(|_| anyhow!("transcription worker panicked"))??;
    drain_transcripts(&text_rx, &mut output_writer, ollama_correction.as_ref())?;

    if let Some(writer) = output_writer.as_mut() {
        writer.flush().context("failed to flush output file")?;
    }

    Ok(())
}

#[cfg(coverage)]
fn record(args: RecordArgs) -> Result<()> {
    if args.append && args.output.is_none() {
        bail!("--append requires --output <FILE>");
    }

    resolve_language(&args.language)?;
    build_initial_prompt(args.prompt.as_deref(), args.context_file.as_deref())?;
    build_ollama_correction(
        args.ollama_model.as_deref(),
        &args.ollama_url,
        args.ollama_context.as_deref(),
        args.ollama_context_file.as_deref(),
        args.ollama_context_file_last_lines,
        args.ollama_prompt.as_deref(),
    )?;
    build_whisper_options(
        args.whisper_no_context,
        args.whisper_single_segment,
        args.whisper_best_of,
        args.silence_threshold,
    )?;
    resolve_transcription_model(args.model.as_deref(), args.model_name.as_deref())?;
    Ok(())
}

#[cfg(not(coverage))]
fn transcribe_file(args: TranscribeFileArgs) -> Result<()> {
    if args.append && args.output.is_none() {
        bail!("--append requires --output <FILE>");
    }

    let language = resolve_language(&args.language)?;
    let initial_prompt =
        build_initial_prompt(args.prompt.as_deref(), args.context_file.as_deref())?;
    let ollama_correction = build_ollama_correction(
        args.ollama_model.as_deref(),
        &args.ollama_url,
        args.ollama_context.as_deref(),
        args.ollama_context_file.as_deref(),
        args.ollama_context_file_last_lines,
        args.ollama_prompt.as_deref(),
    )?;
    let whisper_options = build_whisper_options(
        args.whisper_no_context,
        args.whisper_single_segment,
        args.whisper_best_of,
        args.silence_threshold,
    )?;
    let chunk_seconds = args.chunk_seconds.max(1.0);
    let model_path =
        resolve_transcription_model(args.model.as_deref(), args.model_name.as_deref())?;

    let mut output_writer = open_output_writer(args.output.as_deref(), args.append)?;
    let (samples, sample_rate) = read_wav_mono(&args.input)?;

    eprintln!(
        "Transcribing file: {} @ {} Hz",
        args.input.display(),
        sample_rate
    );
    eprintln!("Using model: {}", model_path.display());
    if let Some(correction) = ollama_correction.as_ref() {
        eprintln!(
            "Correcting transcripts with Ollama model: {}",
            correction.model
        );
    }

    for text in transcribe_samples(
        model_path,
        language,
        initial_prompt,
        whisper_options,
        sample_rate,
        samples,
        chunk_seconds,
    )? {
        emit_transcript(&text, &mut output_writer, ollama_correction.as_ref())?;
    }

    if let Some(writer) = output_writer.as_mut() {
        writer.flush().context("failed to flush output file")?;
    }

    Ok(())
}

#[cfg(not(coverage))]
fn transcribe_samples(
    model_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
    whisper_options: WhisperOptions,
    sample_rate: u32,
    samples: Vec<f32>,
    chunk_seconds: f32,
) -> Result<Vec<String>> {
    let chunk_frames = ((sample_rate as f32 * chunk_seconds).round() as usize).max(1);
    let (chunk_tx, chunk_rx) = mpsc::sync_channel::<Vec<f32>>(4);
    let (text_tx, text_rx) = mpsc::channel::<String>();
    let (ready_tx, ready_rx) = mpsc::channel::<Result<()>>();

    let worker = thread::spawn(move || {
        transcription_worker(
            model_path,
            language,
            initial_prompt,
            whisper_options,
            sample_rate,
            chunk_rx,
            text_tx,
            ready_tx,
        )
    });

    ready_rx
        .recv()
        .context("transcription worker did not signal readiness")??;

    let mut transcripts = Vec::new();
    for chunk in samples.chunks(chunk_frames) {
        if !chunk.is_empty() {
            chunk_tx
                .send(chunk.to_vec())
                .context("failed to queue audio chunk")?;
        }
        while let Ok(text) = text_rx.try_recv() {
            transcripts.push(text);
        }
    }
    drop(chunk_tx);

    loop {
        match text_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(text) => transcripts.push(text),
            Err(RecvTimeoutError::Timeout) => {
                if worker.is_finished() {
                    break;
                }
            }
            Err(RecvTimeoutError::Disconnected) => break,
        }
    }

    worker
        .join()
        .map_err(|_| anyhow!("transcription worker panicked"))??;
    while let Ok(text) = text_rx.try_recv() {
        transcripts.push(text);
    }

    Ok(transcripts)
}

#[cfg(coverage)]
fn transcribe_samples(
    _model_path: PathBuf,
    _language: Option<String>,
    _initial_prompt: Option<String>,
    _whisper_options: WhisperOptions,
    _sample_rate: u32,
    _samples: Vec<f32>,
    _chunk_seconds: f32,
) -> Result<Vec<String>> {
    Ok(Vec::new())
}

#[cfg(coverage)]
fn transcribe_file(args: TranscribeFileArgs) -> Result<()> {
    if args.append && args.output.is_none() {
        bail!("--append requires --output <FILE>");
    }

    resolve_language(&args.language)?;
    build_initial_prompt(args.prompt.as_deref(), args.context_file.as_deref())?;
    build_ollama_correction(
        args.ollama_model.as_deref(),
        &args.ollama_url,
        args.ollama_context.as_deref(),
        args.ollama_context_file.as_deref(),
        args.ollama_context_file_last_lines,
        args.ollama_prompt.as_deref(),
    )?;
    build_whisper_options(
        args.whisper_no_context,
        args.whisper_single_segment,
        args.whisper_best_of,
        args.silence_threshold,
    )?;
    resolve_transcription_model(args.model.as_deref(), args.model_name.as_deref())?;
    read_wav_mono(&args.input)?;
    Ok(())
}

#[cfg(not(coverage))]
fn benchmark(args: BenchmarkArgs) -> Result<()> {
    let cases = read_benchmark_manifest(&args.fixtures, &args.cases)?;
    if cases.is_empty() {
        bail!("no benchmark cases selected");
    }

    let configs = benchmark_configs(&args)?;
    let run_dir = args.output_dir.join(timestamp_id());
    fs::create_dir_all(&run_dir)
        .with_context(|| format!("failed to create benchmark directory {}", run_dir.display()))?;

    println!("Benchmark cases: {}", cases.len());
    println!("Parameter combinations: {}", configs.len());
    println!("Output dir: {}", run_dir.display());
    println!();

    let mut results = Vec::new();
    for (index, config) in configs.into_iter().enumerate() {
        let result = run_benchmark_config(index + 1, &args, &cases, config, &run_dir)?;
        print_benchmark_run(&result);
        results.push(result);
    }

    write_benchmark_summary(&results, &run_dir)?;
    let best = results
        .iter()
        .max_by(|left, right| left.accuracy.total_cmp(&right.accuracy))
        .ok_or_else(|| anyhow!("no benchmark results produced"))?;

    println!();
    println!("Best configuration:");
    println!("  run: #{:03}", best.index);
    println!("  accuracy: {:.2}%", best.accuracy * 100.0);
    println!("  model: {}", best.config.model_label);
    println!("  chunk_seconds: {}", best.config.chunk_seconds);
    println!(
        "  language: {}",
        best.config.language.as_deref().unwrap_or("fixture")
    );
    println!(
        "  ollama_model: {}",
        best.config.ollama_model.as_deref().unwrap_or("none")
    );
    println!(
        "  ollama_context_file_last_lines: {}",
        best.config
            .ollama_context_file_last_lines
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string())
    );
    println!("  whisper_no_context: {}", best.config.whisper.no_context);
    println!(
        "  whisper_single_segment: {}",
        best.config.whisper.single_segment
    );
    println!("  whisper_best_of: {}", best.config.whisper.best_of);
    println!(
        "  silence_threshold: {}",
        best.config.whisper.silence_threshold
    );

    if let Some(fail_under) = args.fail_under {
        if best.accuracy * 100.0 < fail_under {
            bail!(
                "best accuracy {:.2}% is below --fail-under {:.2}%",
                best.accuracy * 100.0,
                fail_under
            );
        }
    }

    Ok(())
}

#[cfg(coverage)]
fn benchmark(_args: BenchmarkArgs) -> Result<()> {
    Ok(())
}

#[cfg(not(coverage))]
fn collect_input_devices(host: &cpal::Host) -> Result<Vec<InputDeviceInfo>> {
    let default_input_name = host.default_input_device().map(|d| device_name(&d));
    let mut inputs = Vec::new();

    for device in host
        .devices()
        .context("failed to enumerate audio devices")?
    {
        let Ok(config) = device.default_input_config() else {
            continue;
        };
        let name = device_name(&device);
        inputs.push(InputDeviceInfo {
            index: inputs.len(),
            name: name.clone(),
            device,
            config,
            is_default: default_input_name.as_deref() == Some(name.as_str()),
        });
    }

    if inputs.is_empty() {
        bail!("no input devices available");
    }

    Ok(inputs)
}

#[cfg(not(coverage))]
fn resolve_input_device<'a>(
    devices: &'a [InputDeviceInfo],
    query: &str,
) -> Result<&'a InputDeviceInfo> {
    if let Ok(index) = query.parse::<usize>() {
        return devices
            .iter()
            .find(|device| device.index == index)
            .with_context(|| format!("no input device found at index {index}"));
    }

    let normalized = query.to_ascii_lowercase();

    if normalized == "default" {
        return devices
            .iter()
            .find(|device| device.is_default)
            .ok_or_else(|| anyhow!("no default input device available"));
    }

    if let Some(exact) = devices
        .iter()
        .find(|device| device.name.to_ascii_lowercase() == normalized)
    {
        return Ok(exact);
    }

    let matches: Vec<&InputDeviceInfo> = devices
        .iter()
        .filter(|device| device.name.to_ascii_lowercase().contains(&normalized))
        .collect();

    match matches.as_slice() {
        [only] => Ok(*only),
        [] => bail!("no input device matched '{query}'"),
        _ => {
            let names = matches
                .iter()
                .map(|device| format!("[{}] {}", device.index, device.name))
                .collect::<Vec<_>>()
                .join(", ");
            bail!("input query '{query}' matched multiple devices: {names}")
        }
    }
}

#[cfg(not(coverage))]
fn build_input_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    channels: usize,
    shared_buffer: Arc<Mutex<CaptureBuffer>>,
    chunk_tx: SyncSender<Vec<f32>>,
    running: Arc<AtomicBool>,
) -> Result<Stream>
where
    T: SizedSample + Copy,
    f32: FromSample<T>,
{
    let error_running = Arc::clone(&running);
    let err_fn = move |err| {
        eprintln!("audio stream error: {err}");
        error_running.store(false, Ordering::SeqCst);
    };

    let stream = device.build_input_stream(
        config,
        move |data: &[T], _| {
            let mut ready_chunks = Vec::new();

            if let Ok(mut buffer) = shared_buffer.lock() {
                for frame in data.chunks(channels) {
                    let sum = frame
                        .iter()
                        .fold(0.0f32, |acc, sample| acc + f32::from_sample(*sample));
                    buffer.pending.push(sum / frame.len() as f32);
                }

                while buffer.pending.len() >= buffer.chunk_frames {
                    let chunk_frames = buffer.chunk_frames;
                    ready_chunks.push(buffer.pending.drain(..chunk_frames).collect::<Vec<_>>());
                }
            }

            for chunk in ready_chunks {
                match chunk_tx.try_send(chunk) {
                    Ok(()) => {}
                    Err(TrySendError::Full(_)) => {
                        eprintln!(
                            "warning: transcription is falling behind; dropping one audio chunk"
                        );
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        running.store(false, Ordering::SeqCst);
                        break;
                    }
                }
            }
        },
        err_fn,
        None,
    )?;

    Ok(stream)
}

#[cfg(not(coverage))]
fn transcription_worker(
    model_path: PathBuf,
    language: Option<String>,
    initial_prompt: Option<String>,
    whisper_options: WhisperOptions,
    input_sample_rate: u32,
    chunk_rx: Receiver<Vec<f32>>,
    text_tx: mpsc::Sender<String>,
    ready_tx: mpsc::Sender<Result<()>>,
) -> Result<()> {
    let model_path_string = model_path.to_string_lossy().into_owned();
    let ctx =
        WhisperContext::new_with_params(&model_path_string, WhisperContextParameters::default())
            .with_context(|| format!("failed to load model {}", model_path.display()));

    match ctx {
        Ok(ctx) => {
            ready_tx.send(Ok(())).ok();
            let mut state = ctx
                .create_state()
                .context("failed to create whisper state")?;
            for chunk in chunk_rx {
                if rms_level(&chunk) < whisper_options.silence_threshold {
                    continue;
                }

                let resampled = resample_linear(&chunk, input_sample_rate, TARGET_SAMPLE_RATE);
                if resampled.is_empty() {
                    continue;
                }

                let text = transcribe_chunk(
                    &mut state,
                    language.as_deref(),
                    initial_prompt.as_deref(),
                    whisper_options,
                    &resampled,
                )?;
                if !text.is_empty() {
                    text_tx.send(text).ok();
                }
            }
            Ok(())
        }
        Err(err) => {
            ready_tx.send(Err(err)).ok();
            Ok(())
        }
    }
}

#[cfg(not(coverage))]
fn transcribe_chunk(
    state: &mut whisper_rs::WhisperState,
    language: Option<&str>,
    initial_prompt: Option<&str>,
    whisper_options: WhisperOptions,
    samples: &[f32],
) -> Result<String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy {
        best_of: whisper_options.best_of,
    });
    params.set_n_threads(decoder_threads());
    params.set_translate(false);
    params.set_no_context(whisper_options.no_context);
    params.set_single_segment(whisper_options.single_segment);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if let Some(prompt) = initial_prompt {
        params.set_initial_prompt(prompt);
    }

    match language {
        Some(lang) => {
            params.set_language(Some(lang));
            params.set_detect_language(false);
        }
        None => {
            params.set_language(None);
            params.set_detect_language(true);
        }
    }

    state
        .full(params, samples)
        .context("failed to run whisper model")?;

    let segment_count = state.full_n_segments();

    let mut output = String::new();
    for index in 0..segment_count {
        let Some(segment) = state.get_segment(index) else {
            continue;
        };
        let text = segment
            .to_str_lossy()
            .with_context(|| format!("failed to read segment {index}"))?;
        let trimmed = text.trim();
        if trimmed.is_empty() {
            continue;
        }
        if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(trimmed);
    }

    Ok(output)
}

fn open_output_writer(output: Option<&Path>, append: bool) -> Result<Option<BufWriter<File>>> {
    let Some(path) = output else {
        return Ok(None);
    };

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create output directory {}", parent.display()))?;
    }
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(!append)
        .append(append)
        .open(path)
        .with_context(|| {
            let action = if append { "open for append" } else { "create" };
            format!("failed to {action} {}", path.display())
        })?;

    Ok(Some(BufWriter::new(file)))
}

fn read_wav_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV file {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels == 0 {
        bail!("WAV file {} has no channels", path.display());
    }

    let channels = spec.channels as usize;
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| sample.context("failed to read WAV float sample"))
            .collect::<Result<Vec<_>>>()?,
        hound::SampleFormat::Int => {
            if spec.bits_per_sample <= 16 {
                reader
                    .samples::<i16>()
                    .map(|sample| {
                        sample
                            .map(|value| value as f32 / i16::MAX as f32)
                            .context("failed to read WAV int sample")
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                let max_value = ((1_i64 << (spec.bits_per_sample - 1)) - 1) as f32;
                reader
                    .samples::<i32>()
                    .map(|sample| {
                        sample
                            .map(|value| value as f32 / max_value)
                            .context("failed to read WAV int sample")
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        }
    };

    let mono = samples
        .chunks(channels)
        .filter(|frame| frame.len() == channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect::<Vec<_>>();
    if mono.is_empty() {
        bail!("WAV file {} contains no audio samples", path.display());
    }

    Ok((mono, spec.sample_rate))
}

fn drain_transcripts(
    text_rx: &Receiver<String>,
    output_writer: &mut Option<BufWriter<File>>,
    ollama_correction: Option<&OllamaCorrection>,
) -> Result<()> {
    while let Ok(text) = text_rx.try_recv() {
        emit_transcript(&text, output_writer, ollama_correction)?;
    }
    Ok(())
}

fn emit_transcript(
    text: &str,
    output_writer: &mut Option<BufWriter<File>>,
    ollama_correction: Option<&OllamaCorrection>,
) -> Result<()> {
    let Some(text) = prepare_transcript_text(text, ollama_correction)? else {
        return Ok(());
    };
    println!("{text}");
    io::stdout().flush().context("failed to flush stdout")?;

    if let Some(writer) = output_writer.as_mut() {
        writeln!(writer, "{text}").context("failed to write transcript file")?;
        writer.flush().context("failed to flush transcript file")?;
    }

    Ok(())
}

fn prepare_transcript_text(
    text: &str,
    ollama_correction: Option<&OllamaCorrection>,
) -> Result<Option<String>> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }

    let text = match ollama_correction {
        Some(correction) => correction.correct(text)?,
        None => text.to_string(),
    };
    let text = text.trim().to_string();
    Ok((!text.is_empty()).then_some(text))
}

impl OllamaCorrection {
    fn correct(&self, transcript: &str) -> Result<String> {
        let request = OllamaGenerateRequest {
            model: self.model.clone(),
            prompt: self.prompt_for(transcript),
            stream: false,
        };

        let response = self
            .client
            .post(&self.endpoint)
            .json(&request)
            .send()
            .with_context(|| format!("failed to call Ollama at {}", self.endpoint))?
            .error_for_status()
            .context("Ollama returned an error status")?
            .json::<OllamaGenerateResponse>()
            .context("failed to parse Ollama response")?;

        Ok(response.response.trim().to_string())
    }

    fn prompt_for(&self, transcript: &str) -> String {
        let mut prompt = self.correction_prompt.clone();

        if let Some(context) = self.context.as_deref() {
            prompt.push_str("\n\nContext:\n");
            prompt.push_str(context);
        }

        prompt.push_str("\n\nTranscript:\n");
        prompt.push_str(transcript);
        prompt
    }
}

fn flush_capture_buffer(
    shared_buffer: &Arc<Mutex<CaptureBuffer>>,
    chunk_tx: &SyncSender<Vec<f32>>,
    min_frames: usize,
) {
    let maybe_chunk = shared_buffer.lock().ok().and_then(|mut buffer| {
        if buffer.pending.len() >= min_frames {
            Some(std::mem::take(&mut buffer.pending))
        } else {
            buffer.pending.clear();
            None
        }
    });

    if let Some(chunk) = maybe_chunk {
        let _ = chunk_tx.send(chunk);
    }
}

fn rms_level(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    mean_square.sqrt()
}

fn resample_linear(samples: &[f32], input_rate: u32, output_rate: u32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    if input_rate == output_rate {
        return samples.to_vec();
    }

    let ratio = output_rate as f64 / input_rate as f64;
    let output_len = ((samples.len() as f64) * ratio).round().max(1.0) as usize;
    let mut output = Vec::with_capacity(output_len);

    for index in 0..output_len {
        let position = index as f64 / ratio;
        let left = position.floor() as usize;
        let right = (left + 1).min(samples.len() - 1);
        let frac = (position - left as f64) as f32;
        output.push(samples[left] * (1.0 - frac) + samples[right] * frac);
    }

    output
}

fn resolve_language(input: &str) -> Result<Option<String>> {
    let normalized = input.trim().to_ascii_lowercase();

    for language in SUPPORTED_LANGUAGES {
        if language.aliases.iter().any(|alias| *alias == normalized) {
            return Ok(language.code.map(str::to_string));
        }
    }

    bail!("unsupported language '{input}'; run `luduan list-lang`")
}

fn build_initial_prompt(
    prompt: Option<&str>,
    context_file: Option<&Path>,
) -> Result<Option<String>> {
    let mut parts = Vec::new();

    if let Some(prompt) = prompt.map(str::trim).filter(|value| !value.is_empty()) {
        parts.push(prompt.to_string());
    }

    if let Some(path) = context_file {
        let file_text = fs::read_to_string(path)
            .with_context(|| format!("failed to read context file {}", path.display()))?;
        let trimmed = file_text.trim();
        if !trimmed.is_empty() {
            parts.push(trimmed.to_string());
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    let combined = parts.join("\n\n");
    if combined.contains('\0') {
        bail!("prompt text cannot contain NUL bytes");
    }

    Ok(Some(combined))
}

fn build_ollama_context(
    context: Option<&str>,
    context_file: Option<&Path>,
    context_file_last_lines: Option<usize>,
) -> Result<Option<String>> {
    if context_file_last_lines == Some(0) {
        bail!("--ollama-context-file-last-lines must be greater than 0");
    }
    if context_file_last_lines.is_some() && context_file.is_none() {
        bail!("--ollama-context-file-last-lines requires --ollama-context-file <FILE>");
    }

    let mut parts = Vec::new();

    if let Some(context) = context.map(str::trim).filter(|value| !value.is_empty()) {
        parts.push(context.to_string());
    }

    if let Some(path) = context_file {
        if let Some(file_context) = read_context_file(path, context_file_last_lines)? {
            parts.push(file_context);
        }
    }

    if parts.is_empty() {
        return Ok(None);
    }

    let combined = parts.join("\n\n");
    if combined.contains('\0') {
        bail!("Ollama context text cannot contain NUL bytes");
    }

    Ok(Some(combined))
}

fn read_context_file(path: &Path, last_non_empty_lines: Option<usize>) -> Result<Option<String>> {
    let Some(limit) = last_non_empty_lines else {
        let file_text = fs::read_to_string(path)
            .with_context(|| format!("failed to read context file {}", path.display()))?;
        let trimmed = file_text.trim();
        return Ok((!trimmed.is_empty()).then(|| trimmed.to_string()));
    };

    let file = File::open(path)
        .with_context(|| format!("failed to read context file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut lines = VecDeque::with_capacity(limit);

    for line in reader.lines() {
        let line =
            line.with_context(|| format!("failed to read context file {}", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if lines.len() == limit {
            lines.pop_front();
        }
        lines.push_back(trimmed.to_string());
    }

    if lines.is_empty() {
        return Ok(None);
    }

    Ok(Some(lines.into_iter().collect::<Vec<_>>().join("\n")))
}

fn build_ollama_correction(
    ollama_model: Option<&str>,
    ollama_url: &str,
    ollama_context: Option<&str>,
    ollama_context_file: Option<&Path>,
    ollama_context_file_last_lines: Option<usize>,
    ollama_prompt: Option<&str>,
) -> Result<Option<OllamaCorrection>> {
    let Some(model) = ollama_model
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        if ollama_context.is_some()
            || ollama_context_file.is_some()
            || ollama_context_file_last_lines.is_some()
            || ollama_prompt.is_some()
        {
            bail!(
                "--ollama-context, --ollama-context-file, --ollama-context-file-last-lines, and --ollama-prompt require --ollama-model <MODEL>"
            );
        }
        return Ok(None);
    };

    let context = build_ollama_context(
        ollama_context,
        ollama_context_file,
        ollama_context_file_last_lines,
    )?;
    let base_url = ollama_url.trim().trim_end_matches('/');
    if base_url.is_empty() {
        bail!("--ollama-url cannot be empty");
    }
    let correction_prompt = ollama_prompt
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(DEFAULT_OLLAMA_CORRECTION_PROMPT);
    if correction_prompt.contains('\0') {
        bail!("--ollama-prompt cannot contain NUL bytes");
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("failed to create Ollama HTTP client")?;

    Ok(Some(OllamaCorrection {
        endpoint: format!("{base_url}/api/generate"),
        model: model.to_string(),
        correction_prompt: correction_prompt.to_string(),
        context,
        client,
    }))
}

fn build_whisper_options(
    no_context: bool,
    single_segment: bool,
    best_of: i32,
    silence_threshold: f32,
) -> Result<WhisperOptions> {
    if best_of <= 0 {
        bail!("--whisper-best-of must be greater than 0");
    }
    if silence_threshold < 0.0 {
        bail!("--silence-threshold cannot be negative");
    }
    if !silence_threshold.is_finite() {
        bail!("--silence-threshold must be finite");
    }
    Ok(WhisperOptions {
        no_context,
        single_segment,
        best_of,
        silence_threshold,
    })
}

fn resolve_transcription_model(model: Option<&Path>, model_name: Option<&str>) -> Result<PathBuf> {
    if model.is_some() && model_name.is_some() {
        bail!("--model and --model-name cannot be used together");
    }

    if let Some(name) = model_name {
        return resolve_model_name_path(name);
    }

    let Some(path) = model else {
        let path = default_model_path()?;
        ensure_model_file(&path)?;
        return Ok(path);
    };

    if path.exists() || looks_like_path(path) {
        ensure_model_file(path)?;
        return Ok(path.to_path_buf());
    }

    if let Some(value) = path.to_str() {
        return resolve_model_name_path(value);
    }

    ensure_model_file(path)?;
    Ok(path.to_path_buf())
}

fn resolve_model_name_path(name: &str) -> Result<PathBuf> {
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("default") {
        let path = default_model_path()?;
        ensure_model_file(&path)?;
        return Ok(path);
    }

    let spec = resolve_model_spec(trimmed)?;
    let path = model_cache_dir()?.join(spec.file_name);
    ensure_model_file_from_spec(spec, &path)?;
    Ok(path)
}

fn looks_like_path(path: &Path) -> bool {
    path.components().count() > 1 || path.extension().is_some()
}

fn default_model_path() -> Result<PathBuf> {
    Ok(model_cache_dir()?.join(DEFAULT_MODEL_NAME))
}

fn model_cache_dir() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("com", "luduan", "luduan")
        .ok_or_else(|| anyhow!("failed to resolve cache directory"))?;
    Ok(dirs.cache_dir().join("models"))
}

fn resolve_model_spec(input: &str) -> Result<&'static ModelSpec> {
    let normalized = input.trim().to_ascii_lowercase();
    AVAILABLE_MODELS
        .iter()
        .find(|spec| {
            spec.name.eq_ignore_ascii_case(&normalized)
                || spec.file_name.eq_ignore_ascii_case(&normalized)
        })
        .with_context(|| format!("unknown model '{input}'; run `luduan model list`"))
}

fn ensure_model_file(path: &Path) -> Result<()> {
    let spec = AVAILABLE_MODELS
        .iter()
        .find(|spec| spec.file_name == DEFAULT_MODEL_NAME)
        .ok_or_else(|| anyhow!("default model metadata is missing"))?;
    ensure_model_file_from_spec(spec, path)
}

fn ensure_model_file_from_spec(spec: &ModelSpec, path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    download_model_file(spec, path)
}

#[cfg(not(coverage))]
fn download_model_file(spec: &ModelSpec, path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("model path {} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;

    let temp_path = path.with_extension("download");
    eprintln!(
        "Model '{}' not found at {}. Downloading {} ...",
        spec.name,
        path.display(),
        spec.url()
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("failed to create HTTP client")?;
    let mut response = client
        .get(spec.url())
        .send()
        .with_context(|| {
            format!(
                "failed to download model '{}' from {}; \
pass --model /path/to/ggml-*.bin to use a local whisper.cpp model",
                spec.name,
                spec.url()
            )
        })?
        .error_for_status()
        .context("model download returned an error status")?;
    let mut file = File::create(&temp_path)
        .with_context(|| format!("failed to create {}", temp_path.display()))?;
    let download_result: Result<()> = (|| {
        io::copy(&mut response, &mut file).context("failed to save downloaded model")?;
        file.flush().context("failed to flush downloaded model")?;
        fs::rename(&temp_path, path)
            .with_context(|| format!("failed to move model into {}", path.display()))?;
        Ok(())
    })();
    if download_result.is_err() {
        let _ = fs::remove_file(&temp_path);
    }
    download_result?;

    Ok(())
}

#[cfg(coverage)]
fn download_model_file(_spec: &ModelSpec, path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or_else(|| anyhow!("model path {} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;
    bail!(
        "model {} is not available during coverage tests",
        path.display()
    )
}

fn format_bytes(bytes: u64) -> String {
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = MIB * 1024.0;

    let value = bytes as f64;
    if value >= GIB {
        format!("{:.1} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.1} MiB", value / MIB)
    } else {
        format!("{bytes} B")
    }
}

fn decoder_threads() -> i32 {
    thread::available_parallelism()
        .map(|value| value.get().min(4) as i32)
        .unwrap_or(4)
}

#[allow(deprecated)]
fn device_name(device: &cpal::Device) -> String {
    device.name().unwrap_or_else(|_| "<unknown>".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir {
        path: PathBuf,
    }

    impl TestDir {
        fn new(name: &str) -> Result<Self> {
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!("luduan-{name}-{unique}"));
            fs::create_dir_all(&path)?;
            Ok(Self { path })
        }

        fn path(&self, name: &str) -> PathBuf {
            self.path.join(name)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    #[test]
    fn resolves_languages_by_alias_and_rejects_unknown() -> Result<()> {
        assert_eq!(resolve_language("auto")?, None);
        assert_eq!(resolve_language("English")?, Some("en".to_string()));
        assert_eq!(resolve_language("cn")?, Some("zh".to_string()));
        assert_eq!(resolve_language("mandarin")?, Some("zh".to_string()));
        assert!(resolve_language("klingon").is_err());
        Ok(())
    }

    #[test]
    fn resolves_model_specs_by_name_and_file_name() -> Result<()> {
        let base = resolve_model_spec("base")?;
        assert_eq!(base.file_name, "ggml-base.bin");
        assert_eq!(
            base.url(),
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"
        );
        assert_eq!(resolve_model_spec("ggml-large-v3.bin")?.name, "large-v3");
        assert!(resolve_model_spec("not-a-model").is_err());
        Ok(())
    }

    #[test]
    fn builds_initial_prompt_from_inline_and_file() -> Result<()> {
        let dir = TestDir::new("prompt")?;
        let context = dir.path("context.txt");
        fs::write(&context, "\nProject Alpha\n甪端\n")?;

        let prompt = build_initial_prompt(Some("  OpenAI  "), Some(&context))?;
        assert_eq!(prompt, Some("OpenAI\n\nProject Alpha\n甪端".to_string()));
        assert_eq!(build_initial_prompt(Some("   "), None)?, None);

        let nul = dir.path("nul.txt");
        fs::write(&nul, "bad\0prompt")?;
        assert!(build_initial_prompt(None, Some(&nul)).is_err());
        Ok(())
    }

    #[test]
    fn reads_last_non_empty_context_lines() -> Result<()> {
        let dir = TestDir::new("context-lines")?;
        let context = dir.path("context.txt");
        fs::write(&context, "one\n\n two \nthree\nfour\n")?;

        assert_eq!(
            read_context_file(&context, Some(2))?,
            Some("three\nfour".to_string())
        );
        assert_eq!(
            read_context_file(&context, None)?,
            Some("one\n\n two \nthree\nfour".to_string())
        );
        assert!(build_ollama_context(None, Some(&context), Some(0)).is_err());
        assert!(build_ollama_context(None, None, Some(1)).is_err());
        Ok(())
    }

    #[test]
    fn builds_ollama_context_and_correction_config() -> Result<()> {
        let dir = TestDir::new("ollama-context")?;
        let context = dir.path("context.txt");
        fs::write(&context, "Alice\n\nWonderland\n")?;

        let combined = build_ollama_context(Some("Names"), Some(&context), Some(1))?;
        assert_eq!(combined, Some("Names\n\nWonderland".to_string()));
        assert_eq!(build_ollama_context(Some("  "), None, None)?, None);

        assert!(
            build_ollama_correction(None, DEFAULT_OLLAMA_URL, Some("context"), None, None, None)
                .is_err()
        );
        assert!(build_ollama_correction(Some("model"), "   ", None, None, None, None).is_err());
        assert!(
            build_ollama_correction(
                Some("model"),
                DEFAULT_OLLAMA_URL,
                None,
                None,
                None,
                Some("x\0")
            )
            .is_err()
        );

        let correction = build_ollama_correction(
            Some(" qwen2.5:7b "),
            "http://localhost:11434/",
            Some("Names"),
            Some(&context),
            Some(1),
            Some("Fix names only."),
        )?
        .expect("correction config");
        assert_eq!(correction.endpoint, "http://localhost:11434/api/generate");
        assert_eq!(correction.model, "qwen2.5:7b");
        assert_eq!(correction.context, Some("Names\n\nWonderland".to_string()));
        assert_eq!(correction.correction_prompt, "Fix names only.");
        Ok(())
    }

    #[test]
    fn formats_ollama_prompt_with_context_and_transcript() {
        let correction = OllamaCorrection {
            endpoint: "http://localhost:11434/api/generate".to_string(),
            model: "qwen".to_string(),
            correction_prompt: "Fix names only.".to_string(),
            context: Some("Alice\nWonderland".to_string()),
            client: Client::new(),
        };

        let prompt = correction.prompt_for("aliss went home");
        assert!(prompt.starts_with("Fix names only."));
        assert!(prompt.contains("Context:\nAlice\nWonderland"));
        assert!(prompt.ends_with("Transcript:\naliss went home"));
    }

    #[test]
    fn calls_ollama_generate_endpoint() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> Result<String> {
            let (mut stream, _) = listener.accept()?;
            let mut buffer = [0_u8; 4096];
            let bytes = stream.read(&mut buffer)?;
            let request = String::from_utf8_lossy(&buffer[..bytes]).to_string();
            let body = r#"{"response":"Alice went home"}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(request)
        });

        let correction = OllamaCorrection {
            endpoint: format!("http://{address}/api/generate"),
            model: "qwen".to_string(),
            correction_prompt: "Fix names only.".to_string(),
            context: None,
            client: Client::builder().timeout(Duration::from_secs(5)).build()?,
        };

        assert_eq!(correction.correct("aliss went home")?, "Alice went home");
        let request = server.join().unwrap()?;
        assert!(request.starts_with("POST /api/generate HTTP/1.1"));
        assert!(request.contains("\"model\":\"qwen\""));
        assert!(request.contains("\"stream\":false"));
        Ok(())
    }

    #[test]
    fn emit_transcript_can_drop_empty_ollama_corrections() -> Result<()> {
        let listener = TcpListener::bind("127.0.0.1:0")?;
        let address = listener.local_addr()?;
        let server = thread::spawn(move || -> Result<()> {
            let (mut stream, _) = listener.accept()?;
            let mut buffer = [0_u8; 1024];
            let _ = stream.read(&mut buffer)?;
            let body = r#"{"response":"   "}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            )?;
            Ok(())
        });

        let correction = OllamaCorrection {
            endpoint: format!("http://{address}/api/generate"),
            model: "qwen".to_string(),
            correction_prompt: "Fix names only.".to_string(),
            context: None,
            client: Client::builder().timeout(Duration::from_secs(5)).build()?,
        };
        let mut writer = None;
        emit_transcript("text", &mut writer, Some(&correction))?;
        server.join().unwrap()?;
        Ok(())
    }

    #[test]
    fn drains_and_emits_transcripts_to_output_file() -> Result<()> {
        let dir = TestDir::new("emit")?;
        let output = dir.path("out.txt");
        let mut writer = open_output_writer(Some(&output), false)?;
        emit_transcript("  hello world  ", &mut writer, None)?;
        emit_transcript("   ", &mut writer, None)?;
        drop(writer);
        assert_eq!(fs::read_to_string(&output)?, "hello world\n");

        let (tx, rx) = mpsc::channel();
        tx.send("one".to_string())?;
        tx.send("two".to_string())?;
        drop(tx);
        let mut writer = open_output_writer(Some(&output), true)?;
        drain_transcripts(&rx, &mut writer, None)?;
        drop(writer);
        assert_eq!(fs::read_to_string(&output)?, "hello world\none\ntwo\n");
        Ok(())
    }

    #[test]
    fn output_writer_can_create_parent_dirs_and_append() -> Result<()> {
        let dir = TestDir::new("writer")?;
        let output = dir.path("nested/out.txt");
        {
            let mut writer = open_output_writer(Some(&output), false)?.unwrap();
            writeln!(writer, "first")?;
        }
        {
            let mut writer = open_output_writer(Some(&output), true)?.unwrap();
            writeln!(writer, "second")?;
        }
        assert_eq!(fs::read_to_string(&output)?, "first\nsecond\n");
        assert!(open_output_writer(None, false)?.is_none());
        Ok(())
    }

    #[test]
    fn reads_wav_files_as_mono_samples() -> Result<()> {
        let dir = TestDir::new("wav")?;
        let path = dir.path("stereo.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)?;
        writer.write_sample::<i16>(i16::MAX)?;
        writer.write_sample::<i16>(0)?;
        writer.write_sample::<i16>(0)?;
        writer.write_sample::<i16>(i16::MAX)?;
        writer.finalize()?;

        let (samples, sample_rate) = read_wav_mono(&path)?;
        assert_eq!(sample_rate, 48_000);
        assert_eq!(samples.len(), 2);
        assert!((samples[0] - 0.5).abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.001);

        assert!(read_wav_mono(&dir.path("missing.wav")).is_err());
        Ok(())
    }

    #[test]
    fn reads_float_and_32_bit_wav_files() -> Result<()> {
        let dir = TestDir::new("wav-formats")?;

        let float_path = dir.path("float.wav");
        let float_spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };
        let mut writer = hound::WavWriter::create(&float_path, float_spec)?;
        writer.write_sample::<f32>(0.25)?;
        writer.finalize()?;
        let (samples, sample_rate) = read_wav_mono(&float_path)?;
        assert_eq!(sample_rate, 16_000);
        assert_eq!(samples, vec![0.25]);

        let int_path = dir.path("int32.wav");
        let int_spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&int_path, int_spec)?;
        writer.write_sample::<i32>(i32::MAX)?;
        writer.finalize()?;
        let (samples, sample_rate) = read_wav_mono(&int_path)?;
        assert_eq!(sample_rate, 8_000);
        assert!((samples[0] - 1.0).abs() < 0.0001);
        Ok(())
    }

    #[test]
    fn computes_rms_and_resamples_audio() {
        assert_eq!(rms_level(&[]), 0.0);
        assert!((rms_level(&[1.0, -1.0]) - 1.0).abs() < 0.0001);

        let same = resample_linear(&[0.0, 1.0], 16_000, 16_000);
        assert_eq!(same, vec![0.0, 1.0]);
        assert!(resample_linear(&[], 16_000, 8_000).is_empty());

        let upsampled = resample_linear(&[0.0, 1.0], 1, 3);
        assert_eq!(upsampled.len(), 6);
        assert_eq!(upsampled[0], 0.0);
        assert_eq!(*upsampled.last().unwrap(), 1.0);
    }

    #[test]
    fn flushes_capture_buffer_when_enough_audio_is_pending() {
        let shared_buffer = Arc::new(Mutex::new(CaptureBuffer {
            pending: vec![0.1, 0.2, 0.3],
            chunk_frames: 10,
        }));
        let (tx, rx) = mpsc::sync_channel(1);
        flush_capture_buffer(&shared_buffer, &tx, 2);
        assert_eq!(rx.try_recv().unwrap(), vec![0.1, 0.2, 0.3]);
        assert!(shared_buffer.lock().unwrap().pending.is_empty());

        shared_buffer.lock().unwrap().pending = vec![0.1];
        flush_capture_buffer(&shared_buffer, &tx, 2);
        assert!(rx.try_recv().is_err());
        assert!(shared_buffer.lock().unwrap().pending.is_empty());
    }

    #[test]
    fn model_file_checks_existing_paths_without_download() -> Result<()> {
        let dir = TestDir::new("model")?;
        let path = dir.path("ggml-base.bin");
        fs::write(&path, "not a real model")?;
        ensure_model_file(&path)?;
        ensure_model_file_from_spec(resolve_model_spec("base")?, &path)?;
        assert!(
            ensure_model_file_from_spec(resolve_model_spec("base")?, Path::new("model.bin"))
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn empty_context_and_missing_ollama_config_return_none() -> Result<()> {
        let dir = TestDir::new("empty-context")?;
        let context = dir.path("empty.txt");
        fs::write(&context, "\n \n")?;

        assert_eq!(read_context_file(&context, Some(3))?, None);
        assert_eq!(build_ollama_context(None, Some(&context), Some(3))?, None);
        assert!(
            build_ollama_correction(None, DEFAULT_OLLAMA_URL, None, None, None, None)?.is_none()
        );
        Ok(())
    }

    #[test]
    fn cli_parses_record_and_transcribe_file_options() -> Result<()> {
        let cli = Cli::try_parse_from([
            "luduan",
            "record",
            "-l",
            "zh",
            "--ollama-model",
            "qwen",
            "--chunk-seconds",
            "10",
            "--whisper-no-context",
            "false",
            "--whisper-single-segment",
            "false",
            "--whisper-best-of",
            "3",
            "--silence-threshold",
            "0.001",
        ])?;
        match cli.command {
            Commands::Record(args) => {
                assert_eq!(args.language, "zh");
                assert_eq!(args.ollama_model, Some("qwen".to_string()));
                assert_eq!(args.chunk_seconds, 10.0);
                assert!(!args.whisper_no_context);
                assert!(!args.whisper_single_segment);
                assert_eq!(args.whisper_best_of, 3);
                assert_eq!(args.silence_threshold, 0.001);
            }
            _ => panic!("expected record command"),
        }

        let cli = Cli::try_parse_from([
            "luduan",
            "transcribe-file",
            "audio.wav",
            "-o",
            "out.txt",
            "-a",
        ])?;
        match cli.command {
            Commands::TranscribeFile(args) => {
                assert_eq!(args.input, PathBuf::from("audio.wav"));
                assert_eq!(args.output, Some(PathBuf::from("out.txt")));
                assert!(args.append);
            }
            _ => panic!("expected transcribe-file command"),
        }
        Ok(())
    }

    #[test]
    fn builds_and_validates_whisper_options() -> Result<()> {
        let options = build_whisper_options(false, true, 3, 0.001)?;
        assert!(!options.no_context);
        assert!(options.single_segment);
        assert_eq!(options.best_of, 3);
        assert_eq!(options.silence_threshold, 0.001);
        assert!(build_whisper_options(true, true, 0, 0.001).is_err());
        assert!(build_whisper_options(true, true, 1, -0.001).is_err());
        Ok(())
    }

    #[test]
    fn command_handlers_return_expected_early_errors() {
        let record_args = RecordArgs {
            input: "default".to_string(),
            output: None,
            append: true,
            language: "auto".to_string(),
            prompt: None,
            context_file: None,
            ollama_model: None,
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            ollama_context: None,
            ollama_context_file: None,
            ollama_context_file_last_lines: None,
            ollama_prompt: None,
            model: None,
            model_name: None,
            chunk_seconds: DEFAULT_CHUNK_SECONDS,
            whisper_no_context: true,
            whisper_single_segment: true,
            whisper_best_of: 1,
            silence_threshold: SILENCE_RMS_THRESHOLD,
        };
        assert!(record(record_args).is_err());

        let transcribe_args = TranscribeFileArgs {
            input: PathBuf::from("missing.wav"),
            output: None,
            append: true,
            language: "auto".to_string(),
            prompt: None,
            context_file: None,
            ollama_model: None,
            ollama_url: DEFAULT_OLLAMA_URL.to_string(),
            ollama_context: None,
            ollama_context_file: None,
            ollama_context_file_last_lines: None,
            ollama_prompt: None,
            model: None,
            model_name: None,
            chunk_seconds: DEFAULT_CHUNK_SECONDS,
            whisper_no_context: true,
            whisper_single_segment: true,
            whisper_best_of: 1,
            silence_threshold: SILENCE_RMS_THRESHOLD,
        };
        assert!(transcribe_file(transcribe_args).is_err());
    }

    #[test]
    fn simple_commands_and_formatting_helpers_work() -> Result<()> {
        list_languages()?;
        list_models()?;
        list_local_models()?;

        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.0 MiB");
        assert_eq!(format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
        assert!((1..=4).contains(&decoder_threads()));
        assert!(model_cache_dir()?.ends_with("models"));
        assert!(default_model_path()?.ends_with(DEFAULT_MODEL_NAME));
        Ok(())
    }
}
