use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Parser, Subcommand};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SizedSample, Stream, StreamConfig, SupportedStreamConfig};
use directories::ProjectDirs;
use reqwest::blocking::Client;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const TARGET_SAMPLE_RATE: u32 = 16_000;
const DEFAULT_CHUNK_SECONDS: f32 = 5.0;
const MIN_FLUSH_SECONDS: f32 = 0.5;
const SILENCE_RMS_THRESHOLD: f32 = 0.003;
const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin";
const DEFAULT_MODEL_NAME: &str = "ggml-base.bin";

#[derive(Parser, Debug)]
#[command(name = "luduan-cli")]
#[command(about = "Local audio device listing and streaming speech-to-text")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List available audio input and output devices
    ListDev,
    /// Record from an input device and stream transcripts to stdout
    Record(RecordArgs),
}

#[derive(Args, Debug)]
struct RecordArgs {
    /// Input device index from `list-dev`, `default`, or part/all of the device name
    #[arg(short = 'i', long)]
    input: String,

    /// Optional output text file for transcribed chunks
    #[arg(short = 'o', long)]
    output: Option<PathBuf>,

    /// Append to the output file instead of overwriting it
    #[arg(short = 'a', long)]
    append: bool,

    /// Language: auto, english/en, swedish/se, chinese/cn, french/fr, russian/ru
    #[arg(short = 'l', long, default_value = "auto")]
    language: String,

    /// Path to a whisper.cpp GGML model file
    #[arg(short = 'm', long)]
    model: Option<PathBuf>,

    /// Chunk size in seconds for streaming transcription
    #[arg(short = 'c', long, default_value_t = DEFAULT_CHUNK_SECONDS)]
    chunk_seconds: f32,
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

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::ListDev => list_devices(),
        Commands::Record(args) => record(args),
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

fn record(args: RecordArgs) -> Result<()> {
    if args.append && args.output.is_none() {
        bail!("--append requires --output <FILE>");
    }

    let language = resolve_language(&args.language)?;
    let chunk_seconds = args.chunk_seconds.max(1.0);
    let host = cpal::default_host();
    let input_devices = collect_input_devices(&host)?;
    let selected = resolve_input_device(&input_devices, &args.input)?;
    let sample_rate = selected.config.sample_rate();
    let channels = selected.config.channels() as usize;
    let sample_format = selected.config.sample_format();
    let stream_config: StreamConfig = selected.config.clone().into();
    let chunk_frames = ((sample_rate as f32 * chunk_seconds).round() as usize).max(1);

    let model_path = args.model.unwrap_or(default_model_path()?);
    ensure_model_file(&model_path)?;

    let mut output_writer = match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).with_context(|| {
                    format!("failed to create output directory {}", parent.display())
                })?;
            }
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(!args.append)
                .append(args.append)
                .open(&path)
                .with_context(|| {
                    let action = if args.append {
                        "open for append"
                    } else {
                        "create"
                    };
                    format!("failed to {action} {}", path.display())
                })?;
            Some(BufWriter::new(file))
        }
        None => None,
    };

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
    let worker = thread::spawn(move || {
        transcription_worker(
            worker_model_path,
            worker_language,
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
    stream.play().context("failed to start audio stream")?;

    while running.load(Ordering::SeqCst) {
        drain_transcripts(&text_rx, &mut output_writer)?;
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
            Ok(text) => emit_transcript(&text, &mut output_writer)?,
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
    drain_transcripts(&text_rx, &mut output_writer)?;

    if let Some(writer) = output_writer.as_mut() {
        writer.flush().context("failed to flush output file")?;
    }

    Ok(())
}

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

fn transcription_worker(
    model_path: PathBuf,
    language: Option<String>,
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
                if rms_level(&chunk) < SILENCE_RMS_THRESHOLD {
                    continue;
                }

                let resampled = resample_linear(&chunk, input_sample_rate, TARGET_SAMPLE_RATE);
                if resampled.is_empty() {
                    continue;
                }

                let text = transcribe_chunk(&mut state, language.as_deref(), &resampled)?;
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

fn transcribe_chunk(
    state: &mut whisper_rs::WhisperState,
    language: Option<&str>,
    samples: &[f32],
) -> Result<String> {
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(decoder_threads());
    params.set_translate(false);
    params.set_no_context(true);
    params.set_single_segment(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

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

fn drain_transcripts(
    text_rx: &Receiver<String>,
    output_writer: &mut Option<BufWriter<File>>,
) -> Result<()> {
    while let Ok(text) = text_rx.try_recv() {
        emit_transcript(&text, output_writer)?;
    }
    Ok(())
}

fn emit_transcript(text: &str, output_writer: &mut Option<BufWriter<File>>) -> Result<()> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(());
    }

    println!("{text}");
    io::stdout().flush().context("failed to flush stdout")?;

    if let Some(writer) = output_writer.as_mut() {
        writeln!(writer, "{text}").context("failed to write transcript file")?;
        writer.flush().context("failed to flush transcript file")?;
    }

    Ok(())
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
    let value = match normalized.as_str() {
        "auto" => None,
        "english" | "en" => Some("en"),
        "swedish" | "se" | "sv" => Some("sv"),
        "chinese" | "cn" | "zh" => Some("zh"),
        "french" | "fr" => Some("fr"),
        "russian" | "ru" => Some("ru"),
        _ => bail!("unsupported language '{input}'"),
    };
    Ok(value.map(str::to_string))
}

fn default_model_path() -> Result<PathBuf> {
    let dirs = ProjectDirs::from("com", "luduan", "rust-cli")
        .ok_or_else(|| anyhow!("failed to resolve cache directory"))?;
    Ok(dirs.cache_dir().join("models").join(DEFAULT_MODEL_NAME))
}

fn ensure_model_file(path: &Path) -> Result<()> {
    if path.exists() {
        return Ok(());
    }

    let parent = path
        .parent()
        .ok_or_else(|| anyhow!("model path {} has no parent directory", path.display()))?;
    fs::create_dir_all(parent)
        .with_context(|| format!("failed to create model directory {}", parent.display()))?;

    let temp_path = path.with_extension("download");
    eprintln!(
        "Model not found at {}. Downloading {} ...",
        path.display(),
        DEFAULT_MODEL_URL
    );

    let client = Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .context("failed to create HTTP client")?;
    let mut response = client
        .get(DEFAULT_MODEL_URL)
        .send()
        .with_context(|| {
            format!(
                "failed to download default model from {DEFAULT_MODEL_URL}; \
pass --model /path/to/ggml-*.bin to use a local whisper.cpp model"
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

fn decoder_threads() -> i32 {
    thread::available_parallelism()
        .map(|value| value.get().min(4) as i32)
        .unwrap_or(4)
}

#[allow(deprecated)]
fn device_name(device: &cpal::Device) -> String {
    device.name().unwrap_or_else(|_| "<unknown>".to_string())
}
