"""Luduan menu bar app — entry point."""

from __future__ import annotations

import sys
import threading
from enum import Enum, auto

import rumps

from luduan import config as cfg
from luduan.audio import AudioRecorder
from luduan.output import TextOutput
from luduan.postprocessor import Postprocessor
from luduan.transcriber import Transcriber


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


_ICONS = {
    State.IDLE: "🎙",
    State.RECORDING: "🔴",
    State.PROCESSING: "⏳",
}


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

class LuduanApp(rumps.App):
    """macOS menu bar application for Luduan."""

    def __init__(self, conf: dict) -> None:
        super().__init__("🎙", quit_button=None)

        self._conf = conf
        self._state = State.IDLE
        self._lock = threading.Lock()

        # Components
        self._recorder = AudioRecorder(
            sample_rate=conf["audio"]["sample_rate"],
            channels=conf["audio"]["channels"],
        )
        self._transcriber = Transcriber(
            model=conf["whisper"]["model"],
            language=conf["whisper"].get("language") or None,
            on_partial=self._on_partial_transcript,
        )
        self._postprocessor = Postprocessor(
            host=conf["ollama"]["host"],
            model=conf["ollama"]["model"],
            enabled=conf["ollama"]["enabled"],
        )
        self._output = TextOutput()
        self._partial_text = ""

        # Menu
        self._toggle_item = rumps.MenuItem("Start Recording", callback=self._on_toggle)
        self._status_item = rumps.MenuItem("Ready", callback=None)
        self._status_item.set_callback(None)  # non-clickable status line

        ollama_label = (
            f"LLM: {conf['ollama']['model']} (enabled)"
            if conf["ollama"]["enabled"]
            else "LLM: disabled"
        )
        self._ollama_item = rumps.MenuItem(ollama_label, callback=None)

        self.menu = [
            self._toggle_item,
            None,  # separator
            self._status_item,
            self._ollama_item,
            None,
            rumps.MenuItem("Quit Luduan", callback=self._on_quit),
        ]

        # Start global hotkey listener
        self._start_hotkey_listener()

    # ------------------------------------------------------------------
    # Recording toggle
    # ------------------------------------------------------------------

    def toggle_recording(self) -> None:
        """Toggle recording on/off (thread-safe, callable from hotkey thread)."""
        with self._lock:
            state = self._state

        if state == State.IDLE:
            self._start_recording()
        elif state == State.RECORDING:
            self._stop_recording()
        # Ignore toggle while PROCESSING

    def _start_recording(self) -> None:
        self._set_state(State.RECORDING)
        self._partial_text = ""
        try:
            self._recorder.start()
        except Exception as exc:
            self._set_state(State.IDLE)
            rumps.notification(
                title="Luduan",
                subtitle="Microphone error",
                message=str(exc),
                sound=False,
            )

    def _stop_recording(self) -> None:
        self._set_state(State.PROCESSING)
        # Run transcription in a background thread to keep the UI responsive
        thread = threading.Thread(target=self._process_audio, daemon=True)
        thread.start()

    def _process_audio(self) -> None:
        audio = self._recorder.stop()

        if len(audio) == 0:
            self._set_state(State.IDLE)
            return

        try:
            raw_text = self._transcriber.transcribe(
                audio,
                sample_rate=self._conf["audio"]["sample_rate"],
            )
        except Exception as exc:
            self._set_state(State.IDLE)
            rumps.notification(
                title="Luduan",
                subtitle="Transcription error",
                message=str(exc),
                sound=False,
            )
            return

        if not raw_text:
            self._set_state(State.IDLE)
            return

        # LLM post-processing
        cleaned_text, used_llm = self._postprocessor.process(raw_text)

        # Output
        success = self._output.paste(cleaned_text)
        if not success:
            rumps.notification(
                title="Luduan",
                subtitle="Copied to clipboard",
                message="Could not auto-paste — text is in your clipboard.",
                sound=False,
            )

        self._set_state(State.IDLE)

    # ------------------------------------------------------------------
    # Partial transcript callback (fired by Transcriber per chunk)
    # ------------------------------------------------------------------

    def _on_partial_transcript(self, text: str) -> None:
        self._partial_text = text
        # Show preview in status menu item (truncated)
        preview = text[:60] + "…" if len(text) > 60 else text
        self._update_status(f'"{preview}"')

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: State) -> None:
        with self._lock:
            self._state = state
        icon = _ICONS[state]
        self.title = icon
        if state == State.IDLE:
            self._toggle_item.title = "Start Recording"
            self._update_status("Ready")
        elif state == State.RECORDING:
            self._toggle_item.title = "Stop Recording"
            self._update_status("Recording…")
        elif state == State.PROCESSING:
            self._toggle_item.title = "Processing…"
            self._update_status("Transcribing…")

    def _update_status(self, message: str) -> None:
        self._status_item.title = message

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_toggle(self, _sender) -> None:
        self.toggle_recording()

    def _on_quit(self, _sender) -> None:
        if self._recorder.is_recording:
            self._recorder.stop()
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Global hotkey listener
    # ------------------------------------------------------------------

    def _start_hotkey_listener(self) -> None:
        keys_conf: list[str] = self._conf["hotkey"]["keys"]
        thread = threading.Thread(
            target=_run_hotkey_listener,
            args=(keys_conf, self.toggle_recording),
            daemon=True,
        )
        thread.start()


# ---------------------------------------------------------------------------
# Global hotkey (runs in background thread via pynput)
# ---------------------------------------------------------------------------

def _run_hotkey_listener(
    keys_conf: list[str],
    on_trigger: callable,
) -> None:
    """Listen for the configured global hotkey combination."""
    from pynput import keyboard

    combo = _parse_hotkey(keys_conf)
    current: set = set()

    def on_press(key):
        current.add(_normalize_key(key))
        if combo <= current:
            # Avoid repeat-firing while held; clear combo keys
            for k in combo:
                current.discard(k)
            on_trigger()

    def on_release(key):
        current.discard(_normalize_key(key))

    with keyboard.Listener(on_press=on_press, on_release=on_release):
        import time
        while True:
            time.sleep(1)


def _parse_hotkey(keys_conf: list[str]) -> set[str]:
    """Convert config key list to a set of normalised key strings."""
    mapping = {
        "cmd": "<cmd>",
        "command": "<cmd>",
        "shift": "<shift>",
        "alt": "<alt>",
        "option": "<alt>",
        "ctrl": "<ctrl>",
        "control": "<ctrl>",
        "space": "<space>",
    }
    result = set()
    for k in keys_conf:
        k_lower = k.lower()
        result.add(mapping.get(k_lower, k_lower))
    return result


def _normalize_key(key) -> str:
    """Normalise a pynput key to a string for comparison."""
    from pynput import keyboard

    if isinstance(key, keyboard.KeyCode):
        return key.char or str(key)
    # Special keys
    key_map = {
        keyboard.Key.cmd: "<cmd>",
        keyboard.Key.cmd_l: "<cmd>",
        keyboard.Key.cmd_r: "<cmd>",
        keyboard.Key.shift: "<shift>",
        keyboard.Key.shift_l: "<shift>",
        keyboard.Key.shift_r: "<shift>",
        keyboard.Key.alt: "<alt>",
        keyboard.Key.alt_l: "<alt>",
        keyboard.Key.alt_r: "<alt>",
        keyboard.Key.ctrl: "<ctrl>",
        keyboard.Key.ctrl_l: "<ctrl>",
        keyboard.Key.ctrl_r: "<ctrl>",
        keyboard.Key.space: "<space>",
    }
    return key_map.get(key, str(key))


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------

def first_run_check(conf: dict) -> None:
    """Warn user if Ollama is unreachable on startup."""
    if not conf["ollama"]["enabled"]:
        return

    pp = Postprocessor(
        host=conf["ollama"]["host"],
        model=conf["ollama"]["model"],
        enabled=True,
    )
    if not pp.is_available():
        rumps.notification(
            title="Luduan",
            subtitle="Ollama not running",
            message=(
                "Start Ollama (`ollama serve`) for LLM post-processing. "
                "Transcription will still work without it."
            ),
            sound=False,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    conf = cfg.load()
    app = LuduanApp(conf)

    # Defer first-run check until after app starts
    threading.Timer(1.5, first_run_check, args=(conf,)).start()

    app.run()


if __name__ == "__main__":
    main()
