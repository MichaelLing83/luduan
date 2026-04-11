"""Luduan menu bar app — entry point."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from enum import Enum, auto
from pathlib import Path

import rumps

from luduan import config as cfg
from luduan.audio import AudioRecorder
from luduan.log import LOG_PATH, get_logger
from luduan.output import PasteResult, TextOutput
from luduan.postprocessor import Postprocessor
from luduan.transcriber import Transcriber

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Icon helpers
# ---------------------------------------------------------------------------

def _find_resource(filename: str) -> str | None:
    """Return path to a file in the .app bundle Resources dir, or None."""
    # When running inside Luduan.app the executable is in Contents/MacOS/
    exe = Path(sys.executable)
    candidates = [
        exe.parent.parent / "Resources" / filename,   # inside .app bundle
        Path(__file__).parent.parent.parent / "build" / filename,  # dev tree
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


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
        menubar_icon = _find_resource("menubar_icon.png")
        if menubar_icon:
            super().__init__("", icon=menubar_icon, template=True, quit_button=None)
        else:
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
        self._accessibility_ok = True  # flips to False on first error 1002

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
        self._fix_perms_item = rumps.MenuItem(
            "⚠️  Grant Accessibility Permission…",
            callback=self._on_fix_accessibility,
        )
        self._lang_menu = self._build_language_menu(conf["whisper"].get("language") or "auto")

        self.menu = [
            self._toggle_item,
            None,  # separator
            self._status_item,
            self._ollama_item,
            self._lang_menu,
            None,
            rumps.MenuItem("Show Log", callback=self._on_show_log),
            rumps.MenuItem("Quit Luduan", callback=self._on_quit),
        ]

        # Start global hotkey listener
        self._start_hotkey_listener()

        # Check Accessibility on startup so the warning shows immediately
        threading.Timer(2.0, self._check_accessibility).start()

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
            log.info("Recording started")
        except Exception as exc:
            log.exception("Failed to start microphone")
            self._set_state(State.IDLE)
            rumps.notification(
                title="Luduan",
                subtitle="Microphone error",
                message=str(exc),
                sound=False,
            )

    def _stop_recording(self) -> None:
        self._set_state(State.PROCESSING)
        log.info("Recording stopped — starting transcription")
        # Run transcription in a background thread to keep the UI responsive
        thread = threading.Thread(target=self._process_audio, daemon=True)
        thread.start()

    def _process_audio(self) -> None:
        audio = self._recorder.stop()

        if len(audio) == 0:
            log.warning("No audio captured")
            self._set_state(State.IDLE)
            return

        log.info("Audio captured: %.1f seconds", len(audio) / self._conf["audio"]["sample_rate"])

        try:
            raw_text = self._transcriber.transcribe(
                audio,
                sample_rate=self._conf["audio"]["sample_rate"],
            )
            log.info("Whisper transcript: %r", raw_text)
        except Exception as exc:
            log.exception("Transcription failed")
            self._set_state(State.IDLE)
            rumps.notification(
                title="Luduan",
                subtitle="Transcription error",
                message=str(exc),
                sound=False,
            )
            return

        if not raw_text:
            log.info("Empty transcript — nothing to output")
            self._set_state(State.IDLE)
            return

        # LLM post-processing
        cleaned_text, used_llm = self._postprocessor.process(raw_text)
        if used_llm:
            log.info("Ollama cleaned text: %r", cleaned_text)
        else:
            log.info("Ollama skipped — using raw transcript")

        # Output
        paste_result = self._output.paste(cleaned_text)
        if paste_result == PasteResult.OK:
            log.info("Text pasted successfully")
        elif paste_result == PasteResult.NO_ACCESSIBILITY:
            log.warning("Paste blocked — Accessibility permission not granted")
            self._on_accessibility_denied()
        else:
            log.warning("Auto-paste failed — text left in clipboard")
            rumps.notification(
                title="Luduan",
                subtitle="Copied to clipboard",
                message="Auto-paste failed. Text is in your clipboard.",
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
        # When using an image icon, show state via the title overlay;
        # when using emoji fallback, replace the title entirely.
        if self.icon:
            self.title = "" if state == State.IDLE else _ICONS[state]
        else:
            self.title = _ICONS[state]
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

    def _on_show_log(self, _sender) -> None:
        subprocess.run(["open", str(LOG_PATH)], check=False)

    def _on_fix_accessibility(self, _sender) -> None:
        TextOutput.open_accessibility_settings()

    def _on_quit(self, _sender) -> None:
        if self._recorder.is_recording:
            self._recorder.stop()
        rumps.quit_application()

    # ------------------------------------------------------------------
    # Language menu
    # ------------------------------------------------------------------

    # (code, display label)
    _LANGUAGES: list[tuple[str, str]] = [
        ("auto",  "Auto-detect"),
        ("zh",    "Mandarin (中文)"),
        ("en",    "English"),
        ("sv",    "Swedish (Svenska)"),
        ("fr",    "French (Français)"),
    ]

    def _build_language_menu(self, current: str) -> rumps.MenuItem:
        submenu = rumps.MenuItem("Language")
        for code, label in self._LANGUAGES:
            item = rumps.MenuItem(label, callback=self._on_language_select)
            item.state = 1 if code == current else 0
            submenu.add(item)
        return submenu

    def _on_language_select(self, sender: rumps.MenuItem) -> None:
        # Find the matching code for the clicked label
        code = next(
            (c for c, lbl in self._LANGUAGES if lbl == sender.title),
            "auto",
        )
        # Uncheck all, check selected
        for item in self._lang_menu.values():
            item.state = 1 if item.title == sender.title else 0

        # Apply to transcriber immediately (no restart needed)
        self._transcriber.language = None if code == "auto" else code
        log.info("Language changed to %s", code)

        # Persist to config
        self._conf["whisper"]["language"] = "" if code == "auto" else code
        cfg.save(self._conf)
        log.info("Config saved")

    # ------------------------------------------------------------------
    # Accessibility permission helpers
    # ------------------------------------------------------------------

    def _check_accessibility(self) -> None:
        """Probe for Accessibility permission by doing a dry-run paste of empty string."""
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke ""'],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 and "1002" in result.stderr:
            log.warning("Accessibility permission not granted — adding Fix menu item")
            self._show_accessibility_warning()

    def _on_accessibility_denied(self) -> None:
        """Called when a real paste attempt is blocked by missing Accessibility."""
        if self._accessibility_ok:
            self._accessibility_ok = False
            self._show_accessibility_warning()

    def _show_accessibility_warning(self) -> None:
        """Insert the Fix Permissions item into the menu and notify the user."""
        # Insert warning item if not already present
        if "⚠️  Grant Accessibility Permission…" not in self.menu:
            self.menu.insert_before("Show Log", self._fix_perms_item)

        rumps.notification(
            title="Luduan — Action Required",
            subtitle="Accessibility permission needed",
            message=(
                "Luduan can't paste text without Accessibility access. "
                "Click '⚠️ Grant Accessibility Permission…' in the menu."
            ),
            sound=False,
        )

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
        log.warning("Ollama not reachable at %s", conf["ollama"]["host"])
        rumps.notification(
            title="Luduan",
            subtitle="Ollama not running",
            message=(
                "Start Ollama (`ollama serve`) for LLM post-processing. "
                "Transcription will still work without it."
            ),
            sound=False,
        )
    else:
        log.info("Ollama reachable at %s, model=%s", conf["ollama"]["host"], conf["ollama"]["model"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    conf = cfg.load()

    # Ensure HuggingFace models land in our config dir, not ~/.cache/huggingface
    # which may be a broken symlink to an unmounted volume.
    import os
    from luduan.log import LOG_PATH
    hf_home = str(cfg.CONFIG_DIR / "models")
    os.makedirs(hf_home, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_home)

    log.info("Luduan starting — log: %s", LOG_PATH)
    log.info("Config: whisper=%s  ollama=%s/%s  hotkey=%s",
             conf["whisper"]["model"],
             conf["ollama"]["host"], conf["ollama"]["model"],
             conf["hotkey"]["keys"])
    app = LuduanApp(conf)

    # Defer first-run check until after app starts
    threading.Timer(1.5, first_run_check, args=(conf,)).start()

    app.run()


if __name__ == "__main__":
    main()
