"""Luduan menu bar app — entry point."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path

import AppKit
import objc
import rumps
from PyObjCTools import AppHelper

from luduan import __version__, config as cfg
from luduan.audio import AudioRecorder
from luduan.context import AppContextReader
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
    exe = Path(sys.executable)
    module = Path(__file__).resolve()
    candidates: list[Path] = []

    # PyInstaller bundle: executable in .../Contents/MacOS/Luduan,
    # resources in .../Contents/Resources/
    candidates.append(exe.parent.parent / "Resources" / filename)

    # Bundled venv app: interpreter in .../Contents/Resources/venv/bin/python,
    # resources in .../Contents/Resources/
    candidates.append(exe.parent.parent.parent / filename)

    # PyInstaller also exposes an extraction/resource dir via sys._MEIPASS.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / filename)

    # Dev tree: .../src/luduan/main.py -> repo/build/
    for parent in module.parents:
        build_path = parent / "build" / filename
        candidates.append(build_path)

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

    _MENU_BAR_LABEL = "L"

    def __init__(self, conf: dict) -> None:
        menubar_icon = _find_resource("menubar_icon.png")
        if menubar_icon:
            super().__init__(
                "Luduan",
                title=self._MENU_BAR_LABEL,
                icon=menubar_icon,
                template=False,
                quit_button=None,
            )
        else:
            super().__init__("Luduan", title=self._MENU_BAR_LABEL, quit_button=None)

        self._conf = conf
        self._state = State.IDLE
        self._lock = threading.Lock()

        # Components
        self._recorder = AudioRecorder(
            sample_rate=conf["audio"]["sample_rate"],
            channels=conf["audio"]["channels"],
            chunk_seconds=3.0,
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
        self._context_reader = AppContextReader()
        self._partial_text = ""
        self._accessibility_ok = True  # flips to False on first error 1002
        self._input_monitoring_ok = True
        self._offline_prepare_in_progress = False
        self._offline_status_message = "Offline: checking…"
        self._recording_started_at: float | None = None
        self._last_input_duration = 0.0
        self._recording_session_id = 0
        self._live_preview_enabled = True
        self._live_preview_notice = "Waiting for speech…"
        self._recording_panel: AppKit.NSPanel | None = None
        self._recording_panel_controller: _RecordingPanelController | None = None
        self._recording_status_label = None
        self._recording_meta_label = None
        self._recording_count_label = None
        self._recording_preview_view = None
        self._recording_stop_button = None

        # Menu
        self._start_item = rumps.MenuItem("Start Recording", callback=self._on_start_recording)
        self._stop_item = rumps.MenuItem("Stop Recording", callback=self._on_stop_recording)
        self._busy_item = rumps.MenuItem("Processing…", callback=None)
        self._busy_item.set_callback(None)
        self._status_item = rumps.MenuItem("Ready", callback=None)
        self._status_item.set_callback(None)  # non-clickable status line
        self._offline_status_item = rumps.MenuItem(self._offline_status_message, callback=None)
        self._offline_status_item.set_callback(None)
        self._version_item = rumps.MenuItem(f"Version: {__version__}", callback=None)
        self._version_item.set_callback(None)

        ollama_label = (
            f"LLM: {conf['ollama']['model']} (enabled)"
            if conf["ollama"]["enabled"]
            else "LLM: disabled"
        )
        self._ollama_item = rumps.MenuItem(ollama_label, callback=None)
        self._context_item = rumps.MenuItem(
            "Use App Context",
            callback=self._on_toggle_context,
        )
        self._context_item.state = 1 if conf["context"]["enabled"] else 0
        self._offline_item = rumps.MenuItem(
            "Prepare Offline Use",
            callback=self._on_prepare_offline,
        )
        self._fix_perms_item = rumps.MenuItem(
            "⚠️  Grant Accessibility Permission…",
            callback=self._on_fix_accessibility,
        )
        self._fix_input_item = rumps.MenuItem(
            "⚠️  Grant Input Monitoring Permission…",
            callback=self._on_fix_input_monitoring,
        )
        self._lang_menu = self._build_language_menu(conf["whisper"].get("language") or "auto")

        self.menu = [
            self._start_item,
            self._stop_item,
            self._busy_item,
            None,  # separator
            self._status_item,
            self._ollama_item,
            self._lang_menu,
            self._context_item,
            self._offline_item,
            self._offline_status_item,
            None,
            self._version_item,
            rumps.MenuItem("Show Log", callback=self._on_show_log),
            rumps.MenuItem("Quit Luduan", callback=self._on_quit),
        ]
        self._set_state(State.IDLE)

        # Start global hotkey listener
        self._start_hotkey_listener()

        # Check Accessibility on startup so the warning shows immediately
        threading.Timer(2.0, self._check_input_monitoring).start()
        threading.Timer(2.0, self._check_accessibility).start()
        threading.Thread(target=self._refresh_offline_status, daemon=True).start()
        threading.Thread(target=self._panel_update_loop, daemon=True).start()

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
        self._partial_text = ""
        self._live_preview_enabled = True
        self._live_preview_notice = "Listening… live preview will appear here."
        self._recording_started_at = time.monotonic()
        self._last_input_duration = 0.0
        with self._lock:
            self._recording_session_id += 1
            session_id = self._recording_session_id
        self._recorder.on_chunk = lambda chunk, sid=session_id: self._on_audio_chunk(sid, chunk)
        try:
            self._recorder.start()
            self._set_state(State.RECORDING)
            AppHelper.callAfter(self._refresh_recording_panel)
            log.info("Recording started")
        except Exception as exc:
            log.exception("Failed to start microphone")
            self._recorder.on_chunk = None
            self._set_state(State.IDLE)
            rumps.notification(
                title="Luduan",
                subtitle="Microphone error",
                message=str(exc),
                sound=False,
            )

    def _stop_recording(self) -> None:
        if self._recording_started_at is not None:
            self._last_input_duration = time.monotonic() - self._recording_started_at
        self._set_state(State.PROCESSING)
        AppHelper.callAfter(self._refresh_recording_panel)
        log.info("Recording stopped — starting transcription")
        # Run transcription in a background thread to keep the UI responsive
        thread = threading.Thread(target=self._process_audio, daemon=True)
        thread.start()

    def _process_audio(self) -> None:
        audio = self._recorder.stop()
        self._recorder.on_chunk = None

        if len(audio) == 0:
            log.warning("No audio captured")
            self._set_state(State.IDLE)
            return

        log.info("Audio captured: %.1f seconds", len(audio) / self._conf["audio"]["sample_rate"])
        self._live_preview_notice = "Final transcription in progress…"
        AppHelper.callAfter(self._refresh_recording_panel)

        try:
            raw_text = self._transcriber.transcribe(
                audio,
                sample_rate=self._conf["audio"]["sample_rate"],
            )
            self._partial_text = raw_text
            AppHelper.callAfter(self._refresh_recording_panel)
            log.info("Whisper transcript: %r", raw_text)
        except Exception as exc:
            log.exception("Transcription failed")
            self._live_preview_notice = "Transcription failed — see log."
            AppHelper.callAfter(self._refresh_recording_panel)
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
            self._live_preview_notice = "No speech detected."
            AppHelper.callAfter(self._refresh_recording_panel)
            self._set_state(State.IDLE)
            return

        context_before_cursor = None
        context_app_name = None
        if self._conf["context"]["enabled"]:
            snapshot = self._context_reader.capture(
                max_chars=self._conf["context"]["max_chars"],
            )
            if snapshot is not None:
                context_before_cursor = snapshot.before_cursor
                context_app_name = snapshot.app_name
                log.info(
                    "Captured %d chars of app context from %s",
                    len(context_before_cursor),
                    context_app_name or "focused app",
                )
            else:
                log.info("App context enabled but no focused text context was available")

        # LLM post-processing
        cleaned_text, used_llm = self._postprocessor.process(
            raw_text,
            context_before_cursor=context_before_cursor,
            app_name=context_app_name,
        )
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
        AppHelper.callAfter(self._refresh_recording_panel)
        AppHelper.callAfter(self._update_status, f'"{_menu_preview(text)}"')

    def _on_audio_chunk(self, session_id: int, chunk) -> None:
        if not self._live_preview_enabled:
            return

        with self._lock:
            if session_id != self._recording_session_id:
                return

        try:
            text = self._transcriber.transcribe(
                chunk,
                sample_rate=self._conf["audio"]["sample_rate"],
                emit_partial=False,
            )
        except Exception:
            log.exception("Partial chunk transcription failed")
            self._live_preview_enabled = False
            self._live_preview_notice = (
                "Live preview unavailable. Recording continues; final transcription will run after stop."
            )
            AppHelper.callAfter(self._refresh_recording_panel)
            return

        if not text:
            return

        with self._lock:
            if session_id != self._recording_session_id:
                return
            self._partial_text = _merge_transcript(self._partial_text, text)

        AppHelper.callAfter(self._refresh_recording_panel)
        AppHelper.callAfter(self._update_status, f'"{_menu_preview(self._partial_text)}"')

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _set_state(self, state: State) -> None:
        with self._lock:
            self._state = state
        # Keep the bundle icon stable in the menu bar so it stays easy to find.
        # Fall back to text/emoji state only when no icon resource is available.
        if self.icon:
            self.title = self._MENU_BAR_LABEL
        else:
            self.title = self._MENU_BAR_LABEL if state == State.IDLE else _ICONS[state]
        if state == State.IDLE:
            AppHelper.callAfter(self._hide_recording_panel)
            self._start_item.show()
            self._stop_item.hide()
            self._busy_item.hide()
            self._update_status("Ready")
        elif state == State.RECORDING:
            AppHelper.callAfter(self._show_recording_panel)
            self._start_item.hide()
            self._stop_item.show()
            self._busy_item.hide()
            self._update_status("Recording…")
        elif state == State.PROCESSING:
            AppHelper.callAfter(self._show_recording_panel)
            self._start_item.hide()
            self._stop_item.hide()
            self._busy_item.show()
            self._update_status("Transcribing…")

    def _update_status(self, message: str) -> None:
        self._status_item.title = message

    def _set_offline_status(self, message: str) -> None:
        self._offline_status_message = message
        self._offline_status_item.title = message

    def _show_recording_panel(self) -> None:
        if self._recording_panel is None:
            self._recording_panel_controller = _RecordingPanelController.alloc().initWithApp_(self)

            style_mask = (
                AppKit.NSWindowStyleMaskTitled
                | AppKit.NSWindowStyleMaskClosable
                | AppKit.NSWindowStyleMaskUtilityWindow
            )
            panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
                AppKit.NSMakeRect(0, 0, 360, 220),
                style_mask,
                AppKit.NSBackingStoreBuffered,
                False,
            )
            panel.setTitle_("Luduan")
            panel.setFloatingPanel_(True)
            panel.setBecomesKeyOnlyIfNeeded_(True)
            panel.setHidesOnDeactivate_(False)
            panel.setReleasedWhenClosed_(False)
            panel.setLevel_(AppKit.NSFloatingWindowLevel)
            panel.setCollectionBehavior_(
                AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
                | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            )

            content = panel.contentView()

            self._recording_status_label = AppKit.NSTextField.labelWithString_("Recording…")
            self._recording_status_label.setFrame_(AppKit.NSMakeRect(20, 184, 320, 22))
            self._recording_status_label.setFont_(AppKit.NSFont.boldSystemFontOfSize_(15))
            content.addSubview_(self._recording_status_label)

            self._recording_meta_label = AppKit.NSTextField.labelWithString_("Input: 0.0 s")
            self._recording_meta_label.setFrame_(AppKit.NSMakeRect(20, 160, 150, 18))
            self._recording_meta_label.setTextColor_(AppKit.NSColor.secondaryLabelColor())
            content.addSubview_(self._recording_meta_label)

            self._recording_count_label = AppKit.NSTextField.labelWithString_("Preview: 0 words")
            self._recording_count_label.setFrame_(AppKit.NSMakeRect(180, 160, 160, 18))
            self._recording_count_label.setAlignment_(AppKit.NSTextAlignmentRight)
            self._recording_count_label.setTextColor_(AppKit.NSColor.secondaryLabelColor())
            content.addSubview_(self._recording_count_label)

            latest_label = AppKit.NSTextField.labelWithString_("Latest transcript")
            latest_label.setFrame_(AppKit.NSMakeRect(20, 138, 160, 16))
            latest_label.setTextColor_(AppKit.NSColor.secondaryLabelColor())
            content.addSubview_(latest_label)

            scroll = AppKit.NSScrollView.alloc().initWithFrame_(AppKit.NSMakeRect(20, 50, 320, 82))
            scroll.setBorderType_(AppKit.NSBezelBorder)
            scroll.setHasVerticalScroller_(True)
            scroll.setAutohidesScrollers_(True)

            self._recording_preview_view = AppKit.NSTextView.alloc().initWithFrame_(AppKit.NSMakeRect(0, 0, 320, 82))
            self._recording_preview_view.setEditable_(False)
            self._recording_preview_view.setSelectable_(False)
            self._recording_preview_view.setRichText_(False)
            self._recording_preview_view.setFont_(AppKit.NSFont.systemFontOfSize_(13))
            self._recording_preview_view.setString_("Waiting for speech…")
            scroll.setDocumentView_(self._recording_preview_view)
            content.addSubview_(scroll)

            self._recording_stop_button = AppKit.NSButton.alloc().initWithFrame_(AppKit.NSMakeRect(20, 14, 320, 28))
            self._recording_stop_button.setTitle_("Stop Recording")
            self._recording_stop_button.setTarget_(self._recording_panel_controller)
            self._recording_stop_button.setAction_("stopClicked:")
            self._recording_stop_button.setBezelStyle_(AppKit.NSBezelStyleRounded)
            content.addSubview_(self._recording_stop_button)

            self._recording_panel = panel

        screen = AppKit.NSScreen.mainScreen() or AppKit.NSScreen.screens()[0]
        visible = screen.visibleFrame()
        frame = self._recording_panel.frame()
        origin_x = visible.origin.x + visible.size.width - frame.size.width - 20
        origin_y = visible.origin.y + visible.size.height - frame.size.height - 40
        self._recording_panel.setFrameOrigin_(AppKit.NSPoint(origin_x, origin_y))
        self._refresh_recording_panel()
        self._recording_panel.orderFrontRegardless()

    def _hide_recording_panel(self) -> None:
        if self._recording_panel is not None:
            self._recording_panel.orderOut_(None)

    def _panel_update_loop(self) -> None:
        while True:
            time.sleep(0.5)
            with self._lock:
                state = self._state
            if state in {State.RECORDING, State.PROCESSING}:
                AppHelper.callAfter(self._refresh_recording_panel)

    def _refresh_recording_panel(self) -> None:
        if self._recording_panel is None:
            return

        with self._lock:
            state = self._state
            transcript = self._partial_text.strip()

        if state == State.RECORDING and self._recording_started_at is not None:
            elapsed = time.monotonic() - self._recording_started_at
        else:
            elapsed = self._last_input_duration

        latest = _latest_sentence(transcript) or self._live_preview_notice
        count_label = _transcript_count_label(
            transcript,
            self._conf["whisper"].get("language") or "auto",
        )

        if self._recording_status_label is not None:
            self._recording_status_label.setStringValue_(
                "Recording…" if state == State.RECORDING else "Transcribing…"
            )
        if self._recording_meta_label is not None:
            self._recording_meta_label.setStringValue_(f"Input: {elapsed:.1f} s")
        if self._recording_count_label is not None:
            self._recording_count_label.setStringValue_(count_label)
        if self._recording_preview_view is not None:
            self._recording_preview_view.setString_(latest)
            self._recording_preview_view.scrollRangeToVisible_(
                AppKit.NSMakeRange(max(0, len(latest) - 1), 1 if latest else 0)
            )
        if self._recording_stop_button is not None:
            if state == State.RECORDING:
                self._recording_stop_button.setEnabled_(True)
                self._recording_stop_button.setTitle_("Stop Recording")
            else:
                self._recording_stop_button.setEnabled_(False)
                self._recording_stop_button.setTitle_("Transcribing…")

    # ------------------------------------------------------------------
    # Menu callbacks
    # ------------------------------------------------------------------

    def _on_start_recording(self, _sender) -> None:
        with self._lock:
            if self._state != State.IDLE:
                return
        log.info("Menu start recording clicked")
        self._start_recording()

    def _on_stop_recording(self, _sender) -> None:
        with self._lock:
            if self._state != State.RECORDING:
                return
        log.info("Menu stop recording clicked")
        self._stop_recording()

    def _on_show_log(self, _sender) -> None:
        subprocess.run(["open", str(LOG_PATH)], check=False)

    def _on_fix_accessibility(self, _sender) -> None:
        TextOutput.open_accessibility_settings()

    def _on_fix_input_monitoring(self, _sender) -> None:
        TextOutput.open_input_monitoring_settings()

    def _on_toggle_context(self, _sender) -> None:
        enabled = not bool(self._conf["context"]["enabled"])
        self._conf["context"]["enabled"] = enabled
        self._context_item.state = 1 if enabled else 0
        cfg.save(self._conf)
        log.info("App context mode %s", "enabled" if enabled else "disabled")

        if enabled and not self._context_reader.is_trusted():
            self._show_accessibility_dialog(on_paste_fail=False)

    def _on_prepare_offline(self, _sender) -> None:
        if self._offline_prepare_in_progress:
            return

        with self._lock:
            if self._state != State.IDLE:
                rumps.notification(
                    title="Luduan",
                    subtitle="Busy",
                    message="Wait until recording/transcription is finished before preparing offline use.",
                    sound=False,
                )
                return

        self._offline_prepare_in_progress = True
        self._offline_item.title = "Preparing Offline Use…"
        self._set_offline_status("Offline: preparing…")
        thread = threading.Thread(target=self._prepare_offline_assets, daemon=True)
        thread.start()

    def _on_quit(self, _sender) -> None:
        if self._recorder.is_recording:
            self._recorder.stop()
        rumps.quit_application()

    def _prepare_offline_assets(self) -> None:
        try:
            self._offline_progress("Checking Whisper model…")
            whisper_downloaded = self._transcriber.ensure_model_cached()
            self._offline_progress(
                "Whisper ready"
                if whisper_downloaded
                else "Whisper already cached"
            )

            self._offline_progress("Checking Ollama model…")
            ollama_downloaded = self._postprocessor.ensure_model_available(
                progress_callback=self._offline_progress,
            )

            parts = [
                f"Whisper {'downloaded' if whisper_downloaded else 'already cached'}",
            ]
            if self._conf["ollama"]["enabled"]:
                parts.append(
                    f"Ollama {'downloaded' if ollama_downloaded else 'already cached'}"
                )
            else:
                parts.append("Ollama skipped (disabled)")

            message = "; ".join(parts)
            log.info("Offline preparation complete: %s", message)
            self._set_offline_status("Offline: ready")
            rumps.notification(
                title="Luduan",
                subtitle="Offline preparation complete",
                message=message,
                sound=False,
            )
        except Exception as exc:
            log.exception("Offline preparation failed")
            self._set_offline_status("Offline: failed — see log")
            rumps.notification(
                title="Luduan",
                subtitle="Offline preparation failed",
                message=str(exc),
                sound=False,
            )
        finally:
            self._offline_prepare_in_progress = False
            self._offline_item.title = "Prepare Offline Use"
            with self._lock:
                if self._state == State.IDLE:
                    self._update_status("Ready")

    def _offline_progress(self, message: str) -> None:
        log.info("Offline preparation: %s", message)
        self._set_offline_status(f"Offline: {message}")

    def _refresh_offline_status(self) -> None:
        try:
            whisper_ready = self._transcriber.is_model_cached()
            if self._conf["ollama"]["enabled"]:
                ollama_ready = self._postprocessor.has_model_available()
            else:
                ollama_ready = True
        except Exception:
            log.exception("Failed to determine offline readiness")
            self._set_offline_status("Offline: unknown")
            return

        if whisper_ready and ollama_ready:
            self._set_offline_status("Offline: ready")
        else:
            self._set_offline_status("Offline: not prepared")

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
    # Permission helpers
    # ------------------------------------------------------------------

    def _check_input_monitoring(self) -> None:
        """Probe for Input Monitoring permission used by the global hotkey."""
        if _has_input_monitoring_access():
            self._input_monitoring_ok = True
            return

        self._input_monitoring_ok = False
        _request_input_monitoring_access()
        if not _has_input_monitoring_access():
            log.warning("Input Monitoring permission not granted — global hotkey unavailable")
            self._show_input_monitoring_dialog()

    def _show_input_monitoring_warning(self) -> None:
        if "⚠️  Grant Input Monitoring Permission…" not in self.menu:
            self.menu.insert_before("Show Log", self._fix_input_item)

    def _show_input_monitoring_dialog(self) -> None:
        self._show_input_monitoring_warning()
        script = '''\
display dialog "Luduan needs Input Monitoring permission for the global hotkey.

Without it, Cmd+Shift+Space will not start or stop recording.

Open System Settings → Privacy & Security → Input Monitoring, then add and enable Luduan.

After enabling it, quit and reopen Luduan." ¬
    buttons {"Later", "Open System Settings"} ¬
    default button "Open System Settings" ¬
    with title "Luduan — Hotkey Permission Required" ¬
    with icon caution
'''
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if "Open System Settings" in result.stdout:
            TextOutput.open_input_monitoring_settings()
            log.info("Opened Input Monitoring settings at user request")

    def _check_accessibility(self) -> None:
        """Probe for Accessibility permission by doing a dry-run keystroke."""
        result = subprocess.run(
            ["osascript", "-e", 'tell application "System Events" to keystroke ""'],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0 and "1002" in result.stderr:
            log.warning("Accessibility permission not granted — showing dialog")
            self._show_accessibility_dialog(on_paste_fail=False)

    def _on_accessibility_denied(self) -> None:
        """Called when a real paste attempt is blocked by missing Accessibility."""
        if self._accessibility_ok:
            self._accessibility_ok = False
            self._show_accessibility_dialog(on_paste_fail=True)

    def _show_accessibility_warning(self) -> None:
        """Insert ⚠️ menu item (kept for reference; dialog is primary UX)."""
        if "⚠️  Grant Accessibility Permission…" not in self.menu:
            self.menu.insert_before("Show Log", self._fix_perms_item)

    def _show_accessibility_dialog(self, *, on_paste_fail: bool) -> None:
        """Show a native macOS dialog with an 'Open System Settings' button."""
        self._show_accessibility_warning()  # also add the menu item

        intro = (
            "Luduan tried to paste transcribed text but was blocked.\n\n"
            if on_paste_fail
            else ""
        )
        script = f'''\
display dialog "{intro}Luduan needs Accessibility permission to paste text into other apps.

Open System Settings → Privacy & Security → Accessibility, then add and enable Luduan.

That permission also lets Luduan read nearby text for context-aware dictation when you enable that option." ¬
    buttons {{"Later", "Open System Settings"}} ¬
    default button "Open System Settings" ¬
    with title "Luduan — Permission Required" ¬
    with icon caution
'''
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=30,
        )
        if "Open System Settings" in result.stdout:
            TextOutput.open_accessibility_settings()
            log.info("Opened Accessibility settings at user request")

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

    hotkey_expr = _format_hotkey(keys_conf)

    def on_activate() -> None:
        log.info("Global hotkey activated")
        on_trigger()

    try:
        hotkey = keyboard.HotKey(
            keyboard.HotKey.parse(hotkey_expr),
            on_activate,
        )
    except ValueError:
        log.exception("Invalid hotkey configuration: %s", keys_conf)
        return

    listener = None

    def for_canonical(callback):
        def handler(key):
            callback(listener.canonical(key))

        return handler

    try:
        listener = keyboard.Listener(
            on_press=for_canonical(hotkey.press),
            on_release=for_canonical(hotkey.release),
        )
        log.info("Global hotkey listener started for %s", hotkey_expr)
        with listener:
            listener.join()
    except Exception:
        log.exception("Global hotkey listener stopped unexpectedly")


def _format_hotkey(keys_conf: list[str]) -> str:
    """Convert config key list to a pynput hotkey expression string."""
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
    result: list[str] = []
    for k in keys_conf:
        k_lower = k.lower()
        result.append(mapping.get(k_lower, k_lower))
    return "+".join(result)


def _has_input_monitoring_access() -> bool:
    try:
        from Quartz import CGPreflightListenEventAccess

        return bool(CGPreflightListenEventAccess())
    except Exception:
        log.exception("Failed to query Input Monitoring trust state")
        return True


def _request_input_monitoring_access() -> bool:
    try:
        from Quartz import CGRequestListenEventAccess

        return bool(CGRequestListenEventAccess())
    except Exception:
        log.exception("Failed to request Input Monitoring permission")
        return False


def _merge_transcript(existing: str, new_text: str) -> str:
    existing = existing.strip()
    new_text = new_text.strip()
    if not existing:
        return new_text
    if not new_text:
        return existing
    return f"{existing} {new_text}".strip()


def _latest_sentence(text: str) -> str:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""

    parts = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", cleaned) if part.strip()]
    if parts:
        return parts[-1]
    return cleaned[-120:]


def _menu_preview(text: str) -> str:
    preview = _latest_sentence(text) or text.strip()
    if len(preview) > 60:
        return preview[:59] + "…"
    return preview


def _transcript_count_label(text: str, language_hint: str) -> str:
    if _should_count_chars(text, language_hint):
        count = sum(1 for ch in text if not ch.isspace())
        unit = "chars"
    else:
        count = len([word for word in re.split(r"\s+", text.strip()) if word])
        unit = "words"
    return f"Preview: {count} {unit}"


def _should_count_chars(text: str, language_hint: str) -> bool:
    if language_hint.startswith("zh"):
        return True
    if language_hint == "auto":
        return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    return False


class _RecordingPanelController(AppKit.NSObject):
    def initWithApp_(self, app):
        self = objc.super(_RecordingPanelController, self).init()
        if self is None:
            return None
        self._app = app
        return self

    def stopClicked_(self, _sender) -> None:
        self._app._on_stop_recording(None)


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

    # Force menu bar app behavior even if macOS ignores LSUIElement for the
    # embedded Python runtime. This hides the Dock icon and keeps Luduan in the
    # status bar where the controls live.
    ns_app = AppKit.NSApplication.sharedApplication()
    ns_app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    app = LuduanApp(conf)

    # Defer first-run check until after app starts
    threading.Timer(1.5, first_run_check, args=(conf,)).start()

    app.run()


if __name__ == "__main__":
    main()
