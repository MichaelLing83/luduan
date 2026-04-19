"""Paste text into the currently focused macOS application."""

from __future__ import annotations

import subprocess
from enum import Enum, auto

import pyperclip

from luduan.log import get_logger

log = get_logger(__name__)

# System Settings URL for the Accessibility pane (macOS Ventura+)
_ACCESSIBILITY_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
)
_INPUT_MONITORING_SETTINGS_URL = (
    "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
)


class PasteResult(Enum):
    OK = auto()
    NO_ACCESSIBILITY = auto()   # error 1002 — permission not granted
    FAILED = auto()             # other osascript error


class TextOutput:
    """Pastes *text* into the active macOS app.

    Strategy:
    1. Copy text to clipboard (always — guaranteed fallback)
    2. Use AppleScript to send Cmd+V to the currently focused app
    3. Distinguish between a missing Accessibility permission (fixable)
       and other failures so the caller can show the right message.
    """

    def paste(self, text: str) -> PasteResult:
        """Paste *text* into the active app. Always copies to clipboard first.

        Returns a PasteResult indicating success or the failure reason.
        """
        if not text.strip():
            return PasteResult.OK

        try:
            pyperclip.copy(text)
        except Exception:
            log.exception("Failed to copy to clipboard")
            return PasteResult.FAILED

        return self._applescript_paste(text)

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy *text* to clipboard only (no paste)."""
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    @staticmethod
    def open_accessibility_settings() -> None:
        """Open System Settings → Privacy & Security → Accessibility."""
        subprocess.run(["open", _ACCESSIBILITY_SETTINGS_URL], check=False)

    @staticmethod
    def open_input_monitoring_settings() -> None:
        """Open System Settings → Privacy & Security → Input Monitoring."""
        subprocess.run(["open", _INPUT_MONITORING_SETTINGS_URL], check=False)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _applescript_paste(self, text: str) -> PasteResult:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        script = f'''\
set theText to "{escaped}"
set the clipboard to theText
delay 0.1
tell application "System Events"
    keystroke "v" using command down
end tell
'''
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            log.exception("osascript subprocess error")
            return PasteResult.FAILED

        if result.returncode == 0:
            return PasteResult.OK

        stderr = result.stderr.strip()
        log.warning("osascript failed (rc=%d): %s", result.returncode, stderr)

        # Error 1002 = Accessibility permission not granted
        if "1002" in stderr or "not allowed to send keystrokes" in stderr:
            return PasteResult.NO_ACCESSIBILITY

        return PasteResult.FAILED
