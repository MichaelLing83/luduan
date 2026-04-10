"""Paste text into the currently focused macOS application."""

from __future__ import annotations

import subprocess
import threading

import pyperclip


# AppleScript: save frontmost app, set clipboard, refocus, paste, restore
_PASTE_APPLESCRIPT = """\
tell application "System Events"
    set frontApp to name of first application process whose frontmost is true
end tell
do shell script "python3 -c \\"import pyperclip; pyperclip.copy({escaped_text})\\"" 2>/dev/null
tell application frontApp to activate
delay 0.15
tell application "System Events"
    keystroke "v" using {command down}
end tell
"""

_PASTE_APPLESCRIPT_V2 = """\
set theText to {escaped_text}
set the clipboard to theText
tell application "System Events"
    keystroke "v" using command down
end tell
"""


class TextOutput:
    """Pastes *text* into the active macOS app.

    Strategy:
    1. Copy text to clipboard
    2. Use AppleScript to send Cmd+V to the currently focused app
    3. If AppleScript fails, leave text in clipboard and return False
       (caller should show a notification)
    """

    def paste(self, text: str) -> bool:
        """Paste *text* into the active app.

        Returns True on success, False on failure (text still in clipboard).
        """
        if not text.strip():
            return True

        # Always put text in clipboard as a guaranteed fallback
        try:
            pyperclip.copy(text)
        except Exception:
            return False

        return self._applescript_paste(text)

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy *text* to clipboard only (no paste)."""
        try:
            pyperclip.copy(text)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _applescript_paste(self, text: str) -> bool:
        # Escape text for AppleScript: backslash-escape backslashes and quotes
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
            return result.returncode == 0
        except Exception:
            return False
