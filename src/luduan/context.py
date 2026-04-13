"""Read nearby text from the focused app for context-aware dictation."""

from __future__ import annotations

from dataclasses import dataclass

import AppKit
import ApplicationServices as AS

from luduan.log import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class AppContextSnapshot:
    """Context captured from the active app."""

    app_name: str | None
    before_cursor: str


class AppContextReader:
    """Reads text before the cursor from the focused accessibility element."""

    def is_trusted(self) -> bool:
        try:
            return bool(AS.AXIsProcessTrusted())
        except Exception:
            log.exception("Failed to query Accessibility trust state")
            return False

    def capture(self, max_chars: int = 500) -> AppContextSnapshot | None:
        """Return nearby text from the focused editable element, if available."""
        if max_chars <= 0:
            return None

        if not self.is_trusted():
            log.warning("Context capture skipped — Accessibility permission missing")
            return None

        focused = self._focused_element()
        if focused is None:
            return None

        before_cursor = self._text_before_cursor(focused, max_chars=max_chars)
        if not before_cursor:
            return None

        app_name = self._frontmost_app_name()
        return AppContextSnapshot(app_name=app_name, before_cursor=before_cursor)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _focused_element(self):
        system = AS.AXUIElementCreateSystemWide()
        err, value = AS.AXUIElementCopyAttributeValue(
            system,
            AS.kAXFocusedUIElementAttribute,
            None,
        )
        if err != 0 or value is None:
            log.debug("No focused accessibility element available (err=%s)", err)
            return None
        return value

    def _text_before_cursor(self, element, *, max_chars: int) -> str | None:
        selected_range = self._selected_range(element)

        if selected_range is not None:
            location, _length = selected_range
            location = max(0, location)

            text = self._string_for_range(
                element,
                start=max(0, location - max_chars),
                length=min(max_chars, location),
            )
            if text:
                return _normalize_context(text)

        value = self._copy_attribute(element, AS.kAXValueAttribute)
        if isinstance(value, str) and value:
            location = len(value)
            if selected_range is not None:
                location = max(0, min(selected_range[0], len(value)))
            return _normalize_context(value[max(0, location - max_chars):location])

        return None

    def _selected_range(self, element) -> tuple[int, int] | None:
        value = self._copy_attribute(element, AS.kAXSelectedTextRangeAttribute)
        if value is None:
            return None

        try:
            ok, cf_range = AS.AXValueGetValue(value, AS.kAXValueCFRangeType, None)
        except Exception:
            log.debug("Focused element does not expose AXSelectedTextRange")
            return None

        if not ok or not isinstance(cf_range, tuple) or len(cf_range) != 2:
            return None

        return int(cf_range[0]), int(cf_range[1])

    def _string_for_range(self, element, *, start: int, length: int) -> str | None:
        names = self._copy_parameterized_attribute_names(element)
        if AS.kAXStringForRangeParameterizedAttribute not in names:
            return None

        try:
            ax_range = AS.AXValueCreate(
                AS.kAXValueCFRangeType,
                AS.CFRangeMake(start, length),
            )
            err, value = AS.AXUIElementCopyParameterizedAttributeValue(
                element,
                AS.kAXStringForRangeParameterizedAttribute,
                ax_range,
                None,
            )
        except Exception:
            log.exception("Failed to read AXStringForRange context")
            return None

        if err != 0 or not isinstance(value, str):
            return None
        return value

    def _copy_parameterized_attribute_names(self, element) -> set[str]:
        try:
            err, value = AS.AXUIElementCopyParameterizedAttributeNames(element, None)
        except Exception:
            return set()
        if err != 0 or value is None:
            return set()
        return set(value)

    def _copy_attribute(self, element, attribute: str):
        try:
            err, value = AS.AXUIElementCopyAttributeValue(element, attribute, None)
        except Exception:
            return None
        if err != 0:
            return None
        return value

    def _frontmost_app_name(self) -> str | None:
        try:
            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            return app.localizedName() if app is not None else None
        except Exception:
            return None


def _normalize_context(text: str) -> str:
    """Reduce whitespace noise before sending context to the LLM."""
    return " ".join(text.split()).strip()
