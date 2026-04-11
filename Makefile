.PHONY: app install uninstall clean

DIST_DIR   := dist
APP_NAME   := Luduan
APP_BUNDLE := $(DIST_DIR)/$(APP_NAME).app
CONTENTS   := $(APP_BUNDLE)/Contents
VENV       := $(CONTENTS)/Resources/venv

# ── Build ─────────────────────────────────────────────────────────────────────

app:
	@echo "→ Building $(APP_NAME).app …"
	@rm -rf $(APP_BUNDLE)
	@mkdir -p $(CONTENTS)/MacOS $(CONTENTS)/Resources

	@# Metadata
	@cp build/Info.plist $(CONTENTS)/Info.plist
	@printf 'APPL????' > $(CONTENTS)/PkgInfo

	@# Bundled Python environment
	@echo "  Creating bundled venv …"
	@python3 -m venv $(VENV) --copies
	@$(VENV)/bin/pip install --quiet -e . mlx-whisper

	@# Icons
	@cp build/AppIcon.icns $(CONTENTS)/Resources/AppIcon.icns
	@cp build/menubar_icon.png $(CONTENTS)/Resources/menubar_icon.png

	@# Compiled launcher stub (gives the process Luduan's identity and icon)
	@echo "  Compiling launcher stub …"
	@cc -O2 -o $(CONTENTS)/MacOS/$(APP_NAME) build/launcher.c
	@chmod +x $(CONTENTS)/MacOS/$(APP_NAME)

	@echo "✅ $(APP_BUNDLE)"
	@echo "   → make install   to copy to /Applications"
	@echo "   → open $(APP_BUNDLE)   to run now"

# ── Install / uninstall ───────────────────────────────────────────────────────

install: $(APP_BUNDLE)
	@echo "→ Installing to /Applications …"
	@rm -rf /Applications/$(APP_NAME).app
	@cp -r $(APP_BUNDLE) /Applications/
	@# Force Launch Services to pick up the new icon immediately
	@/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister \
	    -f /Applications/$(APP_NAME).app
	@touch /Applications/$(APP_NAME).app
	@killall Dock 2>/dev/null || true
	@echo "✅ /Applications/$(APP_NAME).app"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Open System Settings → Privacy & Security → Input Monitoring"
	@echo "  2. Add Luduan and enable it"
	@echo "  3. Launch Luduan from Spotlight or /Applications"

uninstall:
	@rm -rf /Applications/$(APP_NAME).app
	@echo "Removed /Applications/$(APP_NAME).app"

# ── Dev (run without building the app) ───────────────────────────────────────

run:
	@mkdir -p ~/.config/luduan/models
	@HF_HOME=~/.config/luduan/models .venv/bin/python -m luduan.main

dev:
	@python3 -m venv .venv --copies 2>/dev/null || true
	@.venv/bin/pip install --quiet -e . mlx-whisper

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	@rm -rf $(DIST_DIR)
	@echo "Cleaned."
