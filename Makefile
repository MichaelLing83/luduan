.PHONY: app dmg install uninstall clean rust-cli rust-cli-check rust-cli-run

DIST_DIR   := dist
APP_NAME   := Luduan
APP_BUNDLE := $(DIST_DIR)/$(APP_NAME).app
DMG_FILE   := $(DIST_DIR)/$(APP_NAME).dmg
DMG_STAGE  := $(DIST_DIR)/dmg-staging

# ── Build ─────────────────────────────────────────────────────────────────────

app:
	@echo "→ Building $(APP_NAME).app …"
	@rm -rf $(APP_BUNDLE)
	@python3 -m venv .venv --copies 2>/dev/null || true
	@.venv/bin/python -m pip install --quiet -e . pyinstaller
	@.venv/bin/pyinstaller --noconfirm Luduan.spec

	@echo "✅ $(APP_BUNDLE)"
	@echo "   → make install   to copy to /Applications"
	@echo "   → make dmg       to build $(DMG_FILE)"
	@echo "   → open $(APP_BUNDLE)   to run now"

dmg: app
	@echo "→ Building $(DMG_FILE) …"
	@rm -rf $(DMG_STAGE) $(DMG_FILE)
	@mkdir -p $(DMG_STAGE)
	@cp -R $(APP_BUNDLE) $(DMG_STAGE)/
	@ln -s /Applications $(DMG_STAGE)/Applications
	@hdiutil create \
		-volname "$(APP_NAME)" \
		-srcfolder $(DMG_STAGE) \
		-ov \
		-format UDZO \
		$(DMG_FILE) >/dev/null
	@rm -rf $(DMG_STAGE)
	@echo "✅ $(DMG_FILE)"

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
	@rm -rf $(DIST_DIR) build/Luduan
	@cargo clean --manifest-path rust-cli/Cargo.toml 2>/dev/null || true
	@echo "Cleaned."

# ── Rust CLI ────────────────────────────────────────────────────────────────────

rust-cli:
	@cargo build --manifest-path rust-cli/Cargo.toml --release

rust-cli-check:
	@cargo check --manifest-path rust-cli/Cargo.toml

rust-cli-run:
	@cargo run --manifest-path rust-cli/Cargo.toml -- $(ARGS)
