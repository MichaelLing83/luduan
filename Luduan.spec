# -*- mode: python ; coding: utf-8 -*-

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


def read_version() -> str:
    init_py = Path("src/luduan/__init__.py").read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', init_py)
    if not match:
        raise RuntimeError("Could not read Luduan version from src/luduan/__init__.py")
    return match.group(1)


VERSION = read_version()
HIDDEN_IMPORTS = collect_submodules("mlx") + collect_submodules("mlx_whisper")
BINARIES = collect_dynamic_libs("mlx")
DATAS = collect_data_files("mlx") + [('build/menubar_icon.png', '.')]

a = Analysis(
    ['src/luduan/main.py'],
    pathex=['src'],
    binaries=BINARIES,
    datas=DATAS,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Luduan',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['build/AppIcon.icns'],
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Luduan',
)

app = BUNDLE(
    coll,
    name='Luduan.app',
    icon='build/AppIcon.icns',
    bundle_identifier='com.luduan.app',
    version=VERSION,
    info_plist={
        'CFBundleName': 'Luduan',
        'CFBundleDisplayName': 'Luduan',
        'CFBundleShortVersionString': VERSION,
        'CFBundleVersion': VERSION,
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'Luduan records your voice to transcribe speech to text.',
        'NSAppleEventsUsageDescription': 'Luduan uses Apple Events to paste transcribed text into the active app.',
    },
)
