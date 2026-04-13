# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ['src/luduan/main.py'],
    pathex=['src'],
    binaries=[],
    datas=[('build/menubar_icon.png', '.')],
    hiddenimports=[],
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
    version='0.1.0',
    info_plist={
        'CFBundleName': 'Luduan',
        'CFBundleDisplayName': 'Luduan',
        'LSUIElement': True,
        'NSMicrophoneUsageDescription': 'Luduan records your voice to transcribe speech to text.',
        'NSAppleEventsUsageDescription': 'Luduan uses Apple Events to paste transcribed text into the active app.',
    },
)
