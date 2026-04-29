#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/.cargo/bin"
BINARY_NAME="luduan"
SOURCE_BINARY="${SCRIPT_DIR}/target/release/${BINARY_NAME}"
TARGET_BINARY="${INSTALL_DIR}/${BINARY_NAME}"

cd "${SCRIPT_DIR}"

CURRENT_VERSION="$(python3 - <<'PY'
import tomllib
from pathlib import Path

data = tomllib.loads(Path("Cargo.toml").read_text())
print(data["package"]["version"])
PY
)"

requested_version="${RELEASE_VERSION:-}"
if [[ -z "${requested_version}" && -t 0 ]]; then
  read -r -p "Current version is ${CURRENT_VERSION}. Enter new version (leave blank to keep): " requested_version
fi

if [[ -n "${requested_version}" && "${requested_version}" != "${CURRENT_VERSION}" ]]; then
  if [[ ! "${requested_version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-+][0-9A-Za-z.-]+)?$ ]]; then
    echo "Invalid version: ${requested_version}" >&2
    exit 1
  fi

  echo "→ Updating version ${CURRENT_VERSION} → ${requested_version} ..."
  python3 - "${requested_version}" <<'PY'
import re
import sys
from pathlib import Path

version = sys.argv[1]
path = Path("Cargo.toml")
text = path.read_text()
updated = re.sub(
    r'(?m)^(version\s*=\s*")[^"]+(")$',
    rf'\g<1>{version}\2',
    text,
    count=1,
)
if updated == text:
    raise SystemExit("failed to update Cargo.toml package version")
path.write_text(updated)
PY
  cargo generate-lockfile
  CURRENT_VERSION="${requested_version}"
fi

echo "→ Building ${BINARY_NAME} (release) ..."
cargo build --release --bin "${BINARY_NAME}"

mkdir -p "${INSTALL_DIR}"
install -m 755 "${SOURCE_BINARY}" "${TARGET_BINARY}"

echo "✅ Installed ${TARGET_BINARY}"
echo "✅ Version ${CURRENT_VERSION}"
