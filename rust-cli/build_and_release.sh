#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR="${HOME}/.cargo/bin"
BINARY_NAME="luduan"
SOURCE_BINARY="${SCRIPT_DIR}/target/release/${BINARY_NAME}"
TARGET_BINARY="${INSTALL_DIR}/${BINARY_NAME}"

cd "${SCRIPT_DIR}"

echo "→ Building ${BINARY_NAME} (release) ..."
cargo build --release --bin "${BINARY_NAME}"

mkdir -p "${INSTALL_DIR}"
install -m 755 "${SOURCE_BINARY}" "${TARGET_BINARY}"

echo "✅ Installed ${TARGET_BINARY}"
