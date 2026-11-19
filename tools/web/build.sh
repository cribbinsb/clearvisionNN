#!/usr/bin/env bash
# Build the browser module into web/pkg (generated; not source-controlled).
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.105
# (`cargo install wasm-bindgen-cli --version 0.2.105 --locked`), or
# WASM_BINDGEN set to that executable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.cargo/bin:$PATH"
bindgen=${WASM_BINDGEN:-wasm-bindgen}
if [[ "$("$bindgen" --version)" != 'wasm-bindgen 0.2.105' ]]; then
  echo 'cvnn-web needs wasm-bindgen-cli 0.2.105 (matching its Cargo.toml).' >&2
  exit 1
fi
# SIMD128 is in every browser that has WebGPU or WebNN.
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
  cargo build --profile web -p clearvisionnn-web --target wasm32-unknown-unknown
"$bindgen" target/wasm32-unknown-unknown/web/cvnn_web.wasm --target web --out-dir web/pkg --out-name cvnn_web
ls -la web/pkg
