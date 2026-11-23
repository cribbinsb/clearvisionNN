#!/usr/bin/env bash
# Build nanocodec's browser module into web/nanocodec (generated; not
# source-controlled). The demo's simulated sender can encode with it, which
# lets the page show the real quantiser and type of every frame. The sibling
# nanocodec checkout is only read: the build's output goes under target/.
# Needs the wasm32-unknown-unknown target and wasm-bindgen-cli 0.2.105, or
# WASM_BINDGEN set to that executable.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export PATH="$HOME/.cargo/bin:$PATH"
bindgen=${WASM_BINDGEN:-wasm-bindgen}
if [[ "$("$bindgen" --version)" != 'wasm-bindgen 0.2.105' ]]; then
  echo 'nanocodec-web needs wasm-bindgen-cli 0.2.105 (matching its Cargo.toml).' >&2
  exit 1
fi
nanocodec=${NANOCODEC:-../nanocodec}
target=$PWD/target/nanocodec-web
# The flags of nanocodec's own tools/web/build.sh: its kernels need SIMD128,
# and its encoders a deeper stack than the default.
RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-zstack-size=8388608 -C target-feature=+simd128" \
  CARGO_TARGET_DIR=$target \
  cargo build --manifest-path "$nanocodec/Cargo.toml" --frozen --profile web \
  -p nanocodec-web --target wasm32-unknown-unknown
"$bindgen" "$target/wasm32-unknown-unknown/web/nc_web.wasm" --target web --out-dir web/nanocodec --out-name nc_web
ls -la web/nanocodec
