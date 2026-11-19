#!/usr/bin/env bash
# Everything that must pass before a change lands. Run from the repository
# root with the project virtualenv active (docs/PROCEDURES.md).
set -euo pipefail

echo "== rust: format, lint, test"
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --exclude clearvisionnn-py --quiet
# The browser crate must also lint clean for its real target, with the target
# features the build uses (tools/web/build.sh): nanocodec's kernels need them.
RUSTFLAGS="${RUSTFLAGS:-} -C target-feature=+simd128" \
  cargo clippy -p clearvisionnn-web --target wasm32-unknown-unknown -- -D warnings

echo "== python: the extension the tests import must be the one just checked"
(cd crates/cvnn-py && env -u CONDA_PREFIX maturin develop --release --quiet)

echo "== python: format, lint, types, structure, test"
ruff format --check python tools
ruff check python tools
mypy
python tools/ci/check_layering.py
python tools/ci/check_file_size.py
python -m pytest -q

echo "== page: syntax, and the logic that needs no browser"
for module in web/js/*.js; do node --check "$module"; done
node --test tools/web/schedule.test.mjs
