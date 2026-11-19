# clearvisionNN — working rules

Start here: `docs/PROCEDURES.md` (how to
build, run and check), `docs/ARCHITECTURE.md` (structure), `docs/DESIGN.md` (the
design and the reasons behind it). Keep them current: they are how work is
handed over.

Fast, causal 2x video super-resolution for two cases: camera output after the
ISP, and video after a real-time encode and decode. A pre-process in Rust
(motion and side planes, built on nanocodec's analysis) feeds a network that
runs on a PC or in a browser.

## Code quality is a hard requirement

The project MUST be cleanly structured, modular and extensible, with
well-written, understandable code. When a quick hack conflicts with clean
structure, choose clean structure.

### Layout

```
crates/
  cvnn-common/     shared types: block grid, per-frame features, errors
  cvnn-preproc/    the pre-process: motion field and side planes, on nc-analysis
  cvnn-runtime/    per-stream session around the network: upscale, warp, blend, colour
  cvnn-py/         Python bindings to cvnn-preproc and cvnn-runtime (module `clearvisionnn_native`)
  cvnn-web/        browser bindings to cvnn-runtime (WebAssembly)
python/clearvisionnn/
  registry.py    named, swappable components
  contract.py    the network's input contract, read from the Rust pre-process
  config/        typed run configuration, the source registry
  video/         clip container, probing, ffmpeg and in-process reading
  pipeline/      one module per stage; degrade/ has one module per degradation
  data/          samples and batches over the pipeline's artefacts
  models/        network definitions (single source for every export)
  losses/  train/  eval/  export/  cli/
web/             the static demo page (GitHub Pages): one small ES module per job
python/tests/    configs/    tools/ci/    tools/web/    docs/
```

Planned, not yet built: `cvnn-cli` (native upscaler).

### Rules

- **No god files.** One job per file. `tools/ci/check_file_size.py` fails a
  source file over 250 lines of Python or 300 of Rust; split it, do not raise
  the limit. Every file starts with a doc comment saying what it does.
- **One-way dependencies, checked in CI.**
  - Rust (enforced by Cargo): cvnn-py, cvnn-web -> cvnn-runtime -> cvnn-preproc ->
    cvnn-common -> nanocodec crates.
  - Python (`tools/ci/check_layering.py`): registry, contract, config, video -> pipeline
    -> data -> models, losses -> train -> eval -> export -> cli.
- **Stages talk only through declared artefacts** (`pipeline/records.py`,
  stored in the work directory). A stage never calls into another stage.
- **Swappable parts are registered, not dispatched.** Degradation modules,
  layouts, losses, metrics and baselines each have a `Registry`; adding one is
  a new file with a decorator, never an edit to an if-chain.
- **Configuration is typed and strict.** Every setting is a field of a
  pydantic model in `config/`; unknown keys are errors.
- **The pre-process exists once, in Rust.** Training calls it through
  `clearvisionnn_native`; there is no Python or PyTorch copy.
- **The runtime's arithmetic is specified once and has one reference, in
  Rust** (`cvnn-runtime`): everything around the network at run time and, for a
  model in integer form, the network too (`docs/INTEGER.md`). A second
  implementation is allowed only where the reference cannot run (the GPU),
  holds no arithmetic the reference lacks, and is held to it bit for bit by a
  documented check on real frames (`tools/web/int8_check.py` for the network,
  `tools/web/runtime_check.py` for the whole frame). The float runtime must
  match `models/` exactly: `python/tests/test_runtime_parity.py` checks every
  model variant frame for frame, and must be extended with each new one. The
  integer runtime replaces its float steps with fixed-point ones, and is held
  to it within the tolerance `python/tests/test_integer_runtime.py` tests.
- **Bindings hold no logic.** `cvnn-py` and `cvnn-web` only convert types.
- **One model definition.** Networks are defined once in `models/`; exports are
  generated from it. Layout-independent parts live in `trunk.py`.
- **No global mutable state.** State lives in explicit objects.
- **`unsafe` is forbidden** in every Rust crate except the glue the binding
  macros expand to in cvnn-py and cvnn-web.
- **No magic numbers:** named constants with a comment giving the reason, or
  configuration fields.
- **No dead code, no commented-out code,** no placeholder modules for work
  that has not started.
- **Comments explain why, not what.**
- **The test split is sealed.** It is assigned by source file at ingest;
  nothing downstream may move a clip, and tuning decisions never use it.
- **Never write under a source dataset root** (`datasets/`). All derived data
  goes to the run's work directory, which has a disk budget.
- **Artefacts carry their identity.** A clip's ID comes from its source
  file's full relative path; a target's or variant's file name carries the
  fingerprint of everything it depends on. Never reuse a derived file by name
  alone, and raise the stage's version constant when its logic changes.
- **Failures are recorded, never swallowed.** A stage tolerates only the error
  types it names (an unreadable source file) and writes them to
  `<stage>.failures.jsonl`; anything else stops it.
- **A checkpoint is loaded as it was trained.** Evaluation and export build
  the model from the configuration stored in the checkpoint and refuse a
  request that differs.
- **Licence of this project:** BSD-3-Clause (`LICENSE`).
- **Licences.** Read code only from permissively licensed projects; ideas from
  papers are fine. Never copy code from another project without its licence
  allowing it and without attribution in the module doc. nanocodec
  (BSD-3-Clause, same owner) may be reused freely.
- **Tooling:** `tools/ci/check.sh` must pass: `cargo fmt`, `cargo clippy -D
  warnings`, `cargo test`, `ruff format`, `ruff check`, `mypy --strict`, the
  two structure checks and `pytest`.

## Reporting results

Decisions use several metric families plus guards, with an interval over
held-out clips (`docs/DESIGN.md` section 10); never PSNR alone, never a few
favourite clips.
