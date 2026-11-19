# clearvisionnn-web

The bindings the demo page uses: the runtime session, the motion search and the simulated camera as a WebAssembly module (built with wasm-bindgen).

This is one of the crates of [clearvisionNN](https://github.com/cribbinsb/clearvisionNN),
fast causal 2x video super-resolution for the browser and the PC. Most users
want the [`clearvisionnn`](https://crates.io/crates/clearvisionnn) crate, which
re-exports the others. In Rust code this crate's library is `cvnn_web`.

- Repository: <https://github.com/cribbinsb/clearvisionNN>
- Live demo: <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository). The crates contain no model
weights. The models in the repository are licensed separately, for
non-commercial use (`web/models/LICENSE.md` there).
