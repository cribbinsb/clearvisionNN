# clearvisionnn-runtime

Everything around the network at run time: the conventional upscale, warping history along the motion, assembling the network's input and blending its output, per stream. It also holds the reference implementation of the 8-bit integer network, so a model in that form runs here without any other inference engine.

This is one of the crates of [clearvisionNN](https://github.com/cribbinsb/clearvisionNN),
fast causal 2x video super-resolution for the browser and the PC. Most users
want the [`clearvisionnn`](https://crates.io/crates/clearvisionnn) crate, which
re-exports the others. In Rust code this crate's library is `cvnn_runtime`.

- Repository: <https://github.com/cribbinsb/clearvisionNN>
- Live demo: <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository). The crates contain no model
weights. The models in the repository are licensed separately, for
non-commercial use (`web/models/LICENSE.md` there).
