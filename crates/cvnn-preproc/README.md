# clearvisionnn-preproc

The pre-process that runs in front of the network: a block motion search against the previous frame, scene-cut detection and the side planes, built on nanocodec's encoder analysis.

This is one of the crates of [clearvisionNN](https://github.com/cribbinsb/clearvisionNN),
fast causal 2x video super-resolution for the browser and the PC. Most users
want the [`clearvisionnn`](https://crates.io/crates/clearvisionnn) crate, which
re-exports the others. In Rust code this crate's library is `cvnn_preproc`.

- Repository: <https://github.com/cribbinsb/clearvisionNN>
- Live demo: <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository). The crates contain no model
weights. The models in the repository are licensed separately, for
non-commercial use (`web/models/LICENSE.md` there).
