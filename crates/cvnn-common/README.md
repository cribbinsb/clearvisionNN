# clearvisionnn-common

The types the other crates share: the block grid, a frame's features as the pre-process reports them, and the error type.

This is one of the crates of [clearvisionNN](https://github.com/cribbinsb/clearvisionNN),
fast causal 2x video super-resolution for the browser and the PC. Most users
want the [`clearvisionnn`](https://crates.io/crates/clearvisionnn) crate, which
re-exports the others. In Rust code this crate's library is `cvnn_common`.

- Repository: <https://github.com/cribbinsb/clearvisionNN>
- Live demo: <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository). The crates contain no model
weights. The models in the repository are licensed separately, for
non-commercial use (`web/models/LICENSE.md` there).
