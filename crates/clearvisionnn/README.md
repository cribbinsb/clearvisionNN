# clearvisionNN

Fast, causal 2x video super-resolution for camera video and for video after a
real-time encode and decode. This crate gathers the project's Rust crates
under one name:

- `clearvisionnn::common`: shared types;
- `clearvisionnn::preproc`: the motion search and side planes that run in front of the network;
- `clearvisionnn::runtime`: everything around the network for a stream (upscale,
  warp, blend, colour), and the reference implementation of the 8-bit integer
  network, so a model in that form runs without any other inference engine.

- Repository, training recipe and models: <https://github.com/cribbinsb/clearvisionNN>
- Live demo (WebGPU): <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository). The crates contain no model
weights. The models in the repository are licensed separately, for
non-commercial use (`web/models/LICENSE.md` there).
