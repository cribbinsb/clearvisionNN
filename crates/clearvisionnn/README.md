# clearvisionNN

Fast, causal 2x video super-resolution for camera video and for video after a
real-time encode and decode. This crate gathers the project's Rust crates
under one name:

- `clearvisionnn::common`: shared types;
- `clearvisionnn::preproc`: the motion search and side planes that run in front of the network;
- `clearvisionnn::runtime`: everything around the network for a stream (upscale,
  warp, blend, colour), and the reference implementation of the 8-bit integer
  network, so a model in that form runs without any other inference engine.

With the `models` feature, `clearvisionnn::models` holds the released 8-bit
models, N to XL, ready to load:

```toml
clearvisionnn = { version = "0.1", features = ["models"] }
```

- Repository, training recipe and models: <https://github.com/cribbinsb/clearvisionNN>
- Live demo (WebGPU): <https://cribbinsb.github.io/clearvisionNN/>

## Licence

BSD-3-Clause (`LICENSE` in the repository) for the code. The `models`
feature is off by default because the weights it adds
(`clearvisionnn-models`) are under CC BY-NC 4.0, for non-commercial use.
