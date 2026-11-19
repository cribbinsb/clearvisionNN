# clearvisionnn-models

The released 8-bit models of [clearvisionNN](https://github.com/cribbinsb/clearvisionNN),
fast causal 2x video super-resolution, built into the program that uses them:
five sizes, N to XL, each as the words of its integer network and its contract.

```rust
use cvnn_models::Size;
use cvnn_runtime::integer::IntegerModel;

let model = IntegerModel::from_words(&Size::M.words())?;
```

Most users want the [`clearvisionnn`](https://crates.io/crates/clearvisionnn) crate
with its `models` feature, which re-exports this crate as `clearvisionnn::models`.

## Licence: non-commercial

**These weights are under CC BY-NC 4.0**, not the BSD-3-Clause licence of the
project's code. They were trained on video that includes material released
for non-commercial use only (among it Meta's VC-1K set and the UVG
sequences). You may use and share them for non-commercial purposes, with
attribution. Weights trained on permissively licensed video alone can be made
with the recipe in the repository.

- Repository and training recipe: <https://github.com/cribbinsb/clearvisionNN>
- Live demo: <https://cribbinsb.github.io/clearvisionNN/>
