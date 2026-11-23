# The recipe behind the published models

The published models are recurrent 2x luma networks in four sizes (s, m, l,
xl), each trained in three steps and then exported:

1. **Base training** from scratch on live data, with a pixel loss and a light
   steadiness term, until held-out PSNR stops improving.
2. **Sharpening**: a short fine-tune with a perceptual term and a texture
   term beside the pixel loss, which restore fine texture a pixel loss
   averages away, on still and on moving content.
3. **8-bit conversion**: a short quantisation-aware fine-tune under a
   simulation of the integer runtime (`docs/INTEGER.md`), guided by the
   sharpened float model.

This directory holds exactly that, one job per file:

| File | What it does |
| --- | --- |
| `model.py` | the network and its recurrent pass over a sequence |
| `options.py` | the settings a checkpoint records; reading and writing checkpoints |
| `sources.py` | datasets crossed with degradation chains, and where their files live |
| `data.py` | live loaders, baked pools (reading), augmentation |
| `clips.py` | prepared validation clips kept on disk; the held-out sets |
| `losses.py` | pixel loss, steadiness along motion, perceptual term |
| `texture.py` | the texture term: local amplitude of two bands against the original's |
| `schedule.py` | the plateau rule of base training, the cosine of the fine-tune |
| `train.py` | base training and the sharpening fine-tune |
| `bake.py` | baking pools of training sequences |
| `quantise.py` | the 8-bit conversion |
| `export.py` | writing a checkpoint as a demo model directory |
| `metrics.py`, `vmaf.py`, `score.py` | per-clip scoring on held-out datasets |

## What it expects

Run everything from the repository root with the project's virtualenv active
and a CUDA GPU. The `clearvisionnn` package and its native extension must be
installed (`docs/PROCEDURES.md`).

Training data is a set of datasets, each under two degradation chains:
`camera` (camera output after the ISP) and `ncodec` (video after a real-time
encode and decode). One dataset under one chain is one run configuration,
`configs/<dataset>_<chain>_blend.yaml`, and its data must have been built
first:

```
for d in vcd_flat_s sanity vcd2 vc1k; do for c in camera ncodec; do
  clearvisionnn pipeline configs/${d}_${c}_blend.yaml
done; done
```

The published models used those four datasets (the default of `--datasets`)
and both chains (the default of `--chains`). Scoring uses two further
datasets no model trained on: `configs/xiph_<chain>_blend.yaml` and
`configs/vc1k_eval_<chain>_blend.yaml`.

Everything the recipe writes goes under `recipe/out/` (ignored by git):
`models/`, `pools/`, `held_out/`, `eval/` and `scores.jsonl`. The live loaders
also fill the frame cache inside each dataset's own pipeline work directory.

## The steps, in order

Shown for size `m`; the other sizes differ only in `--size` and the file
names that follow from it (`s`, `l`, `xl`).

```
# 1. Base training, the long step (--resume continues an interrupted run).
python -m recipe.train --size m --resume

# 2. Pools for the two short steps: 4,000 sequences per dataset and chain,
#    about 16 GB in all. Baked once and shared by every size.
python -m recipe.bake

# 3. Sharpening fine-tune: 1,500 steps from the pools.
python -m recipe.train --sharpen recipe/out/models/base-m.pt

# 4. 8-bit conversion: 1,500 steps from the pools.
python -m recipe.quantise recipe/out/models/base-m-sharp.pt

# 5. Export: the 8-bit model and the float one it came from.
python -m recipe.export recipe/out/models/base-m-sharp-int8.pt web/models/recurrent-m-int8 --list
python -m recipe.export recipe/out/models/base-m-sharp.pt web/models/recurrent-m-float --list

# 6. Scoring on the held-out datasets, under each chain.
for d in xiph vc1k_eval; do for c in camera ncodec; do
  python -m recipe.score --config configs/${d}_${c}_blend.yaml \
    recipe/out/models/base-m.pt recipe/out/models/base-m-sharp.pt
done; done
```

### Settings

| Step | Settings |
| --- | --- |
| Base | sizes (channels x blocks) s 24x10, m 32x15, l 40x15, xl 64x17; 16 state channels, 22 input planes (docs/INTEGER.md, section 1: nothing about how a frame was coded is an input); batch 8, sequences of 6 frames, 128x128 input crops; rate 4e-4 after 100 warm-up steps; steadiness weight 0.05; mirror and transpose augmentation; at most 250 clips per dataset; at most 45,000 steps |
| Plateau rule | every 1,000 steps the mean PSNR over 16 validation clips of every dataset and chain is taken; after 4 checks without a 0.01 dB gain the rate is cut to 0.3 of itself; training ends when a stall follows the third cut; the best weights are kept |
| Sharpening | the base settings, with perceptual (LPIPS, VGG) weight 0.2 on 16 random frames of each step and the texture term at weight 1, batch 32, rate 2e-4 on a cosine, 1,500 steps |
| Texture term | on textured places of the original (15x15 mean step between neighbours above 3/255), still or moving, in sequences that met a codec: the difference in local amplitude (root of the 15x15 mean energy) between output and original, in the top octave a half-size input can carry and in the band above it |
| 8-bit conversion | the sharpening losses with steadiness weight 0.4, plus following the float model's output (weight 1) and its frame-to-frame change (weight 4); batch 8, rate 1e-4 on a cosine to zero, 1,500 steps |

### Notes

- Runs are not bit-reproducible: only the weight initialisation is seeded.
  Data order, crops, degradations and augmentation differ from run to run.
- The loaders hold whole decoded clips in memory, about 2.5 GB per worker
  process; `--workers` sets how many there are in all.
- `recipe.score` adds VMAF, MS-SSIM and XPSNR when given a `vmaf` program
  (`--vmaf-binary` and `--vmaf-model`, or the `VMAF_BINARY` and `VMAF_MODEL`
  environment variables). The program must be a libvmaf build with the CUDA
  backend and the XPSNR feature. Without one those three are left out.
- Decisions between models use several of the scored measures together, over
  all held-out clips (`docs/DESIGN.md` section 10): never PSNR alone.
- The published checkpoints load with this code (`recipe.options` reads their
  settings and ignores keys it does not use). When exporting one, pass
  `--step` to record its training steps, which those files do not carry.
