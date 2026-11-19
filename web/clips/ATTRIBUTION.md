# Sample clip attribution

The demo's sample clips are short excerpts of openly licensed or public-domain
video, taken from the nanocodec demo (<https://cribbinsb.github.io/nanocodec/>)
and redistributed here under their own terms. Only the 1280×720 files are used; the
page reduces them to 640×360 itself to make the upscaler's input.

Changes made to every clip: cut to the segment below, audio removed, scaled with
a Lanczos filter, and encoded with ffmpeg/libx264 as H.264 High, yuv420p, BT.709.
The page then upscales them live; nothing upscaled is stored or redistributed.

## Blender Foundation open movies (CC BY 3.0)

Licence: Creative Commons Attribution 3.0,
<https://creativecommons.org/licenses/by/3.0/>. The Tears of Steel excerpts
were also centre-cropped from 2.4:1 (1920×800) to 16:9 (1422×800) before scaling.
All three were **retimed to 30 fps (1.25x)**: every 24 fps frame plays once at 30 fps,
timestamps only, no re-encoding (`tools/web/retime-clips.py`).

| Files | Work | Author | Source | Segment | CRF 360p / 720p |
|---|---|---|---|---|---|
| `tos-canal-*.mp4` | *Tears of Steel* (2012) | (CC) Blender Foundation, mango.blender.org | <https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip> | 4:18.0–4:30.0 (12 s, 24 fps) | 22 / 24 |
| `tos-bridge-*.mp4` | *Tears of Steel* (2012) | (CC) Blender Foundation, mango.blender.org | <https://download.blender.org/demo/movies/ToS/tears_of_steel_1080p.mov.zip> | 0:26.0–0:39.0 (13 s, 24 fps) | 22 / 24 |
| `bbb-tree-*.mp4` | *Big Buck Bunny* (2008) | (c) copyright 2008, Blender Foundation, peach.blender.org | <https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_720p_h264.mov.zip> | 2:07.0–2:21.0 (14 s, 24 fps) | 22 / 24 |

The Tears of Steel soundtrack is under a different licence (CC BY-ND 3.0); no
audio is included.

## SVT Open Content Video Test Suite 2022 – Natural Complexity (CC BY 4.0)

*SVT Open Content Video Test Suite 2022 – Natural Complexity* by Sveriges
Television AB (SVT), Video Core team, is licensed under Creative Commons
Attribution 4.0 International, <https://creativecommons.org/licenses/by/4.0/>.

- Suite page and licence statement: <https://svt.github.io/en/content/>
- Documentation (v1.3), whose "Intentions, license and restrictions of use"
  section states the CC BY 4.0 licence:
  <https://svt.github.io/docs/SVT_Open_Content_Video_Test_Suite_2022_Natural_Complexity_v1-3-reduced.pdf>
- Masters: `ftp://svtopencontent.svt.se/pub/svt_videotestsuite_natural_complexity/REC709/`
  (SDR Rec. 709 mezzanines, 3840×2160p50, JPEG 2000 lossless, YUV 4:4:4 12-bit)

Additional changes: 3840×2160 downscaled to 1280×720 and 640×360, every second
frame dropped (50 → 25 fps), 4:4:4 12-bit converted to 4:2:0 8-bit, then **retimed to
30 fps (1.2x)**: every frame plays once at 30 fps (timestamps only, `tools/web/retime-clips.py`).

| Files | Sequence | Master file | Segment | CRF 360p / 720p |
|---|---|---|---|---|
| `svt-waterfall-*.mp4` | waterfall (SDR) | `waterfall_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |
| `svt-midnight-sun-*.mp4` | midnight_sun (SDR) | `midnight_sun_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |
| `svt-smoke-sauna-*.mp4` | smoke_sauna (SDR) | `smoke_sauna_JPEG2000_SDR_3840x2160p50_YUV444_12bit_Lossless.mov` | 0:10.0–0:20.0 (10 s) | 25 / 28 |

## Xiph.org derf collection: JCT-VC / Vidyo conferencing sequences (public domain)

The Xiph.org Video Test Media page (derf's collection),
<https://media.xiph.org/video/derf/>, labels each of these sequences
"16:9 | 600 frames | public domain" (checked 2026-11-10; the page gives no
per-file notes for them). They were originally contributed by Vidyo to the
JCT-VC HEVC test set (class E, videoconferencing). Source files: lossless
1280×720p60 YUV 4:2:0 8-bit y4m, `https://media.xiph.org/video/derf/y4m/<file>`.

Additional changes: every second frame dropped (60 → 30 fps); the 720p file is
not rescaled.

| Files | Sequence | Source file | Segment | CRF 360p / 720p |
|---|---|---|---|---|
| `xiph-johnny-*.mp4` | Johnny | `Johnny_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-kristen-sara-*.mp4` | KristenAndSara | `KristenAndSara_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-four-people-*.mp4` | FourPeople | `FourPeople_1280x720_60.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo1-*.mp4` | vidyo1 | `vidyo1_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo3-*.mp4` | vidyo3 | `vidyo3_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
| `xiph-vidyo4-*.mp4` | vidyo4 | `vidyo4_720p_60fps.y4m` | 0:00.0–0:10.0 (10 s, the whole sequence) | 22 / 24 |
