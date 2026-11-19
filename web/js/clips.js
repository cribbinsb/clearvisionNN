// Bundled sample clips: openly licensed or public-domain excerpts at 1280x720,
// lightly H.264-compressed. They play the part of the original; the page
// makes the low-resolution input from them.
// Full attribution and changes: clips/ATTRIBUTION.md.

const BLENDER = '© Blender Foundation · CC BY 3.0';
const SVT = '© SVT (Sveriges Television) · CC BY 4.0';
const XIPH = 'Public domain · Xiph.org derf collection';

export const CLIP_WIDTH = 1280;
export const CLIP_HEIGHT = 720;

export const CLIPS = [
  { id: 'xiph-johnny', title: 'Talking head', credit: XIPH },
  { id: 'xiph-kristen-sara', title: 'Video call · two people', credit: XIPH },
  { id: 'xiph-four-people', title: 'Meeting · four people', credit: XIPH },
  { id: 'xiph-vidyo1', title: 'Meeting table · three people', credit: XIPH },
  { id: 'xiph-vidyo4', title: 'Webcam · talking with hands', credit: XIPH },
  { id: 'xiph-vidyo3', title: 'Whiteboard · presenter walking', credit: XIPH },
  { id: 'tos-bridge', title: 'Tears of Steel · bridge talk', credit: BLENDER },
  { id: 'tos-canal', title: 'Tears of Steel · canal walk', credit: BLENDER },
  { id: 'bbb-tree', title: 'Big Buck Bunny · tree', credit: BLENDER },
  { id: 'svt-waterfall', title: 'Waterfall · fast detail', credit: SVT },
  { id: 'svt-midnight-sun', title: 'Midnight sun · wind and glitter', credit: SVT },
  { id: 'svt-smoke-sauna', title: 'Smoke sauna · smoke and fire', credit: SVT },
];

/** The URL of a clip's file. */
export function clipUrl(id) {
  return new URL(`../clips/${id}-720p.mp4`, import.meta.url).href;
}
