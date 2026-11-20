// A clip or video file played frame by frame: every frame is shown, in order.
// The clip keeps its own pace while the page can keep up, and simply plays
// more slowly when it cannot, where a video element would drop frames. Each
// frame carries its time in the clip, so a codec budgets for the clip's frame
// rate whatever speed it is played at.

import { ClipSource } from './clip-frames.js';

const sleep = (ms) => new Promise((resolve) => { setTimeout(resolve, ms); });

/**
 * A source (see source.js) for the video file at `url`, scaled to `width` x
 * `height`; `release` is called when it stops. Resolves to null where the
 * browser cannot decode the file frame by frame: the caller then plays it in
 * a video element.
 */
export async function openClipPlayer(url, width, height, release) {
  const clip = await ClipSource.open(url, null);
  if (!clip) return null;
  const interval = 1000 / clip.fps;
  let played = 0; // frames handed out; the clip loops, so this passes its length
  let due = performance.now(); // when the next frame may be shown, at the clip's own pace
  return {
    async grab() {
      const frame = await clip.frame(played % clip.frameCount);
      played += 1;
      const time = (played - 1) * interval;
      // A decoded frame is itself a picture that can be drawn, sent to a worker and closed:
      // at the clip's own size it is handed on as it is, with no copy made on this thread.
      if (frame.displayWidth === width && frame.displayHeight === height) return { picture: frame, time };
      try {
        return { picture: await createImageBitmap(frame, { resizeWidth: width, resizeHeight: height }), time };
      } finally {
        frame.close();
      }
    },
    async nextFrame() {
      due += interval;
      const early = due - performance.now();
      if (early > 0) await sleep(early);
      else due = performance.now(); // behind: the clip slows down, it does not catch up
    },
    pause() {},
    resume() {
      due = performance.now();
    },
    stop() {
      clip.close();
      release();
    },
  };
}
