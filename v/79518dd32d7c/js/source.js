// Frame sources. Each yields pictures at the requested size, the size of the
// original; a video of another size is scaled to it. The pictures are
// ImageBitmaps, which stay on the GPU where the browser has them there and
// can be handed to the pipeline's worker without copying. A clip or file is
// played frame by frame (clip-player.js) wherever the browser can do that; a
// camera is live, so it is sampled as it plays.

import { openClipPlayer } from './clip-player.js';

/** A moving synthetic scene: needs no media, so the page always has something to show. */
function drawPattern(context, width, height, time) {
  const gradient = context.createLinearGradient(0, 0, width, height);
  gradient.addColorStop(0, '#203a5c');
  gradient.addColorStop(1, '#c9b27c');
  context.fillStyle = gradient;
  context.fillRect(0, 0, width, height);
  const shift = (time * 20) % width;
  context.strokeStyle = '#f2f2f2';
  context.lineWidth = 1;
  for (let i = 0; i < 24; i += 1) {
    context.beginPath();
    context.moveTo(((i * width) / 12 + shift) % width, 0);
    context.lineTo(((i * width) / 12 + shift) % width - height * 0.4, height);
    context.stroke();
  }
  context.fillStyle = '#101010';
  context.font = `${Math.round(height / 9)}px serif`;
  context.fillText('clearvisionnn 0123456789', width * 0.08 + 6 * Math.sin(time), height * 0.5);
  context.font = `${Math.round(height / 18)}px sans-serif`;
  context.fillText('The quick brown fox jumps over the lazy dog', width * 0.08, height * 0.72 + 4 * Math.cos(time * 0.7));
}

const nextDisplayFrame = () => new Promise((resolve) => { requestAnimationFrame(resolve); });

/** Resolves when `video` presents a frame not yet seen (or on the next display frame where the browser cannot say). */
function nextVideoFrame(video) {
  if (!('requestVideoFrameCallback' in video)) return nextDisplayFrame();
  return new Promise((resolve) => { video.requestVideoFrameCallback(resolve); });
}

/** A playing, looping, muted video; `release` undoes whatever `configure` acquired. */
async function playingVideo(configure, release) {
  const video = document.createElement('video');
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  try {
    configure(video);
    await video.play();
  } catch (error) {
    release();
    throw error;
  }
  return video;
}

/** A playing video element for a camera stream or the file at `url`, with what releases it. */
async function openVideo(kind, url, release) {
  if (kind === 'camera') {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    const stop = () => stream.getTracks().forEach((track) => track.stop());
    return { video: await playingVideo((video) => { video.srcObject = stream; }, stop), release: stop };
  }
  return { video: await playingVideo((video) => { video.src = url; }, release), release };
}

/**
 * A source of the given kind: 'pattern', 'clip' (options.url), 'camera' or 'file' (options.file).
 * `grab()` resolves to the current frame as `{ picture, time }`: a picture the
 * caller owns (an ImageBitmap or a VideoFrame: either can be drawn, sent to a
 * worker and closed) and when it was captured (ms; in a clip, its time there).
 * `nextFrame()` resolves when a new one is due, `pause()` and `resume()`
 * hold and release the source's own clock, and `stop()` releases
 * everything the source holds.
 */
export async function openSource(kind, width, height, options = {}) {
  if (kind === 'pattern') {
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    return {
      async grab() {
        const time = performance.now();
        drawPattern(context, width, height, time / 1000);
        return { picture: canvas.transferToImageBitmap(), time };
      },
      nextFrame: nextDisplayFrame,
      pause() {},
      resume() {},
      stop() {},
    };
  }
  if (kind === 'file' && !options.file) throw new Error('Choose a video file first.');
  const url = kind === 'file' ? URL.createObjectURL(options.file) : options.url;
  const release = () => { if (kind === 'file') URL.revokeObjectURL(url); };
  if (kind !== 'camera') {
    const player = await openClipPlayer(url, width, height, release);
    if (player) return player;
  }
  const { video, release: stopVideo } = await openVideo(kind, url, release);
  return {
    async grab() {
      const scaled = video.videoWidth !== width || video.videoHeight !== height;
      const picture = await createImageBitmap(video, scaled ? { resizeWidth: width, resizeHeight: height } : {});
      return { picture, time: performance.now() };
    },
    nextFrame: () => nextVideoFrame(video),
    pause: () => video.pause(),
    resume: () => video.play(),
    stop() {
      video.pause();
      video.removeAttribute('src');
      video.srcObject = null;
      stopVideo();
    },
  };
}
