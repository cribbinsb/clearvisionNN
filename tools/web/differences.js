// How far words read back from the GPU are from the reference's, for the
// browser checks (int8_check.js, runtime_check.js): the count of differing
// words, which must be zero for an exact form of the shaders, and for the
// approximate form the size of the differences, in steps of the 8-bit scale
// and as PSNR. An ES module the checks' Python hands to the page as text.

const PEAK = 255; // of an 8-bit scale, for PSNR
const PICTURE_WORDS = 2; // of the network's output: four 16-bit picture outputs
const PICTURE_STEPS = 32; // picture outputs count 1/32 of an 8-bit level
const COLOUR_BYTES = 3; // of a picture's word; the fourth is alpha, which is constant

const psnr = (squares, count) => (squares ? Number((10 * Math.log10((PEAK * PEAK * count) / squares)).toFixed(2)) : null);

/** A tally of differences between values on one scale; `unit` is the values to an 8-bit step. */
function createTally(unit) {
  let count = 0;
  let differing = 0;
  let largest = 0;
  let squares = 0;
  let worstFrame = Infinity; // PSNR of the frame furthest off
  let frameSquares = 0;
  let frameCount = 0;
  return {
    add(got, expected) {
      const difference = Math.abs(got - expected);
      frameCount += 1;
      if (!difference) return;
      differing += 1;
      largest = Math.max(largest, difference);
      frameSquares += difference * difference;
    },
    endFrame() {
      if (frameSquares) worstFrame = Math.min(worstFrame, psnr(frameSquares / (unit * unit), frameCount));
      count += frameCount;
      squares += frameSquares;
      frameSquares = 0;
      frameCount = 0;
    },
    /** `largestSteps`: in 8-bit steps; `differingShare`: of the values; the PSNRs are null where nothing differs. */
    report: () => ({
      largestSteps: largest / unit,
      differingShare: Number((differing / Math.max(count, 1)).toPrecision(3)),
      psnr: psnr(squares / (unit * unit), count),
      worstFramePsnr: Number.isFinite(worstFrame) ? worstFrame : null,
    }),
  };
}

const signedByte = (word, byte) => (word << (24 - 8 * byte)) >> 24;
const unsignedByte = (word, byte) => (word >>> (8 * byte)) & 255;
const signedHalf = (word, half) => (word << (16 - 16 * half)) >> 16;

/**
 * A running comparison of buffers of one layout, a frame to each `add`:
 * 'signed' (four signed bytes to a word: activations, the network's input),
 * 'picture' (RGBA bytes) or 'output' (the network's, `outputWords` to a
 * pixel: two words of 16-bit picture outputs, then the state's signed bytes).
 */
export function createDifference(layout, outputWords = 1) {
  let differingWords = 0;
  const tallies = layout === 'output' ? { picture: createTally(PICTURE_STEPS), state: createTally(1) } : { values: createTally(1) };
  const compare = {
    signed(got, expected) {
      for (let byte = 0; byte < 4; byte += 1) tallies.values.add(signedByte(got, byte), signedByte(expected, byte));
    },
    picture(got, expected) {
      for (let byte = 0; byte < COLOUR_BYTES; byte += 1) tallies.values.add(unsignedByte(got, byte), unsignedByte(expected, byte));
    },
    output(got, expected, index) {
      if (index % outputWords < PICTURE_WORDS) {
        for (let half = 0; half < 2; half += 1) tallies.picture.add(signedHalf(got, half), signedHalf(expected, half));
      } else {
        for (let byte = 0; byte < 4; byte += 1) tallies.state.add(signedByte(got, byte), signedByte(expected, byte));
      }
    },
  }[layout];
  return {
    add(got, expected) {
      for (let index = 0; index < expected.length; index += 1) {
        differingWords += got[index] !== expected[index];
        compare(got[index], expected[index], index);
      }
      Object.values(tallies).forEach((tally) => tally.endFrame());
    },
    report: () => ({ differingWords, ...Object.fromEntries(Object.entries(tallies).map(([name, tally]) => [name, tally.report()])) }),
  };
}

/** Resolves to the words `buffer` holds on `device`. */
export async function readWords(device, buffer) {
  const staging = device.createBuffer({ size: buffer.size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(buffer, 0, staging, 0, buffer.size);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const words = new Uint32Array(staging.getMappedRange().slice(0));
  staging.destroy();
  return words;
}
