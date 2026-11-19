// Reading a decoded WebCodecs frame as the packed planes the runtime takes.

/** The decoded picture as packed 8-bit 4:2:0 (Y, then U, then V). */
export async function toI420(frame, width, height) {
  const luma = width * height;
  const chroma = luma / 4;
  const packed = new Uint8Array(luma + 2 * chroma);
  if (frame.format === 'I420') {
    await frame.copyTo(packed, { layout: [
      { offset: 0, stride: width },
      { offset: luma, stride: width / 2 },
      { offset: luma + chroma, stride: width / 2 },
    ] });
    return packed;
  }
  if (frame.format !== 'NV12') throw new Error(`The decoder produced ${frame.format} frames, which this page cannot read.`);
  // Hardware decoders interleave the two chroma planes.
  await frame.copyTo(packed, { layout: [{ offset: 0, stride: width }, { offset: luma, stride: width }] });
  const planar = packed.slice();
  for (let i = 0; i < chroma; i += 1) {
    planar[luma + i] = packed[luma + 2 * i];
    planar[luma + chroma + i] = packed[luma + 2 * i + 1];
  }
  return planar;
}
