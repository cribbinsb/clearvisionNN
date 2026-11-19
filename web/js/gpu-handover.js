// Whether a picture drawn by WebGPU on `device` reaches the rest of the
// browser. On a laptop with two GPUs Chrome draws the page on the integrated
// one; a canvas drawn on the other can fail to cross over, and arrives black.
// One small frame at start-up shows which: a known colour is drawn, handed
// over the way a frame's pictures are, and looked at.

const COLOUR = [0.25, 0.5, 0.75, 1]; // any colour that is not black
const LEAST = 32; // a channel this bright or more has arrived (of 255)

const SHADER = `
@vertex fn corner(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  return vec4<f32>(f32(index & 1u) * 4.0 - 1.0, f32(index >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}
@fragment fn fill() -> @location(0) vec4<f32> { return vec4<f32>(${COLOUR.join(', ')}); }`;

/**
 * Resolves to whether a `width` x `height` WebGPU canvas drawn on `device`
 * arrives as it was drawn. The size is the pictures' own: the fault depends on it.
 */
export async function canvasArrives(device, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto', vertex: { module, entryPoint: 'corner' }, fragment: { module, entryPoint: 'fill', targets: [{ format }] },
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }] });
  pass.setPipeline(pipeline);
  pass.draw(3);
  pass.end();
  device.queue.submit([encoder.finish()]);
  const picture = canvas.transferToImageBitmap();
  await device.queue.onSubmittedWorkDone();
  // Looked at through an ordinary canvas, as the page will draw it.
  const reader = new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true });
  reader.drawImage(picture, 0, 0);
  picture.close();
  const [red, green, blue] = reader.getImageData(width >> 1, height >> 1, 1, 1).data;
  return Math.max(red, green, blue) >= LEAST;
}
