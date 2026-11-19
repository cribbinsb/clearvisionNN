// A conventional 2x upscale of the input for the comparison view, with a
// choice of the standard filters. The network adds its detail to one particular
// bicubic (Keys' cubic with a = -0.75, as PyTorch and OpenCV have it), and
// that is what the page shows as the input by default; people comparing
// upscalers often prefer another member of the family as the baseline, so
// the view can be given one. This is display only: nothing the network sees
// changes. It is a small render pass on a device of its own.

import { requestDevice } from './gpu-device.js';

// The standard resampling filters: `kernel` is [kind, first parameter, second]. A cubic is of
// the family of Mitchell and Netravali, by its (B, C); a Lanczos filter has its number of lobes.
const [CUBIC, LANCZOS, TRIANGLE, BOX] = [0, 1, 2, 3];
export const COMPARISONS = [
  { id: 'base', label: 'Bicubic, a = −0.75 (what the network builds on)', kernel: null },
  { id: 'catmull-rom', label: 'Catmull-Rom (bicubic, a = −0.5)', kernel: [CUBIC, 0, 0.5] },
  { id: 'mitchell', label: 'Mitchell-Netravali (B = C = 1/3)', kernel: [CUBIC, 1 / 3, 1 / 3] },
  { id: 'lanczos3', label: 'Lanczos, 3 lobes', kernel: [LANCZOS, 3, 0] },
  { id: 'lanczos2', label: 'Lanczos, 2 lobes', kernel: [LANCZOS, 2, 0] },
  { id: 'b-spline', label: 'Cubic B-spline (smoothest)', kernel: [CUBIC, 1, 0] },
  { id: 'bilinear', label: 'Bilinear', kernel: [TRIANGLE, 0, 0] },
  { id: 'nearest', label: 'Nearest neighbour', kernel: [BOX, 0, 0] },
];

const SHADER = `
struct Kernel { kind: f32, b: f32, c: f32 }
@group(0) @binding(0) var luma: texture_2d<f32>;
@group(0) @binding(1) var blue: texture_2d<f32>;
@group(0) @binding(2) var red: texture_2d<f32>;
@group(0) @binding(3) var chromaSampler: sampler;
@group(0) @binding(4) var<uniform> kernel: Kernel;

@vertex fn corner(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
  // One triangle that covers the picture.
  return vec4<f32>(f32(index & 1u) * 4.0 - 1.0, f32(index >> 1u) * 4.0 - 1.0, 0.0, 1.0);
}

const PI = 3.14159265;
fn sinc(x: f32) -> f32 { if (abs(x) < 1e-5) { return 1.0; } return sin(PI * x) / (PI * x); }

// The filter's weight at a distance, in input samples.
fn weight(distance: f32) -> f32 {
  let x = abs(distance); let b = kernel.b; let c = kernel.c;
  if (kernel.kind == ${LANCZOS}.0) { if (x < b) { return sinc(x) * sinc(x / b); } return 0.0; }
  if (kernel.kind == ${TRIANGLE}.0) { return max(1.0 - x, 0.0); }
  if (kernel.kind == ${BOX}.0) { if (distance > -0.5 && distance <= 0.5) { return 1.0; } return 0.0; }
  // The cubic of Mitchell and Netravali (1988).
  if (x < 1.0) { return ((12.0 - 9.0 * b - 6.0 * c) * x * x * x + (-18.0 + 12.0 * b + 6.0 * c) * x * x + (6.0 - 2.0 * b)) / 6.0; }
  if (x < 2.0) { return ((-b - 6.0 * c) * x * x * x + (6.0 * b + 30.0 * c) * x * x + (-12.0 * b - 48.0 * c) * x + (8.0 * b + 24.0 * c)) / 6.0; }
  return 0.0;
}

@fragment fn upscale(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
  let size = vec2<i32>(textureDimensions(luma));
  // The output sample's place among the input's, sample centres at whole numbers.
  let place = at.xy / 2.0 - 0.5;
  let first = vec2<i32>(floor(place));
  let part = place - floor(place);
  // Six taps each way reach as far as the widest filter here; the weights are made to sum to one.
  var y = 0.0;
  var total = 0.0;
  for (var j = -2; j <= 3; j++) {
    for (var i = -2; i <= 3; i++) {
      let tap = clamp(first + vec2<i32>(i, j), vec2<i32>(0), size - 1);
      let w = weight(f32(i) - part.x) * weight(f32(j) - part.y);
      y += w * textureLoad(luma, tap, 0).r;
      total += w;
    }
  }
  y = y / total;
  // Colour carries little detail: it is taken smoothly, as the network's own base takes it.
  let spot = at.xy / vec2<f32>(2 * size);
  let u = textureSampleLevel(blue, chromaSampler, spot, 0.0).r - 128.0 / 255.0;
  let v = textureSampleLevel(red, chromaSampler, spot, 0.0).r - 128.0 / 255.0;
  // 8-bit BT.709, limited range, to RGB.
  let l = (y - 16.0 / 255.0) * 255.0 / 219.0;
  let s = 255.0 / 224.0;
  return vec4<f32>(l + 1.5748 * s * v, l - 0.1873 * s * u - 0.4681 * s * v, l + 1.8556 * s * u, 1.0);
}`;

/**
 * An upscaler of `width` x `height` 4:2:0 input frames with `filter`, a `kernel` of COMPARISONS.
 * `render(frame)` resolves to the picture, twice the size, as an ImageBitmap
 * the caller owns; `release()` frees the device.
 */
export async function createComparisonUpscale(width, height, filter) {
  const device = await requestDevice();
  const canvas = new OffscreenCanvas(2 * width, 2 * height);
  const context = canvas.getContext('webgpu');
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({ device, format, alphaMode: 'opaque' });
  const plane = (planeWidth, planeHeight) => device.createTexture({
    size: [planeWidth, planeHeight], format: 'r8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const planes = [plane(width, height), plane(width / 2, height / 2), plane(width / 2, height / 2)];
  const kernel = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(kernel, 0, new Float32Array([...filter, 0]));
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = device.createRenderPipeline({
    layout: 'auto', vertex: { module, entryPoint: 'corner' }, fragment: { module, entryPoint: 'upscale', targets: [{ format }] },
  });
  const group = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      ...planes.map((texture, binding) => ({ binding, resource: texture.createView() })),
      { binding: 3, resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }) },
      { binding: 4, resource: { buffer: kernel } },
    ],
  });
  const pixels = width * height;
  return {
    async render(frame) {
      const offsets = [0, pixels, pixels + pixels / 4];
      planes.forEach((texture, index) => {
        const [planeWidth, planeHeight] = index ? [width / 2, height / 2] : [width, height];
        device.queue.writeTexture({ texture }, frame, { offset: frame.byteOffset + offsets[index], bytesPerRow: planeWidth }, [planeWidth, planeHeight]);
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginRenderPass({ colorAttachments: [{ view: context.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }] });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.draw(3);
      pass.end();
      device.queue.submit([encoder.finish()]);
      const picture = canvas.transferToImageBitmap();
      // Handed over only once the GPU has drawn it: a browser shows a picture taken sooner as black.
      await device.queue.onSubmittedWorkDone();
      return picture;
    },
    release() {
      device.destroy();
    },
  };
}
