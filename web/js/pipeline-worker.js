// The worker the page's pipeline lives in: the sender, the motion search and
// the GPU work all happen here, so the page's own thread only fetches frames
// and shows results. remote-pipeline.js is the page's side of this protocol.

import { createPipeline } from './frame-pipeline.js';

/**
 * Answer `scope`'s messages: `open` a pipeline, pass it each `frame`, `release` it.
 */
function serve(scope) {
  let pipeline = null;
  const handlers = {
    async open({ model, backendId, width, height, degrade, tuning, comparison }) {
      pipeline = await createPipeline(model, backendId, width, height, degrade, tuning, comparison);
    },
    frame: ({ picture, time }) => pipeline.process(picture, time),
    async release() {
      const released = pipeline;
      pipeline = null;
      await released?.release();
    },
  };
  scope.onmessage = async ({ data }) => {
    try {
      const value = await handlers[data.type](data);
      // A frame's pictures move to the page; nothing here uses them again.
      const pictures = value ? [value.original, value.input, value.output] : [];
      scope.postMessage({ id: data.id, value }, pictures);
    } catch (error) {
      scope.postMessage({ id: data.id, error: error?.message ?? String(error) });
    }
  };
}

serve(self);
