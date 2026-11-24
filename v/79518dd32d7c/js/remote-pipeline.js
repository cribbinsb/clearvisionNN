// The page's side of the pipeline worker (pipeline-worker.js): the same
// pipeline as frame-pipeline.js makes, reached through messages, so that none of
// its work runs on the page's own thread.

/**
 * A host for pipelines, one at a time, in a worker that lives as long as the
 * page. `createPipeline` takes what frame-pipeline.js's does and resolves to a
 * pipeline whose `process(picture, time)` takes an ImageBitmap of the original
 * (which it consumes) and when it was captured and resolves to that pipeline's result with a `close()`
 * that frees its pictures, or to null.
 */
export function createPipelineHost() {
  const worker = new Worker(new URL('./pipeline-worker.js', import.meta.url), { type: 'module' });
  const waiting = new Map(); // calls not yet answered, by id
  let nextId = 0;
  worker.onmessage = ({ data }) => {
    const { resolve, reject } = waiting.get(data.id);
    waiting.delete(data.id);
    if (data.error === undefined) resolve(data.value);
    else reject(new Error(data.error));
  };
  // The worker itself failed (it could not load, say): nothing waiting will be answered.
  worker.onerror = (event) => {
    for (const { reject } of waiting.values()) reject(new Error(event.message || 'The pipeline worker failed.'));
    waiting.clear();
  };

  function call(type, payload = {}, transfer = []) {
    return new Promise((resolve, reject) => {
      waiting.set(nextId, { resolve, reject });
      worker.postMessage({ id: nextId, type, ...payload }, transfer);
      nextId += 1;
    });
  }

  return {
    async createPipeline(model, backendId, width, height, degrade, tuning, comparison) {
      await call('open', { model, backendId, width, height, degrade, tuning, comparison });
      return {
        async process(picture, time) {
          const result = await call('frame', { picture, time }, [picture]);
          if (!result) return null;
          const close = () => [result.original, result.input, result.output].forEach((each) => each.close());
          return { ...result, close };
        },
        release: () => call('release'),
      };
    },
  };
}
