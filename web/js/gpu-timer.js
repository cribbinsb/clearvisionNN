// The time passes take on the GPU itself, from timestamp queries. The wall
// clock around a frame also counts the queue behind other work and the hops
// between processes, which on a busy machine are several times the GPU's
// time and hide what a choice of schedule changes. Some GPUs can only take a
// timestamp where a pass begins or ends, so what is to be timed apart has to
// be a pass of its own.

const STAMP_BYTES = 8; // a timestamp: unsigned 64-bit nanoseconds
const STAMPS = 2; // a pass's beginning and end
const MS = 1e-6; // milliseconds per nanosecond

/**
 * A timer for up to `passes` passes per frame on `device`, or null where the
 * device has no timestamp queries. Begin pass `index` with `writes(index)` as
 * its `timestampWrites`, call `resolve(encoder)` after the last pass ends,
 * and `read()` once the encoder is submitted; a frame must be read before the
 * next is resolved.
 */
export function createPassTimer(device, passes = 1) {
  if (!device.features.has('timestamp-query')) return null;
  const count = STAMPS * passes;
  const bytes = count * STAMP_BYTES;
  const querySet = device.createQuerySet({ type: 'timestamp', count });
  const resolved = device.createBuffer({ size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  return {
    writes: (index = 0) => ({ querySet, beginningOfPassWriteIndex: STAMPS * index, endOfPassWriteIndex: STAMPS * index + 1 }),
    resolve(encoder) {
      encoder.resolveQuerySet(querySet, 0, count, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, staging, 0, bytes);
    },
    /**
     * Resolves when the GPU has finished the frame, to each pass's milliseconds in order (null
     * for one that cannot be known; a pass not begun this frame has no meaning), or to null if
     * none can be.
     */
    async read() {
      try {
        await staging.mapAsync(GPUMapMode.READ);
      } catch {
        return null; // the device was lost with the frame on it; its owner finds that out from the device
      }
      const stamps = new BigInt64Array(staging.getMappedRange().slice(0));
      staging.unmap();
      // Browsers may coarsen timestamps, and two of a short pass can then come out of order.
      return Array.from({ length: passes }, (_, index) => {
        const [begin, end] = stamps.subarray(STAMPS * index, STAMPS * index + STAMPS);
        return end > begin ? Number(end - begin) * MS : null;
      });
    },
    destroy() {
      querySet.destroy();
      resolved.destroy();
      staging.destroy();
    },
  };
}
