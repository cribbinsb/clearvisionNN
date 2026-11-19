// The time a pass takes on the GPU itself, from timestamp queries. The wall
// clock around a frame also counts the queue behind other work and the hops
// between processes, which on a busy machine are several times the GPU's
// time and hide what a choice of schedule changes.

const STAMP_BYTES = 8; // a timestamp: unsigned 64-bit nanoseconds
const STAMPS = 2; // the pass's beginning and end
const MS = 1e-6; // milliseconds per nanosecond

/**
 * A timer for one pass per frame on `device`, or null where the device has no
 * timestamp queries. Begin the pass with `writes` as its `timestampWrites`,
 * call `resolve(encoder)` after the pass ends, and `read()` once the encoder
 * is submitted; a frame must be read before the next is resolved.
 */
export function createPassTimer(device) {
  if (!device.features.has('timestamp-query')) return null;
  const bytes = STAMPS * STAMP_BYTES;
  const querySet = device.createQuerySet({ type: 'timestamp', count: STAMPS });
  const resolved = device.createBuffer({ size: bytes, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  return {
    writes: { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
    resolve(encoder) {
      encoder.resolveQuerySet(querySet, 0, STAMPS, resolved, 0);
      encoder.copyBufferToBuffer(resolved, 0, staging, 0, bytes);
    },
    /** Resolves when the GPU has finished the frame, to the pass's milliseconds; null if they cannot be known. */
    async read() {
      try {
        await staging.mapAsync(GPUMapMode.READ);
      } catch {
        return null; // the device was lost with the frame on it; its owner finds that out from the device
      }
      const [begin, end] = new BigInt64Array(staging.getMappedRange());
      staging.unmap();
      // Browsers may coarsen timestamps, and two of a short pass can then come out of order.
      return end > begin ? Number(end - begin) * MS : null;
    },
    destroy() {
      querySet.destroy();
      resolved.destroy();
      staging.destroy();
    },
  };
}
