// The WebAssembly runtime module, initialised once for the whole page.

import init from '../pkg/cvnn_web.js';

let ready = null;

/** Resolves when the module's classes can be constructed. */
export function runtimeReady() {
  ready ??= init();
  return ready;
}
