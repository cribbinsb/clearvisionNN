// The tile-group tuner (web/js/int8-schedule.js), run by tools/ci/check.sh
// with `node --test`: probing, keeping the fastest, and when it looks again.

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { createSchedule, groupCandidates } from '../../web/js/int8-schedule.js';

const PROBE_FRAMES = 6; // as the tuner's: frames per candidate, the first a warm-up

/** Run frames until the tuner has settled, timing each by `cost(group)`; returns the groups used. */
function settle(schedule, cost, candidates) {
  const used = [];
  for (let frame = 0; frame < PROBE_FRAMES * candidates.length; frame += 1) {
    used.push(schedule.current());
    schedule.record(cost(schedule.current(), frame));
  }
  return used;
}

test('candidates halve from all tiles down to a few', () => {
  assert.deepEqual(groupCandidates(60), [60, 30, 15, 8, 4, 2]);
  assert.deepEqual(groupCandidates(6), [6, 3, 2]);
  assert.deepEqual(groupCandidates(1), [1]);
});

test('every candidate is tried in turn and the fastest by median is kept', () => {
  const candidates = [15, 8, 4];
  const schedule = createSchedule(candidates);
  const used = settle(schedule, (group) => ({ 15: 5, 8: 3, 4: 9 })[group], candidates);
  assert.deepEqual(used, candidates.flatMap((group) => Array(PROBE_FRAMES).fill(group)));
  assert.equal(schedule.current(), 8);
});

test('a warm-up frame and a stray slow frame do not decide', () => {
  const candidates = [15, 8];
  const schedule = createSchedule(candidates);
  // 15 is the faster but starts with a slow first frame and has one outlier.
  settle(schedule, (group, frame) => (group === 15 ? (frame === 0 || frame === 3 ? 50 : 3) : 4), candidates);
  assert.equal(schedule.current(), 15);
});

test('a lone candidate is never probed', () => {
  const schedule = createSchedule([1]);
  for (let frame = 0; frame < 2000; frame += 1) schedule.record(frame % 7);
  assert.equal(schedule.current(), 1);
});

test('frames slowing down start a new probe, and a marginal rival does not replace the choice', () => {
  const candidates = [15, 8];
  const schedule = createSchedule(candidates);
  settle(schedule, (group) => (group === 15 ? 4 : 5), candidates);
  assert.equal(schedule.current(), 15);
  for (let frame = 0; frame < 50; frame += 1) schedule.record(4);
  assert.equal(schedule.current(), 15, 'steady frames leave the choice alone');
  // Everything gets slower, the rival 2% less so: the tuner looks again
  // within a hundred frames, and does not change over for so little.
  const slowed = (group) => (group === 15 ? 8 : 7.84);
  for (let frame = 0; frame < 100 && schedule.current() === 15; frame += 1) schedule.record(slowed(15));
  assert.equal(schedule.current(), 8, 'a new probe reached the rival');
  for (let frame = 0; frame < PROBE_FRAMES; frame += 1) schedule.record(slowed(8));
  assert.equal(schedule.current(), 15);
  // Long after, it is checked again, and a clearly faster rival takes over.
  for (let frame = 0; frame < 700; frame += 1) schedule.record(schedule.current() === 15 ? 8 : 4);
  assert.equal(schedule.current(), 8);
});
