// Worker-thread solver for the T4 cables (used by run.mjs when the native addon cannot be loaded).
// All cables are copied into SharedArrayBuffers; each worker thread solves a contiguous range of cells (balanced
// by compartment count) with the same implicit Hines step as fe() in sim.worker.js. The main thread and the
// workers synchronise with Atomics on a small control array. The cable states then point into the shared
// buffers, so the model keeps reading and writing the same fields.
// createParallelCableSolver({threads}) returns solve(states, dtSec), which sim.worker.js calls through
// globalThis.__YIFEI_SOLVE_T4A_CABLES__; it returns false (and the worker solves serially) when it cannot run.
import { Worker, isMainThread, workerData } from "node:worker_threads";

// Tag of this module's worker threads, and the slots of the shared control array.
const WORKER_KIND = "yifei-parallel-cable-v1";
const GENERATION = 0;
const DONE = 1;
const STOP = 2;
const READY = 3;
const ERROR = 4;
const CONTROL_LENGTH = 5;

function sharedTypedArray(Type, length) {
  return new Type(new SharedArrayBuffer(Type.BYTES_PER_ELEMENT * length));
}

function view(Type, buffer) {
  return new Type(buffer);
}

// One implicit (backward-Euler) step of cells [cellStart, cellEnd), the same arithmetic as fe() in
// sim.worker.js.
function solveCellRange(data, cellStart, cellEnd, voltage, next, dtSec) {
  const {
    offsets,
    parents,
    coupling,
    matrixDiagonal,
    capacitance,
    leak,
    forwardOffsets,
    forwardOrder,
    backwardOffsets,
    backwardOrder,
    conductance,
    current,
    diagonal,
    rhs,
  } = data;

  for (let cellIndex = cellStart; cellIndex < cellEnd; cellIndex += 1) {
    const start = offsets[cellIndex];
    const end = offsets[cellIndex + 1];
    for (let nodeIndex = start; nodeIndex < end; nodeIndex += 1) {
      const capacitanceOverDt = capacitance[nodeIndex] / dtSec;
      diagonal[nodeIndex] = matrixDiagonal[nodeIndex] + capacitanceOverDt + conductance[nodeIndex];
      rhs[nodeIndex] = capacitanceOverDt * voltage[nodeIndex]
        + leak[nodeIndex] * (-65 / 1e3)
        + current[nodeIndex];
    }

    const forwardStart = forwardOffsets[cellIndex];
    const forwardEnd = forwardOffsets[cellIndex + 1];
    for (let orderIndex = forwardStart; orderIndex < forwardEnd; orderIndex += 1) {
      const nodeIndex = forwardOrder[orderIndex];
      const parentIndex = parents[nodeIndex];
      const factor = coupling[nodeIndex] / diagonal[nodeIndex];
      diagonal[parentIndex] -= coupling[nodeIndex] * factor;
      rhs[parentIndex] -= factor * rhs[nodeIndex];
    }

    const backwardStart = backwardOffsets[cellIndex];
    const backwardEnd = backwardOffsets[cellIndex + 1];
    for (let orderIndex = backwardStart; orderIndex < backwardEnd; orderIndex += 1) {
      const nodeIndex = backwardOrder[orderIndex];
      const parentIndex = parents[nodeIndex];
      next[nodeIndex] = parentIndex < 0
        ? rhs[nodeIndex] / diagonal[nodeIndex]
        : (rhs[nodeIndex] - coupling[nodeIndex] * next[parentIndex]) / diagonal[nodeIndex];
    }
  }
}

// Worker thread: wait for a new generation, solve its cells, count itself DONE; stop on STOP.
function workerMain(data) {
  const control = view(Int32Array, data.control);
  try {
    const arrays = {
      offsets: view(Int32Array, data.offsets),
      parents: view(Int32Array, data.parents),
      coupling: view(Float64Array, data.coupling),
      matrixDiagonal: view(Float64Array, data.matrixDiagonal),
      capacitance: view(Float64Array, data.capacitance),
      leak: view(Float64Array, data.leak),
      forwardOffsets: view(Int32Array, data.forwardOffsets),
      forwardOrder: view(Int32Array, data.forwardOrder),
      backwardOffsets: view(Int32Array, data.backwardOffsets),
      backwardOrder: view(Int32Array, data.backwardOrder),
      conductance: view(Float64Array, data.conductance),
      current: view(Float64Array, data.current),
      diagonal: view(Float64Array, data.diagonal),
      rhs: view(Float64Array, data.rhs),
    };
    const dt = view(Float64Array, data.dt);
    let voltage = view(Float64Array, data.voltageA);
    let next = view(Float64Array, data.voltageB);
    let generation = Atomics.load(control, GENERATION);

    Atomics.add(control, READY, 1);
    Atomics.notify(control, READY);

    while (true) {
      Atomics.wait(control, GENERATION, generation);
      const nextGeneration = Atomics.load(control, GENERATION);
      if (Atomics.load(control, STOP)) break;
      if (nextGeneration === generation) continue;
      generation = nextGeneration;

      try {
        solveCellRange(arrays, data.cellStart, data.cellEnd, voltage, next, dt[0]);
        const swap = voltage;
        voltage = next;
        next = swap;
      } catch {
        Atomics.store(control, ERROR, 1);
      }
      Atomics.add(control, DONE, 1);
      Atomics.notify(control, DONE);
    }
  } catch {
    Atomics.store(control, ERROR, 1);
    Atomics.add(control, READY, 1);
    Atomics.notify(control, READY);
  }
}

// This file is also the entry point of the worker threads.
if (!isMainThread && workerData?.kind === WORKER_KIND) {
  workerMain(workerData);
}

// Split the cells into contiguous ranges with about equal numbers of compartments.
function partitionCells(offsets, workerCount) {
  const cellCount = offsets.length - 1;
  const totalCompartments = offsets[cellCount];
  const partitions = [];
  let start = 0;
  for (let workerIndex = 0; workerIndex < workerCount; workerIndex += 1) {
    const workersRemaining = workerCount - workerIndex;
    let end;
    if (workersRemaining === 1) {
      end = cellCount;
    } else {
      const minimumEnd = start + 1;
      const maximumEnd = cellCount - (workersRemaining - 1);
      const target = offsets[start]
        + (totalCompartments - offsets[start]) / workersRemaining;
      end = minimumEnd;
      while (end < maximumEnd && offsets[end] < target) end += 1;
    }
    partitions.push([start, end]);
    start = end;
  }
  return partitions;
}

// Block until control[index] >= expected.
function waitForValue(control, index, expected, timeoutMs, label) {
  const deadline = Date.now() + timeoutMs;
  while (Atomics.load(control, index) < expected) {
    const current = Atomics.load(control, index);
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error(`Timed out waiting for ${label}`);
    Atomics.wait(control, index, current, remaining);
  }
}

// Copy all cable states into shared buffers and start the worker threads.
function buildPool(states, requestedThreads) {
  const cellCount = states.length;
  const workerCount = Math.max(1, Math.min(Math.trunc(requestedThreads), cellCount));
  const offsets = sharedTypedArray(Int32Array, cellCount + 1);
  const forwardOffsets = sharedTypedArray(Int32Array, cellCount + 1);
  const backwardOffsets = sharedTypedArray(Int32Array, cellCount + 1);

  for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
    const state = states[cellIndex];
    const nodeCount = state?.kernel?.nodes?.length ?? 0;
    if (!nodeCount || state.voltage.length !== nodeCount) {
      throw new Error(`Invalid cable state at cell ${cellIndex}`);
    }
    offsets[cellIndex + 1] = offsets[cellIndex] + nodeCount;
    forwardOffsets[cellIndex + 1] = forwardOffsets[cellIndex] + state.kernel.forwardOrder.length;
    backwardOffsets[cellIndex + 1] = backwardOffsets[cellIndex] + state.kernel.backwardOrder.length;
  }

  const totalCompartments = offsets[cellCount];
  const parents = sharedTypedArray(Int32Array, totalCompartments);
  const coupling = sharedTypedArray(Float64Array, totalCompartments);
  const matrixDiagonal = sharedTypedArray(Float64Array, totalCompartments);
  const capacitance = sharedTypedArray(Float64Array, totalCompartments);
  const leak = sharedTypedArray(Float64Array, totalCompartments);
  const forwardOrder = sharedTypedArray(Int32Array, forwardOffsets[cellCount]);
  const backwardOrder = sharedTypedArray(Int32Array, backwardOffsets[cellCount]);
  const voltageA = sharedTypedArray(Float64Array, totalCompartments);
  const voltageB = sharedTypedArray(Float64Array, totalCompartments);
  const conductance = sharedTypedArray(Float64Array, totalCompartments);
  const current = sharedTypedArray(Float64Array, totalCompartments);
  const diagonal = sharedTypedArray(Float64Array, totalCompartments);
  const rhs = sharedTypedArray(Float64Array, totalCompartments);
  const dt = sharedTypedArray(Float64Array, 1);
  const control = sharedTypedArray(Int32Array, CONTROL_LENGTH);

  for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
    const state = states[cellIndex];
    const kernel = state.kernel;
    const start = offsets[cellIndex];
    const forwardStart = forwardOffsets[cellIndex];
    const backwardStart = backwardOffsets[cellIndex];
    for (let localIndex = 0; localIndex < kernel.nodes.length; localIndex += 1) {
      const parentIndex = kernel.parents[localIndex];
      parents[start + localIndex] = parentIndex < 0 ? -1 : start + parentIndex;
    }
    coupling.set(kernel.coupling, start);
    matrixDiagonal.set(kernel.matrixDiagonal, start);
    capacitance.set(kernel.capacitance, start);
    leak.set(kernel.leak, start);
    for (let orderIndex = 0; orderIndex < kernel.forwardOrder.length; orderIndex += 1) {
      forwardOrder[forwardStart + orderIndex] = start + kernel.forwardOrder[orderIndex];
    }
    for (let orderIndex = 0; orderIndex < kernel.backwardOrder.length; orderIndex += 1) {
      backwardOrder[backwardStart + orderIndex] = start + kernel.backwardOrder[orderIndex];
    }
    voltageA.set(state.voltage, start);
    voltageB.set(state.next, start);
    conductance.set(state.conductance, start);
    current.set(state.current, start);
    diagonal.set(state.diagonal, start);
    rhs.set(state.rhs, start);
  }

  const shared = {
    kind: WORKER_KIND,
    control: control.buffer,
    offsets: offsets.buffer,
    parents: parents.buffer,
    coupling: coupling.buffer,
    matrixDiagonal: matrixDiagonal.buffer,
    capacitance: capacitance.buffer,
    leak: leak.buffer,
    forwardOffsets: forwardOffsets.buffer,
    forwardOrder: forwardOrder.buffer,
    backwardOffsets: backwardOffsets.buffer,
    backwardOrder: backwardOrder.buffer,
    voltageA: voltageA.buffer,
    voltageB: voltageB.buffer,
    conductance: conductance.buffer,
    current: current.buffer,
    diagonal: diagonal.buffer,
    rhs: rhs.buffer,
    dt: dt.buffer,
  };
  const partitions = partitionCells(offsets, workerCount);
  const workers = partitions.map(([cellStart, cellEnd]) => {
    const worker = new Worker(new URL(import.meta.url), {
      workerData: { ...shared, cellStart, cellEnd },
    });
    worker.unref();
    return worker;
  });

  try {
    waitForValue(control, READY, workerCount, 60_000, "parallel cable workers");
    if (Atomics.load(control, ERROR)) throw new Error("A parallel cable worker failed to initialize");
  } catch (error) {
    Atomics.store(control, STOP, 1);
    Atomics.add(control, GENERATION, 1);
    Atomics.notify(control, GENERATION);
    for (const worker of workers) void worker.terminate();
    throw error;
  }

  // Adopt SharedArrayBuffer-backed views only after every worker is ready.  All
  // model code continues to read/write the same per-cell fields without copies.
  for (let cellIndex = 0; cellIndex < cellCount; cellIndex += 1) {
    const start = offsets[cellIndex];
    const end = offsets[cellIndex + 1];
    const state = states[cellIndex];
    state.voltage = voltageA.subarray(start, end);
    state.next = voltageB.subarray(start, end);
    state.conductance = conductance.subarray(start, end);
    state.current = current.subarray(start, end);
    state.diagonal = diagonal.subarray(start, end);
    state.rhs = rhs.subarray(start, end);
  }

  return {
    states,
    workers,
    workerCount,
    totalCompartments,
    control,
    dt,
    close() {
      Atomics.store(control, STOP, 1);
      Atomics.add(control, GENERATION, 1);
      Atomics.notify(control, GENERATION);
      for (const worker of workers) void worker.terminate();
    },
    solve(dtSec) {
      dt[0] = dtSec;
      Atomics.store(control, DONE, 0);
      Atomics.store(control, ERROR, 0);
      Atomics.add(control, GENERATION, 1);
      Atomics.notify(control, GENERATION);
      waitForValue(control, DONE, workerCount, 60_000, "parallel cable step");
      if (Atomics.load(control, ERROR)) throw new Error("A parallel cable worker failed during solve");
      for (const state of states) {
        const swap = state.voltage;
        state.voltage = state.next;
        state.next = swap;
      }
    },
  };
}

// The pool is built on the first call, when the cable states exist.
export function createParallelCableSolver({ threads = 1 } = {}) {
  const requestedThreads = Math.max(1, Math.trunc(Number(threads) || 1));
  let pool = null;
  let setupFailed = false;
  let closed = false;
  const info = {
    implementation: "node-worker-threads-shared-hines-v1",
    requestedThreads,
    threads: 1,
    enabled: false,
    fallback: requestedThreads < 2 ? "single-cpu-allocation" : "not-initialized",
    steps: 0,
    totalCompartments: 0,
  };

  function solve(states, dtSec) {
    if (closed || setupFailed || requestedThreads < 2) return false;
    if (!pool) {
      try {
        pool = buildPool(states, requestedThreads);
        info.threads = pool.workerCount;
        info.enabled = pool.workerCount > 1;
        info.fallback = info.enabled ? "" : "single-cable-worker";
        info.totalCompartments = pool.totalCompartments;
      } catch (error) {
        setupFailed = true;
        info.enabled = false;
        info.fallback = error instanceof Error ? error.message : String(error);
        return false;
      }
    }
    if (pool.states !== states) throw new Error("Parallel cable state collection changed during a run");
    pool.solve(dtSec);
    info.steps += 1;
    return true;
  }

  solve.info = info;
  solve.close = () => {
    if (closed) return;
    closed = true;
    pool?.close();
  };
  return solve;
}
