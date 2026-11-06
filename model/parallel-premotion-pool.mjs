// Parallel precomputation of the input model (photoreceptors -> Mi1, Tm3, Mi9, Mi4 and C3 conductances) for the
// whole run. A cell's input model does not depend on the circuit, so the cells are split into equal slices and
// each worker thread loads sim.worker.js and runs its 'precompute-premotion-chunk' message on one slice. The
// conductances of every cell at every step go into SharedArrayBuffers (5 x (steps + 1) x cells values).
// createParallelPremotionPrecomputer({threads, workerPath}) returns precompute(...), which sim.worker.js calls
// through globalThis.__YIFEI_PRECOMPUTE_PREMOTION__; it returns null (and the worker runs the input model itself)
// when it cannot run.
import { Worker, isMainThread, workerData } from "node:worker_threads";
import { pathToFileURL } from "node:url";

// Tag of this module's worker threads, and the slots of the shared control array.
const WORKER_KIND = "yifei-parallel-premotion-v1";
const DONE = 0;
const ERROR = 1;

function waitForWorkers(control, workerCount, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Atomics.load(control, DONE) < workerCount) {
    const current = Atomics.load(control, DONE);
    const remaining = deadline - Date.now();
    if (remaining <= 0) throw new Error("Timed out waiting for parallel premotion workers");
    Atomics.wait(control, DONE, current, remaining);
  }
}

// Worker thread: load the simulation worker and run its precompute message on one slice of cells.
async function runWorker(data) {
  const control = new Int32Array(data.control);
  let failure = "";
  try {
    globalThis.self = {
      postMessage(message) {
        if (message?.type === "error") failure = String(message.error || "Premotion worker failed");
      },
    };
    await import(`${pathToFileURL(data.workerPath).href}?premotion=${data.workerIndex}-${Date.now()}`);
    self.onmessage({ data: data.payload });
    if (failure) throw new Error(failure);
  } catch (error) {
    Atomics.store(control, ERROR, 1);
  } finally {
    Atomics.add(control, DONE, 1);
    Atomics.notify(control, DONE);
  }
}

// This file is also the entry point of the worker threads.
if (!isMainThread && workerData?.kind === WORKER_KIND) {
  await runWorker(workerData);
}

function sharedFloat64(length) {
  return new Float64Array(new SharedArrayBuffer(Float64Array.BYTES_PER_ELEMENT * length));
}

// Equal slices of cells.
function partitions(cellCount, workerCount) {
  const result = [];
  for (let workerIndex = 0; workerIndex < workerCount; workerIndex += 1) {
    const start = Math.floor(cellCount * workerIndex / workerCount);
    const end = Math.floor(cellCount * (workerIndex + 1) / workerCount);
    result.push([start, end]);
  }
  return result;
}

export function createParallelPremotionPrecomputer({ threads = 1, workerPath } = {}) {
  const requestedThreads = Math.max(1, Math.trunc(Number(threads) || 1));
  const resolvedWorkerPath = String(workerPath || "");
  const info = {
    implementation: "node-worker-threads-shared-premotion-v1",
    requestedThreads,
    threads: 1,
    enabled: false,
    fallback: requestedThreads < 2 ? "single-cpu-allocation" : "not-run",
    cells: 0,
    steps: 0,
    sharedBytes: 0,
    elapsedMs: 0,
  };

  // Called once per run by sim.worker.js; returns the shared arrays, or null.
  function precompute({
    premotionCells,
    stimulus,
    dtMs,
    stepCount,
    premotionOptions,
    targetIndex,
  }) {
    const started = performance.now();
    const cellCount = premotionCells.length;
    const workerCount = Math.max(1, Math.min(requestedThreads, cellCount));
    if (workerCount < 2 || !resolvedWorkerPath) return null;

    const frameLength = (stepCount + 1) * cellCount;
    const conductance = Array.from({ length: 5 }, () => sharedFloat64(frameLength));
    const targetActivity = Array.from({ length: 5 }, () => sharedFloat64(stepCount + 1));
    const targetAccepted = Array.from({ length: 3 }, () => sharedFloat64(stepCount + 1));
    const targetMi9Off = sharedFloat64(stepCount + 1);
    const targetMi9On = sharedFloat64(stepCount + 1);
    const targetMi9Dark = sharedFloat64(stepCount + 1);
    const control = new Int32Array(new SharedArrayBuffer(Int32Array.BYTES_PER_ELEMENT * 2));
    const shared = {
      conductance: conductance.map((array) => array.buffer),
      targetActivity: targetActivity.map((array) => array.buffer),
      targetAccepted: targetAccepted.map((array) => array.buffer),
      targetMi9Off: targetMi9Off.buffer,
      targetMi9On: targetMi9On.buffer,
      targetMi9Dark: targetMi9Dark.buffer,
    };
    const workers = partitions(cellCount, workerCount).map(([cellStart, cellEnd], workerIndex) => {
      const cells = premotionCells.slice(cellStart, cellEnd).map((cell) => ({
        premotion: cell.premotion,
      }));
      const worker = new Worker(new URL(import.meta.url), {
        workerData: {
          kind: WORKER_KIND,
          workerIndex,
          workerPath: resolvedWorkerPath,
          control: control.buffer,
          payload: {
            type: "precompute-premotion-chunk",
            cells,
            cellStart,
            cellCount,
            targetIndex,
            stimulus,
            dtMs,
            stepCount,
            premotionOptions,
            shared,
          },
        },
      });
      worker.unref();
      return worker;
    });

    try {
      waitForWorkers(control, workerCount, 30 * 60_000);
      if (Atomics.load(control, ERROR)) throw new Error("A parallel premotion worker failed");
      info.threads = workerCount;
      info.enabled = true;
      info.fallback = "";
      info.cells = cellCount;
      info.steps = stepCount;
      info.sharedBytes = conductance.reduce((sum, array) => sum + array.byteLength, 0)
        + targetActivity.reduce((sum, array) => sum + array.byteLength, 0)
        + targetAccepted.reduce((sum, array) => sum + array.byteLength, 0)
        + targetMi9Off.byteLength + targetMi9On.byteLength + targetMi9Dark.byteLength;
      info.elapsedMs = performance.now() - started;
      return {
        conductance,
        targetActivity,
        targetAccepted,
        targetMi9Off,
        targetMi9On,
        targetMi9Dark,
      };
    } catch (error) {
      info.enabled = false;
      info.fallback = error instanceof Error ? error.message : String(error);
      info.elapsedMs = performance.now() - started;
      return null;
    } finally {
      for (const worker of workers) void worker.terminate();
    }
  }

  precompute.info = info;
  return precompute;
}
