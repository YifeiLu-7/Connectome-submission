// Node wrapper of the native cable solver (native-cable-solver.node, built from native-cable-solver.cc with
// build-native.sh). run.mjs uses it when the addon is given and more than one thread is requested.
// All T4 cables are packed into flat arrays that the addon reads and writes in place; persistent C++ threads
// solve contiguous ranges of cells with the same implicit Hines step as fe() in sim.worker.js. The addon also
// assembles the visual input conductances (step (2) of the worker's time loop).
// createNativeCableSolver({addonPath, threads}) returns solve(states, dtSec) with prepareInputAssembly() and
// prepareLpi15InputAssembly(); sim.worker.js finds it as globalThis.__YIFEI_SOLVE_T4A_CABLES__.
import { createRequire } from "node:module";

function typed(Type, length) {
  return new Type(length);
}

// Pack the cable states into flat arrays, point the states at them and hand the arrays to the addon.
function buildNativePool(states, addon, threads) {
  const cellCount = states.length;
  const offsets = typed(Int32Array, cellCount + 1);
  const forwardOffsets = typed(Int32Array, cellCount + 1);
  const backwardOffsets = typed(Int32Array, cellCount + 1);
  for (let cell = 0; cell < cellCount; cell += 1) {
    offsets[cell + 1] = offsets[cell] + states[cell].kernel.nodes.length;
    forwardOffsets[cell + 1] = forwardOffsets[cell] + states[cell].kernel.forwardOrder.length;
    backwardOffsets[cell + 1] = backwardOffsets[cell] + states[cell].kernel.backwardOrder.length;
  }
  const total = offsets[cellCount];
  const parents = typed(Int32Array, total);
  const coupling = typed(Float64Array, total);
  const matrixDiagonal = typed(Float64Array, total);
  const capacitance = typed(Float64Array, total);
  const leak = typed(Float64Array, total);
  const forwardOrder = typed(Int32Array, forwardOffsets[cellCount]);
  const backwardOrder = typed(Int32Array, backwardOffsets[cellCount]);
  let voltage = typed(Float64Array, total);
  let next = typed(Float64Array, total);
  const conductance = typed(Float64Array, total);
  const current = typed(Float64Array, total);
  const diagonal = typed(Float64Array, total);
  const rhs = typed(Float64Array, total);
  for (let cell = 0; cell < cellCount; cell += 1) {
    const state = states[cell];
    const kernel = state.kernel;
    const start = offsets[cell];
    for (let local = 0; local < kernel.nodes.length; local += 1) {
      const parent = kernel.parents[local];
      parents[start + local] = parent < 0 ? -1 : start + parent;
    }
    coupling.set(kernel.coupling, start);
    matrixDiagonal.set(kernel.matrixDiagonal, start);
    capacitance.set(kernel.capacitance, start);
    leak.set(kernel.leak, start);
    for (let order = 0; order < kernel.forwardOrder.length; order += 1) {
      forwardOrder[forwardOffsets[cell] + order] = start + kernel.forwardOrder[order];
    }
    for (let order = 0; order < kernel.backwardOrder.length; order += 1) {
      backwardOrder[backwardOffsets[cell] + order] = start + kernel.backwardOrder[order];
    }
    voltage.set(state.voltage, start);
    next.set(state.next, start);
    conductance.set(state.conductance, start);
    current.set(state.current, start);
    diagonal.set(state.diagonal, start);
    rhs.set(state.rhs, start);
  }
  // The states read and write the packed arrays directly (redone after every solve, which swaps voltage
  // and next).
  const adopt = () => {
    for (let cell = 0; cell < cellCount; cell += 1) {
      const start = offsets[cell], end = offsets[cell + 1], state = states[cell];
      state.voltage = voltage.subarray(start, end);
      state.next = next.subarray(start, end);
      state.conductance = conductance.subarray(start, end);
      state.current = current.subarray(start, end);
      state.diagonal = diagonal.subarray(start, end);
      state.rhs = rhs.subarray(start, end);
    }
  };
  adopt();
  addon.configure(
    offsets, parents, coupling, matrixDiagonal, capacitance, leak,
    forwardOffsets, forwardOrder, backwardOffsets, backwardOrder,
    voltage, next, conductance, current, diagonal, rhs, threads,
  );
  const retainedArrays = [
    offsets, parents, coupling, matrixDiagonal, capacitance, leak,
    forwardOffsets, forwardOrder, backwardOffsets, backwardOrder,
    conductance, current, diagonal, rhs,
  ];
  return {
    total,
    retainedArrays,
    solve(dtSec) {
      addon.solve(dtSec);
      const swap = voltage; voltage = next; next = swap;
      adopt();
    },
    // Visual input assembly: the compartments to reset, every input contact, conductance units and reversals.
    configureInputs(config) {
      const kinds = config.kinds;
      const kindCount = kinds.length;
      const touchedOffsets = typed(Int32Array, cellCount + 1);
      const contactOffsets = typed(Int32Array, cellCount * kindCount + 1);
      for (let cell = 0; cell < cellCount; cell += 1) {
        touchedOffsets[cell + 1] = touchedOffsets[cell] + config.touchedCompartments[cell].length;
        for (let kind = 0; kind < kindCount; kind += 1) {
          const mapping = cell * kindCount + kind;
          const contacts = config.mappings[cell].upstreamTargetByType[kinds[kind]] ?? [];
          contactOffsets[mapping + 1] = contactOffsets[mapping] + contacts.length;
        }
      }
      const touchedCompartments = typed(Int32Array, touchedOffsets[cellCount]);
      const contactCompartments = typed(Int32Array, contactOffsets[cellCount * kindCount]);
      for (let cell = 0; cell < cellCount; cell += 1) {
        touchedCompartments.set(config.touchedCompartments[cell], touchedOffsets[cell]);
        for (let kind = 0; kind < kindCount; kind += 1) {
          const mapping = cell * kindCount + kind;
          contactCompartments.set(
            config.mappings[cell].upstreamTargetByType[kinds[kind]] ?? [],
            contactOffsets[mapping],
          );
        }
      }
      const units = Float64Array.from(states, (state) => state.kernel.leakTotalS / 0.5);
      const reversals = Float64Array.from(config.reversals);
      const frames = typed(Float64Array, cellCount * kindCount);
      const perContact = Int32Array.of(config.upstreamScaling === "perContact" ? 1 : 0);
      addon.configureInputs(
        touchedOffsets, touchedCompartments, contactOffsets, contactCompartments,
        units, reversals, frames, perContact,
      );
      retainedArrays.push(
        touchedOffsets, touchedCompartments, contactOffsets, contactCompartments,
        units, reversals, frames, perContact,
      );
      return (activeFrames) => {
        for (let kind = 0; kind < kindCount; kind += 1) {
          frames.set(activeFrames[kind].subarray(0, cellCount), kind * cellCount);
        }
        addon.assembleInputs();
        return true;
      };
    },
    // T4b -> LPi15 assembly for point-model T4b (inactive in the circuit runs).
    configureLpi15Inputs(state, config) {
      const contactOffsets = typed(Int32Array, config.contactCompartments.length + 1);
      for (let cell = 0; cell < config.contactCompartments.length; cell += 1) {
        contactOffsets[cell + 1] = contactOffsets[cell] + config.contactCompartments[cell].length;
      }
      const contactCompartments = typed(Int32Array, contactOffsets[config.contactCompartments.length]);
      for (let cell = 0; cell < config.contactCompartments.length; cell += 1) {
        contactCompartments.set(config.contactCompartments[cell], contactOffsets[cell]);
      }
      const release = typed(Float64Array, config.contactCompartments.length);
      const parameters = Float64Array.of(config.perContactNS, config.reversalMv);
      addon.configureLpiInputs(
        contactOffsets, contactCompartments, release,
        state.conductance, state.current, parameters,
      );
      retainedArrays.push(
        contactOffsets, contactCompartments, release,
        state.conductance, state.current, parameters,
      );
      return (activeRelease) => {
        release.set(activeRelease.subarray(0, release.length));
        addon.assembleLpiInputs();
        return true;
      };
    },
    close() { addon.close(); },
  };
}

export function createNativeCableSolver({ addonPath, threads = 1 } = {}) {
  const requestedThreads = Math.max(1, Math.trunc(Number(threads) || 1));
  const require = createRequire(import.meta.url);
  const addon = require(addonPath);
  let pool = null;
  let closed = false;
  const info = {
    implementation: "napi-persistent-threads-hines-v4",
    requestedThreads,
    threads: requestedThreads,
    enabled: true,
    fallback: "",
    steps: 0,
    parallelInputAssembly: false,
    inputAssemblySteps: 0,
    parallelLpi15InputAssembly: false,
    lpi15InputAssemblySteps: 0,
    totalCompartments: 0,
  };
  function ensurePool(states) {
    if (!pool) {
      pool = buildNativePool(states, addon, requestedThreads);
      info.totalCompartments = pool.total;
    }
    return pool;
  }
  function solve(states, dtSec) {
    if (closed) return false;
    ensurePool(states).solve(dtSec);
    info.steps += 1;
    return true;
  }
  solve.info = info;
  solve.prepareInputAssembly = (states, config) => {
    if (closed) return null;
    const assemble = ensurePool(states).configureInputs(config);
    info.parallelInputAssembly = true;
    return (activeFrames) => {
      if (closed) return false;
      assemble(activeFrames);
      info.inputAssemblySteps += 1;
      return true;
    };
  };
  solve.prepareLpi15InputAssembly = (states, state, config) => {
    if (closed) return null;
    const assemble = ensurePool(states).configureLpi15Inputs(state, config);
    info.parallelLpi15InputAssembly = true;
    return (activeRelease) => {
      if (closed) return false;
      assemble(activeRelease);
      info.lpi15InputAssemblySteps += 1;
      return true;
    };
  };
  solve.close = () => { if (!closed) pool?.close(); closed = true; };
  return solve;
}
