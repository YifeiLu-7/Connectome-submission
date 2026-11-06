// Run one simulation under Node and write the worker's result as gzipped JSON (one-eye or two-eye circuit).
//
//   node run.mjs --circuit circuit.json --worker sim.worker.js --config configs/<run>.json \
//                --skeleton-dir skeletons/ --output result.json.gz \
//                [--native-solver native-cable-solver.mjs --native-addon native-cable-solver.node] \
//                [--parallel-solver parallel-cable-pool.mjs] [--parallel-premotion parallel-premotion-pool.mjs]
//
// The config holds targetRootId, stimulus and settings (see configs/). Every T4a and T4b needs a skeleton
// <skeleton-dir>/<rootId>.csv with the columns node_id, x, y, z, parent_id, radius (nm).
// Threads: YIFEI_PARALLEL_THREADS (default SLURM_CPUS_PER_TASK, else 1) and YIFEI_CABLE_THREADS (cable solver).
// With more than one thread the T4 cables are solved by the native addon (or, if it cannot be loaded, by the
// worker-thread pool) and the input model is precomputed in parallel; with one thread the worker does it all.
import { readFile, writeFile } from "node:fs/promises";
import { gzipSync } from "node:zlib";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { pathToFileURL } from "node:url";

// Command-line options.
function option(name) {
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 ? process.argv[index + 1] : "";
}

function requiredOption(name) {
  const value = option(name);
  if (!value) throw new Error(`Missing --${name}`);
  return path.resolve(value);
}

// Skeleton CSV -> nodes {id, type (1 = root, 3 = other), x, y, z, radius, parent}.
function parseSkeletonCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const columns = lines[0].split(",").map((value) => value.trim());
  const index = Object.fromEntries(
    ["node_id", "x", "y", "z", "parent_id", "radius"].map((name) => [name, columns.indexOf(name)]),
  );
  if (Object.values(index).some((value) => value < 0)) {
    throw new Error("Skeleton CSV is missing a required column");
  }
  return lines.slice(1).map((line) => {
    const fields = line.split(",");
    const parent = Number(fields[index.parent_id]);
    return {
      id: Number(fields[index.node_id]),
      type: parent < 0 ? 1 : 3,
      x: Number(fields[index.x]),
      y: Number(fields[index.y]),
      z: Number(fields[index.z]),
      radius: Math.max(0, Number(fields[index.radius]) || 0),
      parent,
    };
  }).filter((node) => Number.isFinite(node.id));
}

const circuitPath = requiredOption("circuit");
const workerPath = requiredOption("worker");
const parallelSolverPath = option("parallel-solver") ? requiredOption("parallel-solver") : "";
const parallelPremotionPath = option("parallel-premotion") ? requiredOption("parallel-premotion") : "";
const nativeSolverPath = option("native-solver") ? requiredOption("native-solver") : "";
const nativeAddonPath = option("native-addon") ? requiredOption("native-addon") : "";
const configPath = requiredOption("config");
const skeletonDir = requiredOption("skeleton-dir");
const outputPath = requiredOption("output");

const [circuitText, configText] = await Promise.all([
  readFile(circuitPath, "utf8"),
  readFile(configPath, "utf8"),
]);
const circuit = JSON.parse(circuitText);
const config = JSON.parse(configText);
const targetRootId = String(config.targetRootId || "");
if (!targetRootId) throw new Error("The run config has no targetRootId");
// A one-eye circuit keeps cells and t4b at the top level, a two-eye circuit per hemicircuit. The code below
// only needs the list of these blocks; the circuit is passed to the worker as read.
const monocular = !circuit.hemicircuits;
const blocks = monocular
  ? [{
      key: "monocular",
      cells: () => circuit.cells,
      setCells: (v) => { circuit.cells = v; },
      t4b: () => circuit.t4b ?? [],
    }]
  : Object.keys(circuit.hemicircuits).map((key) => ({
      key,
      cells: () => circuit.hemicircuits[key].cells,
      setCells: (v) => { circuit.hemicircuits[key].cells = v; },
      t4b: () => circuit.hemicircuits[key].t4b,
    }));
console.log(`circuit: ${monocular ? "monocular" : "binocular"}, ${blocks.length} hemicircuit(s)`);
const activeRootIds = new Set((config.activeRootIds || []).map(String));
let targetCell = null;
for (const block of blocks) {
  const found = block.cells().find((cell) => String(cell.rootId) === targetRootId);
  if (found) targetCell = found;
}
if (!targetCell) throw new Error(`Target ${targetRootId} is absent from the circuit`);
// activeRootIds (optional): simulate only these T4a (and the target) in the hemicircuits the list names;
// a hemicircuit the list does not name keeps all its T4a.
if (activeRootIds.size) {
  activeRootIds.add(targetRootId);
  for (const block of blocks) {
    const named = block.cells().filter((cell) => activeRootIds.has(String(cell.rootId)));
    if (named.length) block.setCells(named);
    else console.warn(`NOTE: activeRootIds names no cell of hemicircuit ${block.key}; that eye runs whole (${block.cells().length} T4a)`);
  }
}

// Every T4a and T4b must be a cable with its input contacts.
if (config.settings?.t4bModelVariant !== "multicompartment-real-synapses") {
  throw new Error("V0 requires t4bModelVariant=multicompartment-real-synapses");
}

const cableCells = [];
for (const block of blocks) {
  for (const cell of block.cells()) cableCells.push(cell);
  for (const cell of block.t4b()) cableCells.push(cell);
}
const invalid = cableCells.find((cell) => (
  !cell?.premotion || !Array.isArray(cell.upstreamContacts)
));
if (invalid) throw new Error(`Cell ${invalid.rootId} is missing premotion or upstream contacts`);

// Read the skeleton of every T4a and T4b.
const skeletonEntries = await Promise.all(cableCells.map(async (cell) => {
  const rootId = String(cell.rootId);
  const text = await readFile(path.join(skeletonDir, `${rootId}.csv`), "utf8");
  const nodes = parseSkeletonCsv(text);
  if (!nodes.length) throw new Error(`Skeleton ${rootId} has no nodes`);
  return [rootId, nodes];
}));
const t4aSkeletons = Object.fromEntries(skeletonEntries);
const targetSkeleton = t4aSkeletons[targetRootId];

// The worker posts its result through self.postMessage; collect it here.
let result = null;
let simulationError = null;
globalThis.self = {
  postMessage(message) {
    if (message?.type === "result") result = message.result;
    if (message?.type === "error") simulationError = new Error(message.error || "Simulation failed");
  },
};

// Thread counts and the optional parallel solvers, installed as the globals that sim.worker.js looks for.
const slurmCpus = Math.max(1, Math.trunc(Number(process.env.SLURM_CPUS_PER_TASK) || 1));
const requestedParallelThreads = Math.max(
  1,
  Math.trunc(Number(process.env.YIFEI_PARALLEL_THREADS) || slurmCpus),
);
const requestedCableThreads = Math.max(
  1,
  Math.min(
    requestedParallelThreads,
    Math.trunc(Number(process.env.YIFEI_CABLE_THREADS) || requestedParallelThreads),
  ),
);
let parallelCableSolver = null;
let parallelPremotionPrecomputer = null;
if (nativeSolverPath && nativeAddonPath && requestedParallelThreads > 1) {
  try {
    const { createNativeCableSolver } = await import(pathToFileURL(nativeSolverPath).href);
    parallelCableSolver = createNativeCableSolver({
      addonPath: nativeAddonPath,
      threads: requestedCableThreads,
    });
  } catch (error) {
    console.warn(`Native cable solver unavailable: ${error instanceof Error ? error.message : error}`);
  }
}
if (!parallelCableSolver && parallelSolverPath && requestedParallelThreads > 1) {
  const { createParallelCableSolver } = await import(pathToFileURL(parallelSolverPath).href);
  parallelCableSolver = createParallelCableSolver({ threads: requestedCableThreads });
}
if (parallelCableSolver) {
  globalThis.__YIFEI_SOLVE_T4A_CABLES__ = parallelCableSolver;
  globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__ = parallelCableSolver.info;
}
if (parallelPremotionPath && requestedParallelThreads > 1) {
  const { createParallelPremotionPrecomputer } = await import(pathToFileURL(parallelPremotionPath).href);
  parallelPremotionPrecomputer = createParallelPremotionPrecomputer({
    threads: requestedParallelThreads,
    workerPath,
  });
  globalThis.__YIFEI_PRECOMPUTE_PREMOTION__ = parallelPremotionPrecomputer;
  globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__ = parallelPremotionPrecomputer.info;
}

// Load the worker module and run the simulation (synchronously).
try {
  await import(`${pathToFileURL(workerPath).href}?run=${encodeURIComponent(config.runId || "v0")}`);
  const simulationStartedAt = performance.now();
  self.onmessage({
    data: {
      type: config.messageType,
      circuit,
      targetCell,
      targetSkeleton,
      t4aSkeletons,
      stimulus: config.stimulus,
      settings: config.settings,
      probes: config.probes || [],
    },
  });
  if (result?.meta) result.meta.runnerSimulationElapsedMs = performance.now() - simulationStartedAt;
} finally {
  parallelCableSolver?.close();
  delete globalThis.__YIFEI_SOLVE_T4A_CABLES__;
  delete globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__;
  delete globalThis.__YIFEI_PRECOMPUTE_PREMOTION__;
  delete globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__;
}

// Reject a result in which not every cell was a cable, or that ran a different circuit.
if (simulationError) throw simulationError;
if (!result) throw new Error("The simulation worker returned no result");
const expectedT4a = blocks.reduce((n, block) => n + block.cells().length, 0);
const expectedT4b = blocks.reduce((n, block) => n + block.t4b().length, 0);
if (result.meta?.t4aMulticompartmentCount !== expectedT4a
    || result.meta?.premotionPointCellCount !== 0
    || result.meta?.t4bPointCellCount !== 0
    || result.meta?.opticFlowCellCount !== 0
    || result.meta?.premotionDriveRouting !== "all-cable-delayed-frame-per-cell") {
  throw new Error("Rejected result because not every active T4a/T4b used the required cable model");
}
if (result.meta?.binocular !== !monocular || result.meta?.hemicircuitCount !== blocks.length) {
  throw new Error(`Rejected result: expected ${monocular ? "a monocular" : "a binocular"} run over ${blocks.length} hemicircuit(s), worker reported binocular=${result.meta?.binocular} hemicircuitCount=${result.meta?.hemicircuitCount}`);
}

// Run information added to meta.
result.meta = {
  ...(result.meta || {}),
  runsOn: "engaging-slurm",
  engagingOnly: true,
  model: "yifei-multi-v0-binocular",
  sourceModel: "yifei-multi-v7",
  modelRoleRule: "all T4a and T4b on both eyes are multicompartment; target and trace are readout roles only",
  engagingSubmissionRoute: String(config.submissionRoute || "unknown"),
  engagingRunId: String(config.runId || ""),
  slurmJobId: String(process.env.SLURM_JOB_ID || ""),
  slurmCpusPerTask: slurmCpus,
  targetRootId,
  expectedT4a,
  expectedT4b,
  requestedDurationSec: Number(config.stimulus?.durationSec || 0),
  actualDurationSec: Number(result.time?.at(-1) || 0),
};

// Typed arrays -> plain arrays; compress with pigz when it is installed, else with zlib.
const json = JSON.stringify(result, (_key, value) => (
  ArrayBuffer.isView(value) && !(value instanceof DataView) ? Array.from(value) : value
));
const jsonBuffer = Buffer.from(json);
const pigz = spawnSync(
  "/usr/bin/pigz",
  ["-1", "-p", String(Math.min(16, requestedParallelThreads)), "-c"],
  { input: jsonBuffer, maxBuffer: jsonBuffer.length + 1024 * 1024 },
);
const compressed = pigz.status === 0 && pigz.stdout?.length
  ? pigz.stdout
  : gzipSync(jsonBuffer, { level: 1 });
await writeFile(outputPath, compressed);
console.log(`RESULT=${outputPath}`);
console.log(`TARGET=${targetRootId}`);
console.log(`T4A=${expectedT4a} T4B=${expectedT4b}`);
console.log(`TIMING_SIMULATION_MS=${result.meta.runnerSimulationElapsedMs.toFixed(1)}`);
