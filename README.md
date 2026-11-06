# T4a figure-ground model

Simulation code of a connectome-constrained model of figure-ground discrimination in the fly lobula plate.
T4a and T4b cells of both eyes drive the vCH, LPi15 and H2 cells; every cell is a passive multicompartment cable built
from its FlyWire (FAFB v783) skeleton, and every synapse sits at its measured position.

## Contents

```
model/
  sim.worker.js               the model: stimulus, T4 input model, cables, synapses, time loop, recording
  run.mjs                     runs one simulation under Node and writes the result (gzipped JSON)
  native-cable-solver.cc/.mjs multithreaded C++ solver for the T4 cables (optional; build-native.sh)
  parallel-cable-pool.mjs     worker-thread solver for the T4 cables (used when the native solver is missing)
  parallel-premotion-pool.mjs parallel precomputation of the T4 input model
  opticflow-model-v17.js      imported by sim.worker.js; used only by an inactive code path
  build-native.sh             builds native-cable-solver.node
configs/                      run configurations (stimulus + settings) of the paper's main conditions
cluster/run-array.sh          Slurm job array that runs a list of configurations
extract/                      reduce a result to peaks / traces / release-site histograms
tools/compare-results.mjs     compare two results value by value
```

## Model

A hemicircuit is one vCH and one LPi15 with their T4a and T4b; the two-eye circuit has one hemicircuit per eye
(898 T4a and 1446 T4b in total; the one-eye circuit has 424 T4a and 713 T4b, without H2).

- **T4a, T4b**: five input types computed from the luminance in small receptive fields (photoreceptor, temporal
  filters, rectification): Mi1 and Tm3 (excitatory), Mi9, Mi4 and C3 (inhibitory). Each T4 has three receptive-field
  positions along its preferred direction, taken from its measured input offset.
- **vCH**: excited by T4a and H2; inhibits T4a, T4b, LPi15 and H2.
- **LPi15**: excited by T4b; inhibits vCH, T4a, T4b and H2.
- **H2**: excited by the T4b of one eye; excites the vCH of the other side (the only cell crossing the midline).
- All other measured connections between these cells are included as well (`v2FullConnectivity`).

Synaptic release is computed at every synapse from the presynaptic voltage at its own compartment,
r(V) = clip((gain * max(0, V - threshold))^exponent, 0, 1), and the conductance r * g (per synapse) is added at the
postsynaptic compartment. Each time step computes all synaptic conductances from the voltages of the previous step,
then advances every cable by one implicit (backward-Euler) step, solved exactly with the Hines algorithm.

Parameters of the paper's model (from `configs/`; the code defaults differ):

| | |
|---|---|
| time step / output interval | 1 ms / 10 ms |
| cables | T4: Ra 400 Ohm cm, Rm 8 kOhm cm2; vCH, LPi15: Ra 350 Ohm cm, Rm 17 kOhm cm2; H2: Ra 180 Ohm cm, Rm 17 kOhm cm2; Cm 1 uF/cm2 |
| reversal potentials | leak -65 mV, excitatory -21 mV, inhibitory (vCH and LPi15 outputs) -68 mV |
| release: T4a, T4b, H2 | threshold -59.5 mV, gain 0.06 /mV, exponent 1 |
| release: vCH | threshold -58 mV, gain 0.18 /mV, exponent 1 |
| release: LPi15 | threshold -45 mV, gain 0.10 /mV, exponent 1.4 |
| conductance per synapse | excitatory 0.02 nS; vCH outputs 0.17 nS; LPi15 outputs 0.01 nS |

## Requirements

- Node.js 22 or newer.
- Optional: g++ with C++17 and the Node.js headers for the native solver (`bash model/build-native.sh`), and `pigz`.
- Data, **not included in this repository**:
  - the circuit file (JSON): for each hemicircuit the T4a and T4b with their input contacts, receptive-field
    positions and synapse coordinates, the vCH and LPi15 with their skeletons and synapses, and the H2 cells;
  - one skeleton per T4a and T4b, `<rootId>.csv` with the columns `node_id, x, y, z, parent_id, radius` (nm).

## Running

One simulation (multithreaded when more than one thread is allowed):

```bash
YIFEI_PARALLEL_THREADS=8 node model/run.mjs \
  --circuit circuit.json --skeleton-dir skeletons/ \
  --worker model/sim.worker.js \
  --native-solver model/native-cable-solver.mjs --native-addon model/native-cable-solver.node \
  --parallel-solver model/parallel-cable-pool.mjs --parallel-premotion model/parallel-premotion-pool.mjs \
  --config configs/hc-e680-ap5-both-180.json --output result.json.gz
```

A 10-s run of the two-eye circuit takes a few minutes on 8 cores and writes about 90 MB.
Many runs on a Slurm cluster: put the configs into `$WORK/configs/`, list their names in `$WORK/manifest.txt` and
submit `cluster/run-array.sh` (see the comments at its top).

## Configurations

All use the model parameters above. Stimulus (except the one-eye run): random black and white squares of 5 x 5 deg
(azimuth x elevation); the figure is a 10 x 120 deg bar at azimuth -30 deg with its own square texture.

| config | condition |
|---|---|
| `hc-e680-ap5-both-0` | figure and ground oscillate together (+-10 deg, 0.5 Hz, in phase), 10 s |
| `hc-e680-ap5-both-180` | the same in antiphase (relative motion) |
| `hc-e680-ap5-contra-0` | in phase, ground shown only at positive azimuth |
| `hc-e680-ap5-vchabl-both-0` | in phase, without vCH -> T4 synapses |
| `hc-e680-ap5-lpiabl-both-0` | in phase, without LPi15 outputs |
| `hc-e680-ap5-h2abl-both-0` | in phase, without H2 outputs |
| `hc-e680-r2v2-intact-0` | one-eye circuit: a white 5-deg bar between two 10-deg black bands, on 10-deg black and white stripes; in phase |
| `fga-WF-intact` | the figure's texture moves at -30 deg/s inside a fixed 10-deg window; the ground is still |
| `fga-WG-intact` | the ground moves at -30 deg/s; the figure is still |

`targetRootId` selects the T4a whose detailed traces are recorded; it does not change the model.

## Result

`result.json.gz` contains, among others:

- `time` (s) and, per T4a root id, `pointVoltageByRoot` (axon voltage, mV) and, with `recordAllRegions`,
  `regionVoltageByRoot` (dendrite and soma);
- `sides`: per hemicircuit the vCH and LPi15 voltages and the synaptic conductances between them and the T4a;
- `h2Sides`: per H2 its voltage and its input and output conductances;
- `t4bVoltageByRoot`, `t4bReleaseByRoot`; traces of the target cell (`target...`);
- `releaseSiteHist`: per pathway, histograms of the presynaptic voltage at all release sites;
- `meta`: every parameter used and summary values.

## Notes on the code

`sim.worker.js` reproduces the four worker variants used for the paper runs (two-eye model, the separate vCH -> T4a
reversal option, the fixed-window stimulus option, and the one-eye model); 1-s runs of each setting were checked to
give identical results with `tools/compare-results.mjs`. The short helper names (`n`, `r`, `i`, `a`, ...) come from
the original minified browser build; each helper is described by a comment. Code paths for point-model T4b and an
optic-flow T4b drive are inactive and kept only so that the published runs are reproduced exactly.
