// Simulation worker of the T4a figure-ground model (fly lobula plate, both eyes).
//
// Every cell is a passive cable built from its FlyWire (FAFB v783) skeleton, and every synapse sits at its
// measured position: its release is computed from the presynaptic voltage at the synapse's own compartment, and
// its conductance is delivered to the matching compartment of the postsynaptic cell.
//
// A hemicircuit is one vCH and one LPi15 with their T4a and T4b; a two-eye circuit has one hemicircuit per eye.
// Main connections:
//   T4a, T4b  driven by five input types computed from the luminance in small receptive fields:
//             Mi1 and Tm3 (excitatory), Mi9, Mi4 and C3 (inhibitory)
//   vCH       excited by T4a and H2; inhibits T4a, T4b, LPi15 and H2
//   LPi15     excited by T4b; inhibits vCH, T4a, T4b and H2
//   H2        excited by the T4b of one eye; excites the vCH of the other side (the only cell crossing the midline)
// The other measured connections between these cells are added with settings.v2FullConnectivity (see V()).
//
// Synaptic release: r(V) = clip((gain * max(0, V - threshold))^exponent, 0, 1), one curve per presynaptic cell
// type; conductance = r * (conductance per synapse). Excitatory synapses reverse at -21 mV, inhibitory ones at
// settings.vchToT4ReversalMv (vCH outputs) and settings.lpi15ReversalMv (LPi15 outputs); leak reversal -65 mV.
//
// One time step (settings.dtMs): (1) advance the input model; (2) put the visual input conductances on every T4;
// (3) compute every synaptic conductance from the voltages at the end of the previous step; (4) advance every
// cable by one implicit (backward-Euler) step, solved exactly with the Hines algorithm; (5) record the outputs.
//
// Units: settings and results use mV, nS, ms and degrees; cable states store volts, siemens and amperes.
//
// Messages (a Web Worker in the browser; run-v4.mjs and run-v2mono.mjs load it under Node):
//   {type: 'precompute-premotion-chunk', ...}  input model for a slice of cells (parallel-premotion-pool.mjs)
//   {type: 'prepare-fit-traces', ...}          input-model traces of one cell (not used by the circuit runs)
//   any other message                          full simulation, V(payload)
//
// The short helper names (n, r, i, a, ...) come from the original minified browser build; each helper has a
// one-line description. Branches marked "inactive" (point-model T4b, optic-flow T4b drive) are never taken by
// the circuit runs and are kept only so that this file reproduces the published runs exactly.
// Optic-flow model: used only by the inactive optic-flow drive of point-model T4b.
import {
  createOpticFlowDriveCache,
  createOpticFlowTemporalFilter,
  filterOpticFlowDrive,
  opticFlowDriveBatch,
  precomputeOpticFlowCells,
} from "./opticflow-model-v17.js";

(function() {
  // Default cable parameters (Ra in Ohm cm, Rm in Ohm cm2, Cm in F/cm2): e = T4a and T4b, t = vCH, LPi15 and H2.
  // Each skeleton is reduced to at most maxCompartments compartments.
  let e = { raOhmCm: 400, rmOhmCm2: 8e3, cmFPerCm2: 1e-6, maxCompartments: 2500 }, t = { raOhmCm: 350, rmOhmCm2: 17e3, cmFPerCm2: 1e-6, maxCompartments: 2500 };
  // n(nodes, maxCount): reduce a skeleton to at most maxCount nodes. Keeps the roots, every k-th node and the last
  // node; each kept node is re-attached to its nearest kept ancestor.
  function n(e2, t2) {
    if (t2 <= 0 || e2.length <= t2) return e2.map((e3) => ({ ...e3 }));
    let n2 = Math.max(2, Math.ceil(e2.length / t2)), r2 = new Set(e2.filter((e3) => e3.parent < 0).map((e3) => e3.id)), i2 = new Set(r2);
    for (let t3 = 0; t3 < e2.length; t3 += n2) i2.add(e2[t3].id);
    i2.add(e2[e2.length - 1].id);
    let a2 = new Map(e2.map((e3) => [e3.id, e3.parent])), o2 = e2.filter((e3) => i2.has(e3.id)), s2 = new Map(o2.map((e3, t3) => [e3.id, t3 + 1]));
    return o2.map((e3) => {
      let t3 = e3.parent;
      for (; t3 >= 0 && !i2.has(t3); ) t3 = a2.get(t3) ?? -1;
      return { ...e3, id: s2.get(e3.id), parent: s2.get(t3) ?? -1 };
    });
  }
  // r(nodes, cableOptions): cable kernel of one cell (coordinates and radii in nm). Per compartment: leak
  // conductance (membrane area / Rm), capacitance (area * Cm) and axial coupling to the parent
  // (pi * mean radius^2 / (Ra * length)); also the elimination orders of the Hines solver (leaves first / root first).
  function r(e2, t2) {
    let r2 = n(e2, t2.maxCompartments);
    if (!r2.length) throw Error(`Cannot build a cable without SWC nodes.`);
    let i2 = 1e-7, a2 = r2.length, o2 = new Map(r2.map((e3, t3) => [e3.id, t3])), s2 = new Int32Array(a2).fill(-1), c2 = new Float64Array(a2), l2 = new Float64Array(a2), u2 = new Float64Array(a2), d2 = new Float64Array(a2), f2 = new Float64Array(a2), p2 = 0;
    for (let e3 = 0; e3 < a2; e3 += 1) {
      let n2 = r2[e3], a3 = o2.get(n2.parent) ?? -1;
      s2[e3] = a3, a3 < 0 && (p2 = e3);
      let l3 = Math.max(n2.radius * i2, 2e-7), m3;
      if (a3 >= 0) {
        let o3 = r2[a3], s3 = (n2.x - o3.x) * i2, u3 = (n2.y - o3.y) * i2, d3 = (n2.z - o3.z) * i2;
        m3 = Math.max(Math.hypot(s3, u3, d3), l3);
        let p3 = (l3 + Math.max(o3.radius * i2, 2e-7)) / 2, h4 = Math.PI * p3 * p3 / (t2.raOhmCm * m3);
        f2[e3] = h4, c2[e3] = -h4;
      } else m3 = Math.max(l3 * 2, 1e-6);
      let h3 = Math.max(2 * Math.PI * l3 * m3, 1e-12);
      d2[e3] = h3 / t2.rmOhmCm2, u2[e3] = h3 * t2.cmFPerCm2;
    }
    for (let e3 = 0; e3 < a2; e3 += 1) {
      l2[e3] += d2[e3];
      let t3 = s2[e3];
      t3 >= 0 && (l2[e3] += f2[e3], l2[t3] += f2[e3]);
    }
    let m2 = new Int32Array(a2);
    for (let e3 = 0; e3 < a2; e3 += 1) s2[e3] >= 0 && (m2[s2[e3]] += 1);
    let h2 = Int32Array.from(m2), g2 = new Int32Array(a2), _2 = [];
    for (let e3 = 0; e3 < a2; e3 += 1) h2[e3] === 0 && _2.push(e3);
    for (let e3 = 0; e3 < _2.length; e3 += 1) {
      let t3 = _2[e3], n2 = s2[t3];
      n2 >= 0 && (g2[n2] = Math.max(g2[n2], g2[t3] + 1), --h2[n2], h2[n2] === 0 && _2.push(n2));
    }
    let v2 = Array.from({ length: a2 }, (e3, t3) => t3).filter((e3) => s2[e3] >= 0).sort((e3, t3) => g2[e3] - g2[t3]), y2 = Array.from({ length: a2 }, (e3, t3) => t3).sort((e3, t3) => g2[t3] - g2[e3]), b2 = 0;
    for (let e3 of d2) b2 += e3;
    return { nodes: r2, parents: s2, coupling: c2, matrixDiagonal: l2, capacitance: u2, leak: d2, forwardOrder: Int32Array.from(v2), backwardOrder: Int32Array.from(y2), rootIndex: p2, leakTotalS: b2 };
  }
  // i(nodes, x, y, z): index of the node nearest to (x, y, z); maps every synapse onto a compartment.
  function i(e2, t2, n2, r2) {
    let i2 = 0, a2 = 1 / 0;
    for (let o2 = 0; o2 < e2.length; o2 += 1) {
      let s2 = e2[o2], c2 = (s2.x - t2) ** 2 + (s2.y - n2) ** 2 + (s2.z - r2) ** 2;
      c2 < a2 && (a2 = c2, i2 = o2);
    }
    return i2;
  }
  // a(deg): wrap an angle to [-180, 180). Bit-identical to ((x + 180) % 360 + 360) % 360 - 180 but avoids the two
  // % operations; this is the most frequently called function of the model.
  function a(e2) {
    let u2 = e2 + 180;
    let v2 = u2 >= 0 && u2 < 360 ? u2 : u2 % 360;
    let w2 = v2 + 360;
    return (w2 >= 360 ? w2 - 360 : w2) - 180;
  }
  // o(a, b): absolute angular distance (deg).
  function o(e2, t2) {
    return Math.abs(a(e2 - t2));
  }
  // s(seed, x, y): deterministic hash of three integers to [0, 1]; used for every random texture.
  function s(e2, t2, n2) {
    let r2 = Math.imul(e2, 374761393) + Math.imul(t2, 668265263) + Math.imul(n2, 1442695041) >>> 0;
    return r2 = Math.imul(r2 ^ r2 >>> 13, 1274126177) >>> 0, ((r2 ^ r2 >>> 16) >>> 0) / 4294967295;
  }
  // One-dimensional ground texture (when groundTexture is not 'random-black-squares'): signal 'random' gives
  // alternating black and white stripes of width clip(spatialPeriodDeg, 1, 10) deg with a seeded offset; any other
  // signal gives a sine grating. cTexPrime caches the constants for the current spatialPeriodDeg and textureSeed.
  let cTexPeriodKey = NaN, cTexSeedKey = NaN, cTexCellDeg = 0, cTexOffsetDeg = 0, cTexPeriodDeg = 0;
  function cTexPrime(e2) {
    let n3 = Math.max(1, Math.min(e2.spatialPeriodDeg, 10));
    cTexCellDeg = n3;
    cTexOffsetDeg = (s(e2.textureSeed, 11, 503) - 0.5) * n3;
    cTexPeriodDeg = Math.max(3, e2.spatialPeriodDeg * 4);
    cTexPeriodKey = e2.spatialPeriodDeg;
    cTexSeedKey = e2.textureSeed;
  }
  // A fixed texture of 10-deg black and white squares (textureSeed 17), stored as a table so that it is reproduced
  // exactly (blackSquareSizeDeg 10 without blackSquareEnforceMinimum).
  let BS_SIZE = 10, BS_AZOFF = 3, BS_TXMIN = -19, BS_TYMIN = -6, BS_NX = 37, BS_NY = 12;
  let BS_ROWS = [
    `1110010001100111100100011110000010111`,
    `0100110001110100110000101001000100100`,
    `1011011111111100011100001000000001110`,
    `1111101111010100100101001110010010010`,
    `1001000010110000111111110010001100001`,
    `1011000101110111111100001110110101110`,
    `1001001001100110101010101001101010010`,
    `1100011100010110110111110010110010001`,
    `0000010011001101110101111010101000001`,
    `0101100011001111010111100000100100111`,
    `0000011001001111101101110111110010111`,
    `0110001011010100001101111110000001001`,
  ];
  // Random black squares with blackSquareEnforceMinimum (the texture used in the paper): square tiles of
  // 180 / floor(180 / blackSquareMinSizeDeg) deg, i.e. the smallest size >= blackSquareMinSizeDeg that tiles 180 deg
  // of elevation and 360 deg of azimuth exactly. A tile is black when s(textureSeed, column, row) <
  // blackSquareHashThreshold. The tile table is built once per stimulus object.
  let minimumBlackSquareStimulus = null;
  let minimumBlackSquareRowCount = 0;
  let minimumBlackSquareTileSize = 0;
  let minimumBlackSquareTiles = null;
  function primeMinimumBlackSquares(e2) {
    let minimumSize = Math.max(0.5, Math.min(30, Number(
      e2.blackSquareMinSizeDeg ?? e2.blackSquareSizeDeg ?? 5,
    )));
    let rowCount = Math.max(1, Math.floor(180 / minimumSize + 1e-10));
    let seed = Number(e2.textureSeed ?? 17);
    let threshold = Math.max(0, Math.min(1, Number(e2.blackSquareHashThreshold ?? 0.48)));
    let columnCount = rowCount * 2;
    let tiles = new Uint8Array(columnCount * rowCount);
    for (let ty = 0; ty < rowCount; ty += 1) {
      for (let tx = 0; tx < columnCount; tx += 1) {
        tiles[ty * columnCount + tx] = s(seed, tx, ty) < threshold ? 0 : 1;
      }
    }
    minimumBlackSquareStimulus = e2;
    minimumBlackSquareRowCount = rowCount;
    minimumBlackSquareTileSize = 180 / rowCount;
    minimumBlackSquareTiles = tiles;
  }
  // Luminance (0 or 1) of that texture at (azimuth, elevation).
  function minimumBlackSquareLuminance(e2, azimuthDeg, elevationDeg) {
    if (minimumBlackSquareStimulus !== e2) primeMinimumBlackSquares(e2);
    let rowCount = minimumBlackSquareRowCount;
    let tileSize = minimumBlackSquareTileSize;
    let wrappedAzimuth = a(azimuthDeg) + 180;
    let boundedElevation = Math.max(-90, Math.min(90 - Number.EPSILON * 180, elevationDeg));
    let tx = Math.min(rowCount * 2 - 1, Math.floor(wrappedAzimuth / tileSize));
    let ty = Math.min(rowCount - 1, Math.floor((boundedElevation + 90) / tileSize));
    return minimumBlackSquareTiles[ty * rowCount * 2 + tx];
  }
  // Ground luminance for groundTexture 'random-black-squares'.
  function blackSquareLuminance(e2, azimuthDeg, elevationDeg) {
    if (e2.blackSquareEnforceMinimum === true) {
      return minimumBlackSquareLuminance(e2, azimuthDeg, elevationDeg);
    }
    let size = Math.max(0.25, Number(e2.blackSquareSizeDeg ?? BS_SIZE));
    let seed = Number(e2.textureSeed ?? 17);
    let tx = Math.floor((azimuthDeg - BS_AZOFF) / size);
    let ty = Math.floor(elevationDeg / size);
    // The stored texture.
    if (size === BS_SIZE && seed === 17) {
      let ix = ((tx - BS_TXMIN) % BS_NX + BS_NX) % BS_NX;
      let iy = ((ty - BS_TYMIN) % BS_NY + BS_NY) % BS_NY;
      return BS_ROWS[iy].charCodeAt(ix) - 48;
    }
    // Any other size or seed: one hashed draw per tile.
    let threshold = Number(e2.blackSquareHashThreshold ?? 0.48);
    return s(seed, tx, ty) < threshold ? 0 : 1;
  }
  // Luminance of the one-dimensional texture at azimuth t2 (see cTexPrime).
  function legacyTextureLuminance(e2, t2) {
    if (e2.spatialPeriodDeg !== cTexPeriodKey || e2.textureSeed !== cTexSeedKey) cTexPrime(e2);
    if (e2.signal === `random`) {
      return Math.floor((t2 + cTexOffsetDeg) / cTexCellDeg) + e2.textureSeed & 1 ? 1 : 0;
    }
    return 0.5 + 0.45 * Math.sin(t2 * Math.PI * 2 / cTexPeriodDeg);
  }
  // c(stimulus, az, el): ground luminance at (az, el) in the ground's own frame.
  function c(e2, t2, n2) {
    if (e2.groundTexture === `random-black-squares`) return blackSquareLuminance(e2, t2, n2);
    return legacyTextureLuminance(e2, t2);
  }
  // l(stimulus, t, layer): displacement (deg) of the 'figure' or the 'ground' at time t (s): sinusoidal
  // (oscillationAmplitudeDeg, temporalFrequencyHz, figurePhaseDeg / groundPhaseDeg) or constant velocity.
  function l(e2, t2, n2) {
    if (e2.motion === `sinusoidal`) {
      let r2 = n2 === `ground` ? e2.groundPhaseDeg : e2.figurePhaseDeg;
      return e2.oscillationAmplitudeDeg * Math.sin(2 * Math.PI * e2.temporalFrequencyHz * t2 + r2 * Math.PI / 180);
    }
    return (n2 === `ground` ? e2.groundVelocityDps : e2.figureVelocityDps) * t2;
  }
  // u(stimulus, t, layer): as l(), except that with pattern 'FG' and relativeMotionDelaySec > 0 the figure moves
  // with the ground until that delay.
  function u(e2, t2, n2) {
    let r2 = Math.max(0, e2.relativeMotionDelaySec);
    return e2.pattern === `FG` && n2 === `figure` && r2 > 0 && t2 < r2 ? l(e2, t2, `ground`) : l(e2, t2, n2);
  }
  // d(stimulus, t, layer): velocity (deg/s) of a layer.
  function d(e2, t2, n2) {
    if (e2.motion !== `sinusoidal`) return n2 === `ground` ? e2.groundVelocityDps : e2.figureVelocityDps;
    let r2 = n2 === `ground` ? e2.groundPhaseDeg : e2.figurePhaseDeg, i2 = 2 * Math.PI * e2.temporalFrequencyHz * (t2 + 1e-7) + r2 * Math.PI / 180;
    return e2.oscillationAmplitudeDeg * 2 * Math.PI * e2.temporalFrequencyHz * Math.cos(i2);
  }
  // f(stimulus, t, layer): direction of motion, 'left' (negative velocity) or 'right'.
  function f(e2, t2, n2 = `figure`) {
    return d(e2, t2, n2) < 0 ? `left` : `right`;
  }
  // p(stimulus, t): contrast polarity ('on' / 'off') of the moving edge for the edge patterns pE, pEG, rE and rEG;
  // null for every other pattern.
  function p(e2, t2 = 0) {
    return [`pE`, `pEG`, `rE`, `rEG`].includes(e2.pattern) ? e2.pattern.startsWith(`p`) === (f(e2, t2) === `right`) ? `on` : `off` : null;
  }
  // Luminance of the single-cell test protocols (validationProtocol 'fig1-grating', 'fig5-bar', 'fig3-edge'); not
  // used by the circuit runs.
  function validationLuminance(e2, t2, n2, r2) {
    let i2 = a(t2 - (e2.validationCenterAzimuthDeg ?? 0)), o2 = n2 - (e2.validationCenterElevationDeg ?? 0), s2 = (e2.validationDirectionDeg ?? 0) * Math.PI / 180, c2 = i2 * Math.cos(s2) + o2 * Math.sin(s2), l2 = -i2 * Math.sin(s2) + o2 * Math.cos(s2), u2 = e2.validationDirection === `PD` ? -1 : 1, d2 = S(e2.contrast ?? 1, 0, 1), f2 = 0.5 - 0.5 * d2, p2 = 0.5 + 0.5 * d2;
    if (e2.validationProtocol === `fig1-grating`) {
      let t3 = Math.max(0.5, e2.spatialPeriodDeg ?? 30), n3 = u2 * Math.abs(e2.validationSpeedDps ?? 30) * r2, a2 = c2 - n3, o3 = Math.sin(2 * Math.PI * (a2 / t3 + (e2.validationPhaseDeg ?? 0) / 360)) >= 0 ? 1 : -1;
      return S(0.5 + 0.5 * d2 * o3, 0, 1);
    }
    if (e2.validationProtocol === `fig5-bar`) {
      let t3 = Math.max(0, e2.validationOnsetSec ?? 1), n3 = Math.max(1e-3, e2.validationStepSec ?? 0.08), a2 = Math.max(1, Math.round(e2.validationSpanSteps ?? 9)), o3 = Math.floor((r2 - t3) / n3), s3 = (a2 - 1) / 2, m2 = u2 * (o3 - s3) * (e2.validationBarStepDeg ?? 2.25), h2 = Math.abs(c2 - m2) <= (e2.validationBarWidthDeg ?? 2.25) / 2 && Math.abs(l2) <= (e2.validationBarHeightDeg ?? 20.25) / 2;
      return o3 >= 0 && o3 < a2 && h2 ? p2 : f2;
    }
    if (e2.validationProtocol === `fig3-edge`) {
      let t3 = Math.max(1e-3, Math.abs(e2.validationSpeedDps ?? 30)), n3 = Math.max(0.1, e2.durationSec ?? 3), a2 = -u2 * t3 * n3 * 0.45 + u2 * t3 * r2, o3 = u2 * (c2 - a2) < 0, s3 = o3 ? p2 : f2;
      return e2.validationPolarity === `off` ? 1 - s3 : s3;
    }
    return 0.5;
  }
  // Quantities of the luminance field that depend on time but not on direction. mPrime() computes them once per
  // (stimulus, time); every receptive-field sample of that time step shares them.
  let mStim = null, mTime = NaN;
  let mFigureOffset = 0, mGroundOffset = 0, mFigureEdgeDeg = 0, mStaticCenterDeg = 0;
  let mHalfWidthDeg = 0, mHalfHeightDeg = 0, mBlackScreenDeg = 0, mContrast = 0;
  let mCoverageHalfDeg = 0, mCoverageIsFull = false, mTfGain = 1;
  let mIsG = false, mIsFigureGroup = false, mIsFG = false, mIsEdgeGroup = false;
  let mEdgeStartsP = false, mEdgeEndsG = false, mIsTF = false, mIsRandom = false;
  let mCenterDeg = 0, mEdgeLowDeg = 0, mEdgeHighDeg = 0;
  // groundHemifield: 'positiveAz' / 'negativeAz' show the ground only at azimuth > 0 / < 0 (the two eyes meet
  // near azimuth 0, so this shows the ground to one eye); 'none' removes it; 'both' (default) shows it everywhere.
  // Only the ground is affected; the figure and its black band are not. mGroundBlanked() is true where the ground
  // is replaced by grey (0.5).
  let mGroundHemifield = `both`;
  function mGroundBlanked(azimuthDeg) {
    if (mGroundHemifield === `positiveAz`) return azimuthDeg < 0;
    if (mGroundHemifield === `negativeAz`) return azimuthDeg > 0;
    if (mGroundHemifield === `none`) return true;
    return false;
  }
  // Compute the per-time quantities above for stimulus e2 at time r2 (s).
  function mPrime(e2, r2) {
    mStim = e2;
    mTime = r2;
    let i2 = u(e2, r2, `figure`), s2 = u(e2, r2, `ground`);
    mIsTF = e2.pattern === `tF`;
    mIsTF && (i2 = 0);
    mFigureOffset = i2;
    mGroundOffset = s2;
    // figureAperture true: the figure window stays at figureCenterDeg while the texture inside it moves (motion seen
    // through a fixed window).
    mFigureEdgeDeg = e2.figureAperture === true ? a(e2.figureCenterDeg) : a(e2.figureCenterDeg + i2);
    mStaticCenterDeg = e2.motion === `constant` ? mFigureEdgeDeg : a(e2.figureCenterDeg);
    mTfGain = mIsTF ? Math.cos(2 * Math.PI * e2.temporalFrequencyHz * r2) : 1;
    mHalfWidthDeg = e2.figureWidthDeg / 2;
    mHalfHeightDeg = e2.figureHeightDeg / 2;
    mBlackScreenDeg = e2.staticBlackScreenDeg;
    mContrast = Math.max(0, Math.min(1, e2.contrast));
    mCoverageHalfDeg = e2.backgroundCoverageDeg / 2;
    mCoverageIsFull = mCoverageHalfDeg >= 180;
    mIsG = e2.pattern === `G`;
    mIsFG = e2.pattern === `FG`;
    mIsFigureGroup = mIsFG || e2.pattern === `F` || mIsTF;
    mIsEdgeGroup = e2.pattern === `pE` || e2.pattern === `pEG`
      || e2.pattern === `rE` || e2.pattern === `rEG`;
    mEdgeStartsP = e2.pattern.startsWith(`p`);
    mEdgeEndsG = e2.pattern.endsWith(`G`);
    mIsRandom = e2.signal === `random`;
    mCenterDeg = e2.figureCenterDeg;
    mEdgeLowDeg = e2.figureCenterDeg - 180;
    mEdgeHighDeg = e2.figureCenterDeg + 360;
    mGroundHemifield = e2.groundHemifield === `positiveAz`
      || e2.groundHemifield === `negativeAz`
      || e2.groundHemifield === `none`
      ? e2.groundHemifield
      : `both`;
  }
  // Random black-square texture of the figure (figureTexture 'random-black-squares'): the ground's tile size and
  // threshold, seed figureTextureSeed (default textureSeed + 1000). Sampled in the figure's own frame, so the
  // squares move with the figure; columns start at the figure's left edge at rest.
  function figureSquareLuminance(e2, azimuthInFigureDeg, elevationDeg) {
    let minimumSize = Math.max(0.5, Math.min(30, Number(
      e2.blackSquareMinSizeDeg ?? e2.blackSquareSizeDeg ?? 5,
    )));
    let tile = 180 / Math.max(1, Math.floor(180 / minimumSize + 1e-10));
    let seed = Number(e2.figureTextureSeed ?? Number(e2.textureSeed ?? 17) + 1000);
    let threshold = Math.max(0, Math.min(1, Number(e2.blackSquareHashThreshold ?? 0.48)));
    let left = e2.figureCenterDeg - e2.figureWidthDeg / 2;
    let columns = Math.max(1, Math.ceil(e2.figureWidthDeg / tile - 1e-9));
    let tx = Math.floor(a(azimuthInFigureDeg - left) / tile + 1e-9);
    // Behind a fixed window (figureAperture) the moving texture continues the same hash beyond the window's columns.
    // Otherwise, and while the texture has not moved, the column is clamped, so the scene at rest is the same in both
    // modes.
    if (e2.figureAperture !== true || mFigureOffset === 0) tx = tx < 0 ? 0 : tx > columns - 1 ? columns - 1 : tx;
    let boundedElevation = Math.max(-90, Math.min(90 - Number.EPSILON * 180, elevationDeg));
    let ty = Math.floor((boundedElevation + 90) / tile);
    return s(seed, tx, ty) < threshold ? 0 : 1;
  }
  // Luminance inside the figure: its square texture, white (signal 'random'), or the one-dimensional texture moved
  // with the figure. Pattern 'tF' (a still figure) reverses its contrast as cos(2 pi f t).
  function mFigureLuminance(e2, t2, n2) {
    let _2 = e2.figureTexture === `random-black-squares`
      ? figureSquareLuminance(e2, t2 - mFigureOffset, n2)
      : mIsRandom ? 1 : legacyTextureLuminance(e2, t2 - mFigureOffset);
    return mIsTF ? 0.5 + (_2 - 0.5) * mTfGain : _2;
  }
  // m(stimulus, az, el, t): luminance (0-1) at direction (az, el) deg and time t s. Patterns: 'G' ground only;
  // 'F' figure moving on a still ground; 'FG' figure and ground both moving; 'tF' contrast-reversing still figure;
  // 'pE', 'pEG', 'rE', 'rEG' moving edges. Outside backgroundCoverageDeg (centred on azimuth 0) the screen is
  // grey; with signal 'random', staticBlackScreenDeg adds a black band on each side of the figure's rest position.
  function m(e2, t2, n2, r2) {
    if (e2.validationProtocol) return validationLuminance(e2, t2, n2, r2);
    if (mStim !== e2 || mTime !== r2) mPrime(e2, r2);
    if (!mCoverageIsFull && o(t2, 0) > mCoverageHalfDeg) return 0.5;
    let f2 = Math.abs(n2) <= mHalfHeightDeg, v2;
    if (mIsG) v2 = mGroundBlanked(t2) ? 0.5 : c(e2, t2 - mGroundOffset, n2);
    else if (mIsFigureGroup) {
      if (f2 && o(t2, mFigureEdgeDeg) <= mHalfWidthDeg) v2 = mFigureLuminance(e2, t2, n2);
      else if (mIsRandom && f2 && mBlackScreenDeg > 0
        && o(t2, mStaticCenterDeg) <= mHalfWidthDeg + mBlackScreenDeg) v2 = 0;
      else if (mGroundBlanked(t2)) v2 = 0.5;
      else v2 = mIsFG ? c(e2, t2 - mGroundOffset, n2) : c(e2, t2, n2);
    } else if (mIsEdgeGroup) {
      let n3 = mCenterDeg + mFigureOffset, r3 = t2;
      for (; r3 < mEdgeLowDeg; ) r3 += 360;
      for (; r3 > mEdgeHighDeg; ) r3 -= 360;
      let a2 = r3 <= n3, o2 = mEdgeStartsP ? a2 : !a2;
      v2 = o2
        ? mFigureLuminance(e2, t2, n2)
        : mEdgeEndsG ? c(e2, t2 - mGroundOffset, n2) : c(e2, t2, n2);
    } else v2 = c(e2, t2, n2);
    let y2 = 0.5 + (v2 - 0.5) * mContrast;
    return Math.max(0, Math.min(1, y2));
  }
  // The five input types of T4 (this order is used throughout).
  let h = [`Mi9`, `Tm3`, `Mi1`, `Mi4`, `C3`];
  // g(row): skeleton row [id, type, x, y, z, radius, parent] -> node object.
  function g(e2) {
    return { id: e2[0], type: e2[1], x: e2[2], y: e2[3], z: e2[4], radius: e2[5], parent: e2[6] };
  }
  // Default input-model parameters (the calibration F below overrides most of them): _ photoreceptor; v time
  // constants (ms) of the running mean and the background mean, and the drive scale; y per-type filter (ON or OFF
  // drive, rise and decay time constants in ms, sustained fraction); b per-type gain and threshold; x tonic floor.
  let _ = { kappa: 300, tr0: 22, trmin: 17, ta: 55, tm0: 20, tmmin: 3, vmax: 65 }, v = { tauMean: 250, tauBg: 3e3, driveScale: 12 }, y = { Mi1: { drive: `on`, tr: 10, td: 60, sus: 0.35 }, Tm3: { drive: `on`, tr: 5, td: 40, sus: 0.15 }, Mi9: { drive: `off`, tr: 40, td: 150, sus: 0.9 }, Mi4: { drive: `on`, tr: 40, td: 200, sus: 0.9 }, C3: { drive: `on`, tr: 50, td: 250, sus: 0.7 } }, b = { Mi9: { gain: 0.92, threshold: 0.2 }, Tm3: { gain: 0.35, threshold: 0.35 }, Mi1: { gain: 0.65, threshold: 0.88 }, Mi4: { gain: 1.1, threshold: 0.44 }, C3: { gain: 1.5, threshold: 0.7 } }, x = { Mi9: 0.8, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 };
  // S(x, lo, hi): clamp.
  function S(e2, t2, n2) {
    return e2 < t2 ? t2 : e2 > n2 ? n2 : e2;
  }
  // C(luminance): photoreceptor input intensity.
  function C(e2) {
    return 300 * (0.2 + 1.6 * S(e2, 0, 1));
  }
  // ee(luminance): steady-state photoreceptor voltage; w and te are its values for black and white.
  function ee(e2) {
    let t2 = C(e2), n2 = 1 / (1 + t2 / _.kappa) * t2;
    return _.vmax * n2 / (n2 + _.kappa);
  }
  let w = ee(0), te = ee(1);
  // ne(intensity): photoreceptor state at steady state.
  function ne(e2) {
    let t2 = 1 / (1 + e2 / _.kappa) * e2;
    return { u1: e2, u2: e2, a: e2, voltage: _.vmax * t2 / (t2 + _.kappa) };
  }
  // T(state, intensity, dtMs): advance the photoreceptor in sub-steps of <= 1 ms: two low-pass stages, a slow
  // adaptation state that sets the gain and the time constants, and a saturating output.
  function T(e2, t2, n2) {
    let r2 = Math.max(1, Math.ceil(n2)), i2 = n2 / r2;
    for (let n3 = 0; n3 < r2; n3 += 1) {
      let n4 = 1 / (1 + e2.a / _.kappa), r3 = _.trmin + (_.tr0 - _.trmin) * n4, a2 = _.tmmin + (_.tm0 - _.tmmin) * n4;
      e2.u1 += i2 * (t2 - e2.u1) / r3, e2.u2 += i2 * (e2.u1 - e2.u2) / r3, e2.a += i2 * (e2.u2 - e2.a) / _.ta;
      let o2 = n4 * e2.u2, s2 = _.vmax * o2 / (o2 + _.kappa);
      e2.voltage += i2 * (s2 - e2.voltage) / a2;
    }
  }
  // E(): fresh filter state of the five input types.
  function E() {
    return { Mi9: { h: 0, f: 0 }, Tm3: { h: 0, f: 0 }, Mi1: { h: 0, f: 0 }, Mi4: { h: 0, f: 0 }, C3: { h: 0, f: 0 } };
  }
  // Per-type filter constants of a calibration (cached).
  let filterConstKey = -1, filterConstTable = null;
  function filterConstants(calibration) {
    if (calibration !== filterConstKey) {
      filterConstTable = h.map((n3) => {
        let s2 = y[n3];
        let scale = S(Number(calibration?.filterTimeScales?.[n3] ?? 1), 0.25, 2);
        return {
          key: n3,
          driveIsOn: s2.drive === `on`,
          sus: S(Number(calibration?.sustainedFractions?.[n3] ?? s2.sus), 0, 1),
          riseMs: Math.max(1, Number(calibration?.filterRiseMs?.[n3] ?? s2.tr * scale)),
          decayMs: Math.max(1, Number(calibration?.filterDecayMs?.[n3] ?? s2.td * scale)),
        };
      });
      filterConstKey = calibration;
    }
    return filterConstTable;
  }
  // D(path, dtMs, tauMeanMs, calibration): advance the temporal filters of one receptive-field path in sub-steps
  // of <= 2 ms. The photoreceptor output is compared with a running mean (tauMeanMs) and a slow background mean; each
  // type mixes the two deviations (sustained fraction), keeps its sign (ON or OFF) and passes two low-pass stages.
  function D(e2, t2, n2 = v.tauMean, calibration = {}) {
    let r2 = Math.max(1, Math.ceil(t2 / 2));
    let i2 = t2 / r2;
    let f2 = filterConstants(calibration);
    let backgroundMeanMs = S(Number(calibration?.backgroundMeanMs ?? v.tauBg), 100, 1e4);
    for (let t3 = 0; t3 < r2; t3 += 1) {
      e2.mean += i2 * (e2.pr.voltage - e2.mean) / n2;
      e2.background += i2 * (e2.pr.voltage - e2.background) / backgroundMeanMs;
      let t4 = Math.max(0, e2.pr.voltage - e2.mean) / v.driveScale, r3 = Math.max(0, e2.mean - e2.pr.voltage) / v.driveScale, a2 = Math.max(0, e2.pr.voltage - e2.background) / v.driveScale, o2 = Math.max(0, e2.background - e2.pr.voltage) / v.driveScale;
      for (let k2 = 0; k2 < f2.length; k2 += 1) {
        let s2 = f2[k2], c2 = s2.driveIsOn ? t4 : r3, l2 = s2.driveIsOn ? a2 : o2, u2 = s2.sus * l2 + (1 - s2.sus) * c2, d2 = e2.filters[s2.key];
        d2.h += i2 * (u2 - d2.h) / s2.riseMs, d2.f += i2 * (d2.h - d2.f) / s2.decayMs;
      }
    }
  }
  // Receptive-field sampling: the centre plus 12 points on two rings at 1 and 2 standard deviations, with Gaussian
  // weights (cached per receptive-field width).
  let rfFwhmKey = NaN, rfDeltaAzDeg = null, rfDeltaElDeg = null, rfWeight = null, rfWeightSum = 1;
  function rfPrime(i2) {
    let a2 = S(i2, 1, 30) / 2.3548;
    let rings = [{ radius: 1, count: 6, offset: 0 }, { radius: 2, count: 6, offset: 0.5 }];
    let sampleCount = 0;
    for (let ring of rings) sampleCount += ring.count;
    rfDeltaAzDeg = new Float64Array(sampleCount);
    rfDeltaElDeg = new Float64Array(sampleCount);
    rfWeight = new Float64Array(sampleCount);
    let s2 = 1, index = 0;
    for (let i3 of rings) {
      let c2 = Math.exp(-0.5 * i3.radius * i3.radius);
      for (let l2 = 0; l2 < i3.count; l2 += 1) {
        let u2 = 2 * Math.PI * (l2 + i3.offset) / i3.count;
        rfDeltaAzDeg[index] = i3.radius * a2 * Math.cos(u2);
        rfDeltaElDeg[index] = i3.radius * a2 * Math.sin(u2);
        rfWeight[index] = c2;
        s2 += c2;
        index += 1;
      }
    }
    rfWeightSum = s2;
    rfFwhmKey = i2;
  }
  // O(stimulus, az, el, t, fwhmDeg): Gaussian-weighted luminance in a receptive field.
  function O(e2, t2, n2, r2, i2 = 8) {
    if (i2 !== rfFwhmKey) rfPrime(i2);
    let o2 = m(e2, t2, n2, r2);
    for (let index = 0; index < rfDeltaAzDeg.length; index += 1) {
      o2 += rfWeight[index] * m(e2, t2 + rfDeltaAzDeg[index], n2 + rfDeltaElDeg[index], r2);
    }
    return o2 / rfWeightSum;
  }
  // k(cell, path, stimulus, initialLuminance, options): state of one receptive-field path of a T4. Each T4 has
  // three paths: 'home' at its receptive-field centre (Mi1, Tm3), 'preferred' at centre - offset (Mi9) and 'null'
  // at centre + offset (Mi4, C3), where offset is the cell's premotion.offsetAzimuth / offsetElevation.
  function k(e2, t2, n2, r2, i2) {
    let a2 = t2 === `null` ? 1 : t2 === `preferred` ? -1 : 0;
    let o2 = e2.premotion.azimuth + a2 * e2.premotion.offsetAzimuth;
    let s2 = e2.premotion.elevation + a2 * e2.premotion.offsetElevation;
    let u2 = t2 === `preferred`
      ? i2.calibration?.receptiveFieldFwhmDeg?.Mi9
      : t2 === `null`
        ? i2.calibration?.receptiveFieldFwhmDeg?.C3
        : i2.calibration?.receptiveFieldFwhmDeg?.Mi1;
    let rfFwhmDeg = Number(u2 ?? i2.rfFwhmDeg);
    let c2 = r2 ?? oe(n2, o2, s2, { ...i2, rfFwhmDeg });
    let l2 = ne(C(c2));
    return {
      pr: l2,
      mean: l2.voltage,
      background: l2.voltage,
      filters: E(),
      azimuth: o2,
      elevation: s2,
      rfFwhmDeg,
      acceptedLuminance: c2,
      burnLuminance: c2,
    };
  }
  // A(v): photoreceptor mean voltage scaled to 0 (black) .. 1 (white).
  let aSpan = Math.max(1e-9, te - w);
  function A(e2) {
    return S((e2 - w) / aSpan, 0, 1);
  }
  // j(type): the path that feeds a type (Mi9: preferred; Mi4, C3: null; Mi1, Tm3: home).
  function j(e2) {
    return e2 === `Mi9` ? `preferred` : e2 === `Mi4` || e2 === `C3` ? `null` : `home`;
  }
  // M(type, calibration): conductance time shift (ms) of a type; 0 for every type in the calibration used.
  function M(e2, t2) {
    return t2?.conductanceShiftsMs?.[e2] ?? 0;
  }
  // ie(step, stepCount, type, dtMs, calibration): the step whose conductance a type reads at this step.
  function ie(e2, t2, n2, r2, i2) {
    return S(e2 - Math.round(M(n2, i2) / r2), 0, t2);
  }
  // N(stimulus, t): edge polarity ('on' unless validationPolarity or an edge pattern says otherwise).
  function N(e2, t2) {
    if (e2.validationPolarity === `on` || e2.validationPolarity === `off`) return e2.validationPolarity;
    return p(e2, t2) ?? `on`;
  }
  // ae(stimulus, t): for the edge patterns, the uniform luminance seen before the edge arrives; null otherwise.
  function ae(e2, t2) {
    let n2 = p(e2, t2);
    if (n2 === null) return null;
    let r2 = S(e2.contrast, 0, 1);
    return n2 === `on` ? 0.5 - 0.5 * r2 : 0.5 + 0.5 * r2;
  }
  // oe(stimulus, az, el, options): luminance a path sees before the stimulus starts (preStimulusMode:
  // 'model-default' = the scene at t = 0, 'mean-gray', 'black' or 'ground-only').
  function oe(e2, t2, n2, r2) {
    let i2 = r2.preStimulusMode ?? `model-default`;
    return i2 === `mean-gray` ? 0.5 : i2 === `black` ? 0 : O(i2 === `ground-only` ? { ...e2, pattern: `G` } : e2, t2, n2, 0, r2.rfFwhmDeg);
  }
  // Per-type constants of the activity -> conductance step of a calibration (cached).
  let cascadeKey = -1, cascadeTable = null;
  function cascadeConstants(n2) {
    if (n2 !== cascadeKey) {
      cascadeKey = n2;
      cascadeTable = null;
    }
    if (cascadeTable) return cascadeTable;
    cascadeTable = h.map((r2) => {
      let gainSpec = b[r2];
      return {
        key: r2,
        pathKey: j(r2),
        onActivityScale: n2?.onActivityScales?.[r2] ?? 1,
        driveIsOff: y[r2].drive === `off`,
        tonicFloor: x[r2],
        threshold: n2?.thresholds?.[r2] ?? gainSpec.threshold,
        conductanceFactor: (n2?.conductanceScales?.[r2] ?? 1) * gainSpec.gain,
        saturation: Number(n2?.conductanceSaturationLevels?.[r2] ?? 0),
      };
    });
    return cascadeTable;
  }
  // P(cellState, calibration): activity and conductance of the five input types of one cell.
  // activity = max(filter output * onActivityScale, tonic floor * mean brightness (mean darkness for Mi9));
  // conductance = scale * max(0, activity - threshold), saturating as x / (1 + x / level) where a level is set.
  function P(e2, n2) {
    let table = cascadeConstants(n2);
    for (let k2 = 0; k2 < table.length; k2 += 1) {
      let entry = table[k2], i2 = e2.paths[entry.pathKey];
      let a2 = Math.max(0, i2.filters[entry.key].f);
      let o2 = a2 * entry.onActivityScale;
      let meanDrive = A(i2.mean);
      let s2 = entry.driveIsOff ? 1 - meanDrive : meanDrive;
      let l2 = Math.max(o2, entry.tonicFloor * s2);
      let excess = Math.max(0, l2 - entry.threshold);
      let bounded = entry.saturation > 0
        ? excess / (1 + excess / entry.saturation)
        : excess;
      e2.activity[entry.key] = l2;
      e2.conductance[entry.key] = entry.conductanceFactor * bounded;
    }
  }
  // se(cells, stimulus, dtMs, options): input-model states of all cells, run for 1 s on the pre-stimulus luminance
  // so that every cell starts at steady state.
  function se(e2, t2, n2, r2 = {}) {
    let i2 = N(t2, 0), e3 = Number(t2.validationInitialLuminance), a2 = Number.isFinite(e3) ? S(e3, 0, 1) : ae(t2, 0), o2 = e2.map((e4) => ({ cell: e4, paths: { home: k(e4, `home`, t2, a2, r2), null: k(e4, `null`, t2, a2, r2), preferred: k(e4, `preferred`, t2, a2, r2) }, activity: { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 }, conductance: { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 } })), s2 = r2.calibration?.temporalMeanMs ?? v.tauMean, m2 = r2.calibration ?? {}, c2 = Math.round(1e3 / n2);
    let burnPath = (t3) => {
      let e5 = t3.burnLuminance;
      T(t3.pr, C(e5), n2), D(t3, n2, s2, m2);
    };
    for (let e3 = 0; e3 < c2; e3 += 1) for (let e4 of o2) {
      burnPath(e4.paths.home), burnPath(e4.paths.null), burnPath(e4.paths.preferred);
    }
    for (let e3 of o2) P(e3, r2.calibration);
    return o2;
  }
  // ce(states, stimulus, tSec, dtMs, options): advance the input model of all cells by one step.
  function ce(e2, t2, n2, r2, i2 = {}) {
    let a2 = N(t2, n2), o2 = i2.calibration?.temporalMeanMs ?? v.tauMean, m2 = i2.calibration ?? {};
    let advancePath = (e3) => {
      let a3 = O(t2, e3.azimuth, e3.elevation, n2, e3.rfFwhmDeg ?? i2.rfFwhmDeg);
      e3.acceptedLuminance = a3, T(e3.pr, C(a3), r2), D(e3, r2, o2, m2);
    };
    for (let c2 of e2) {
      let paths = c2.paths;
      advancePath(paths.home), advancePath(paths.null), advancePath(paths.preferred);
      P(c2, i2.calibration);
    }
  }
  // Message 'precompute-premotion-chunk' (from parallel-premotion-pool.mjs): run the input model of a slice of
  // cells over the whole run and write their conductances into shared buffers.
  function precomputePremotionChunk(payload) {
    let {
      cells, cellStart, cellCount, targetIndex, stimulus, dtMs, stepCount,
      premotionOptions, shared,
    } = payload;
    let dtSec = dtMs / 1e3;
    let states = se(cells, stimulus, dtMs, premotionOptions);
    let conductance = shared.conductance.map((buffer) => new Float64Array(buffer));
    let targetActivity = shared.targetActivity.map((buffer) => new Float64Array(buffer));
    let targetAccepted = shared.targetAccepted.map((buffer) => new Float64Array(buffer));
    let targetMi9Off = new Float64Array(shared.targetMi9Off);
    let targetMi9On = new Float64Array(shared.targetMi9On);
    let targetMi9Dark = new Float64Array(shared.targetMi9Dark);
    let targetLocalIndex = targetIndex - cellStart;
    let ownsTarget = targetLocalIndex >= 0 && targetLocalIndex < states.length;
    let writeStep = (step) => {
      let frameOffset = step * cellCount + cellStart;
      for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
        let kind = h[kindIndex];
        let frame = conductance[kindIndex];
        for (let localIndex = 0; localIndex < states.length; localIndex += 1) {
          frame[frameOffset + localIndex] = states[localIndex].conductance[kind];
        }
      }
      if (!ownsTarget) return;
      let targetState = states[targetLocalIndex];
      for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
        targetActivity[kindIndex][step] = targetState.activity[h[kindIndex]];
      }
      targetAccepted[0][step] = targetState.paths.home.acceptedLuminance;
      targetAccepted[1][step] = targetState.paths.null.acceptedLuminance;
      targetAccepted[2][step] = targetState.paths.preferred.acceptedLuminance;
      targetMi9Off[step] = Math.max(0, targetState.paths.preferred.filters.Mi9.f);
      targetMi9On[step] = Math.max(0, targetState.paths.preferred.filters.Mi1.f);
      targetMi9Dark[step] = targetState.paths.preferred.mean;
    };
    writeStep(0);
    for (let step = 1; step <= stepCount; step += 1) {
      ce(states, stimulus, step * dtSec, dtMs, premotionOptions);
      writeStep(step);
    }
    return { cellStart, cellEnd: cellStart + states.length, stepCount };
  }
  // Calibrated input-model parameters (premotionVariant 'speedTuned', the default): per-type thresholds,
  // conductance scales, saturation levels, filter time constants, sustained fractions, receptive-field widths and
  // reversal potentials, and the time constants of the running and background means.
  let F = { label: `soma-dual-refit-zero-shift-tradeoff-0-rf3-c3decay80`, thresholds: { Mi9: .14, Tm3: .15, Mi1: 0, Mi4: .4, C3: .3 }, conductanceScales: { Mi9: .62, Tm3: 1.9, Mi1: 2, Mi4: 1.4, C3: 2.2 }, onActivityScales: { Mi4: 5 }, conductanceShiftsMs: { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 }, spatialOffsetsDeg: { Mi9: { x: 0, y: 0 }, Tm3: { x: 0, y: 0 }, Mi1: { x: 0, y: 0 }, Mi4: { x: 0, y: 0 }, C3: { x: 0, y: 0 } }, filterRiseMs: { Mi9: 37, Tm3: 1.9, Mi1: 17, Mi4: 48, C3: 10 }, filterDecayMs: { Mi9: 68, Tm3: 15, Mi1: 55, Mi4: 230, C3: 80 }, sustainedFractions: { Mi9: .86, Tm3: .22, Mi1: .15, Mi4: .93, C3: .9 }, conductanceSaturationLevels: { Tm3: 3.2, Mi1: .52 }, receptiveFieldFwhmDeg: { Mi9: 3, Tm3: 3, Mi1: 3, Mi4: 3, C3: 3 }, reversalPotentialsMv: { Mi9: -71.8, Tm3: -23.5, Mi1: -23.5, Mi4: -68.2, C3: -68.2 }, temporalMeanMs: 550, backgroundMeanMs: 1200 };
  // le(variant): calibration of a premotionVariant.
  function le(e2) {
    return e2 === `speedTuned` ? F : void 0;
  }
  // Reversal potentials (mV) of the input synapses; L maps the type index used in the circuit file to the type.
  let I = { Mi9: -71.8, Tm3: -23.5, Mi1: -23.5, Mi4: -68.2, C3: -68.2 }, L = h;
  // The same reversal potentials as an array in type order (for the per-step loops).
  let upstreamReversalByIndex = h.map((kind) => I[kind]);
  // R(x, lo, hi): clamp.
  function R(e2, t2, n2) {
    return e2 < t2 ? t2 : e2 > n2 ? n2 : e2;
  }
  // Display-only activity of a T4a: sigmoid of its axon voltage (half point -45 mV, slope 4 mV), 0 at rest.
  function sigmoidMembraneActivity(voltageMv, leakMv = -65) {
    let sigmoid = (value) => 1 / (1 + Math.exp(-R(value, -40, 40)));
    let raw = sigmoid((voltageMv + 45) / 4);
    let resting = sigmoid((leakMv + 45) / 4);
    return R((raw - resting) / Math.max(1e-9, 1 - resting), 0, 1);
  }
  // ue(values): mean (-65 mV, in volts, for an empty list).
  function ue(e2) {
    let t2 = 0;
    for (let n2 of e2) t2 += n2;
    return e2.length ? t2 / e2.length : -65 / 1e3;
  }
  // z(V, thresholdMv, gainPerMv, exponent): release fraction clip((gain * max(0, V - threshold))^exponent, 0, 1)
  // from the presynaptic voltage V (volts).
  function z(e2, t2, n2, r2 = 1) {
    let i2 = Math.max(0, (e2 * 1e3 - t2) * n2);
    // Shortcuts for the common cases (below threshold, exponent 1); they return the same values.
    if (i2 === 0) return 0;
    return R(r2 === 1 ? i2 : i2 ** r2, 0, 1);
  }
  // de(kernel): cable state at rest (-65 mV): voltages (V), this step's synaptic conductance (S) and current (A),
  // and the solver's work arrays.
  function de(e2) {
    let t2 = e2.nodes.length;
    return { kernel: e2, voltage: new Float64Array(t2).fill(-65 / 1e3), next: new Float64Array(t2), conductance: new Float64Array(t2), current: new Float64Array(t2), diagonal: new Float64Array(t2), rhs: new Float64Array(t2) };
  }
  // fe(state, dtSec): one implicit (backward-Euler) step of a passive cable, solved exactly in O(n) with the Hines
  // algorithm (leak reversal -65 mV), using the synaptic conductances and currents assembled for this step.
  function fe(e2, t2) {
    let { kernel: n2, voltage: r2, next: i2, conductance: a2, current: o2, diagonal: s2, rhs: c2 } = e2, l2 = n2.nodes.length;
    for (let e3 = 0; e3 < l2; e3 += 1) {
      let i3 = n2.capacitance[e3] / t2;
      s2[e3] = n2.matrixDiagonal[e3] + i3 + a2[e3], c2[e3] = i3 * r2[e3] + n2.leak[e3] * (-65 / 1e3) + o2[e3];
    }
    for (let e3 of n2.forwardOrder) {
      let t3 = n2.parents[e3], r3 = n2.coupling[e3] / s2[e3];
      s2[t3] -= n2.coupling[e3] * r3, c2[t3] -= r3 * c2[e3];
    }
    for (let e3 of n2.backwardOrder) {
      let t3 = n2.parents[e3];
      i2[e3] = t3 < 0 ? c2[e3] / s2[e3] : (c2[e3] - n2.coupling[e3] * i2[t3]) / s2[e3];
    }
    e2.voltage = i2, e2.next = r2;
  }
  // pe(circuit, targetCell, targetKernel, vchKernels, hemiOfT4a): map the vCH side of every T4a -> vCH and vCH ->
  // T4a synapse onto the vCH of that T4a's own hemicircuit, and the target cell's own synapses onto its cable.
  function pe(e2, t2, n2, vchKernels, hemiOfT4a) {
    let a2 = { Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [] };
    for (let e3 of t2.upstreamContacts) {
      let t3 = L[e3[0]];
      t3 && a2[t3].push(i(n2.nodes, e3[1], e3[2], e3[3]));
    }
    let o2 = [], s2 = [], c2 = [], l2 = [];
    for (let cellIndex = 0; cellIndex < e2.cells.length; cellIndex += 1) {
      let a3 = e2.cells[cellIndex];
      let r2 = vchKernels[hemiOfT4a[cellIndex]];
      o2.push(a3.t4ToVchContacts.map((e3) => i(r2.nodes, e3[3], e3[4], e3[5])));
      s2.push(a3.vchToT4Contacts.map((e3) => i(r2.nodes, e3[0], e3[1], e3[2])));
      if (a3.rootId === t2.rootId) {
        for (let e3 of a3.t4ToVchContacts) c2.push(i(n2.nodes, e3[0], e3[1], e3[2]));
        for (let e3 of a3.vchToT4Contacts) l2.push(i(n2.nodes, e3[3], e3[4], e3[5]));
      }
    }
    return { upstreamTargetByType: a2, t4ToVchByCell: o2, vchToT4ByCell: s2, targetToVchTargetCompartments: c2, vchToTargetTargetCompartments: l2 };
  }
  // Map one T4a's input contacts and its T4a -> vCH and vCH -> T4a synapses onto the cell's own cable.
  function mapT4aCable(cell, kernel) {
    let upstreamTargetByType = { Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [] };
    for (let contact of cell.upstreamContacts ?? []) {
      let kind = L[contact[0]];
      if (kind) upstreamTargetByType[kind].push(
        i(kernel.nodes, contact[1], contact[2], contact[3]),
      );
    }
    return {
      upstreamTargetByType,
      targetToVchTargetCompartments: (cell.t4ToVchContacts ?? []).map(
        (contact) => i(kernel.nodes, contact[0], contact[1], contact[2]),
      ),
      vchToTargetTargetCompartments: (cell.vchToT4Contacts ?? []).map(
        (contact) => i(kernel.nodes, contact[3], contact[4], contact[5]),
      ),
    };
  }
  // The same for one T4b. The field names are those of T4a: targetToVchTargetCompartments holds the T4b -> LPi15
  // release sites and vchToTargetTargetCompartments the vCH -> T4b input sites.
  function mapT4bCable(cell, kernel) {
    let upstreamTargetByType = { Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [] };
    for (let contact of cell.upstreamContacts ?? []) {
      let kind = L[contact[0]];
      if (kind) upstreamTargetByType[kind].push(
        i(kernel.nodes, contact[1], contact[2], contact[3]),
      );
    }
    return {
      upstreamTargetByType,
      targetToVchTargetCompartments: (cell.t4bToLpi15CableContacts ?? []).map(
        (contact) => i(kernel.nodes, contact[0], contact[1], contact[2]),
      ),
      vchToTargetTargetCompartments: (cell.vchToT4bCableContacts ?? []).map(
        (contact) => i(kernel.nodes, contact[3], contact[4], contact[5]),
      ),
    };
  }
  // Readout compartments of a T4 cable: dendrite = its input sites; soma = the root; axon = the compartment
  // farthest from the root (path length) that lies on the root paths of at least half of the cell's output synapses.
  function selectTargetAnatomicalReadouts(e2, t2) {
    let n2 = h.flatMap((e3) => t2.upstreamTargetByType[e3] ?? []);
    let r2 = Array.from(new Set(
      h.flatMap((e3) => t2.upstreamTargetByType[e3] ?? []),
    ));
    n2.length || (n2 = r2.length ? r2 : [e2.rootIndex]);
    let i2 = e2.rootIndex;
    let a2 = t2.targetToVchTargetCompartments.filter((t3) => (
      Number.isInteger(t3) && t3 >= 0 && t3 < e2.nodes.length
    ));
    let o2 = new Int32Array(e2.nodes.length);
    for (let t3 of a2) {
      let n3 = t3, r3 = new Set();
      for (; n3 >= 0 && !r3.has(n3); ) {
        r3.add(n3), o2[n3] += 1, n3 = e2.parents[n3];
      }
    }
    let s2 = new Float64Array(e2.nodes.length).fill(Number.NaN);
    s2[i2] = 0;
    let c2 = (t3) => {
      if (Number.isFinite(s2[t3])) return s2[t3];
      let n3 = [], r3 = t3, i3 = new Set();
      for (; r3 >= 0 && !Number.isFinite(s2[r3]) && !i3.has(r3); ) {
        i3.add(r3), n3.push(r3), r3 = e2.parents[r3];
      }
      let a3 = r3 >= 0 && Number.isFinite(s2[r3]) ? s2[r3] : 0;
      for (let t4 = n3.length - 1; t4 >= 0; t4 -= 1) {
        let r4 = n3[t4], i4 = e2.parents[r4];
        if (i4 >= 0) {
          let t5 = e2.nodes[r4], n4 = e2.nodes[i4];
          a3 += Math.hypot(t5.x - n4.x, t5.y - n4.y, t5.z - n4.z);
        }
        s2[r4] = a3;
      }
      return s2[t3];
    };
    let l2 = Math.max(1, Math.ceil(a2.length / 2)), u2 = -1, d2 = -1;
    if (a2.length) {
      for (let t3 = 0; t3 < e2.nodes.length; t3 += 1) {
        if (t3 === i2 || o2[t3] < l2) continue;
        let n3 = c2(t3);
        (n3 > d2 || n3 === d2 && o2[t3] > o2[u2]) && (u2 = t3, d2 = n3);
      }
      if (u2 < 0) {
        u2 = a2.reduce((t3, n3) => c2(n3) < c2(t3) ? n3 : t3, a2[0]);
      }
    }
    u2 < 0 && (u2 = i2);
    return {
      dendriteCompartments: n2,
      allInputCompartments: r2,
      somaCompartment: i2,
      axonCompartment: u2,
      axonOutputContactCount: a2.length,
      axonTrunkCoverageCount: o2[u2] ?? 0,
      axonTrunkCoverageThreshold: a2.length ? l2 : 0,
    };
  }
  // B(state, compartment, conductanceS, reversalMv): add a synaptic conductance for this step.
  function B(e2, t2, n2, r2) {
    t2 < 0 || t2 >= e2.conductance.length || n2 <= 0 || (e2.conductance[t2] += n2, e2.current[t2] += n2 * r2 / 1e3);
  }
  // Convert a circuit file into flat lists of T4a and T4b, each cell tagged with the index of its hemicircuit,
  // plus the H2 cells with their inputs and outputs resolved to cell roles. A one-eye circuit (no 'hemicircuits'
  // key) becomes a single hemicircuit without H2.
  function normalizeBinocularCircuit(raw) {
    if (!raw?.hemicircuits) {
      let cells = raw.cells.map((cell) => (cell.hemiIndex === 0 ? cell : { ...cell, hemiIndex: 0 }));
      let t4b = (raw.t4b ?? []).map((cell) => (
        cell.hemiIndex === 0 ? cell : { ...cell, hemiIndex: 0 }
      ));
      let monoRoles = new Map();
      if (raw.vch?.rootId !== undefined) {
        monoRoles.set(String(raw.vch.rootId), { kind: `VCH`, hemiIndex: 0 });
      }
      if (raw.lpi15?.rootId !== undefined) {
        monoRoles.set(String(raw.lpi15.rootId), { kind: `LPi15`, hemiIndex: 0 });
      }
      return {
        roleByRoot: monoRoles,
        circuit: { ...raw, cells, t4b },
        hemis: [{ key: `monocular`, eye: raw.cells[0]?.side ?? `left`, vch: raw.vch, lpi15: raw.lpi15 }],
        h2Units: [],
        binocular: false,
      };
    }
    let order = Object.keys(raw.hemicircuits);
    let hemis = order.map((key) => {
      let block = raw.hemicircuits[key];
      return { key, eye: block.eye, vch: block.vch, lpi15: block.lpi15, h2RootId: block.h2RootId };
    });
    let cells = [];
    let t4b = [];
    let hemiOfEye = {};
    order.forEach((key, hemiIndex) => {
      let block = raw.hemicircuits[key];
      hemiOfEye[block.eye] = hemiIndex;
      for (let cell of block.cells) cells.push({ ...cell, hemiIndex });
      for (let cell of block.t4b) t4b.push({ ...cell, hemiIndex });
    });
    // Role of every large cell (vCH or LPi15 with its hemicircuit, or H2), so that the H2 synapses are wired from the
    // data.
    let roleByRoot = new Map();
    hemis.forEach((hemi, hemiIndex) => {
      roleByRoot.set(String(hemi.vch.rootId), { kind: `VCH`, hemiIndex });
      roleByRoot.set(String(hemi.lpi15.rootId), { kind: `LPi15`, hemiIndex });
    });
    let h2Sides = Object.keys(raw.h2 ?? {});
    h2Sides.forEach((side, h2Index) => {
      roleByRoot.set(String(raw.h2[side].rootId), { kind: `H2`, h2Index });
    });
    let h2Units = h2Sides.map((side, h2Index) => {
      let unit = raw.h2[side];
      return {
        side,
        h2Index,
        rootId: String(unit.rootId),
        skeleton: unit.skeleton,
        // Hemicircuit whose T4b drive this H2.
        driveHemiIndex: hemiOfEye[unit.driveEye],
        t4bDrive: unit.t4bDrive ?? {},
        outputs: (unit.outputs ?? []).map((edge) => ({
          ...edge,
          role: roleByRoot.get(String(edge.postRootId)) ?? null,
        })).filter((edge) => edge.role),
        inputs: (unit.inputs ?? []).map((edge) => ({
          ...edge,
          role: roleByRoot.get(String(edge.preRootId)) ?? null,
        })).filter((edge) => edge.role),
      };
    });
    return {
      roleByRoot,
      circuit: {
        ...raw,
        cells,
        t4b,
        vch: hemis[0].vch,
        lpi15: hemis[0].lpi15,
        upstreamTypes: raw.upstreamTypes,
      },
      hemis,
      h2Units,
      binocular: true,
    };
  }
  // me() and he(): not called (earlier forms of the input assembly that V() now does inline).
  function me(e2, t2, n2, r2, activeFrames = null, cellIndex = -1) {
    let i2 = e2.kernel.leakTotalS / 0.5;
    for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
      let a2 = h[kindIndex];
      let o2 = n2.upstreamTargetByType[a2];
      if (!o2.length) continue;
      let premotionConductance = activeFrames && cellIndex >= 0
        ? activeFrames[kindIndex][cellIndex]
        : t2.conductance[a2];
      let s2 = premotionConductance * i2, c2 = r2.upstreamScaling === `perContact` ? s2 : s2 / o2.length;
      if (c2 <= 0) continue;
      let conductance = e2.conductance;
      let current = e2.current;
      let currentIncrement = c2 * I[a2] / 1e3;
      for (let contactIndex = 0; contactIndex < o2.length; contactIndex += 1) {
        let compartment = o2[contactIndex];
        if (compartment < 0 || compartment >= conductance.length) continue;
        conductance[compartment] += c2;
        current[compartment] += currentIncrement;
      }
    }
  }
  function he(e2, t2, n2) {
    let r2 = n2.pointConductanceUnitNS, i2 = 0, a2 = 0;
    for (let o2 of h) {
      let s2 = n2.upstreamScaling === `perContact` ? t2.upstreamCounts[o2] : 1, c2 = e2.conductance[o2] * r2 * s2;
      i2 += c2, a2 += c2 * I[o2];
    }
    return { totalG: i2, totalGE: a2 };
  }
  // Message 'prepare-fit-traces': input-model traces of a single cell for fitting the input model; not used by the
  // circuit runs.
  function fitRawSample(e2) {
    let n2 = e2.paths.preferred, r2 = { Mi9: Math.max(Math.max(0, n2.filters.Mi9.f), x.Mi9 * (1 - A(n2.mean))) };
    for (let i2 of h) {
      if (i2 === `Mi9`) continue;
      let n3 = e2.paths[j(i2)], a2 = Math.max(0, n3.filters[i2].f), o2 = y[i2].drive === `off` ? 1 - A(n3.mean) : A(n3.mean);
      r2[i2] = Math.max(a2, x[i2] * o2);
    }
    return r2;
  }
  function prepareFitTraces(e2, t2) {
    let { targetCell: n2, stimulus: r2, settings: i2 } = e2, a2 = R(i2.dtMs ?? 2, 0.5, 5), o2 = a2 / 1e3, s2 = R(r2.durationSec, 0.12, r2.validationProtocol ? 12 : 64), c2 = Math.max(1, Math.round(s2 / o2)), l2 = { rfFwhmDeg: i2.premotionRfFwhmDeg ?? 3, preStimulusMode: `model-default`, calibration: e2.premotionOptions?.calibration }, u2 = se([n2], r2, a2, l2), d2 = N(r2, 0), f2 = { time: [], Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [], luminancePreferred: [], luminanceHome: [], luminanceNull: [] }, p2 = (e3) => {
      let t3 = fitRawSample(u2[0]);
      f2.time.push(e3 * o2);
      for (let e4 of h) f2[e4].push(t3[e4]);
      f2.luminancePreferred.push(u2[0].paths.preferred.acceptedLuminance), f2.luminanceHome.push(u2[0].paths.home.acceptedLuminance), f2.luminanceNull.push(u2[0].paths.null.acceptedLuminance);
    };
    p2(0);
    for (let e3 = 1; e3 <= c2; e3 += 1) ce(u2, r2, e3 * o2, a2, l2), p2(e3), e3 % Math.max(1, Math.round(c2 / 20)) === 0 && t2?.(e3 / c2, `Preparing upstream traces ${Math.round(100 * e3 / c2)}%`);
    return { targetRootId: n2.rootId, dtMs: a2, durationSec: s2, polarity: d2, trace: f2, meta: { formula: `Mi9=max(Mi9_OFF, tonicFloor*darkMean)`, filterTimeScales: l2.calibration?.filterTimeScales ?? {}, conductanceShiftsMs: l2.calibration?.conductanceShiftsMs ?? {}, rfFwhmDeg: l2.rfFwhmDeg } };
  }
  // V(payload, progress): the circuit simulation. payload = {circuit, targetCell, t4aSkeletons (the skeletons of
  // all T4a and T4b, by root id), stimulus, settings, probes}. targetCell is the T4a whose detailed traces are
  // recorded; it is a readout role only and does not change the model.
  function V(payload, progress) {
    let simulationStartedMs = performance.now();
    let {
      circuit,
      targetCell,
      targetSkeleton,
      t4aSkeletons,
      stimulus,
      settings,
      probes = [],
    } = payload;
    // Hemicircuits, the flat T4a list and the H2 cells.
    let binocular = normalizeBinocularCircuit(circuit);
    circuit = binocular.circuit;
    let hemis = binocular.hemis;
    let hemiCount = hemis.length;
    let h2Enabled = settings.h2Enabled !== false && binocular.h2Units.length > 0;
    let h2Units = h2Enabled ? binocular.h2Units : [];
    let premotionVariant = settings.premotionVariant ?? `speedTuned`;
    let premotionOptions = {
      rfFwhmDeg: settings.premotionRfFwhmDeg ?? 3,
      preStimulusMode: `model-default`,
      calibration: le(premotionVariant),
      ...payload.premotionOptions,
    };
    // Cable parameters of T4a and T4b (settings targetRaOhmCm, targetRmOhmCm2, targetCmUfPerCm2).
    let targetCableOptions = {
      ...e,
      raOhmCm: R(Number(settings.targetRaOhmCm ?? e.raOhmCm), 10, 5e3),
      rmOhmCm2: R(Number(settings.targetRmOhmCm2 ?? e.rmOhmCm2), 100, 1e7),
      cmFPerCm2: R(
        Number(settings.targetCmUfPerCm2 ?? e.cmFPerCm2 * 1e6),
        0.01,
        100,
      ) * 1e-6,
    };
    let targetIndex = circuit.cells.findIndex((cell) => cell.rootId === targetCell.rootId);
    if (targetIndex < 0) throw Error(`Selected target is not part of the VCH-connected T4a set.`);
    // Skeletons by root id; every T4a and T4b needs one.
    let skeletonByRoot = new Map();
    if (t4aSkeletons && !Array.isArray(t4aSkeletons)) {
      for (let [rootId, skeleton] of Object.entries(t4aSkeletons)) {
        skeletonByRoot.set(String(rootId), skeleton);
      }
    } else if (Array.isArray(t4aSkeletons)) {
      for (let entry of t4aSkeletons) {
        if (entry?.rootId && Array.isArray(entry.nodes)) {
          skeletonByRoot.set(String(entry.rootId), entry.nodes);
        }
      }
    }
    if (!skeletonByRoot.has(String(targetCell.rootId)) && Array.isArray(targetSkeleton)) {
      skeletonByRoot.set(String(targetCell.rootId), targetSkeleton);
    }
    let missingSkeletons = circuit.cells.filter(
      (cell) => !skeletonByRoot.has(String(cell.rootId)),
    );
    if (missingSkeletons.length) {
      throw Error(`All-cable v4 is missing ${missingSkeletons.length} T4a skeletons; first missing ${missingSkeletons[0].rootId}.`);
    }
    // One cable per T4a and one per vCH.
    let t4Kernels = circuit.cells.map((cell) => r(
      skeletonByRoot.get(String(cell.rootId)),
      targetCableOptions,
    ));
    let t4States = t4Kernels.map((kernel) => de(kernel));
    let targetKernel = t4Kernels[targetIndex];
    let hemiOfT4a = Int32Array.from(circuit.cells, (cell) => cell.hemiIndex | 0);
    let vchKernels = hemis.map((hemi) => r(hemi.vch.skeleton.map(g), t));
    let vchStates = vchKernels.map((kernel) => de(kernel));
    let targetHemi = hemiOfT4a[targetIndex];
    // Singular names (vchKernel, vchState, ...) refer to the target cell's hemicircuit; they feed the single-cell
    // traces of the result.
    let vchKernel = vchKernels[targetHemi];
    let vchState = vchStates[targetHemi];
    let targetState = t4States[targetIndex];
    progress?.(0.05, `Built ${t4States.length} T4a cable models`);

    // Map every T4a synapse onto compartments.
    let mapping = pe(circuit, targetCell, targetKernel, vchKernels, hemiOfT4a);
    let t4Mappings = circuit.cells.map((cell, cellIndex) => (
      cellIndex === targetIndex ? {
        upstreamTargetByType: mapping.upstreamTargetByType,
        targetToVchTargetCompartments: mapping.targetToVchTargetCompartments,
        vchToTargetTargetCompartments: mapping.vchToTargetTargetCompartments,
      } : mapT4aCable(cell, t4Kernels[cellIndex])
    ));
    let t4Readouts = t4Kernels.map((kernel, cellIndex) => (
      selectTargetAnatomicalReadouts(kernel, t4Mappings[cellIndex])
    ));
    progress?.(0.15, `Mapped all T4a, VCH and premotion contacts to cable compartments`);

    // Time step (0.5-5 ms), run length and output interval (settings.outputDtMs).
    let dtMs = R(settings.dtMs, 0.5, 5);
    let dtSec = dtMs / 1e3;
    let durationSec = R(stimulus.durationSec, 0.12, stimulus.validationProtocol ? 12 : 64);
    let stepCount = Math.max(1, Math.round(durationSec / dtSec));
    let outputEverySteps = Math.max(1, Math.round(settings.outputDtMs / dtMs));

    // T4b (the cells with at least settings.minT4bLpiSynapses synapses onto LPi15) and LPi15. Every T4b is a cable
    // driven by the same input model as T4a.
    let t4bEnabled = settings.t4bEnabled !== false && Array.isArray(circuit.t4b);
    let t4bCells = t4bEnabled
      ? circuit.t4b.filter((cell) => (cell.lpi15Synapses ?? 0) >= (settings.minT4bLpiSynapses ?? 1))
      : [];
    let lpi15Enabled = settings.lpi15Enabled !== false;
    let lpi15Cells = hemis.map((hemi) => (lpi15Enabled ? (hemi.lpi15 ?? null) : null));
    let lpi15Cell = lpi15Cells[targetHemi];
    let hemiOfT4b = Int32Array.from(t4bCells, (cell) => cell.hemiIndex | 0);
    let t4aCount = circuit.cells.length;
    let t4bCount = t4bCells.length;
    if (settings.t4bModelVariant !== `multicompartment-real-synapses`) {
      throw Error(`V7 requires every T4b to use the multicompartment-real-synapses model.`);
    }
    let t4bMulticompartment = true;
    let t4bOpticFlowFallback = false;
    if (t4bMulticompartment) {
      let missingT4bSkeletons = t4bCells.filter(
        (cell) => !skeletonByRoot.has(String(cell.rootId)),
      );
      if (missingT4bSkeletons.length) {
        throw Error(
          `V7 is missing ${missingT4bSkeletons.length} T4b skeletons; first missing ${missingT4bSkeletons[0].rootId}.`,
        );
      }
      let missingContactMappings = t4bCells.filter((cell) => (
        !Array.isArray(cell.upstreamContacts)
        || !Array.isArray(cell.vchToT4bCableContacts)
        || !Array.isArray(cell.t4bToLpi15CableContacts)
      ));
      if (missingContactMappings.length) {
        throw Error(
          `V7 is missing exact T4b contact maps; first missing ${missingContactMappings[0].rootId}.`,
        );
      }
    }
    let t4bKernels = t4bMulticompartment
      ? t4bCells.map((cell) => r(
        skeletonByRoot.get(String(cell.rootId)),
        targetCableOptions,
      ))
      : [];
    let t4bStates = t4bKernels.map((kernel) => de(kernel));
    let t4bMappings = t4bMulticompartment
      ? t4bCells.map((cell, cellIndex) => mapT4bCable(cell, t4bKernels[cellIndex]))
      : [];
    let t4bReadouts = t4bMulticompartment
      ? t4bKernels.map((kernel, cellIndex) => (
        selectTargetAnatomicalReadouts(kernel, t4bMappings[cellIndex])
      ))
      : [];
    let allT4CableStates = t4bMulticompartment
      ? t4States.concat(t4bStates)
      : t4States;
    let allT4CableMappings = t4bMulticompartment
      ? t4Mappings.concat(t4bMappings)
      : t4Mappings;
    if (t4bMulticompartment) {
      progress?.(
        0.155,
        `Built ${t4bStates.length} T4b cable models with exact FlyWire contact maps`,
      );
    }
    // Inactive: optic-flow drive of point-model T4b (opticflow-model-v17.js); t4bOpticFlowFallback is false.
    let t4bOpticFlowSettings = Object.freeze({
      inputTauMS: 20,
      opticFlowRfFwhmDeg: 3,
      opticFlowDelayMS: 0,
      opticFlowLowPassTauMS: 80,
      opticFlowEdgeWindowMS: 20,
      opticFlowEdgeGain: 8,
      opticFlowGain: 2,
      opticFlowVelocity50DPS: 60,
      opticFlowSpeedModel: `referenceNormalizedPdLowPassNdBoundedLogQuadratic`,
      opticFlowReferenceSpeedDPS: 30,
      opticFlowReferenceFactor: 1 / 3,
      opticFlowSpeedHalfDPS: 22.184142066,
      opticFlowSpeedExponent: 5.792829505,
      opticFlowSpeedFloor: 0.85239462,
      opticFlowNullLogLinear: 0.089351016,
      opticFlowNullLogQuadratic: 0.610548953,
      opticFlowNullLogOffsetDPS: 3,
      opticFlowNullOnsetHalfDPS: 3,
      opticFlowNullOnsetExponent: 2,
      opticFlowNullFactorCap: 4,
      opticFlowNullLeak: 0.132908605,
    });
    let t4bOpticFlowArrays = t4bOpticFlowFallback
      ? precomputeOpticFlowCells(t4bCells, -1)
      : null;
    if (t4bOpticFlowFallback && t4bOpticFlowArrays.missingPremotionRootIds.length) {
      throw Error(
        `T4b optic-flow fallback is missing premotion coordinates for ${t4bOpticFlowArrays.missingPremotionRootIds.length} cells.`,
      );
    }
    let t4bOpticFlowCache = t4bOpticFlowFallback
      ? createOpticFlowDriveCache(
        stimulus,
        t4bOpticFlowArrays,
        dtSec,
        t4bOpticFlowSettings,
      )
      : null;
    let t4bOpticFlowFilter = t4bOpticFlowFallback
      ? createOpticFlowTemporalFilter(t4bCount, dtSec, t4bOpticFlowSettings)
      : null;
    let t4bOpticExternalConductanceNS = 1.264404297;
    // All T4 in one input-model population: T4a are cells [0, t4aCount), T4b [t4aCount, populationCount). A T4b's
    // opposite preferred direction comes from its own input offset in the circuit file.
    let premotionCells = t4bCount ? circuit.cells.concat(t4bCells) : circuit.cells;
    let populationCount = premotionCells.length;

    // The input model runs either before the simulation in a worker pool (__YIFEI_PRECOMPUTE_PREMOTION__, installed by
    // run-v4.mjs) or step by step below.
    let precomputedPremotion = null;
    if (typeof globalThis.__YIFEI_PRECOMPUTE_PREMOTION__ === `function`) {
      try {
        precomputedPremotion = globalThis.__YIFEI_PRECOMPUTE_PREMOTION__({
          premotionCells, stimulus, dtMs, stepCount, premotionOptions, targetIndex,
        });
      } catch {
        precomputedPremotion = null;
      }
    }
    let populationPremotionStates = precomputedPremotion
      ? null
      : se(premotionCells, stimulus, dtMs, premotionOptions);
    // Per-type conductance delays (all 0 in the calibration used) and a ring buffer of past conductance frames.
    let conductanceDelaySteps = Object.fromEntries(h.map((kind) => [
      kind,
      Math.max(0, Math.round(M(kind, premotionOptions.calibration) / dtMs)),
    ]));
    let c3ConductanceShiftMs = M(`C3`, premotionOptions.calibration);
    let c3LookaheadStep = ie(0, stepCount, `C3`, dtMs, premotionOptions.calibration);
    let maximumDelaySteps = Math.max(0, ...Object.values(conductanceDelaySteps));
    let premotionHistoryLength = c3LookaheadStep + maximumDelaySteps + 2;
    let conductanceHistory = precomputedPremotion ? null : Object.fromEntries(h.map((kind) => [
      kind,
      Array.from(
        { length: premotionHistoryLength },
        () => new Float64Array(populationCount),
      ),
    ]));
    let targetActivityHistory = precomputedPremotion ? null : Object.fromEntries(h.map((kind) => [
      kind,
      new Float64Array(premotionHistoryLength),
    ]));
    let targetAcceptedHistory = precomputedPremotion ? null : Object.fromEntries(
      [`home`, `null`, `preferred`].map((path) => [path, new Float64Array(premotionHistoryLength)]),
    );
    let targetMi9OffHistory = precomputedPremotion ? null : new Float64Array(premotionHistoryLength);
    let targetMi9OnHistory = precomputedPremotion ? null : new Float64Array(premotionHistoryLength);
    let targetMi9DarkHistory = precomputedPremotion ? null : new Float64Array(premotionHistoryLength);
    let writePremotionHistory = (historyStep) => {
      let slot = historyStep % premotionHistoryLength;
      for (let kind of h) {
        let frame = conductanceHistory[kind][slot];
        for (let cellIndex = 0; cellIndex < populationCount; cellIndex += 1) {
          frame[cellIndex] = populationPremotionStates[cellIndex].conductance[kind];
        }
        targetActivityHistory[kind][slot] =
          populationPremotionStates[targetIndex].activity[kind];
      }
      let targetStateAtStep = populationPremotionStates[targetIndex];
      for (let path of [`home`, `null`, `preferred`]) {
        targetAcceptedHistory[path][slot] = targetStateAtStep.paths[path].acceptedLuminance;
      }
      let preferredPath = targetStateAtStep.paths.preferred;
      targetMi9OffHistory[slot] = Math.max(0, preferredPath.filters.Mi9.f);
      targetMi9OnHistory[slot] = Math.max(0, preferredPath.filters.Mi1.f);
      targetMi9DarkHistory[slot] = preferredPath.mean;
    };
    precomputedPremotion || writePremotionHistory(0);
    let premotionCursor = precomputedPremotion ? stepCount : 0;
    for (; premotionCursor < c3LookaheadStep; ) {
      premotionCursor += 1;
      ce(populationPremotionStates, stimulus, premotionCursor * dtSec, dtMs, premotionOptions);
      writePremotionHistory(premotionCursor);
    }
    precomputedPremotion
      ? progress?.(0.18, `Precomputed population premotion on Engaging CPUs`)
      : c3LookaheadStep > 0 && progress?.(0.18, `Prepared population C3 look-ahead in the browser`);

    let populationPremotion = premotionCells.map((cell) => ({
      cell,
      conductance: { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 },
    }));
    let targetPremotion = {
      ...populationPremotion[targetIndex],
      activity: { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 },
      paths: {
        home: { acceptedLuminance: 0 },
        null: { acceptedLuminance: 0 },
        preferred: {
          acceptedLuminance: 0,
          mean: 0,
          filters: { Mi9: { f: 0 }, Mi1: { f: 0 } },
        },
      },
    };
    populationPremotion[targetIndex] = targetPremotion;
    // The conductance frame (one value per cell) each input type reads at the current step; applyPremotionHistory
    // selects them and copies the target cell's input-model values for its traces.
    let activeConductanceFrames = new Array(h.length);
    let applyPremotionHistory = (step, applyConductanceShifts = true) => {
      let currentSlot = precomputedPremotion ? step : step % premotionHistoryLength;
      for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
        let kind = h[kindIndex];
        let sourceStep = applyConductanceShifts && kind === `C3`
          ? ie(step, stepCount, kind, dtMs, premotionOptions.calibration)
          : Math.max(0, step - (applyConductanceShifts ? conductanceDelaySteps[kind] : 0));
        let frame = precomputedPremotion
          ? precomputedPremotion.conductance[kindIndex].subarray(
            sourceStep * populationCount,
            (sourceStep + 1) * populationCount,
          )
          : conductanceHistory[kind][sourceStep % premotionHistoryLength];
        activeConductanceFrames[kindIndex] = frame;
        targetPremotion.conductance[kind] = frame[targetIndex];
        targetPremotion.activity[kind] = precomputedPremotion
          ? precomputedPremotion.targetActivity[kindIndex][currentSlot]
          : targetActivityHistory[kind][currentSlot];
      }
      let targetPaths = [`home`, `null`, `preferred`];
      for (let pathIndex = 0; pathIndex < targetPaths.length; pathIndex += 1) {
        let path = targetPaths[pathIndex];
        targetPremotion.paths[path].acceptedLuminance = precomputedPremotion
          ? precomputedPremotion.targetAccepted[pathIndex][currentSlot]
          : targetAcceptedHistory[path][currentSlot];
      }
      targetPremotion.paths.preferred.filters.Mi9.f = precomputedPremotion
        ? precomputedPremotion.targetMi9Off[currentSlot] : targetMi9OffHistory[currentSlot];
      targetPremotion.paths.preferred.filters.Mi1.f = precomputedPremotion
        ? precomputedPremotion.targetMi9On[currentSlot] : targetMi9OnHistory[currentSlot];
      targetPremotion.paths.preferred.mean = precomputedPremotion
        ? precomputedPremotion.targetMi9Dark[currentSlot] : targetMi9DarkHistory[currentSlot];
    };
    applyPremotionHistory(0, false);

    // Readout compartments of the target cell.
    let targetReadouts = t4Readouts[targetIndex];
    let targetInputCompartments = targetReadouts.allInputCompartments;
    let targetOutputCompartments = Array.from(new Set(
      mapping.targetToVchTargetCompartments.filter((compartment) => (
        Number.isInteger(compartment) && compartment >= 0 && compartment < targetKernel.nodes.length
      )),
    ));
    let pointELeakMv = -65;
    // Release curves, one per presynaptic cell type: T4 (all T4a outputs), vCH (all vCH outputs; vCH -> LPi15 can
    // differ) and LPi15; T4b -> LPi15, T4b -> H2 and H2 follow below. Several settings keys are accepted for each
    // parameter; the first one that is set wins.
    let excitatoryReleaseThresholdMv = Number(
      settings.t4ReleaseThresholdMv
        ?? settings.excitatoryReleaseThresholdMv ?? settings.releaseThresholdMv ?? -56,
    );
    let excitatoryReleaseGainPerMv = Math.max(0, Number(
      settings.t4ReleaseGainInsidePerMv
        ?? settings.excitatoryReleaseGainPerMv ?? settings.releaseGainPerMv ?? 0.25,
    ));
    let excitatoryReleaseExponent = Math.max(0.05, Number(
      settings.t4ReleaseExponent ?? settings.excitatoryReleaseExponent ?? 1.5,
    ));
    let inhibitoryReleaseThresholdMv = Number(
      settings.vchReleaseThresholdMv
        ?? settings.targetInhibitoryReleaseThresholdMv
        ?? settings.inhibitoryReleaseThresholdMv ?? -55,
    );
    let inhibitoryReleaseGainPerMv = Math.max(0, Number(
      settings.vchReleaseGainInsidePerMv
        ?? settings.targetInhibitoryReleaseGainPerMv
        ?? settings.inhibitoryReleaseGainPerMv ?? 0.03,
    ));
    let inhibitoryReleaseExponent = Math.max(0.05, Number(
      settings.vchReleaseExponent
        ?? settings.targetInhibitoryReleaseExponent
        ?? settings.inhibitoryReleaseExponent ?? 1,
    ));
    let lpi15ReleaseThresholdMv = Number(
      settings.lpi15ReleaseThresholdMv ?? settings.lpi15ToVchReleaseThresholdMv ?? -55,
    );
    let lpi15ReleaseGainPerMv = Math.max(0, Number(
      settings.lpi15ReleaseGainInsidePerMv ?? settings.inhibitoryReleaseGainPerMv ?? 0.06,
    ));
    let lpi15ReleaseExponent = Math.max(0.05, Number(
      settings.lpi15ReleaseExponent ?? settings.inhibitoryReleaseExponent ?? 1,
    ));
    let vchToLpi15ReleaseThresholdMv = Number(
      settings.vchToLpi15ReleaseThresholdMv ?? inhibitoryReleaseThresholdMv,
    );
    let vchToLpi15ReleaseGainPerMv = Math.max(0, Number(
      settings.vchToLpi15ReleaseGainInsidePerMv ?? inhibitoryReleaseGainPerMv,
    ));
    let vchToLpi15ReleaseExponent = Math.max(0.05, Number(
      settings.vchToLpi15ReleaseExponent ?? inhibitoryReleaseExponent,
    ));
    // Release follows the presynaptic voltage without delay (the low-pass filter on the vCH release voltage is off).
    let inhibitoryReleaseTauMs = 0;
    let targetInhibitoryReleaseTauMs = 0;
    // T4b -> LPi15 release curve (also used for T4b -> vCH); defaults to the T4 curve.
    let t4bToLpi15ReleaseThresholdMv = Number(
      settings.t4bToLpi15ReleaseThresholdMv ?? excitatoryReleaseThresholdMv,
    );
    let t4bToLpi15ReleaseGainPerMv = Math.max(0, Number(
      settings.t4bToLpi15ReleaseGainInsidePerMv ?? excitatoryReleaseGainPerMv,
    ));
    let t4bToLpi15ReleaseExponent = Math.max(0.05, Number(
      settings.t4bToLpi15ReleaseExponent ?? excitatoryReleaseExponent,
    ));
    let lpi15ToVchReleaseThresholdMv = lpi15ReleaseThresholdMv;
    let targetInhibitoryContactScale = 1;
    let targetInhibitoryReleaseThresholdMv = inhibitoryReleaseThresholdMv;
    let targetInhibitoryReleaseGainPerMv = inhibitoryReleaseGainPerMv;
    let targetInhibitoryReleaseExponent = inhibitoryReleaseExponent;
    // Conductance per synapse at full release (nS).
    let t4ToVchPerContactNS = Math.max(0, Number(
      settings.t4ToVchPerContactNS ?? settings.excitatoryPerContactNS ?? 0.1,
    ));
    let vchToT4PerContactNS = Math.max(0, Number(
      settings.vchToT4PerContactNS
        ?? settings.inhibitoryPerContactNS ?? 0.3,
    ));
    // Single-compartment equivalents of the target cell: used only by the inactive point-model T4b and reported in
    // the result.
    let targetCableTotalLeakNS = targetKernel.leakTotalS * 1e9;
    let targetCableTotalCapacitancePF = targetKernel.capacitance.reduce(
      (sum, value) => sum + value,
      0,
    ) * 1e12;
    let targetEquivalentConductanceUnitNS = targetKernel.leakTotalS / 0.5 * 1e9;
    let targetShadowLeakNS = Math.max(1e-9, targetCableTotalLeakNS);
    let targetShadowTauMS = Math.max(
      1e-9,
      targetCableTotalCapacitancePF / targetShadowLeakNS,
    );
    let t4bPointLeakNS = t4bOpticFlowFallback ? 0.767556973609843 : targetShadowLeakNS;
    let t4bPointTauMS = t4bOpticFlowFallback ? 17 : targetShadowTauMS;
    let t4bPointELeakMv = t4bOpticFlowFallback ? -67.17 : pointELeakMv;
    let upstreamConductanceUnitNS = targetEquivalentConductanceUnitNS;
    let upstreamIsPerContact = settings.upstreamScaling === `perContact`;
    let pointVoltage = new Float64Array(populationCount).fill(pointELeakMv / 1e3);
    let pointNext = new Float64Array(populationCount).fill(pointELeakMv / 1e3);
    if (t4bOpticFlowFallback) {
      pointVoltage.fill(t4bPointELeakMv / 1e3, t4aCount);
      pointNext.fill(t4bPointELeakMv / 1e3, t4aCount);
    }
    // Per-cell totals of the visual input and of the inhibitory feedback (nS), for the traces.
    let rawOpticDriveByCell = new Float64Array(populationCount);
    let opticDriveByCell = new Float64Array(populationCount);
    let upstreamConductanceTimesReversalByCell = new Float64Array(populationCount);
    let feedbackConductanceNS = new Float64Array(populationCount);
    let feedbackConductanceTimesReversal = new Float64Array(populationCount);
    // vCH voltage used for release (equal to the voltage while the release time constants are 0).
    let filteredVchReleaseVoltageByHemi = vchKernels.map(
      (kernel) => new Float64Array(kernel.nodes.length).fill(-65 / 1e3),
    );
    let filteredTargetVchReleaseVoltageByHemi = vchKernels.map(
      (kernel) => new Float64Array(kernel.nodes.length).fill(-65 / 1e3),
    );
    let filteredVchReleaseVoltage = filteredVchReleaseVoltageByHemi[targetHemi];
    let filteredTargetVchReleaseVoltage = filteredTargetVchReleaseVoltageByHemi[targetHemi];

    // LPi15 -> vCH conductance per synapse (nS).
    let lpi15ToVchPerContactNS = Math.max(0, Number(settings.lpi15ToVchPerContactNS ?? 0.005));
    // Reversal potentials (mV) of the vCH and the LPi15 outputs. Default -75; the paper uses -68 mV, the GABA
    // reversal measured in T4 by Groschner et al. 2022.
    let vchToT4ReversalMv = Number(settings.vchToT4ReversalMv ?? -75);
    let lpi15ReversalMv = Number(settings.lpi15ReversalMv ?? -75);
    // Optional separate reversal for vCH -> T4a alone (default: vchToT4ReversalMv).
    let vchToT4aReversalMv = Number(settings.vchToT4aReversalMv ?? vchToT4ReversalMv);
    // Point-model LPi15 (only for a one-eye circuit without an LPi15 skeleton).
    let lpi15DriveNS = Math.max(0, Number(settings.lpi15DriveNS ?? 1));
    let lpi15LeakNS = Math.max(1e-6, Number(
      settings.lpi15LeakNS ?? 0.5 * Number(
        settings.pointConductanceUnitNS ?? targetEquivalentConductanceUnitNS,
      ),
    ));
    let lpi15TauMs = Math.max(0.1, Number(settings.lpi15TauMs ?? settings.pointTauMs ?? 17));
    let lpi15VchCompartments = lpi15Cell
      ? lpi15Cell.vchContacts.map((p) => i(vchKernel.nodes, p[0], p[1], p[2]))
      : [];

    // LPi15 as a cable (required for two-eye circuits). Every T4b -> LPi15, LPi15 -> vCH and vCH -> LPi15 synapse
    // is read and delivered at its own compartments.
    let lpi15Multi = settings.lpi15Multicompartment !== false
      && Boolean(lpi15Cell?.skeleton?.length);
    if (binocular.binocular && !lpi15Multi) {
      throw Error(`V0 binocular requires lpi15Multicompartment; the point-LPi15 path is monocular only.`);
    }
    let lpi15Kernels = new Array(hemiCount).fill(null);
    let lpi15StatesByHemi = new Array(hemiCount).fill(null);
    let lpi15ReleaseCompsByHemi = hemis.map(() => []);
    let lpi15ReleaseVchCompsByHemi = hemis.map(() => []);
    let lpi15FbVchCompsByHemi = hemis.map(() => []);
    let lpi15FbLpiCompsByHemi = hemis.map(() => []);
    let lpi15Kernel = null;
    let lpi15State = null;
    let lpi15InputCompsByT4b = [];
    let lpi15ReleaseComps = [];
    let lpi15ReleaseVchComps = [];
    let lpi15FbVchComps = [];
    let lpi15FbLpiComps = [];
    let t4bToLpi15PerContactNS = Math.max(0, Number(settings.t4bToLpi15PerContactNS ?? 0.02));
    let vchToLpi15PerContactNS = Math.max(0, Number(
      settings.vchToLpi15PerContactNS ?? settings.inhibitoryPerContactNS ?? 0.3,
    ));
    if (lpi15Multi) {
      let lpi15CableOptions = {
        ...t,
        raOhmCm: R(Number(settings.lpi15RaOhmCm ?? t.raOhmCm), 10, 5e3),
        rmOhmCm2: R(Number(settings.lpi15RmOhmCm2 ?? t.rmOhmCm2), 100, 1e7),
        cmFPerCm2: R(Number(settings.lpi15CmUfPerCm2 ?? t.cmFPerCm2 * 1e6), 0.01, 100) * 1e-6,
      };
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        let cell = lpi15Cells[hemiIndex];
        if (!cell?.skeleton?.length) continue;
        lpi15Kernels[hemiIndex] = r(cell.skeleton.map(g), lpi15CableOptions);
        lpi15StatesByHemi[hemiIndex] = de(lpi15Kernels[hemiIndex]);
      }
      lpi15Kernel = lpi15Kernels[targetHemi];
      lpi15State = lpi15StatesByHemi[targetHemi];
      // Each T4b synapses onto the LPi15 of its own hemicircuit.
      lpi15InputCompsByT4b = t4bCells.map((cell, k) => {
        let kernel = lpi15Kernels[hemiOfT4b[k]];
        if (!kernel) return [];
        return t4bMulticompartment
          ? (cell.t4bToLpi15CableContacts ?? []).map(
            (p) => i(kernel.nodes, p[3], p[4], p[5]),
          )
          : (cell.lpi15Contacts ?? []).map(
            (p) => i(kernel.nodes, p[0], p[1], p[2]),
          );
      });
      if (t4bMulticompartment) {
        for (let k = 0; k < t4bCount; k += 1) {
          if (
            lpi15InputCompsByT4b[k].length
            !== t4bMappings[k].targetToVchTargetCompartments.length
          ) {
            throw Error(`T4b -> LPi15 paired contact mismatch for ${t4bCells[k].rootId}.`);
          }
        }
      }
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        let kernel = lpi15Kernels[hemiIndex];
        let cell = lpi15Cells[hemiIndex];
        if (!kernel || !cell) continue;
        let vchNodes = vchKernels[hemiIndex].nodes;
        let outputs = cell.vchOutputs ?? [];
        lpi15ReleaseCompsByHemi[hemiIndex] = outputs.map((o) => i(kernel.nodes, o[0], o[1], o[2]));
        lpi15ReleaseVchCompsByHemi[hemiIndex] = outputs.map((o) => i(vchNodes, o[3], o[4], o[5]));
        let inputs = cell.vchInputs ?? [];
        lpi15FbVchCompsByHemi[hemiIndex] = inputs.map((o) => i(vchNodes, o[0], o[1], o[2]));
        lpi15FbLpiCompsByHemi[hemiIndex] = inputs.map((o) => i(kernel.nodes, o[3], o[4], o[5]));
      }
      lpi15ReleaseComps = lpi15ReleaseCompsByHemi[targetHemi];
      lpi15ReleaseVchComps = lpi15ReleaseVchCompsByHemi[targetHemi];
      lpi15FbVchComps = lpi15FbVchCompsByHemi[targetHemi];
      lpi15FbLpiComps = lpi15FbLpiCompsByHemi[targetHemi];
      progress?.(0.16, `Built ${lpi15Kernels.filter(Boolean).length} LPi15 cable(s) (${lpi15Kernel.nodes.length} compartments each side)`);
    }
    // Compartments for the LPi15 -> vCH traces (each counted once).
    let lpi15VchRecipientCompartments = Array.from(new Set(
      lpi15ReleaseVchComps.length ? lpi15ReleaseVchComps : lpi15VchCompartments,
    ));
    let lpi15VchReleaseCompartments = Array.from(new Set(lpi15ReleaseComps));
    // LPi15 -> T4a (settings.lpi15ToT4Enabled; on in the paper). These synapses sit on the lobula-plate part of the
    // T4a arbor, where vCH contacts T4a too.
    let lpi15ToT4Enabled = settings.lpi15ToT4Enabled === true && Boolean(lpi15Cell);
    let lpi15ToT4PerContactNS = Math.max(0, Number(
      settings.lpi15ToT4PerContactNS ?? settings.inhibitoryPerContactNS ?? 0.3,
    ));
    let lpi15ToT4ReleaseComps = [];
    let lpi15ToT4CableCompsByCell = [];
    let lpi15ToT4ContactTotal = 0;
    if (lpi15ToT4Enabled) {
      for (let ci = 0; ci < t4aCount; ci += 1) {
        let contacts = circuit.cells[ci].lpi15ToT4Contacts ?? [];
        let ownLpi15Kernel = lpi15Kernels[hemiOfT4a[ci]];
        lpi15ToT4ContactTotal += contacts.length;
        lpi15ToT4ReleaseComps[ci] = ownLpi15Kernel
          ? contacts.map((q) => i(ownLpi15Kernel.nodes, q[0], q[1], q[2]))
          : new Array(contacts.length).fill(0);
        lpi15ToT4CableCompsByCell[ci] = contacts.map(
          (q) => i(t4Kernels[ci].nodes, q[3], q[4], q[5]),
        );
      }
    }
    // Compartments of each T4 cable that receive synapses. Their conductance and current are reset at the start of
    // every step (the cable arrays are not cleared as a whole).
    let t4TouchedCompartments = allT4CableMappings.map((cellMapping, cellIndex) => {
      let touched = new Set();
      for (let kind of h) {
        for (let compartment of cellMapping.upstreamTargetByType[kind] ?? []) touched.add(compartment);
      }
      for (let compartment of cellMapping.vchToTargetTargetCompartments ?? []) touched.add(compartment);
      if (cellIndex < t4aCount) {
        for (let compartment of lpi15ToT4CableCompsByCell[cellIndex] ?? []) {
          touched.add(compartment);
        }
      }
      if (settings.v2FullConnectivity !== false) {
        // Include the targets of H2 -> T4a, H2 -> T4b and LPi15 -> T4b; otherwise their conductance would not be reset
        // and would accumulate over the run.
        let nodes = cellIndex < t4aCount
          ? t4Kernels[cellIndex]?.nodes
          : t4bKernels[cellIndex - t4aCount]?.nodes;
        let cell = cellIndex < t4aCount
          ? circuit.cells[cellIndex]
          : t4bCells[cellIndex - t4aCount];
        if (nodes && cell) {
          let addTargets = (byRoot) => {
            for (let contacts of Object.values(byRoot ?? {})) {
              for (let q of contacts) touched.add(i(nodes, q[3], q[4], q[5]));
            }
          };
          if (cellIndex < t4aCount) addTargets(cell.h2ToT4aContacts);
          else {
            addTargets(cell.h2ToT4bContacts);
            addTargets(cell.lpi15ToT4bContacts);
          }
        }
      }
      return Int32Array.from(touched);
    });
    // Visual input: the conductance unit of each T4 is its total leak conductance / 0.5. A type's conductance is
    // divided equally among its contacts (settings.upstreamScaling 'perContact' gives each contact the full value).
    let t4InputConductanceUnits = Float64Array.from(
      allT4CableStates,
      (state) => state.kernel.leakTotalS / 0.5,
    );
    let t4InputCompartments = allT4CableMappings.map((cellMapping) => (
      h.map((kind) => Int32Array.from(cellMapping.upstreamTargetByType[kind] ?? []))
    ));
    let t4InputPerContact = settings.upstreamScaling === `perContact`;
    // Optional parallel input assembly, provided by parallel-cable-pool.mjs.
    let parallelT4InputAssembly = null;
    let configuredCableSolver = globalThis.__YIFEI_SOLVE_T4A_CABLES__;
    if (typeof configuredCableSolver?.prepareInputAssembly === `function`) {
      parallelT4InputAssembly = configuredCableSolver.prepareInputAssembly(allT4CableStates, {
        kinds: h,
        reversals: upstreamReversalByIndex,
        mappings: allT4CableMappings,
        touchedCompartments: t4TouchedCompartments,
        upstreamScaling: settings.upstreamScaling,
      });
    }
    // Parallel T4b -> LPi15 assembly: only for point-model T4b (inactive).
    let t4bToLpi15ReleaseFrames = new Float64Array(t4bCount);
    let parallelLpi15InputAssembly = null;
    if (
      lpi15Multi
      && !t4bMulticompartment
      && typeof configuredCableSolver?.prepareLpi15InputAssembly === `function`
    ) {
      parallelLpi15InputAssembly = configuredCableSolver.prepareLpi15InputAssembly(
        allT4CableStates,
        lpi15State,
        {
          contactCompartments: lpi15InputCompsByT4b,
          perContactNS: t4bToLpi15PerContactNS,
          reversalMv: -21,
        },
      );
    }

    // vCH -> T4b synapses: release sites on the vCH of the T4b's hemicircuit.
    let vchToT4bCompartments = t4bCells.map((cell, k) => {
      let nodes = vchKernels[hemiOfT4b[k]].nodes;
      return t4bMulticompartment
        ? (cell.vchToT4bCableContacts ?? []).map((p) => i(nodes, p[0], p[1], p[2]))
        : (cell.vchToT4bContacts ?? []).map((p) => i(nodes, p[0], p[1], p[2]));
    });
    // H2: release curve of its outputs and conductances per synapse (nS) of T4b -> H2 and H2 -> vCH.
    let h2ExcitatoryReleaseThresholdMv = Number(settings.h2ExcitatoryReleaseThresholdMv ?? -55);
    let h2ExcitatoryReleaseGainPerMv = Math.max(0, Number(
      settings.h2ExcitatoryReleaseGainInsidePerMv ?? settings.h2ExcitatoryReleaseGainPerMv ?? 0.25,
    ));
    let h2ExcitatoryReleaseExponent = Math.max(0.05, Number(settings.h2ExcitatoryReleaseExponent ?? 1.5));
    let t4bToH2PerContactNS = Math.max(0, Number(settings.t4bToH2PerContactNS ?? 0.1));
    let h2ToVchPerContactNS = Math.max(0, Number(settings.h2ToVchPerContactNS ?? 0.1));
    // T4b -> H2 release curve (defaults to the H2 curve).
    let t4bToH2ReleaseThresholdMv = Number(
      settings.t4bToH2ReleaseThresholdMv ?? h2ExcitatoryReleaseThresholdMv,
    );
    let t4bToH2ReleaseGainPerMv = Math.max(0, Number(
      settings.t4bToH2ReleaseGainInsidePerMv
        ?? settings.t4bToH2ReleaseGainPerMv ?? h2ExcitatoryReleaseGainPerMv,
    ));
    let t4bToH2ReleaseExponent = Math.max(0.05, Number(
      settings.t4bToH2ReleaseExponent ?? h2ExcitatoryReleaseExponent,
    ));
    // The other measured connections (settings.v2FullConnectivity): T4a -> LPi15, T4a -> H2, H2 -> T4a, T4b -> vCH,
    // vCH -> H2, LPi15 -> T4b, LPi15 -> H2, H2 -> LPi15, H2 -> T4b and H2 -> H2. Each uses the release curve and
    // reversal of its presynaptic cell; conductances per synapse (nS):
    let v2Enabled = settings.v2FullConnectivity !== false;
    let t4aToLpi15PerContactNS = Math.max(0, Number(settings.t4aToLpi15PerContactNS ?? 0.3));
    let t4aToH2PerContactNS = Math.max(0, Number(settings.t4aToH2PerContactNS ?? 0.3));
    let t4bToVchDirectPerContactNS = Math.max(0, Number(settings.t4bToVchDirectPerContactNS ?? 0.0042));
    let vchToH2PerContactNS = Math.max(0, Number(settings.vchToH2PerContactNS ?? 0.3));
    let lpi15ToT4bPerContactNS = Math.max(0, Number(settings.lpi15ToT4bPerContactNS ?? 0.06));
    let lpi15ToH2PerContactNS = Math.max(0, Number(settings.lpi15ToH2PerContactNS ?? 0.06));
    let h2ToLpi15PerContactNS = Math.max(0, Number(settings.h2ToLpi15PerContactNS ?? 0.1));
    let h2ToT4bPerContactNS = Math.max(0, Number(settings.h2ToT4bPerContactNS ?? 0.1));
    let h2ToT4aPerContactNS = Math.max(0, Number(settings.h2ToT4aPerContactNS ?? 0.1));
    let h2ToH2PerContactNS = Math.max(0, Number(settings.h2ToH2PerContactNS ?? 0.1));
    let roleByRoot = binocular.roleByRoot ?? new Map();
    let h2IndexByRoot = new Map(binocular.h2Units.map((u, index) => [String(u.rootId), index]));

    let h2Kernels = [];
    let h2States = [];
    let h2DriveByUnit = [];
    let h2OutputsByUnit = [];
    let t4bIndexByRoot = new Map(t4bCells.map((cell, index) => [String(cell.rootId), index]));
    // H2 cables with their T4b -> H2 and H2 -> vCH synapses.
    if (h2Enabled) {
      let h2CableOptions = {
        ...t,
        raOhmCm: R(Number(settings.h2RaOhmCm ?? t.raOhmCm), 10, 5e3),
        rmOhmCm2: R(Number(settings.h2RmOhmCm2 ?? t.rmOhmCm2), 100, 1e7),
        cmFPerCm2: R(Number(settings.h2CmUfPerCm2 ?? t.cmFPerCm2 * 1e6), 0.01, 100) * 1e-6,
      };
      for (let unit of h2Units) {
        let kernel = r(unit.skeleton.map(g), h2CableOptions);
        h2Kernels.push(kernel);
        h2States.push(de(kernel));
        // T4b -> H2: release site on the T4b, target on H2.
        let drive = [];
        for (let [rootId, contacts] of Object.entries(unit.t4bDrive)) {
          let t4bIndex = t4bIndexByRoot.get(String(rootId));
          if (t4bIndex === undefined || !contacts.length) continue;
          let t4bNodes = t4bKernels[t4bIndex]?.nodes;
          if (!t4bNodes) continue;
          drive.push({
            t4bIndex,
            sourceComps: Int32Array.from(contacts, (c) => i(t4bNodes, c[0], c[1], c[2])),
            h2Comps: Int32Array.from(contacts, (c) => i(kernel.nodes, c[3], c[4], c[5])),
          });
        }
        h2DriveByUnit.push(drive);
        // H2 -> vCH: release site on H2, target on the recipient vCH.
        let outputs = [];
        for (let edge of unit.outputs) {
          if (edge.role.kind !== `VCH`) continue;
          let vchNodes = vchKernels[edge.role.hemiIndex].nodes;
          outputs.push({
            hemiIndex: edge.role.hemiIndex,
            releaseComps: Int32Array.from(edge.contacts, (c) => i(kernel.nodes, c[0], c[1], c[2])),
            targetComps: Int32Array.from(edge.contacts, (c) => i(vchNodes, c[3], c[4], c[5])),
          });
        }
        h2OutputsByUnit.push(outputs);
      }
      progress?.(
        0.165,
        `Built ${h2Kernels.length} H2 cable(s); ${h2DriveByUnit.reduce((sum, d) => sum + d.length, 0)} T4b drive them`,
      );
    }
    let h2Count = h2Kernels.length;

    // Synapse maps of the other connections: src = release compartments on the presynaptic cable, dst = the matching
    // target compartments on the postsynaptic cable.
    let mapPairs = (contacts, srcNodes, dstNodes) => ({
      src: Int32Array.from(contacts, (c) => i(srcNodes, c[0], c[1], c[2])),
      dst: Int32Array.from(contacts, (c) => i(dstNodes, c[3], c[4], c[5])),
    });
    let t4aToLpi15ByCell = [];
    let t4aToH2ByCell = [];
    let h2ToT4aByCell = [];
    let lpi15ToT4bByCell = [];
    let h2ToT4bByCell = [];
    let t4bToVchDirectByCell = [];
    let h2ToLpi15ByUnit = h2Units.map(() => []);
    let lpi15ToH2ByUnit = h2Units.map(() => []);
    let vchToH2ByUnit = h2Units.map(() => []);
    let h2ToH2ByUnit = h2Units.map(() => []);
    let v2Counts = { t4aToLpi15: 0, t4aToH2: 0, h2ToT4a: 0, lpi15ToT4b: 0, h2ToT4b: 0,
      t4bToVchDirect: 0, h2ToLpi15: 0, lpi15ToH2: 0, vchToH2: 0, h2ToH2: 0 };
    if (v2Enabled) {
      for (let ci = 0; ci < t4aCount; ci += 1) {
        let cell = circuit.cells[ci];
        let ownNodes = t4Kernels[ci].nodes;
        let toLpi = [];
        for (let [rootId, contacts] of Object.entries(cell.t4aToLpi15Contacts ?? {})) {
          let role = roleByRoot.get(String(rootId));
          let kernel = role && lpi15Kernels[role.hemiIndex];
          if (!kernel || !contacts.length) continue;
          toLpi.push({ hemiIndex: role.hemiIndex, ...mapPairs(contacts, ownNodes, kernel.nodes) });
          v2Counts.t4aToLpi15 += contacts.length;
        }
        t4aToLpi15ByCell[ci] = toLpi;
        let toH2 = [];
        for (let [rootId, contacts] of Object.entries(cell.t4aToH2Contacts ?? {})) {
          let unitIndex = h2IndexByRoot.get(String(rootId));
          if (unitIndex === undefined || !h2Kernels[unitIndex] || !contacts.length) continue;
          toH2.push({ unitIndex, ...mapPairs(contacts, ownNodes, h2Kernels[unitIndex].nodes) });
          v2Counts.t4aToH2 += contacts.length;
        }
        t4aToH2ByCell[ci] = toH2;
        let fromH2 = [];
        for (let [rootId, contacts] of Object.entries(cell.h2ToT4aContacts ?? {})) {
          let unitIndex = h2IndexByRoot.get(String(rootId));
          if (unitIndex === undefined || !h2Kernels[unitIndex] || !contacts.length) continue;
          fromH2.push({ unitIndex, ...mapPairs(contacts, h2Kernels[unitIndex].nodes, ownNodes) });
          v2Counts.h2ToT4a += contacts.length;
        }
        h2ToT4aByCell[ci] = fromH2;
      }
      for (let k = 0; k < t4bCount; k += 1) {
        let cell = t4bCells[k];
        let ownNodes = t4bKernels[k]?.nodes;
        let fromLpi = [];
        let fromH2 = [];
        if (ownNodes) {
          for (let [rootId, contacts] of Object.entries(cell.lpi15ToT4bContacts ?? {})) {
            let role = roleByRoot.get(String(rootId));
            let kernel = role && lpi15Kernels[role.hemiIndex];
            if (!kernel || !contacts.length) continue;
            fromLpi.push({ hemiIndex: role.hemiIndex, ...mapPairs(contacts, kernel.nodes, ownNodes) });
            v2Counts.lpi15ToT4b += contacts.length;
          }
          for (let [rootId, contacts] of Object.entries(cell.h2ToT4bContacts ?? {})) {
            let unitIndex = h2IndexByRoot.get(String(rootId));
            if (unitIndex === undefined || !h2Kernels[unitIndex] || !contacts.length) continue;
            fromH2.push({ unitIndex, ...mapPairs(contacts, h2Kernels[unitIndex].nodes, ownNodes) });
            v2Counts.h2ToT4b += contacts.length;
          }
          // T4b -> vCH contacts carry no target cell in the circuit file; they go to the vCH of the T4b's own
          // hemicircuit.
          let direct = cell.directVchContacts ?? [];
          if (direct.length) {
            let hemiIndex = hemiOfT4b[k];
            t4bToVchDirectByCell[k] = {
              hemiIndex, ...mapPairs(direct, ownNodes, vchKernels[hemiIndex].nodes),
            };
            v2Counts.t4bToVchDirect += direct.length;
          }
        }
        lpi15ToT4bByCell[k] = fromLpi;
        h2ToT4bByCell[k] = fromH2;
      }
      for (let u = 0; u < h2Count; u += 1) {
        let unit = h2Units[u];
        let ownNodes = h2Kernels[u].nodes;
        for (let edge of unit.outputs) {
          if (edge.role.kind === `LPi15`) {
            let kernel = lpi15Kernels[edge.role.hemiIndex];
            if (!kernel) continue;
            h2ToLpi15ByUnit[u].push({ hemiIndex: edge.role.hemiIndex,
              ...mapPairs(edge.contacts, ownNodes, kernel.nodes) });
            v2Counts.h2ToLpi15 += edge.contacts.length;
          } else if (edge.role.kind === `H2`) {
            let target = h2Kernels[edge.role.h2Index];
            if (!target) continue;
            h2ToH2ByUnit[u].push({ unitIndex: edge.role.h2Index,
              ...mapPairs(edge.contacts, ownNodes, target.nodes) });
            v2Counts.h2ToH2 += edge.contacts.length;
          }
        }
        for (let edge of unit.inputs) {
          if (edge.role.kind === `LPi15`) {
            let kernel = lpi15Kernels[edge.role.hemiIndex];
            if (!kernel) continue;
            lpi15ToH2ByUnit[u].push({ hemiIndex: edge.role.hemiIndex,
              ...mapPairs(edge.contacts, kernel.nodes, ownNodes) });
            v2Counts.lpi15ToH2 += edge.contacts.length;
          } else if (edge.role.kind === `VCH`) {
            vchToH2ByUnit[u].push({ hemiIndex: edge.role.hemiIndex,
              ...mapPairs(edge.contacts, vchKernels[edge.role.hemiIndex].nodes, ownNodes) });
            v2Counts.vchToH2 += edge.contacts.length;
          }
        }
      }
      progress?.(0.168, `v2: wired ${Object.values(v2Counts).reduce((a, b) => a + b, 0)} additional synapses`);
    }

    // Per-step totals and peaks for the result.
    let currentT4bToH2ConductanceNS = new Float64Array(Math.max(1, h2Count));
    let currentH2ToVchConductanceNS = new Float64Array(Math.max(1, h2Count));
    let peakH2VoltageMv = new Float64Array(Math.max(1, h2Count)).fill(-65);
    let peakH2ToVchConductanceNS = 0;
    let peakT4bToH2ConductanceNS = 0;
    let peakH2ReleaseFraction = 0;

    // T4b -> LPi15 synapse counts: weights of the LPi15 drive-fraction trace.
    let t4bLpiWeight = new Float64Array(t4bCount);
    let t4bLpiWeightTotal = 0;
    let t4bLpiWeightTotalByHemi = new Float64Array(hemiCount);
    for (let k = 0; k < t4bCount; k += 1) {
      t4bLpiWeight[k] = t4bCells[k].lpi15Synapses ?? 0;
      t4bLpiWeightTotal += t4bLpiWeight[k];
      t4bLpiWeightTotalByHemi[hemiOfT4b[k]] += t4bLpiWeight[k];
    }
    let lpi15DriveFractionByHemi = new Float64Array(hemiCount);
    let lpi15Voltage = pointELeakMv / 1e3;
    let lpi15DriveFraction = 0;
    let lpi15ConductanceOnVchNS = 0;
    let lpi15ToT4ConductanceNS = 0;
    let vchToT4ConductanceNS = 0;
    if (lpi15Cell) {
      progress?.(0.17, `Mapped LPi15 and ${t4bCount} T4b onto the VCH cable`);
    }

    // Recorded traces (one value per output frame).
    let time = [];
    let targetMeanMv = [];
    let targetCableMeanMv = [];
    let targetRootMv = [];
    let targetInputMeanMv = [];
    let targetDendriteMv = [];
    let targetSomaMv = [];
    let targetAxonMv = [];
    let targetOutputContactMeanMv = [];
    let targetVchFeedbackConductanceNS = [];
    let targetVchContactMeanMv = [];
    let targetVchContactMaxMv = [];
    let targetVchReleaseMeanMv = [];
    let targetVchReleaseMaxMv = [];
    let vchMeanMv = [];
    let lpi15VchReleaseMeanMv = [];
    let lpi15VchContactMeanMv = [];
    let vchRootMv = [];
    // Traces per hemicircuit and per H2.
    let vchMeanMvByHemi = hemis.map(() => []);
    let vchRootMvByHemi = hemis.map(() => []);
    let lpi15MvByHemi = hemis.map(() => []);
    let lpi15SpreadMvByHemi = hemis.map(() => []);
    let lpi15ConductanceOnVchNSByHemi = hemis.map(() => []);
    let t4ToVchConductanceNSByHemi = hemis.map(() => []);
    let vchToT4ConductanceNSByHemi = hemis.map(() => []);
    let vchToLpi15ConductanceNSByHemi = hemis.map(() => []);
    let lpi15ToT4ConductanceNSByHemi = hemis.map(() => []);
    let h2MeanMvByUnit = h2Units.map(() => []);
    let h2RootMvByUnit = h2Units.map(() => []);
    let t4bToH2ConductanceNSByUnit = h2Units.map(() => []);
    let h2ToVchConductanceNSByUnit = h2Units.map(() => []);
    let lpi15VoltageByHemi = new Float64Array(hemiCount).fill(pointELeakMv / 1e3);
    let lpi15SpreadByHemi = new Float64Array(hemiCount);
    let lpi15ConductanceOnVchByHemi = new Float64Array(hemiCount);
    let t4ToVchConductanceByHemi = new Float64Array(hemiCount);
    let vchToT4ConductanceByHemi = new Float64Array(hemiCount);
    let vchToLpi15ConductanceByHemi = new Float64Array(hemiCount);
    let lpi15ToT4ConductanceByHemi = new Float64Array(hemiCount);
    let targetFrames = [];
    let vchFrames = [];
    // recordAllRegions: also record the dendrite and soma voltage of every T4a (the axon is always recorded).
    let recordAllRegions = settings.recordAllRegions === true;
    let regionVoltageByRoot = recordAllRegions
      ? Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, { dendrite: [], soma: [] }]))
      : null;
    let pointVoltageByRoot = Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, []]));
    let pointActivityByRoot = Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, []]));
    let pointReleaseByRoot = Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, []]));
    let rawOpticFlowDriveByRoot = Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, []]));
    let opticFlowDriveByRoot = Object.fromEntries(circuit.cells.map((cell) => [cell.rootId, []]));
    let probeTraces = Object.fromEntries(probes.map((probe) => [probe.id, []]));
    // LPi15 and T4b traces.
    let lpi15Mv = [];
    let lpi15DriveFractionTrace = [];
    let lpi15ConductanceOnVchNSTrace = [];
    let lpi15ToT4ConductanceNSTrace = [];
    let vchToT4ConductanceNSTrace = [];
    let t4ToVchConductanceNSTrace = [];
    let vchToLpi15ConductanceNSTrace = [];
    let lpi15SpreadMvTrace = [];
    let t4bMeanMv = [];
    let t4bReleasingCount = [];
    let t4bVoltageByRoot = Object.fromEntries(t4bCells.map((cell) => [cell.rootId, []]));
    let t4bReleaseByRoot = Object.fromEntries(t4bCells.map((cell) => [cell.rootId, []]));
    let peakLpi15VoltageMv = pointELeakMv;
    let lpi15SpreadMv = 0;
    let peakLpi15SpreadMv = 0;
    let peakLpi15MaxVoltageMv = pointELeakMv;
    let peakLpi15ConductanceOnVchNS = 0;
    let currentT4ToVchConductanceNS = 0;
    let currentVchToLpi15ConductanceNS = 0;
    let peakT4bVoltageMv = pointELeakMv;
    let peakT4bReleasingCount = 0;

    // Peak values for the result summary.
    let peakT4VoltageMv = -65;
    let peakPopulationPointVoltageMv = pointELeakMv;
    let peakPopulationPointRootId = circuit.cells.find((_, index) => index !== targetIndex)?.rootId
      ?? circuit.cells[0]?.rootId
      ?? ``;
    let peakTargetContactVoltageMv = -65;
    let peakTargetShadowVoltageMv = -65;
    let peakTargetRootVoltageMv = -65;
    let peakTargetMeanVoltageMv = -65;
    let peakTargetDendriteVoltageMv = -65;
    let peakTargetSomaVoltageMv = -65;
    let peakTargetAxonVoltageMv = -65;
    let peakTargetOutputContactMeanMv = -65;
    let peakT4ReleaseFraction = 0;
    let peakReleasingT4Count = 0;
    let peakT4ToVchConductanceNS = 0;
    let peakTargetShadowUpstreamConductanceNS = 0;
    let peakRawOpticFlowDrive = 0;
    let peakOpticFlowDrive = 0;
    let peakMeanOpticFlowDrive = 0;
    let peakOpticFlowVisualConductanceNS = 0;
    let everReleased = new Uint8Array(circuit.cells.length);
    let everPremotionDriven = new Uint8Array(populationCount);
    let targetPeakActivity = { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 };
    let targetPeakConductance = { Mi9: 0, Tm3: 0, Mi1: 0, Mi4: 0, C3: 0 };
    let targetAcceptedMin = { home: 1, null: 1, preferred: 1 };
    let targetAcceptedMax = { home: 0, null: 0, preferred: 0 };
    let upstreamTrace = {
      activity: { Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [] },
      conductance: { Mi9: [], Tm3: [], Mi1: [], Mi4: [], C3: [] },
      mi9: { off: [], on: [], dark: [] },
    };

    // Release-site histograms (recording only; reads voltages and changes nothing): for every pathway, a histogram of
    // the presynaptic voltage at all its release sites (0.25-mV bins from -90 mV), one per output frame.
    const __RS_LO = -90, __RS_BIN = 0.25, __RS_N = 320;
    let __rsGroups = null;
    const __rsFrames = [];
    const __rsBuild = () => {
      const groups = new Map();
      const add = (name, state, comps) => {
        if (!state || !comps || !comps.length) return;
        let g = groups.get(name);
        if (!g) { g = { name, refs: [], n: 0 }; groups.set(name, g); }
        g.refs.push({ state, comps: Int32Array.from(comps) });
        g.n += comps.length;
      };
      const hk = (h) => hemis[h]?.key ?? `h${h}`;
      const uk = (u) => `H2-${h2Units[u]?.side ?? u}`;
      for (let ci = 0; ci < circuit.cells.length; ci += 1) {
        const h = hemiOfT4a[ci];
        const n = (mapping.t4ToVchByCell[ci] ?? []).length;
        add(`T4a>vCH|${hk(h)}`, t4States[ci],
          Array.from({ length: n }, (_, c) => t4Mappings[ci].targetToVchTargetCompartments[c]));
        for (const e of (t4aToLpi15ByCell[ci] ?? [])) add(`T4a>LPi15|${hk(e.hemiIndex)}`, t4States[ci], e.src);
        for (const e of (t4aToH2ByCell[ci] ?? [])) add(`T4a>H2|${uk(e.unitIndex)}`, t4States[ci], e.src);
        for (const e of (h2ToT4aByCell[ci] ?? [])) add(`H2>T4a|${uk(e.unitIndex)}`, h2States[e.unitIndex], e.src);
        add(`vCH>T4a|${hk(h)}`, vchStates[h], mapping.vchToT4ByCell[ci]);
        if (lpi15ToT4Enabled && lpi15Multi && lpi15ToT4ReleaseComps[ci]) {
          add(`LPi15>T4a|${hk(h)}`, lpi15StatesByHemi[h], lpi15ToT4ReleaseComps[ci]);
        }
      }
      for (let k = 0; k < t4bCount; k += 1) {
        const h = hemiOfT4b[k];
        if (t4bMulticompartment && lpi15Multi) {
          const n = (lpi15InputCompsByT4b[k] ?? []).length;
          add(`T4b>LPi15|${hk(h)}`, t4bStates[k],
            Array.from({ length: n }, (_, c) => t4bMappings[k].targetToVchTargetCompartments[c]));
        }
        const d = t4bToVchDirectByCell[k];
        if (d) add(`T4b>vCH|${hk(d.hemiIndex)}`, t4bStates[k], d.src);
        for (const e of (h2ToT4bByCell[k] ?? [])) add(`H2>T4b|${uk(e.unitIndex)}`, h2States[e.unitIndex], e.src);
        for (const e of (lpi15ToT4bByCell[k] ?? [])) add(`LPi15>T4b|${hk(e.hemiIndex)}`, lpi15StatesByHemi[e.hemiIndex], e.src);
        add(`vCH>T4b|${hk(h)}`, vchStates[h], vchToT4bCompartments[k]);
      }
      for (let u = 0; u < h2Count; u += 1) {
        for (const e of (h2DriveByUnit[u] ?? [])) {
          add(`T4b>H2|${uk(u)}`, t4bStates[e.t4bIndex], e.sourceComps);
          add(`H2dendrite|${uk(u)}`, h2States[u], e.h2Comps);
        }
        for (const e of (h2OutputsByUnit[u] ?? [])) add(`H2>vCH|${uk(u)}>${hk(e.hemiIndex)}`, h2States[u], e.releaseComps);
        for (const e of (h2ToLpi15ByUnit[u] ?? [])) add(`H2>LPi15|${uk(u)}`, h2States[u], e.src);
        for (const e of (h2ToH2ByUnit[u] ?? [])) add(`H2>H2|${uk(u)}`, h2States[u], e.src);
        for (const e of (lpi15ToH2ByUnit[u] ?? [])) add(`LPi15>H2|${hk(e.hemiIndex)}`, lpi15StatesByHemi[e.hemiIndex], e.src);
        for (const e of (vchToH2ByUnit[u] ?? [])) add(`vCH>H2|${hk(e.hemiIndex)}`, vchStates[e.hemiIndex], e.src);
      }
      for (let h = 0; h < hemiCount; h += 1) {
        if (lpi15Multi && lpi15StatesByHemi[h]) add(`LPi15>vCH|${hk(h)}`, lpi15StatesByHemi[h], lpi15ReleaseCompsByHemi[h]);
        add(`vCH>LPi15|${hk(h)}`, vchStates[h], lpi15FbVchCompsByHemi[h]);
      }
      return Array.from(groups.values());
    };
    const __rsRecord = () => {
      if (!__rsGroups) __rsGroups = __rsBuild();
      const frame = [];
      for (const g of __rsGroups) {
        const hist = new Uint32Array(__RS_N);
        for (const ref of g.refs) {
          const v = ref.state.voltage;
          const cs = ref.comps;
          for (let q = 0; q < cs.length; q += 1) {
            let b = Math.floor((v[cs[q]] * 1e3 - __RS_LO) / __RS_BIN);
            if (!(b >= 0)) b = 0;
            else if (b >= __RS_N) b = __RS_N - 1;
            hist[b] += 1;
          }
        }
        frame.push(hist);
      }
      __rsFrames.push(frame);
    };
    const __rsExport = () => ({
      loMv: __RS_LO, binMv: __RS_BIN, bins: __RS_N,
      groups: (__rsGroups ?? []).map((g) => ({ name: g.name, contacts: g.n })),
      frames: __rsFrames.map((frame) => frame.map((hist) => {
        let a = 0; while (a < hist.length && hist[a] === 0) a += 1;
        let b = hist.length - 1; while (b >= a && hist[b] === 0) b -= 1;
        return a > b ? [0] : [a, ...Array.from(hist.subarray(a, b + 1))];
      })),
    });
    // Record one output frame.
    let captureOutput = (step) => {
      __rsRecord();
      time.push(step * dtSec);
      let targetCableMean = ue(targetState.voltage) * 1e3;
      let targetRoot = targetState.voltage[targetKernel.rootIndex] * 1e3;
      let targetInputMean = targetInputCompartments.length
        ? targetInputCompartments.reduce((sum, compartment) => sum + targetState.voltage[compartment], 0)
          / targetInputCompartments.length * 1e3
        : targetCableMean;
      let targetDendrite = targetReadouts.dendriteCompartments.reduce(
        (sum, compartment) => sum + targetState.voltage[compartment],
        0,
      ) / targetReadouts.dendriteCompartments.length * 1e3;
      let targetSoma = targetState.voltage[targetReadouts.somaCompartment] * 1e3;
      let targetAxon = targetState.voltage[targetReadouts.axonCompartment] * 1e3;
      let targetOutputContactMean = targetOutputCompartments.length
        ? targetOutputCompartments.reduce(
          (sum, compartment) => sum + targetState.voltage[compartment],
          0,
        ) / targetOutputCompartments.length * 1e3
        : targetAxon;
      let preferredPath = targetPremotion.paths.preferred;
      peakTargetMeanVoltageMv = Math.max(peakTargetMeanVoltageMv, targetDendrite);
      peakTargetRootVoltageMv = Math.max(peakTargetRootVoltageMv, targetRoot);
      peakTargetDendriteVoltageMv = Math.max(peakTargetDendriteVoltageMv, targetDendrite);
      peakTargetSomaVoltageMv = Math.max(peakTargetSomaVoltageMv, targetSoma);
      peakTargetAxonVoltageMv = Math.max(peakTargetAxonVoltageMv, targetAxon);
      peakTargetOutputContactMeanMv = Math.max(
        peakTargetOutputContactMeanMv,
        targetOutputContactMean,
      );
      // The 'shadow' fields are kept for older readers; they equal the target's axon voltage.
      peakTargetShadowVoltageMv = Math.max(peakTargetShadowVoltageMv, targetAxon);
      targetMeanMv.push(targetDendrite);
      targetCableMeanMv.push(targetCableMean);
      targetRootMv.push(targetSoma);
      targetInputMeanMv.push(targetInputMean);
      targetDendriteMv.push(targetDendrite);
      targetSomaMv.push(targetSoma);
      targetAxonMv.push(targetAxon);
      targetOutputContactMeanMv.push(targetOutputContactMean);
      targetVchFeedbackConductanceNS.push(feedbackConductanceNS[targetIndex]);
      let targetVchCompartments = mapping.vchToT4ByCell[targetIndex];
      let targetVchContactVoltages = targetVchCompartments.map(
        (compartment) => vchState.voltage[compartment] * 1e3,
      );
      targetVchContactMeanMv.push(targetVchContactVoltages.length
        ? ue(targetVchContactVoltages)
        : ue(vchState.voltage) * 1e3);
      targetVchContactMaxMv.push(targetVchContactVoltages.length
        ? Math.max(...targetVchContactVoltages)
        : vchState.voltage[vchKernel.rootIndex] * 1e3);
      let targetVchReleaseVoltages = targetVchCompartments.map(
        (compartment) => filteredTargetVchReleaseVoltage[compartment] * 1e3,
      );
      targetVchReleaseMeanMv.push(targetVchReleaseVoltages.length
        ? ue(targetVchReleaseVoltages)
        : ue(filteredTargetVchReleaseVoltage) * 1e3);
      targetVchReleaseMaxMv.push(targetVchReleaseVoltages.length
        ? Math.max(...targetVchReleaseVoltages)
        : filteredTargetVchReleaseVoltage[vchKernel.rootIndex] * 1e3);
      vchMeanMv.push(ue(vchState.voltage) * 1e3);
      lpi15VchReleaseMeanMv.push(lpi15Multi && lpi15VchReleaseCompartments.length
        ? ue(lpi15VchReleaseCompartments.map((compartment) => lpi15State.voltage[compartment])) * 1e3
        : lpi15Voltage * 1e3);
      lpi15VchContactMeanMv.push(lpi15VchRecipientCompartments.length
        ? ue(lpi15VchRecipientCompartments.map((compartment) => vchState.voltage[compartment])) * 1e3
        : ue(vchState.voltage) * 1e3);
      vchRootMv.push(vchState.voltage[vchKernel.rootIndex] * 1e3);
      targetFrames.push(Array.from(targetState.voltage, (value) => value * 1e3));
      vchFrames.push(Array.from(vchState.voltage, (value) => value * 1e3));
      for (let kind of h) {
        upstreamTrace.activity[kind].push(targetPremotion.activity[kind]);
        upstreamTrace.conductance[kind].push(targetPremotion.conductance[kind]);
      }
      upstreamTrace.mi9.off.push(Math.max(0, preferredPath.filters.Mi9.f));
      upstreamTrace.mi9.on.push(Math.max(0, preferredPath.filters.Mi1.f));
      upstreamTrace.mi9.dark.push(1 - A(preferredPath.mean));
      for (let cellIndex = 0; cellIndex < circuit.cells.length; cellIndex += 1) {
        let rootId = circuit.cells[cellIndex].rootId;
        let cableState = t4States[cellIndex];
        let axonCompartment = t4Readouts[cellIndex].axonCompartment;
        let voltageV = cableState.voltage[axonCompartment];
        let voltageMv = voltageV * 1e3;
        // pointVoltageByRoot: the axon voltage of every T4a (the name is historical).
        pointVoltageByRoot[rootId].push(voltageMv);
        if (recordAllRegions) {
          let readouts = t4Readouts[cellIndex];
          let dendriteComps = readouts.dendriteCompartments;
          let dendriteSum = 0;
          for (let q = 0; q < dendriteComps.length; q += 1) {
            dendriteSum += cableState.voltage[dendriteComps[q]];
          }
          let bucket = regionVoltageByRoot[rootId];
          bucket.dendrite.push(dendriteComps.length
            ? dendriteSum / dendriteComps.length * 1e3
            : voltageMv);
          bucket.soma.push(cableState.voltage[readouts.somaCompartment] * 1e3);
        }
        pointActivityByRoot[rootId].push(sigmoidMembraneActivity(voltageMv, pointELeakMv));
        pointReleaseByRoot[rootId].push(z(
          voltageV,
          excitatoryReleaseThresholdMv,
          excitatoryReleaseGainPerMv,
          excitatoryReleaseExponent,
        ));
        rawOpticFlowDriveByRoot[rootId].push(rawOpticDriveByCell[cellIndex]);
        opticFlowDriveByRoot[rootId].push(opticDriveByCell[cellIndex]);
      }
      for (let probe of probes) {
        let state = probe.cell === `target` ? targetState : vchState;
        let compartment = R(Math.round(probe.compartment), 0, state.voltage.length - 1);
        probeTraces[probe.id].push(state.voltage[compartment] * 1e3);
      }
      // Traces of the target cell's hemicircuit.
      lpi15Mv.push(lpi15Voltage * 1e3);
      lpi15DriveFractionTrace.push(lpi15DriveFraction);
      lpi15ConductanceOnVchNSTrace.push(lpi15ConductanceOnVchNS);
      lpi15ToT4ConductanceNSTrace.push(lpi15ToT4ConductanceNS);
      vchToT4ConductanceNSTrace.push(vchToT4ConductanceNS);
      t4ToVchConductanceNSTrace.push(currentT4ToVchConductanceNS);
      vchToLpi15ConductanceNSTrace.push(currentVchToLpi15ConductanceNS);
      lpi15SpreadMvTrace.push(lpi15SpreadMv);
      // Traces per hemicircuit and per H2.
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        let state = vchStates[hemiIndex];
        vchMeanMvByHemi[hemiIndex].push(ue(state.voltage) * 1e3);
        vchRootMvByHemi[hemiIndex].push(state.voltage[vchKernels[hemiIndex].rootIndex] * 1e3);
        lpi15MvByHemi[hemiIndex].push(lpi15VoltageByHemi[hemiIndex] * 1e3);
        lpi15SpreadMvByHemi[hemiIndex].push(lpi15SpreadByHemi[hemiIndex]);
        lpi15ConductanceOnVchNSByHemi[hemiIndex].push(lpi15ConductanceOnVchByHemi[hemiIndex]);
        t4ToVchConductanceNSByHemi[hemiIndex].push(t4ToVchConductanceByHemi[hemiIndex]);
        vchToT4ConductanceNSByHemi[hemiIndex].push(vchToT4ConductanceByHemi[hemiIndex]);
        vchToLpi15ConductanceNSByHemi[hemiIndex].push(vchToLpi15ConductanceByHemi[hemiIndex]);
        lpi15ToT4ConductanceNSByHemi[hemiIndex].push(lpi15ToT4ConductanceByHemi[hemiIndex]);
      }
      for (let u = 0; u < h2Count; u += 1) {
        h2MeanMvByUnit[u].push(ue(h2States[u].voltage) * 1e3);
        h2RootMvByUnit[u].push(h2States[u].voltage[h2Kernels[u].rootIndex] * 1e3);
        t4bToH2ConductanceNSByUnit[u].push(currentT4bToH2ConductanceNS[u]);
        h2ToVchConductanceNSByUnit[u].push(currentH2ToVchConductanceNS[u]);
      }
      // T4b voltages and release.
      let t4bSum = 0;
      let t4bReleasing = 0;
      for (let k = 0; k < t4bCount; k += 1) {
        let voltageV = t4bMulticompartment
          ? t4bStates[k].voltage[t4bReadouts[k].axonCompartment]
          : pointVoltage[t4aCount + k];
        let voltageMv = voltageV * 1e3;
        t4bSum += voltageMv;
        let release;
        if (t4bMulticompartment) {
          let sourceComps = t4bMappings[k].targetToVchTargetCompartments;
          let releaseSum = 0;
          for (let compartment of sourceComps) {
            releaseSum += z(
              t4bStates[k].voltage[compartment],
              t4bToLpi15ReleaseThresholdMv,
              t4bToLpi15ReleaseGainPerMv,
              t4bToLpi15ReleaseExponent,
            );
          }
          release = sourceComps.length ? releaseSum / sourceComps.length : 0;
        } else {
          release = z(
            voltageV,
            t4bToLpi15ReleaseThresholdMv,
            t4bToLpi15ReleaseGainPerMv,
            t4bToLpi15ReleaseExponent,
          );
        }
        if (release > 0) t4bReleasing += 1;
        t4bVoltageByRoot[t4bCells[k].rootId].push(voltageMv);
        t4bReleaseByRoot[t4bCells[k].rootId].push(release);
      }
      t4bMeanMv.push(t4bCount ? t4bSum / t4bCount : pointELeakMv);
      t4bReleasingCount.push(t4bReleasing);
    };

    // Time integration; the steps (1)-(5) are those of the header.
    let integrationStartedMs = performance.now();
    let phasePremotionUpstreamMs = 0;
    let phaseT4InputAssemblyMs = 0;
    let phaseFeedbackAssemblyMs = 0;
    let phaseCableSolveMs = 0;
    let phasePostSolveMs = 0;
    captureOutput(0);
    for (let step = 1; step <= stepCount; step += 1) {
      let phaseStartedMs = performance.now();
      let timeSec = step * dtSec;
      // (1) Advance the input model as far as this step needs, and select the conductance frames.
      let desiredPremotionStep = ie(
        step,
        stepCount,
        `C3`,
        dtMs,
        premotionOptions.calibration,
      );
      for (; premotionCursor < desiredPremotionStep; ) {
        premotionCursor += 1;
        ce(
          populationPremotionStates,
          stimulus,
          premotionCursor * dtSec,
          dtMs,
          premotionOptions,
        );
        writePremotionHistory(premotionCursor);
      }
      applyPremotionHistory(step, true);

      // Extremes of the target cell's input, for the result summary.
      for (let path of [`home`, `null`, `preferred`]) {
        let accepted = targetPremotion.paths[path].acceptedLuminance;
        targetAcceptedMin[path] = Math.min(targetAcceptedMin[path], accepted);
        targetAcceptedMax[path] = Math.max(targetAcceptedMax[path], accepted);
      }
      for (let kind of h) {
        targetPeakActivity[kind] = Math.max(targetPeakActivity[kind], targetPremotion.activity[kind]);
        targetPeakConductance[kind] = Math.max(targetPeakConductance[kind], targetPremotion.conductance[kind]);
      }

      // Total visual conductance per cell (nS), for the traces and the summary.
      let populationUpstreamSumNS = 0;
      for (let cellIndex = 0; cellIndex < populationCount; cellIndex += 1) {
        let cell = premotionCells[cellIndex];
        let totalG = 0;
        let totalGE = 0;
        let cellConductanceUnitNS = cellIndex < t4aCount
          ? t4Kernels[cellIndex].leakTotalS / 0.5 * 1e9
          : t4bMulticompartment
            ? t4bKernels[cellIndex - t4aCount].leakTotalS / 0.5 * 1e9
            : upstreamConductanceUnitNS;
        for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
          let contactScale = upstreamIsPerContact ? cell.upstreamCounts[h[kindIndex]] : 1;
          let conductanceNS = activeConductanceFrames[kindIndex][cellIndex]
            * cellConductanceUnitNS * contactScale;
          totalG += conductanceNS;
          totalGE += conductanceNS * upstreamReversalByIndex[kindIndex];
        }
        rawOpticDriveByCell[cellIndex] = totalG;
        opticDriveByCell[cellIndex] = totalG;
        upstreamConductanceTimesReversalByCell[cellIndex] = totalGE;
        if (cellIndex < t4aCount) {
          populationUpstreamSumNS += totalG;
          if (totalG > 1e-12) everPremotionDriven[cellIndex] = 1;
          peakRawOpticFlowDrive = Math.max(peakRawOpticFlowDrive, totalG);
          peakOpticFlowDrive = Math.max(peakOpticFlowDrive, totalG);
        }
      }
      // Inactive: optic-flow drive of point-model T4b.
      if (t4bOpticFlowFallback && t4bCount) {
        let rawT4bOpticDrive = opticFlowDriveBatch(
          timeSec,
          stimulus,
          dtSec,
          t4bOpticFlowSettings,
          t4bOpticFlowArrays,
          t4bOpticFlowCache,
        );
        let filteredT4bOpticDrive = filterOpticFlowDrive(
          rawT4bOpticDrive,
          t4bOpticFlowFilter,
        );
        for (let k = 0; k < t4bCount; k += 1) {
          let cellIndex = t4aCount + k;
          let rawConductanceNS = rawT4bOpticDrive[k] * t4bOpticExternalConductanceNS;
          let filteredConductanceNS = filteredT4bOpticDrive[k]
            * t4bOpticExternalConductanceNS;
          rawOpticDriveByCell[cellIndex] = rawConductanceNS;
          opticDriveByCell[cellIndex] = filteredConductanceNS;
          upstreamConductanceTimesReversalByCell[cellIndex] = filteredConductanceNS * -21;
        }
      }
      peakMeanOpticFlowDrive = Math.max(
        peakMeanOpticFlowDrive,
        t4aCount ? populationUpstreamSumNS / t4aCount : 0,
      );
      peakOpticFlowVisualConductanceNS = Math.max(
        peakOpticFlowVisualConductanceNS,
        peakOpticFlowDrive,
      );
      phasePremotionUpstreamMs += performance.now() - phaseStartedMs;
      phaseStartedMs = performance.now();

      // (2) Visual input on every T4 cable: reset the synapse compartments, then add each type's conductance at its
      // contacts.
      let assembledInputsInParallel = typeof parallelT4InputAssembly === `function`
        && parallelT4InputAssembly(activeConductanceFrames) === true;
      if (!assembledInputsInParallel) {
        for (let cellIndex = 0; cellIndex < allT4CableStates.length; cellIndex += 1) {
          let cableState = allT4CableStates[cellIndex];
          let touched = t4TouchedCompartments[cellIndex];
          for (let touchedIndex = 0; touchedIndex < touched.length; touchedIndex += 1) {
            let compartment = touched[touchedIndex];
            cableState.conductance[compartment] = 0;
            cableState.current[compartment] = 0;
          }
          let inputMappings = t4InputCompartments[cellIndex];
          let conductanceUnit = t4InputConductanceUnits[cellIndex];
          for (let kindIndex = 0; kindIndex < h.length; kindIndex += 1) {
            let contacts = inputMappings[kindIndex];
            if (!contacts.length) continue;
            let conductanceS = activeConductanceFrames[kindIndex][cellIndex] * conductanceUnit;
            if (!t4InputPerContact) conductanceS /= contacts.length;
            if (conductanceS <= 0) continue;
            let currentIncrement = conductanceS * upstreamReversalByIndex[kindIndex] / 1e3;
            for (let contactIndex = 0; contactIndex < contacts.length; contactIndex += 1) {
              let compartment = contacts[contactIndex];
              cableState.conductance[compartment] += conductanceS;
              cableState.current[compartment] += currentIncrement;
            }
          }
        }
      }
      phaseT4InputAssemblyMs += performance.now() - phaseStartedMs;
      phaseStartedMs = performance.now();
      // (3) Synaptic conductances, all from the voltages at the end of the previous step. Reset vCH and H2 (LPi15 is
      // reset further below).
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        vchStates[hemiIndex].conductance.fill(0);
        vchStates[hemiIndex].current.fill(0);
      }
      for (let h2Index = 0; h2Index < h2Count; h2Index += 1) {
        h2States[h2Index].conductance.fill(0);
        h2States[h2Index].current.fill(0);
      }

      feedbackConductanceNS.fill(0);
      feedbackConductanceTimesReversal.fill(0);
      vchToT4ConductanceNS = 0;
      vchToT4ConductanceByHemi.fill(0);
      // vCH release voltage (equal to the voltage while the release time constants are 0).
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        let state = vchStates[hemiIndex];
        let filtered = filteredVchReleaseVoltageByHemi[hemiIndex];
        let targetFiltered = filteredTargetVchReleaseVoltageByHemi[hemiIndex];
        if (inhibitoryReleaseTauMs > 0) {
          let releaseAlpha = 1 - Math.exp(-dtMs / inhibitoryReleaseTauMs);
          for (let compartment = 0; compartment < filtered.length; compartment += 1) {
            filtered[compartment] += releaseAlpha
              * (state.voltage[compartment] - filtered[compartment]);
          }
        } else {
          filtered.set(state.voltage);
        }
        if (targetInhibitoryReleaseTauMs > 0) {
          let targetReleaseAlpha = 1 - Math.exp(-dtMs / targetInhibitoryReleaseTauMs);
          for (let compartment = 0; compartment < targetFiltered.length; compartment += 1) {
            targetFiltered[compartment] += targetReleaseAlpha
              * (state.voltage[compartment] - targetFiltered[compartment]);
          }
        } else {
          targetFiltered.set(state.voltage);
        }
      }
      // vCH -> T4a: release at each synapse's site on vCH, conductance at its site on T4a.
      for (let cellIndex = 0; cellIndex < circuit.cells.length; cellIndex += 1) {
        let cellHemi = hemiOfT4a[cellIndex];
        let ownTargetFiltered = filteredTargetVchReleaseVoltageByHemi[cellHemi];
        let vchCompartments = mapping.vchToT4ByCell[cellIndex];
        for (let contactIndex = 0; contactIndex < vchCompartments.length; contactIndex += 1) {
          let vchCompartment = vchCompartments[contactIndex];
          let conductanceNS = vchToT4PerContactNS * z(
            ownTargetFiltered[vchCompartment],
            inhibitoryReleaseThresholdMv,
            inhibitoryReleaseGainPerMv,
            inhibitoryReleaseExponent,
          );
          if (conductanceNS <= 0) continue;
          feedbackConductanceNS[cellIndex] += conductanceNS;
          feedbackConductanceTimesReversal[cellIndex] += conductanceNS * vchToT4aReversalMv;
          vchToT4ConductanceNS += conductanceNS;
          vchToT4ConductanceByHemi[cellHemi] += conductanceNS;
          let targetCompartment = t4Mappings[cellIndex]
            .vchToTargetTargetCompartments[contactIndex];
          B(t4States[cellIndex], targetCompartment, conductanceNS * 1e-9, vchToT4aReversalMv);
        }
      }
      // LPi15 -> T4a (lpi15ToT4Enabled): release at each synapse's site on LPi15.
      lpi15ToT4ConductanceNS = 0;
      lpi15ToT4ConductanceByHemi.fill(0);
      if (lpi15ToT4Enabled) {
        for (let cellIndex = 0; cellIndex < t4aCount; cellIndex += 1) {
          let comps = lpi15ToT4ReleaseComps[cellIndex];
          if (!comps) continue;
          let cellHemi = hemiOfT4a[cellIndex];
          let ownLpi15State = lpi15StatesByHemi[cellHemi];
          for (let contactIndex = 0; contactIndex < comps.length; contactIndex += 1) {
            let conductanceNS = lpi15ToT4PerContactNS * z(
              lpi15Multi ? ownLpi15State.voltage[comps[contactIndex]] : lpi15VoltageByHemi[cellHemi],
              lpi15ReleaseThresholdMv,
              lpi15ReleaseGainPerMv,
              lpi15ReleaseExponent,
            );
            if (conductanceNS <= 0) continue;
            feedbackConductanceNS[cellIndex] += conductanceNS;
            feedbackConductanceTimesReversal[cellIndex] += conductanceNS * lpi15ReversalMv;
            lpi15ToT4ConductanceNS += conductanceNS;
            lpi15ToT4ConductanceByHemi[cellHemi] += conductanceNS;
            B(
              t4States[cellIndex],
              lpi15ToT4CableCompsByCell[cellIndex][contactIndex],
              conductanceNS * 1e-9,
              lpi15ReversalMv,
            );
          }
        }
      }

      // vCH -> T4b.
      for (let k = 0; k < t4bCount; k += 1) {
        let comps = vchToT4bCompartments[k];
        let cellIndex = t4aCount + k;
        let ownFiltered = filteredVchReleaseVoltageByHemi[hemiOfT4b[k]];
        for (let contactIndex = 0; contactIndex < comps.length; contactIndex += 1) {
          let conductanceNS = vchToT4PerContactNS * z(
            ownFiltered[comps[contactIndex]],
            inhibitoryReleaseThresholdMv,
            inhibitoryReleaseGainPerMv,
            inhibitoryReleaseExponent,
          );
          if (conductanceNS <= 0) continue;
          feedbackConductanceNS[cellIndex] += conductanceNS;
          feedbackConductanceTimesReversal[cellIndex] += conductanceNS * vchToT4ReversalMv;
          if (t4bMulticompartment) {
            let targetCompartment = t4bMappings[k]
              .vchToTargetTargetCompartments[contactIndex];
            B(
              t4bStates[k],
              targetCompartment,
              conductanceNS * 1e-9,
              vchToT4ReversalMv,
            );
          }
        }
      }

      // LPi15 -> vCH: release at each synapse's site on LPi15, conductance at its site on vCH.
      lpi15ConductanceOnVchNS = 0;
      lpi15ConductanceOnVchByHemi.fill(0);
      if (lpi15Multi) {
        for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
          let ownLpi15State = lpi15StatesByHemi[hemiIndex];
          if (!ownLpi15State) continue;
          let releaseComps = lpi15ReleaseCompsByHemi[hemiIndex];
          let vchComps = lpi15ReleaseVchCompsByHemi[hemiIndex];
          let ownVchState = vchStates[hemiIndex];
          let hemiSum = 0;
          for (let c = 0; c < releaseComps.length; c += 1) {
            let release = z(
              ownLpi15State.voltage[releaseComps[c]],
              lpi15ToVchReleaseThresholdMv,
              lpi15ReleaseGainPerMv,
              lpi15ReleaseExponent,
            );
            if (release <= 0) continue;
            let conductanceNS = lpi15ToVchPerContactNS * release;
            B(ownVchState, vchComps[c], conductanceNS * 1e-9, lpi15ReversalMv);
            hemiSum += conductanceNS;
          }
          lpi15ConductanceOnVchByHemi[hemiIndex] = hemiSum;
        }
        lpi15ConductanceOnVchNS = lpi15ConductanceOnVchByHemi[targetHemi];
        peakLpi15ConductanceOnVchNS = Math.max(peakLpi15ConductanceOnVchNS, lpi15ConductanceOnVchNS);
      } else if (lpi15Cell && lpi15VchCompartments.length) {
        // Point-model LPi15 (one-eye circuit without an LPi15 skeleton).
        let lpiRelease = z(
          lpi15Voltage,
          lpi15ToVchReleaseThresholdMv,
          lpi15ReleaseGainPerMv,
          lpi15ReleaseExponent,
        );
        let perContactNS = lpi15ToVchPerContactNS * lpiRelease;
        if (perContactNS > 0) {
          for (let c = 0; c < lpi15VchCompartments.length; c += 1) {
            B(vchState, lpi15VchCompartments[c], perContactNS * 1e-9, lpi15ReversalMv);
          }
          lpi15ConductanceOnVchNS = perContactNS * lpi15VchCompartments.length;
        }
        peakLpi15ConductanceOnVchNS = Math.max(peakLpi15ConductanceOnVchNS, lpi15ConductanceOnVchNS);
      }

      // T4a -> vCH: release at each synapse's site on T4a, conductance at its site on the vCH of the same hemicircuit.
      currentT4ToVchConductanceNS = 0;
      t4ToVchConductanceByHemi.fill(0);
      let releasingT4Count = 0;
      for (let cellIndex = 0; cellIndex < circuit.cells.length; cellIndex += 1) {
        let cellHemi = hemiOfT4a[cellIndex];
        let ownVchState = vchStates[cellHemi];
        let vchCompartments = mapping.t4ToVchByCell[cellIndex];
        let cellPeakRelease = 0;
        for (let contactIndex = 0; contactIndex < vchCompartments.length; contactIndex += 1) {
          let targetCompartment = t4Mappings[cellIndex]
            .targetToVchTargetCompartments[contactIndex];
          let presynapticVoltage = t4States[cellIndex].voltage[targetCompartment];
          if (cellIndex === targetIndex) {
            peakTargetContactVoltageMv = Math.max(peakTargetContactVoltageMv, presynapticVoltage * 1e3);
          }
          let release = z(
            presynapticVoltage,
            excitatoryReleaseThresholdMv,
            excitatoryReleaseGainPerMv,
            excitatoryReleaseExponent,
          );
          cellPeakRelease = Math.max(cellPeakRelease, release);
          peakT4VoltageMv = Math.max(peakT4VoltageMv, presynapticVoltage * 1e3);
          peakT4ReleaseFraction = Math.max(peakT4ReleaseFraction, release);
          let conductanceNS = t4ToVchPerContactNS * release;
          if (conductanceNS > 0) {
            t4ToVchConductanceByHemi[cellHemi] += conductanceNS;
            B(ownVchState, vchCompartments[contactIndex], conductanceNS * 1e-9, -21);
          }
        }
        if (cellPeakRelease > 0) {
          releasingT4Count += 1;
          everReleased[cellIndex] = 1;
        }
      }
      currentT4ToVchConductanceNS = t4ToVchConductanceByHemi[targetHemi];
      peakReleasingT4Count = Math.max(peakReleasingT4Count, releasingT4Count);
      peakT4ToVchConductanceNS = Math.max(peakT4ToVchConductanceNS, currentT4ToVchConductanceNS);

      peakTargetShadowUpstreamConductanceNS = Math.max(
        peakTargetShadowUpstreamConductanceNS,
        opticDriveByCell[targetIndex],
      );
      // Inactive: point-model T4b.
      if (!t4bMulticompartment) {
        for (let cellIndex = t4aCount; cellIndex < populationCount; cellIndex += 1) {
          let upstreamConductanceNS = opticDriveByCell[cellIndex];
          let totalConductanceNS = t4bPointLeakNS
            + upstreamConductanceNS
            + feedbackConductanceNS[cellIndex];
          let conductanceTimesReversal = t4bPointLeakNS * t4bPointELeakMv
            + upstreamConductanceTimesReversalByCell[cellIndex]
            + feedbackConductanceTimesReversal[cellIndex];
          let capacitanceOverDt = t4bPointLeakNS * t4bPointTauMS / dtMs;
          pointNext[cellIndex] = (
            capacitanceOverDt * pointVoltage[cellIndex] * 1e3
            + conductanceTimesReversal
          ) / Math.max(1e-12, capacitanceOverDt + totalConductanceNS) / 1e3;
        }
      }

      // T4b -> LPi15 and vCH -> LPi15.
      let lpi15Next = lpi15Voltage;
      currentVchToLpi15ConductanceNS = 0;
      if (lpi15Multi) {
        // Reset LPi15, then add T4b -> LPi15 at each synapse's own sites.
        if (!parallelLpi15InputAssembly) {
          for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
            let state = lpi15StatesByHemi[hemiIndex];
            if (!state) continue;
            state.conductance.fill(0);
            state.current.fill(0);
          }
        }
        let weightedReleaseByHemi = new Float64Array(hemiCount);
        for (let k = 0; k < t4bCount; k += 1) {
          let comps = lpi15InputCompsByT4b[k];
          if (!comps.length) {
            t4bToLpi15ReleaseFrames[k] = 0;
            continue;
          }
          let cellHemi = hemiOfT4b[k];
          let ownLpi15State = lpi15StatesByHemi[cellHemi];
          if (t4bMulticompartment) {
            let sourceComps = t4bMappings[k].targetToVchTargetCompartments;
            let cellReleaseSum = 0;
            for (let ci = 0; ci < comps.length; ci += 1) {
              let release = z(
                t4bStates[k].voltage[sourceComps[ci]],
                t4bToLpi15ReleaseThresholdMv,
                t4bToLpi15ReleaseGainPerMv,
                t4bToLpi15ReleaseExponent,
              );
              cellReleaseSum += release;
              weightedReleaseByHemi[cellHemi] += release;
              if (release <= 0) continue;
              let conductanceNS = t4bToLpi15PerContactNS * release;
              B(ownLpi15State, comps[ci], conductanceNS * 1e-9, -21);
            }
            t4bToLpi15ReleaseFrames[k] = cellReleaseSum / comps.length;
          } else {
            let release = z(
              pointVoltage[t4aCount + k],
              t4bToLpi15ReleaseThresholdMv,
              t4bToLpi15ReleaseGainPerMv,
              t4bToLpi15ReleaseExponent,
            );
            t4bToLpi15ReleaseFrames[k] = release;
            weightedReleaseByHemi[cellHemi] += t4bLpiWeight[k] * release;
            if (release <= 0 || parallelLpi15InputAssembly) continue;
            let conductanceNS = t4bToLpi15PerContactNS * release;
            for (let ci = 0; ci < comps.length; ci += 1) {
              B(ownLpi15State, comps[ci], conductanceNS * 1e-9, -21);
            }
          }
        }
        if (parallelLpi15InputAssembly) {
          parallelLpi15InputAssembly(t4bToLpi15ReleaseFrames);
        }
        for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
          lpi15DriveFractionByHemi[hemiIndex] = t4bLpiWeightTotalByHemi[hemiIndex] > 0
            ? weightedReleaseByHemi[hemiIndex] / t4bLpiWeightTotalByHemi[hemiIndex]
            : 0;
        }
        lpi15DriveFraction = lpi15DriveFractionByHemi[targetHemi];
        // vCH -> LPi15.
        vchToLpi15ConductanceByHemi.fill(0);
        for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
          let ownLpi15State = lpi15StatesByHemi[hemiIndex];
          if (!ownLpi15State) continue;
          let fbLpiComps = lpi15FbLpiCompsByHemi[hemiIndex];
          let fbVchComps = lpi15FbVchCompsByHemi[hemiIndex];
          let ownFiltered = filteredVchReleaseVoltageByHemi[hemiIndex];
          for (let ci = 0; ci < fbLpiComps.length; ci += 1) {
            let release = z(
              ownFiltered[fbVchComps[ci]],
              vchToLpi15ReleaseThresholdMv,
              vchToLpi15ReleaseGainPerMv,
              vchToLpi15ReleaseExponent,
            );
            if (release <= 0) continue;
            let conductanceNS = vchToLpi15PerContactNS * release;
            vchToLpi15ConductanceByHemi[hemiIndex] += conductanceNS;
            B(ownLpi15State, fbLpiComps[ci], conductanceNS * 1e-9, vchToT4ReversalMv);
          }
        }
        currentVchToLpi15ConductanceNS = vchToLpi15ConductanceByHemi[targetHemi];
      } else if (lpi15Cell) {
        // Point-model LPi15: one membrane driven by the synapse-weighted mean T4b release.
        let weightedRelease = 0;
        for (let k = 0; k < t4bCount; k += 1) {
          if (t4bLpiWeight[k] <= 0) continue;
          if (t4bMulticompartment) {
            for (let compartment of t4bMappings[k].targetToVchTargetCompartments) {
              weightedRelease += z(
                t4bStates[k].voltage[compartment],
                t4bToLpi15ReleaseThresholdMv,
                t4bToLpi15ReleaseGainPerMv,
                t4bToLpi15ReleaseExponent,
              );
            }
          } else {
            weightedRelease += t4bLpiWeight[k] * z(
              pointVoltage[t4aCount + k],
              t4bToLpi15ReleaseThresholdMv,
              t4bToLpi15ReleaseGainPerMv,
              t4bToLpi15ReleaseExponent,
            );
          }
        }
        lpi15DriveFraction = t4bLpiWeightTotal > 0 ? weightedRelease / t4bLpiWeightTotal : 0;
        let excitationNS = lpi15DriveNS * lpi15DriveFraction;
        let capacitanceOverDt = lpi15LeakNS * lpi15TauMs / dtMs;
        lpi15Next = (
          capacitanceOverDt * lpi15Voltage * 1e3
          + lpi15LeakNS * pointELeakMv
          + excitationNS * -21
        ) / Math.max(1e-12, capacitanceOverDt + lpi15LeakNS + excitationNS) / 1e3;
      }

      // T4b -> H2 and H2 -> vCH. H2 is the only cell that crosses the midline: it carries one eye's T4b signal to the
      // vCH of the other side.
      if (h2Enabled && t4bMulticompartment) {
        for (let u = 0; u < h2Count; u += 1) {
          let unitState = h2States[u];
          let drive = h2DriveByUnit[u];
          let driveSum = 0;
          for (let d = 0; d < drive.length; d += 1) {
            let entry = drive[d];
            let sourceState = t4bStates[entry.t4bIndex];
            let sourceComps = entry.sourceComps;
            let h2Comps = entry.h2Comps;
            for (let c = 0; c < sourceComps.length; c += 1) {
              let release = z(
                sourceState.voltage[sourceComps[c]],
                t4bToH2ReleaseThresholdMv,
                t4bToH2ReleaseGainPerMv,
                t4bToH2ReleaseExponent,
              );
              if (release <= 0) continue;
              let conductanceNS = t4bToH2PerContactNS * release;
              driveSum += conductanceNS;
              B(unitState, h2Comps[c], conductanceNS * 1e-9, -21);
            }
          }
          currentT4bToH2ConductanceNS[u] = driveSum;
          peakT4bToH2ConductanceNS = Math.max(peakT4bToH2ConductanceNS, driveSum);

          let outputs = h2OutputsByUnit[u];
          let outSum = 0;
          for (let o = 0; o < outputs.length; o += 1) {
            let edge = outputs[o];
            let recipientVchState = vchStates[edge.hemiIndex];
            for (let c = 0; c < edge.releaseComps.length; c += 1) {
              let release = z(
                unitState.voltage[edge.releaseComps[c]],
                h2ExcitatoryReleaseThresholdMv,
                h2ExcitatoryReleaseGainPerMv,
                h2ExcitatoryReleaseExponent,
              );
              if (release > peakH2ReleaseFraction) peakH2ReleaseFraction = release;
              if (release <= 0) continue;
              let conductanceNS = h2ToVchPerContactNS * release;
              outSum += conductanceNS;
              B(recipientVchState, edge.targetComps[c], conductanceNS * 1e-9, -21);
            }
          }
          currentH2ToVchConductanceNS[u] = outSum;
          peakH2ToVchConductanceNS = Math.max(peakH2ToVchConductanceNS, outSum);
        }
      }

      // The other connections, each with the release curve and reversal of its presynaptic cell.
      if (v2Enabled) {
        for (let ci = 0; ci < t4aCount; ci += 1) {
          let source = t4States[ci];
          let toLpi = t4aToLpi15ByCell[ci];
          for (let e = 0; e < toLpi.length; e += 1) {
            let entry = toLpi[e];
            let target = lpi15StatesByHemi[entry.hemiIndex];
            if (!target) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(source.voltage[entry.src[c]], excitatoryReleaseThresholdMv,
                excitatoryReleaseGainPerMv, excitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(target, entry.dst[c], t4aToLpi15PerContactNS * rel * 1e-9, -21);
            }
          }
          let toH2 = t4aToH2ByCell[ci];
          for (let e = 0; e < toH2.length; e += 1) {
            let entry = toH2[e];
            let target = h2States[entry.unitIndex];
            if (!target) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(source.voltage[entry.src[c]], excitatoryReleaseThresholdMv,
                excitatoryReleaseGainPerMv, excitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(target, entry.dst[c], t4aToH2PerContactNS * rel * 1e-9, -21);
            }
          }
          let fromH2 = h2ToT4aByCell[ci];
          for (let e = 0; e < fromH2.length; e += 1) {
            let entry = fromH2[e];
            let h2State = h2States[entry.unitIndex];
            if (!h2State) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(h2State.voltage[entry.src[c]], h2ExcitatoryReleaseThresholdMv,
                h2ExcitatoryReleaseGainPerMv, h2ExcitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(source, entry.dst[c], h2ToT4aPerContactNS * rel * 1e-9, -21);
            }
          }
        }
        for (let k = 0; k < t4bCount; k += 1) {
          let cellState = t4bStates[k];
          if (!cellState) continue;
          let fromLpi = lpi15ToT4bByCell[k];
          for (let e = 0; e < fromLpi.length; e += 1) {
            let entry = fromLpi[e];
            let source = lpi15StatesByHemi[entry.hemiIndex];
            if (!source) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(source.voltage[entry.src[c]], lpi15ReleaseThresholdMv,
                lpi15ReleaseGainPerMv, lpi15ReleaseExponent);
              if (rel <= 0) continue;
              B(cellState, entry.dst[c], lpi15ToT4bPerContactNS * rel * 1e-9, lpi15ReversalMv);
            }
          }
          let fromH2b = h2ToT4bByCell[k];
          for (let e = 0; e < fromH2b.length; e += 1) {
            let entry = fromH2b[e];
            let source = h2States[entry.unitIndex];
            if (!source) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(source.voltage[entry.src[c]], h2ExcitatoryReleaseThresholdMv,
                h2ExcitatoryReleaseGainPerMv, h2ExcitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(cellState, entry.dst[c], h2ToT4bPerContactNS * rel * 1e-9, -21);
            }
          }
          let direct = t4bToVchDirectByCell[k];
          if (direct) {
            let target = vchStates[direct.hemiIndex];
            for (let c = 0; c < direct.src.length; c += 1) {
              let rel = z(cellState.voltage[direct.src[c]], t4bToLpi15ReleaseThresholdMv,
                t4bToLpi15ReleaseGainPerMv, t4bToLpi15ReleaseExponent);
              if (rel <= 0) continue;
              B(target, direct.dst[c], t4bToVchDirectPerContactNS * rel * 1e-9, -21);
            }
          }
        }
        for (let u = 0; u < h2Count; u += 1) {
          let unitState = h2States[u];
          for (let entry of h2ToLpi15ByUnit[u]) {
            let target = lpi15StatesByHemi[entry.hemiIndex];
            if (!target) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(unitState.voltage[entry.src[c]], h2ExcitatoryReleaseThresholdMv,
                h2ExcitatoryReleaseGainPerMv, h2ExcitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(target, entry.dst[c], h2ToLpi15PerContactNS * rel * 1e-9, -21);
            }
          }
          for (let entry of h2ToH2ByUnit[u]) {
            let target = h2States[entry.unitIndex];
            if (!target) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(unitState.voltage[entry.src[c]], h2ExcitatoryReleaseThresholdMv,
                h2ExcitatoryReleaseGainPerMv, h2ExcitatoryReleaseExponent);
              if (rel <= 0) continue;
              B(target, entry.dst[c], h2ToH2PerContactNS * rel * 1e-9, -21);
            }
          }
          for (let entry of lpi15ToH2ByUnit[u]) {
            let source = lpi15StatesByHemi[entry.hemiIndex];
            if (!source) continue;
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(source.voltage[entry.src[c]], lpi15ReleaseThresholdMv,
                lpi15ReleaseGainPerMv, lpi15ReleaseExponent);
              if (rel <= 0) continue;
              B(unitState, entry.dst[c], lpi15ToH2PerContactNS * rel * 1e-9, lpi15ReversalMv);
            }
          }
          for (let entry of vchToH2ByUnit[u]) {
            let filtered = filteredVchReleaseVoltageByHemi[entry.hemiIndex];
            for (let c = 0; c < entry.src.length; c += 1) {
              let rel = z(filtered[entry.src[c]], inhibitoryReleaseThresholdMv,
                inhibitoryReleaseGainPerMv, inhibitoryReleaseExponent);
              if (rel <= 0) continue;
              B(unitState, entry.dst[c], vchToH2PerContactNS * rel * 1e-9, vchToT4ReversalMv);
            }
          }
        }
      }
      phaseFeedbackAssemblyMs += performance.now() - phaseStartedMs;
      phaseStartedMs = performance.now();

      // (4) One implicit step of every cable: all T4a and T4b (by the native or pooled solver when one is installed),
      // then vCH, LPi15 and H2.
      const parallelCableSolver = globalThis.__YIFEI_SOLVE_T4A_CABLES__;
      const solvedInParallel = typeof parallelCableSolver === `function`
        && parallelCableSolver(allT4CableStates, dtSec) === true;
      if (!solvedInParallel) {
        for (let cellIndex = 0; cellIndex < allT4CableStates.length; cellIndex += 1) {
          fe(allT4CableStates[cellIndex], dtSec);
        }
      }
      for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
        fe(vchStates[hemiIndex], dtSec);
        if (lpi15Multi && lpi15StatesByHemi[hemiIndex]) fe(lpi15StatesByHemi[hemiIndex], dtSec);
      }
      for (let u = 0; u < h2Count; u += 1) fe(h2States[u], dtSec);
      phaseCableSolveMs += performance.now() - phaseStartedMs;
      phaseStartedMs = performance.now();
      // (5) Bookkeeping: point-model T4b (inactive), LPi15 mean voltage and spread (max - min over the arbor), peak
      // values; then record an output frame.
      if (!t4bMulticompartment) {
        for (let cellIndex = t4aCount; cellIndex < populationCount; cellIndex += 1) {
          pointVoltage[cellIndex] = pointNext[cellIndex];
        }
      }
      if (lpi15Multi) {
        for (let hemiIndex = 0; hemiIndex < hemiCount; hemiIndex += 1) {
          let state = lpi15StatesByHemi[hemiIndex];
          if (!state) continue;
          let sum = 0;
          let lo = Infinity;
          let hi = -Infinity;
          for (let q = 0; q < state.voltage.length; q += 1) {
            let v = state.voltage[q];
            sum += v;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
          }
          lpi15VoltageByHemi[hemiIndex] = sum / state.voltage.length;
          lpi15SpreadByHemi[hemiIndex] = (hi - lo) * 1e3;
          if (hemiIndex === targetHemi) {
            peakLpi15SpreadMv = Math.max(peakLpi15SpreadMv, lpi15SpreadByHemi[hemiIndex]);
            peakLpi15MaxVoltageMv = Math.max(peakLpi15MaxVoltageMv, hi * 1e3);
          }
        }
        lpi15Voltage = lpi15VoltageByHemi[targetHemi];
        lpi15SpreadMv = lpi15SpreadByHemi[targetHemi];
      } else {
        lpi15Voltage = lpi15Next;
        lpi15VoltageByHemi[targetHemi] = lpi15Next;
      }
      peakLpi15VoltageMv = Math.max(peakLpi15VoltageMv, lpi15Voltage * 1e3);
      for (let u = 0; u < h2Count; u += 1) {
        let mean = ue(h2States[u].voltage) * 1e3;
        if (mean > peakH2VoltageMv[u]) peakH2VoltageMv[u] = mean;
      }
      for (let cellIndex = 0; cellIndex < t4aCount; cellIndex += 1) {
        let voltageMv = t4States[cellIndex].voltage[t4Readouts[cellIndex].axonCompartment] * 1e3;
        peakT4VoltageMv = Math.max(peakT4VoltageMv, voltageMv);
        if (voltageMv > peakPopulationPointVoltageMv) {
          peakPopulationPointVoltageMv = voltageMv;
          peakPopulationPointRootId = circuit.cells[cellIndex].rootId;
        }
      }
      let t4bReleasingNow = 0;
      for (let k = 0; k < t4bCount; k += 1) {
        let voltageV = t4bMulticompartment
          ? t4bStates[k].voltage[t4bReadouts[k].axonCompartment]
          : pointVoltage[t4aCount + k];
        peakT4bVoltageMv = Math.max(peakT4bVoltageMv, voltageV * 1e3);
        let release = 0;
        if (t4bMulticompartment) {
          for (let compartment of t4bMappings[k].targetToVchTargetCompartments) {
            release = Math.max(release, z(
              t4bStates[k].voltage[compartment],
              t4bToLpi15ReleaseThresholdMv,
              t4bToLpi15ReleaseGainPerMv,
              t4bToLpi15ReleaseExponent,
            ));
          }
        } else {
          release = z(
            voltageV,
            t4bToLpi15ReleaseThresholdMv,
            t4bToLpi15ReleaseGainPerMv,
            t4bToLpi15ReleaseExponent,
          );
        }
        if (release > 0) t4bReleasingNow += 1;
      }
      peakT4bReleasingCount = Math.max(peakT4bReleasingCount, t4bReleasingNow);

      if (step % outputEverySteps === 0 || step === stepCount) captureOutput(step);
      if (step % Math.max(1, Math.round(stepCount / 20)) === 0) {
        progress?.(
          0.15 + 0.85 * step / stepCount,
          `${t4bMulticompartment ? `All-T4a/T4b` : `All-T4a`} cable simulation ${Math.round(100 * step / stepCount)}%`,
        );
      }
      phasePostSolveMs += performance.now() - phaseStartedMs;
    }
    let integrationElapsedMs = performance.now() - integrationStartedMs;

    // Result: traces of the target cell and its hemicircuit, per-hemicircuit traces (sides), per-H2 traces
    // (h2Sides), per-cell traces keyed by root id, release-site histograms, and meta (every parameter used plus
    // summary values).
    return {
      time,
      targetRootId: targetCell.rootId,
      targetMeanMv,
      targetCableMeanMv,
      targetRootMv,
      targetInputMeanMv,
      targetDendriteMv,
      targetSomaMv,
      targetAxonMv,
      targetOutputContactMeanMv,
      targetVchFeedbackConductanceNS,
      targetVchContactMeanMv,
      targetVchContactMaxMv,
      targetVchReleaseMeanMv,
      targetVchReleaseMaxMv,
      upstreamTrace,
      vchMeanMv,
      lpi15VchReleaseMeanMv,
      lpi15VchContactMeanMv,
      vchRootMv,
      // Per-hemicircuit traces.
      sides: hemis.map((hemi, hemiIndex) => ({
        hemicircuit: hemi.key,
        eye: hemi.eye,
        isTargetSide: hemiIndex === targetHemi,
        vchRootId: String(hemi.vch.rootId),
        vchSide: hemi.vch.side ?? ``,
        lpi15RootId: String(hemi.lpi15?.rootId ?? ``),
        lpi15Side: hemi.lpi15?.side ?? ``,
        t4aCount: circuit.cells.reduce((n, _, ci) => n + (hemiOfT4a[ci] === hemiIndex ? 1 : 0), 0),
        t4bCount: t4bCells.reduce((n, _, k) => n + (hemiOfT4b[k] === hemiIndex ? 1 : 0), 0),
        vchMeanMv: vchMeanMvByHemi[hemiIndex],
        vchRootMv: vchRootMvByHemi[hemiIndex],
        lpi15Mv: lpi15MvByHemi[hemiIndex],
        lpi15SpreadMv: lpi15SpreadMvByHemi[hemiIndex],
        lpi15ConductanceOnVchNS: lpi15ConductanceOnVchNSByHemi[hemiIndex],
        t4ToVchConductanceNS: t4ToVchConductanceNSByHemi[hemiIndex],
        vchToT4ConductanceNS: vchToT4ConductanceNSByHemi[hemiIndex],
        vchToLpi15ConductanceNS: vchToLpi15ConductanceNSByHemi[hemiIndex],
        lpi15ToT4ConductanceNS: lpi15ToT4ConductanceNSByHemi[hemiIndex],
      })),
      h2Sides: h2Units.map((unit, u) => ({
        side: unit.side,
        rootId: unit.rootId,
        driveEye: hemis[unit.driveHemiIndex]?.eye ?? ``,
        driveHemicircuit: hemis[unit.driveHemiIndex]?.key ?? ``,
        drivingT4bCount: h2DriveByUnit[u].length,
        compartments: h2Kernels[u].nodes.length,
        targets: h2OutputsByUnit[u].map((edge) => ({
          kind: `VCH`,
          hemicircuit: hemis[edge.hemiIndex].key,
          vchRootId: String(hemis[edge.hemiIndex].vch.rootId),
          synapses: edge.releaseComps.length,
        })),
        h2MeanMv: h2MeanMvByUnit[u],
        h2RootMv: h2RootMvByUnit[u],
        t4bToH2ConductanceNS: t4bToH2ConductanceNSByUnit[u],
        h2ToVchConductanceNS: h2ToVchConductanceNSByUnit[u],
        peakH2MeanMv: peakH2VoltageMv[u],
      })),
      // LPi15, T4b and per-cell traces.
      lpi15Mv,
      lpi15DriveFraction: lpi15DriveFractionTrace,
      lpi15ConductanceOnVchNS: lpi15ConductanceOnVchNSTrace,
      lpi15ToT4ConductanceNS: lpi15ToT4ConductanceNSTrace,
      vchToT4ConductanceNS: vchToT4ConductanceNSTrace,
      t4ToVchConductanceNS: t4ToVchConductanceNSTrace,
      vchToLpi15ConductanceNS: vchToLpi15ConductanceNSTrace,
      lpi15SpreadMv: lpi15SpreadMvTrace,
      t4bMeanMv,
      t4bReleasingCount,
      t4bVoltageByRoot,
      t4bReleaseByRoot,
      targetFrames,
      vchFrames,
      pointVoltageByRoot,
      regionVoltageByRoot,
      pointActivityByRoot,
      pointReleaseByRoot,
      rawOpticFlowDriveByRoot,
      opticFlowDriveByRoot,
      probeTraces,
      releaseSiteHist: __rsExport(),
      // Parameters and summary values; some fields keep older names so that saved results stay readable.
      meta: {
        runsOn: `client-web-worker`,
        modelVariant: t4bMulticompartment
          ? `yifei-multi-v7-all-t4a-t4b-multicompartment-real-synapses`
          : t4bOpticFlowFallback
            ? `noshift-all-t4a-multicompartment-v4-t4b-opticflow-true-pd-tuning-fallback1`
            : `noshift-all-t4a-multicompartment-v4-t4b-lpi15-population-premotion-fix1`,
        targetRootId: targetCell.rootId,
        targetRole: `dynamic-selected-cell`,
        targetModelCellBinding: `none; target and trace are readout roles over the all-cable population`,
        v2Enabled: Boolean(lpi15Cell) || t4bCount > 0,
        t4bCount,
        t4bAvailable: Array.isArray(circuit.t4b) ? circuit.t4b.length : 0,
        t4bToLpi15Synapses: t4bLpiWeightTotal,
        lpi15RootId: lpi15Cell?.rootId ?? ``,
        lpi15VchContactCount: lpi15VchCompartments.length,
        lpi15DistinctVchCompartments: new Set(lpi15VchCompartments).size,
        lpi15VchReleaseCompartmentCount: lpi15VchReleaseCompartments.length,
        lpi15VchRecipientCompartmentCount: lpi15VchRecipientCompartments.length,
        lpi15VchReadoutCompartmentCount: lpi15VchRecipientCompartments.length,
        lpi15VchReadoutDefinition: `paired mean Vm over distinct LPi15 release compartments and their VCH recipient compartments`,
        lpi15ToVchPerContactNS,
        lpi15ToT4Enabled,
        lpi15ToT4PerContactNS,
        lpi15ToT4ContactTotal,
        lpi15DriveNS,
        lpi15LeakNS,
        lpi15TauMs,
        lpi15ReversalMv,
        t4bExcitatoryReversalMv: -21,
        binocular: binocular.binocular,
        hemicircuitCount: hemiCount,
        hemicircuits: hemis.map((hemi) => ({
          key: hemi.key,
          eye: hemi.eye,
          vchRootId: String(hemi.vch.rootId),
          lpi15RootId: String(hemi.lpi15?.rootId ?? ``),
        })),
        targetHemicircuit: hemis[targetHemi].key,
        targetEye: hemis[targetHemi].eye,
        h2Enabled,
        h2Count,
        h2Multicompartment: h2Count > 0,
        h2Compartments: h2Kernels.map((kernel) => kernel.nodes.length),
        h2DrivingT4bCounts: h2DriveByUnit.map((drive) => drive.length),
        h2ExcitatoryReleaseThresholdMv,
        h2ExcitatoryReleaseGainInsidePerMv: h2ExcitatoryReleaseGainPerMv,
        h2ExcitatoryReleaseExponent,
        t4bToH2ReleaseThresholdMv,
        t4bToH2ReleaseGainInsidePerMv: t4bToH2ReleaseGainPerMv,
        t4bToH2ReleaseExponent,
        t4bToH2PerContactNS,
        h2ToVchPerContactNS,
        peakT4bToH2ConductanceNS,
        peakH2ToVchConductanceNS,
        peakH2ReleaseFraction,
        h2ParameterSource: `v9 bilateral H2 arm (threshold/gain/exponent and both per-contact conductances)`,
        parameterSource: `v7 (v5 Set A release on the v7 cable circuit, premotion C3 decay 80 ms) for every T4a/T4b/VCH/LPi15 arm`,
        groundHemifield: [`positiveAz`,`negativeAz`,`none`].includes(stimulus.groundHemifield)
          ? stimulus.groundHemifield
          : `both`,
        groundHemifieldRule: `ground texture is blanked to 0.5 on the excluded side of azimuth 0; figure and static black mask are unaffected`,
        laterality: `within a hemicircuit every T4a/T4b/LPi15/VCH connection is ipsilateral (verified on synapse coordinates against the x = 533000 nm midline); FlyWire side labels are soma-based and VCH's soma sits across the midline from its arbor`,
        binocularLink: `T4b -> H2 -> VCH of the opposite eye's hemicircuit; H2 is the only midline-crossing cell`,
        v2FullConnectivity: v2Enabled,
        v2AddedSynapses: v2Counts,
        v2PerContactNS: {
          t4aToLpi15: t4aToLpi15PerContactNS, t4aToH2: t4aToH2PerContactNS,
          t4bToVchDirect: t4bToVchDirectPerContactNS, vchToH2: vchToH2PerContactNS,
          lpi15ToT4b: lpi15ToT4bPerContactNS, lpi15ToH2: lpi15ToH2PerContactNS,
          h2ToLpi15: h2ToLpi15PerContactNS, h2ToT4b: h2ToT4bPerContactNS,
          h2ToT4a: h2ToT4aPerContactNS, h2ToH2: h2ToH2PerContactNS,
        },
        v2Note: `each new pathway inherits its presynaptic cell weakest already-modelled arm, excluding VCH->LPi15 and LPi15->T4a which were strengthened instead; T4a<->T4b is deliberately not modelled`,
        v2TouchedCompartmentsFix: `2026-08-29: v2 target compartments on T4 cables registered in t4TouchedCompartments so their conductance is cleared every step (pre-fix runs integrated it across the whole run)`,
        peakLpi15VoltageMv,
        lpi15Multicompartment: lpi15Multi,
        lpi15Compartments: lpi15Kernel ? lpi15Kernel.nodes.length : 1,
        lpi15SkeletonNodes: lpi15Cell?.skeletonNodes ?? 0,
        t4bToLpi15PerContactNS,
        vchToLpi15PerContactNS,
        peakLpi15SpreadMv,
        peakLpi15MaxVoltageMv,
        t4bToLpi15SynapseSites: lpi15InputCompsByT4b.reduce((a, b) => a + b.length, 0),
        lpi15DistinctReleaseCompartments: new Set(lpi15ReleaseComps).size,
        peakLpi15ConductanceOnVchNS,
        peakT4bVoltageMv,
        peakT4bReleasingCount,
        v2Motif: `T4b --ACh--> LPi15 --GABA--> VCH; direction-opponent to the direct T4a --ACh--> VCH path`,
        v2DirectionSource: t4bMulticompartment
          ? `T4b keeps its measured premotion coordinates and true PD, with each input mapped to its exact FlyWire postsynaptic cable site`
          : t4bOpticFlowFallback
            ? `noshift-refit optic-flow drive projected onto each T4b cell's normalized measured premotion offset axis`
            : `T4b keeps its own measured premotion offset axis, so its opposite direction preference is data, not an assumption`,
        t4bModelVariant: t4bMulticompartment
          ? `multicompartment-real-synapses`
          : t4bOpticFlowFallback
            ? `noshift-opticflow-true-pd`
            : `population-premotion-point`,
        t4bMulticompartment,
        t4bMulticompartmentCount: t4bStates.length,
        t4bPointCellCount: 0,
        t4bOpticFlowCellCount: 0,
        t4bTotalCompartments: t4bKernels.reduce(
          (sum, kernel) => sum + kernel.nodes.length,
          0,
        ),
        t4bMinCompartments: t4bKernels.length
          ? Math.min(...t4bKernels.map((kernel) => kernel.nodes.length)) : 0,
        t4bMaxCompartments: t4bKernels.length
          ? Math.max(...t4bKernels.map((kernel) => kernel.nodes.length)) : 0,
        t4bCableRaOhmCm: t4bMulticompartment ? targetCableOptions.raOhmCm : 0,
        t4bCableRmOhmCm2: t4bMulticompartment ? targetCableOptions.rmOhmCm2 : 0,
        t4bCableCmUfPerCm2: t4bMulticompartment
          ? targetCableOptions.cmFPerCm2 * 1e6 : 0,
        t4bVisualInputContactTotal: t4bCells.reduce(
          (sum, cell) => sum + (cell.upstreamContacts?.length ?? 0),
          0,
        ),
        t4bVchFeedbackPairedContactTotal: t4bCells.reduce(
          (sum, cell) => sum + (cell.vchToT4bCableContacts?.length ?? 0),
          0,
        ),
        t4bLpi15OutputPairedContactTotal: t4bCells.reduce(
          (sum, cell) => sum + (cell.t4bToLpi15CableContacts?.length ?? 0),
          0,
        ),
        t4bOpticFlowFixedCalibration: t4bOpticFlowFallback,
        t4bOpticFlowMissingPremotionCount:
          t4bOpticFlowArrays?.missingPremotionRootIds.length ?? 0,
        v2Omitted: `direct T4b -> VCH (33 synapses over 23 cells, 0.6% of the T4a input) is not modelled`,
        t4aCount: circuit.cells.length,
        targetPremotionCellCount: populationCount,
        t4aMulticompartment: true,
        t4aMulticompartmentCount: circuit.cells.length,
        parallelCableSolver: Boolean(globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.enabled),
        parallelCableThreads: Number(globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.threads || 1),
        parallelCableSteps: Number(globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.steps || 0),
        parallelCableImplementation: String(
          globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.implementation || `serial-hines`,
        ),
        parallelCableFallback: String(globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.fallback || ``),
        parallelT4InputAssembly: Boolean(
          globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.parallelInputAssembly,
        ),
        parallelT4InputAssemblySteps: Number(
          globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.inputAssemblySteps || 0,
        ),
        parallelLpi15InputAssembly: Boolean(
          globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.parallelLpi15InputAssembly,
        ),
        parallelLpi15InputAssemblySteps: Number(
          globalThis.__YIFEI_PARALLEL_CABLE_SOLVER_INFO__?.lpi15InputAssemblySteps || 0,
        ),
        parallelPremotion: Boolean(globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.enabled),
        parallelPremotionThreads: Number(globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.threads || 1),
        parallelPremotionImplementation: String(
          globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.implementation || `serial-premotion`,
        ),
        parallelPremotionElapsedMs: Number(
          globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.elapsedMs || 0,
        ),
        parallelPremotionSharedBytes: Number(
          globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.sharedBytes || 0,
        ),
        parallelPremotionFallback: String(
          globalThis.__YIFEI_PARALLEL_PREMOTION_INFO__?.fallback || ``,
        ),
        initializationElapsedMs: integrationStartedMs - simulationStartedMs,
        integrationElapsedMs,
        simulationElapsedMs: performance.now() - simulationStartedMs,
        phasePremotionUpstreamMs,
        phaseT4InputAssemblyMs,
        phaseFeedbackAssemblyMs,
        phaseCableSolveMs,
        phasePostSolveMs,
        t4aTotalCompartments: t4Kernels.reduce(
          (sum, kernel) => sum + kernel.nodes.length,
          0,
        ),
        t4aMinCompartments: Math.min(...t4Kernels.map((kernel) => kernel.nodes.length)),
        t4aMaxCompartments: Math.max(...t4Kernels.map((kernel) => kernel.nodes.length)),
        premotionPointCellCount: t4bMulticompartment ? 0 : t4bCount,
        premotionCableCellCount: t4aCount + (t4bMulticompartment ? t4bCount : 0),
        premotionDriveRouting: `all-cable-delayed-frame-per-cell`,
        premotionDrivenCableCellCount: everPremotionDriven.reduce((sum, value) => sum + value, 0),
        premotionDrivenNonTargetCableCellCount: everPremotionDriven.reduce(
          (sum, value, cellIndex) => sum + (cellIndex !== targetIndex ? value : 0),
          0,
        ),
        opticFlowCellCount: t4bOpticFlowFallback ? t4bCount : 0,
        t4bPdMethod: `same as T4a: preferred=center-premotion offset; null=center+premotion offset`,
        groundTextureWidthDeg: Number(
          stimulus.maxStripeWidthDeg ?? stimulus.spatialPeriodDeg ?? 0
        ),
        spatialPeriodDeg: Number(stimulus.spatialPeriodDeg ?? 0),
        targetCompartments: targetKernel.nodes.length,
        vchCompartments: vchKernel.nodes.length,
        dtMs,
        outputDtMs: settings.outputDtMs,
        upstreamScaling: settings.upstreamScaling,
        pointMode: t4bMulticompartment
          ? `disabled; all T4a and T4b are passive cables`
          : `t4b-only; disabled for all T4a`,
        minVchInputContacts: settings.minVchInputContacts ?? 0,
        minT4aOutputContacts: settings.minT4aOutputContacts ?? 0,
        targetDriveModel: `Mi1/Tm3/Mi9/Mi4/C3 premotion to real cable`,
        targetPremotionModelVersion: `soma-dual-refit-zero-shift-tradeoff-0`,
        targetReadoutDefinition: `fit/output readout=equal weight over unique mapped T4a-to-VCH output compartments; interactive readouts retain dendrite, soma and axon`,
        targetPrimaryVoltageDefinition: `output-contact mean for noshift fit compatibility, plus dendrite, soma and axon anatomical traces`,
        targetInputSiteReadoutCount: targetInputCompartments.length,
        targetDendriteReadoutCount: targetReadouts.dendriteCompartments.length,
        targetDendriteInputTypes: h,
        targetSomaCompartment: targetReadouts.somaCompartment,
        targetAxonCompartment: targetReadouts.axonCompartment,
        targetAxonSelectionRule: `most distal compartment traversed by at least half of mapped T4a-to-VCH output-contact paths; closest output-contact fallback`,
        targetAxonOutputContactCount: targetReadouts.axonOutputContactCount,
        targetAxonTrunkCoverageCount: targetReadouts.axonTrunkCoverageCount,
        targetAxonTrunkCoverageThreshold: targetReadouts.axonTrunkCoverageThreshold,
        targetOutputContactReadoutCount: targetOutputCompartments.length,
        targetOutputContactWeighting: `unique-compartment-equal-weight`,
        otherT4DriveModel: `same premotion cascade mapped onto each cell's own passive cable`,
        otherT4UsesUpstream: true,
        otherT4PremotionRfFwhmDeg: 3,
        opticFlowCoordinateSource: t4bOpticFlowFallback
          ? `T4b unshifted premotion RF centers`
          : `disabled; legacy schema alias only`,
        opticFlowDirectionRule: t4bOpticFlowFallback
          ? `horizontal stimulus velocity projected onto each T4b normalized measured offsetAzimuth/offsetElevation PD vector`
          : `disabled; each T4a uses its own premotion home/null/preferred RF coordinates`,
        retinaActivityReadout: `baseline-normalized sigmoid of each T4a cable's anatomical axon voltage`,
        retinaActivityVHalfMv: -45,
        retinaActivitySlopeMv: 4,
        premotionVariant: premotionOptions.calibration ? `speedTuned` : `baseline`,
        premotionCalibration: premotionOptions.calibration?.label ?? `baseline`,
        premotionC3ConductanceShiftMs: c3ConductanceShiftMs,
        premotionMi1ConductanceShiftMs: M(`Mi1`, premotionOptions.calibration),
        premotionMi9ConductanceShiftMs: M(`Mi9`, premotionOptions.calibration),
        premotionRfFwhmDeg: premotionOptions.rfFwhmDeg,
        premotionRfFwhmByPath: premotionOptions.calibration?.receptiveFieldFwhmDeg ?? {
          Mi9: premotionOptions.rfFwhmDeg,
          Tm3: premotionOptions.rfFwhmDeg,
          Mi1: premotionOptions.rfFwhmDeg,
          Mi4: premotionOptions.rfFwhmDeg,
          C3: premotionOptions.rfFwhmDeg,
        },
        premotionPreStimulusMode: premotionOptions.preStimulusMode ?? `model-default`,
        relativeMotionDelaySec: Math.max(0, Number(stimulus.relativeMotionDelaySec ?? 0)),
        targetCableRaOhmCm: targetCableOptions.raOhmCm,
        targetCableRmOhmCm2: targetCableOptions.rmOhmCm2,
        targetCableCmUfPerCm2: targetCableOptions.cmFPerCm2 * 1e6,
        targetCableTotalLeakNS,
        targetCableTotalCapacitancePF,
        targetConductanceUnitNS: targetEquivalentConductanceUnitNS,
        pointConductanceUnitNS: targetEquivalentConductanceUnitNS,
        targetEquivalentPointLeakNS: targetShadowLeakNS,
        targetEquivalentPointCapacitancePF: targetCableTotalCapacitancePF,
        targetEquivalentPointTauMs: targetShadowTauMS,
        targetEquivalentPointConductanceUnitNS: targetEquivalentConductanceUnitNS,
        t4aCableConductanceRule: `per-cell visual conductance unit equals total cable leak / 0.5`,
        opticFlowInputTauMs: t4bOpticFlowFallback ? t4bOpticFlowSettings.inputTauMS : 0,
        opticFlowDriveCacheEnabled: t4bOpticFlowFallback,
        opticFlowDelayMs: t4bOpticFlowFallback ? t4bOpticFlowSettings.opticFlowDelayMS : 0,
        opticFlowLowPassTauMs: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowLowPassTauMS : 0,
        opticFlowEdgeWindowMs: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowEdgeWindowMS : 0,
        opticFlowEdgeGain: t4bOpticFlowFallback ? t4bOpticFlowSettings.opticFlowEdgeGain : 0,
        opticFlowGain: t4bOpticFlowFallback ? t4bOpticFlowSettings.opticFlowGain : 0,
        opticFlowVelocity50Dps: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowVelocity50DPS : 0,
        opticFlowSpeedModel: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowSpeedModel
          : `disabled-replaced-by-premotion`,
        opticFlowReferenceSpeedDps: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowReferenceSpeedDPS : 0,
        opticFlowReferenceFactor: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowReferenceFactor : 0,
        opticFlowSpeedHalfDps: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowSpeedHalfDPS : 0,
        opticFlowSpeedExponent: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowSpeedExponent : 0,
        opticFlowSpeedFloor: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowSpeedFloor : 0,
        opticFlowNullLogLinear: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullLogLinear : 0,
        opticFlowNullLogQuadratic: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullLogQuadratic : 0,
        opticFlowNullLogOffsetDps: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullLogOffsetDPS : 0,
        opticFlowNullOnsetHalfDps: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullOnsetHalfDPS : 0,
        opticFlowNullOnsetExponent: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullOnsetExponent : 0,
        opticFlowNullFactorCap: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullFactorCap : 0,
        opticFlowNullLeak: t4bOpticFlowFallback
          ? t4bOpticFlowSettings.opticFlowNullLeak : 0,
        opticFlowCalibration: t4bOpticFlowFallback
          ? `fixed noshift-refit zero-shift optic-flow calibration applied only to T4b`
          : `disabled; all cells use the noshift zero-shift premotion calibration`,
        opticFlowCalibrationBaselineMv: t4bPointELeakMv,
        opticFlowCalibrationPeakMv: peakPopulationPointVoltageMv,
        opticFlowCalibrationPeakSemMv: 0,
        opticFlowExternalConductanceNS: t4bOpticFlowFallback
          ? t4bOpticExternalConductanceNS : 0,
        opticFlowPointLeakNS: t4bPointLeakNS,
        opticFlowPointTauMs: t4bPointTauMS,
        opticFlowPointELeakMv: t4bPointELeakMv,
        opticFlowExcitatoryReversalMv: -21,
        upstreamMi9ReversalMv: I.Mi9,
        upstreamMi4C3ReversalMv: I.Mi4,
        vchToT4ReversalMv,
        vchToT4aReversalMv,
        t4ToVchReversalMv: -21,
        // Older aliases of the release parameters.
        releaseThresholdMv: excitatoryReleaseThresholdMv,
        releaseGainPerMv: excitatoryReleaseGainPerMv,
        excitatoryReleaseThresholdMv,
        excitatoryReleaseGainPerMv,
        excitatoryReleaseExponent,
        t4ReleaseThresholdMv: excitatoryReleaseThresholdMv,
        t4ReleaseGainInsidePerMv: excitatoryReleaseGainPerMv,
        t4ReleaseExponent: excitatoryReleaseExponent,
        inhibitoryReleaseThresholdMv,
        inhibitoryReleaseGainPerMv,
        inhibitoryReleaseExponent,
        vchReleaseThresholdMv: inhibitoryReleaseThresholdMv,
        vchReleaseGainInsidePerMv: inhibitoryReleaseGainPerMv,
        vchReleaseExponent: inhibitoryReleaseExponent,
        lpi15ReleaseThresholdMv,
        lpi15ReleaseGainInsidePerMv: lpi15ReleaseGainPerMv,
        lpi15ReleaseExponent,
        t4ToVchPerContactNS,
        vchToT4PerContactNS,
        inhibitoryReleaseTauMs,
        t4bToLpi15ReleaseThresholdMv,
        t4bToLpi15ReleaseGainInsidePerMv: t4bToLpi15ReleaseGainPerMv,
        t4bToLpi15ReleaseExponent: t4bToLpi15ReleaseExponent,
        vchToLpi15ReleaseThresholdMv,
        vchToLpi15ReleaseGainInsidePerMv: vchToLpi15ReleaseGainPerMv,
        vchToLpi15ReleaseExponent,
        lpi15ToVchReleaseThresholdMv,
        targetInhibitoryContactScale,
        targetInhibitoryReleaseThresholdMv,
        targetInhibitoryReleaseGainPerMv,
        targetInhibitoryReleaseExponent,
        targetInhibitoryReleaseTauMs,
        targetRetinaReadout: `selected cable axon with vch_sim sigmoid display; target is a readout role only`,
        peakT4VoltageMv,
        peakPopulationPointVoltageMv,
        peakPopulationPointRootId,
        peakPopulationAxonVoltageMv: peakPopulationPointVoltageMv,
        peakPopulationAxonRootId: peakPopulationPointRootId,
        peakTargetContactVoltageMv,
        peakTargetShadowVoltageMv,
        peakTargetRootVoltageMv,
        peakTargetMeanVoltageMv,
        peakTargetDendriteVoltageMv,
        peakTargetSomaVoltageMv,
        peakTargetAxonVoltageMv,
        peakTargetOutputContactMeanMv,
        targetPeakActivityMi9: targetPeakActivity.Mi9,
        targetPeakActivityTm3: targetPeakActivity.Tm3,
        targetPeakActivityMi1: targetPeakActivity.Mi1,
        targetPeakActivityMi4: targetPeakActivity.Mi4,
        targetPeakActivityC3: targetPeakActivity.C3,
        targetPeakRelativeConductanceMi9: targetPeakConductance.Mi9,
        targetPeakRelativeConductanceTm3: targetPeakConductance.Tm3,
        targetPeakRelativeConductanceMi1: targetPeakConductance.Mi1,
        targetPeakRelativeConductanceMi4: targetPeakConductance.Mi4,
        targetPeakRelativeConductanceC3: targetPeakConductance.C3,
        targetAcceptedLuminanceHomeMin: targetAcceptedMin.home,
        targetAcceptedLuminanceHomeMax: targetAcceptedMax.home,
        targetAcceptedLuminanceNullMin: targetAcceptedMin.null,
        targetAcceptedLuminanceNullMax: targetAcceptedMax.null,
        targetAcceptedLuminancePreferredMin: targetAcceptedMin.preferred,
        targetAcceptedLuminancePreferredMax: targetAcceptedMax.preferred,
        peakT4ReleaseFraction,
        peakReleasingT4Count,
        everReleasingT4Count: everReleased.reduce((sum, value) => sum + value, 0),
        peakT4ToVchConductanceNS,
        peakPointUpstreamConductanceNS: peakTargetShadowUpstreamConductanceNS,
        peakTargetShadowUpstreamConductanceNS,
        peakRawOpticFlowDrive,
        peakOpticFlowDrive,
        peakMeanOpticFlowDrive,
        peakOpticFlowVisualConductanceNS,
        vchMeanBaselineMv: vchMeanMv[0] ?? -65,
        vchMeanPeakMv: vchMeanMv.length ? Math.max(...vchMeanMv) : -65,
        vchMeanDeltaMv: vchMeanMv.length
          ? Math.max(...vchMeanMv) - (vchMeanMv[0] ?? -65)
          : 0,
      },
    };
  }
  // Worker entry point (message types: see the header).
  self.onmessage = (e2) => {
    try {
      let t2 = e2.data?.type === `precompute-premotion-chunk`
        ? precomputePremotionChunk(e2.data)
        : e2.data?.type === `prepare-fit-traces` ? prepareFitTraces(e2.data, (e3, t3) => {
        self.postMessage({ type: `progress`, fraction: e3, message: t3 });
      }) : V(e2.data, (e3, t3) => {
        self.postMessage({ type: `progress`, fraction: e3, message: t3 });
      });
      self.postMessage({ type: `result`, result: t2 });
    } catch (e3) {
      self.postMessage({ type: `error`, error: e3 instanceof Error ? e3.message : String(e3) });
    }
  };
})();
