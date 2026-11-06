// Optic-flow drive of an earlier, simpler T4 model. sim.worker.js imports it only for the inactive optic-flow
// branch of point-model T4b; the circuit runs never call it.

const DEFAULT_SCREEN_WIDTH_DEG = 6;
const LOW_PASS_WEIGHTS = [0, 1, 2, 3, 4].map((k) => Math.exp(-k * 0.5));
const LOW_PASS_WEIGHT_SUM = LOW_PASS_WEIGHTS.reduce((sum, value) => sum + value, 0);
const LEGACY_RF_SAMPLES = (() => {
  const samples = [{ sigmaX: 0, sigmaY: 0, weight: 1 }];
  for (const ring of [{ radius: 1, offset: 0 }, { radius: 2, offset: 0.5 }]) {
    const weight = Math.exp(-0.5 * ring.radius * ring.radius);
    for (let index = 0; index < 6; index += 1) {
      const angle = 2 * Math.PI * (index + ring.offset) / 6;
      samples.push({
        sigmaX: ring.radius * Math.cos(angle),
        sigmaY: ring.radius * Math.sin(angle),
        weight,
      });
    }
  }
  return samples;
})();
const RF_SAMPLE_CACHE = new Map([[13, LEGACY_RF_SAMPLES]]);
const GOLDEN_ANGLE_RAD = Math.PI * (3 - Math.sqrt(5));

function opticFlowRfSampleCount(settings) {
  const requested = Math.round(Number(settings.opticFlowRfSampleCount ?? 13));
  if (!Number.isFinite(requested) || requested <= 13) return 13;
  return requested % 2 === 0 ? requested + 1 : requested;
}

function gaussianRfSamples(settings) {
  const sampleCount = opticFlowRfSampleCount(settings);
  const cached = RF_SAMPLE_CACHE.get(sampleCount);
  if (cached) return cached;

  // Deterministic antithetic quasi-Monte-Carlo samples of a unit 2-D Gaussian.
  // Pairing every point with its negative keeps the RF exactly centred, while
  // the golden angle avoids the repeated edge-crossing times produced by the
  // old two-ring, six-spoke quadrature.
  const pairCount = (sampleCount - 1) / 2;
  const samples = [{ sigmaX: 0, sigmaY: 0, weight: 1 }];
  for (let pair = 0; pair < pairCount; pair += 1) {
    const quantile = (pair + 0.5) / pairCount;
    const radius = Math.sqrt(-2 * Math.log(Math.max(1e-12, 1 - quantile)));
    const angle = (pair + 0.5) * GOLDEN_ANGLE_RAD;
    const sigmaX = radius * Math.cos(angle);
    const sigmaY = radius * Math.sin(angle);
    samples.push({ sigmaX, sigmaY, weight: 1 });
    samples.push({ sigmaX: -sigmaX, sigmaY: -sigmaY, weight: 1 });
  }
  RF_SAMPLE_CACHE.set(sampleCount, samples);
  return samples;
}

function clip(value, low, high) {
  return value < low ? low : value > high ? high : value;
}

function wrapDeg(value) {
  let wrapped = (value + 180) % 360;
  if (wrapped < 0) wrapped += 360;
  return wrapped - 180;
}

function angularDistanceDeg(a, b) {
  return Math.abs(wrapDeg(a - b));
}

function validationFig3Coordinate(stimulus, azimuthDeg, elevationDeg) {
  const relativeAzimuthDeg = wrapDeg(
    azimuthDeg - Number(stimulus.validationCenterAzimuthDeg ?? 0),
  );
  const relativeElevationDeg = elevationDeg
    - Number(stimulus.validationCenterElevationDeg ?? 0);
  const directionRad = Number(stimulus.validationDirectionDeg ?? 0) * Math.PI / 180;
  return relativeAzimuthDeg * Math.cos(directionRad)
    + relativeElevationDeg * Math.sin(directionRad);
}

function validationFig3DirectionSign(stimulus) {
  return stimulus.validationDirection === "PD" ? -1 : 1;
}

function validationFig3Luminance(stimulus, azimuthDeg, elevationDeg, timeSec) {
  const coordinateDeg = validationFig3Coordinate(stimulus, azimuthDeg, elevationDeg);
  const directionSign = validationFig3DirectionSign(stimulus);
  const speedDps = Math.max(1e-3, Math.abs(Number(stimulus.validationSpeedDps ?? 30)));
  const durationSec = Math.max(0.1, Number(stimulus.durationSec ?? 3));
  const edgeDeg = -directionSign * speedDps * durationSec * 0.45
    + directionSign * speedDps * timeSec;
  const contrast = clip(Number(stimulus.contrast ?? 1), 0, 1);
  const dark = 0.5 - 0.5 * contrast;
  const bright = 0.5 + 0.5 * contrast;
  const brightSide = directionSign * (coordinateDeg - edgeDeg) < 0;
  const onLuminance = brightSide ? bright : dark;
  return stimulus.validationPolarity === "off" ? 1 - onLuminance : onLuminance;
}

function validationFig5Luminance(stimulus, azimuthDeg, elevationDeg, timeSec) {
  const relativeAzimuthDeg = wrapDeg(
    azimuthDeg - Number(stimulus.validationCenterAzimuthDeg ?? 0),
  );
  const relativeElevationDeg = elevationDeg
    - Number(stimulus.validationCenterElevationDeg ?? 0);
  const directionRad = Number(stimulus.validationDirectionDeg ?? 0) * Math.PI / 180;
  const alongDeg = relativeAzimuthDeg * Math.cos(directionRad)
    + relativeElevationDeg * Math.sin(directionRad);
  const acrossDeg = -relativeAzimuthDeg * Math.sin(directionRad)
    + relativeElevationDeg * Math.cos(directionRad);
  const onsetSec = Math.max(0, Number(stimulus.validationOnsetSec ?? 1));
  const stepSec = Math.max(1e-3, Number(stimulus.validationStepSec ?? 0.08));
  const spanSteps = Math.max(1, Math.round(Number(stimulus.validationSpanSteps ?? 9)));
  const stepIndex = Math.floor((timeSec - onsetSec) / stepSec);
  const contrast = clip(Number(stimulus.contrast ?? 1), 0, 1);
  const dark = 0.5 - 0.5 * contrast;
  if (stepIndex < 0 || stepIndex >= spanSteps) return dark;
  const directionSign = validationFig3DirectionSign(stimulus);
  const barCenterDeg = directionSign
    * (stepIndex - (spanSteps - 1) / 2)
    * Number(stimulus.validationBarStepDeg ?? 2.25);
  const inside = Math.abs(alongDeg - barCenterDeg)
      <= Math.max(0.1, Number(stimulus.validationBarWidthDeg ?? 2.25)) / 2
    && Math.abs(acrossDeg)
      <= Math.max(0.1, Number(stimulus.validationBarHeightDeg ?? 20.25)) / 2;
  return inside ? 0.5 + 0.5 * contrast : dark;
}

function normalCdf(value) {
  const sign = value < 0 ? -1 : 1;
  const scaled = Math.abs(value) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * scaled);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t;
  const erf = sign * (1 - polynomial * Math.exp(-scaled * scaled));
  return 0.5 * (1 + erf);
}

function validationFig3AcceptedLuminance(
  stimulus,
  azimuthDeg,
  elevationDeg,
  timeSec,
  rfFwhmDeg,
) {
  const coordinateDeg = validationFig3Coordinate(stimulus, azimuthDeg, elevationDeg);
  const directionSign = validationFig3DirectionSign(stimulus);
  const speedDps = Math.max(1e-3, Math.abs(Number(stimulus.validationSpeedDps ?? 30)));
  const durationSec = Math.max(0.1, Number(stimulus.durationSec ?? 3));
  const edgeDeg = -directionSign * speedDps * durationSec * 0.45
    + directionSign * speedDps * timeSec;
  const sigmaDeg = Math.max(1e-6, Number(rfFwhmDeg) / 2.3548);
  const contrast = clip(Number(stimulus.contrast ?? 1), 0, 1);
  const dark = 0.5 - 0.5 * contrast;
  const brightProbability = normalCdf(
    directionSign * (edgeDeg - coordinateDeg) / sigmaDeg,
  );
  const onLuminance = dark + contrast * brightProbability;
  return stimulus.validationPolarity === "off" ? 1 - onLuminance : onLuminance;
}

function validationFig5AcceptedLuminance(
  stimulus,
  azimuthDeg,
  elevationDeg,
  timeSec,
  rfFwhmDeg,
) {
  const relativeAzimuthDeg = wrapDeg(
    azimuthDeg - Number(stimulus.validationCenterAzimuthDeg ?? 0),
  );
  const relativeElevationDeg = elevationDeg
    - Number(stimulus.validationCenterElevationDeg ?? 0);
  const directionRad = Number(stimulus.validationDirectionDeg ?? 0) * Math.PI / 180;
  const alongDeg = relativeAzimuthDeg * Math.cos(directionRad)
    + relativeElevationDeg * Math.sin(directionRad);
  const acrossDeg = -relativeAzimuthDeg * Math.sin(directionRad)
    + relativeElevationDeg * Math.cos(directionRad);
  const onsetSec = Math.max(0, Number(stimulus.validationOnsetSec ?? 1));
  const stepSec = Math.max(1e-3, Number(stimulus.validationStepSec ?? 0.08));
  const spanSteps = Math.max(1, Math.round(Number(stimulus.validationSpanSteps ?? 9)));
  const stepIndex = Math.floor((timeSec - onsetSec) / stepSec);
  const contrast = clip(Number(stimulus.contrast ?? 1), 0, 1);
  const dark = 0.5 - 0.5 * contrast;
  if (stepIndex < 0 || stepIndex >= spanSteps) return dark;
  const directionSign = validationFig3DirectionSign(stimulus);
  const barCenterDeg = directionSign
    * (stepIndex - (spanSteps - 1) / 2)
    * Number(stimulus.validationBarStepDeg ?? 2.25);
  const widthDeg = Math.max(0.1, Number(stimulus.validationBarWidthDeg ?? 2.25));
  const heightDeg = Math.max(0.1, Number(stimulus.validationBarHeightDeg ?? 20.25));
  const sigmaDeg = Math.max(1e-6, Number(rfFwhmDeg) / 2.3548);
  const alongWeight = normalCdf((barCenterDeg + widthDeg / 2 - alongDeg) / sigmaDeg)
    - normalCdf((barCenterDeg - widthDeg / 2 - alongDeg) / sigmaDeg);
  const acrossWeight = normalCdf((heightDeg / 2 - acrossDeg) / sigmaDeg)
    - normalCdf((-heightDeg / 2 - acrossDeg) / sigmaDeg);
  return dark + contrast * alongWeight * acrossWeight;
}

function hash01(x, y, seed) {
  let value = (
    Math.imul(x | 0, 374761393)
    + Math.imul(y | 0, 668265263)
    + Math.imul(seed | 0, 1442695041)
  ) >>> 0;
  value = Math.imul((value ^ (value >>> 13)) >>> 0, 1274126177) >>> 0;
  value = (value ^ (value >>> 16)) >>> 0;
  return value / 4294967295;
}

function textureValueAt(stimulus, worldAzimuthDeg) {
  if (stimulus.signal === "random") {
    const scaleDeg = Math.max(1, Math.min(Number(stimulus.spatialPeriodDeg ?? 7.5), 10));
    const seed = Number(stimulus.textureSeed ?? 17) | 0;
    const phaseDeg = (hash01(seed, 11, 503) - 0.5) * scaleDeg;
    const column = Math.floor((worldAzimuthDeg + phaseDeg) / scaleDeg);
    return ((column + seed) & 1) ? 1 : 0;
  }
  const periodDeg = Math.max(3, Number(stimulus.spatialPeriodDeg ?? 7.5) * 4);
  return 0.5 + 0.45 * Math.sin(worldAzimuthDeg * Math.PI * 2 / periodDeg);
}

function displacementAt(stimulus, timeSec, surface) {
  if (stimulus.motion === "sinusoidal") {
    const phaseDeg = surface === "ground"
      ? Number(stimulus.groundPhaseDeg ?? 0)
      : Number(stimulus.figurePhaseDeg ?? 90);
    return Number(stimulus.oscillationAmplitudeDeg ?? 5)
      * Math.sin(2 * Math.PI * Number(stimulus.temporalFrequencyHz ?? 2.5) * timeSec + phaseDeg * Math.PI / 180);
  }
  const velocity = surface === "ground"
    ? Number(stimulus.groundVelocityDps ?? 0)
    : Number(stimulus.figureVelocityDps ?? 90);
  return velocity * timeSec;
}

function delayedDisplacementAt(stimulus, timeSec, surface) {
  const delaySec = Math.max(0, Number(stimulus.relativeMotionDelaySec ?? 0));
  if (stimulus.pattern === "FG" && surface === "figure" && delaySec > 0 && timeSec < delaySec) {
    return displacementAt(stimulus, timeSec, "ground");
  }
  return displacementAt(stimulus, timeSec, surface);
}

function velocityAt(stimulus, timeSec, surface) {
  if (stimulus.motion !== "sinusoidal") {
    return surface === "ground"
      ? Number(stimulus.groundVelocityDps ?? 0)
      : Number(stimulus.figureVelocityDps ?? 90);
  }
  const phaseDeg = surface === "ground"
    ? Number(stimulus.groundPhaseDeg ?? 0)
    : Number(stimulus.figurePhaseDeg ?? 90);
  const frequencyHz = Number(stimulus.temporalFrequencyHz ?? 2.5);
  return Number(stimulus.oscillationAmplitudeDeg ?? 5)
    * 2 * Math.PI * frequencyHz
    * Math.cos(2 * Math.PI * frequencyHz * timeSec + phaseDeg * Math.PI / 180);
}

function delayedVelocityAt(stimulus, timeSec, surface) {
  const delaySec = Math.max(0, Number(stimulus.relativeMotionDelaySec ?? 0));
  if (stimulus.pattern === "FG" && surface === "figure" && delaySec > 0 && timeSec < delaySec) {
    return velocityAt(stimulus, timeSec, "ground");
  }
  return velocityAt(stimulus, timeSec, surface);
}

export function luminanceAt(stimulus, azimuthDeg, elevationDeg, timeSec) {
  if (stimulus.validationProtocol === "fig3-edge") {
    return validationFig3Luminance(stimulus, azimuthDeg, elevationDeg, timeSec);
  }
  if (stimulus.validationProtocol === "fig5-bar") {
    return validationFig5Luminance(stimulus, azimuthDeg, elevationDeg, timeSec);
  }
  if (angularDistanceDeg(azimuthDeg, 0) > Number(stimulus.backgroundCoverageDeg ?? 360) / 2) {
    return 0.5;
  }

  let figureDisplacementDeg = delayedDisplacementAt(stimulus, timeSec, "figure");
  const groundDisplacementDeg = delayedDisplacementAt(stimulus, timeSec, "ground");
  if (stimulus.pattern === "tF") figureDisplacementDeg = 0;

  const figureStartDeg = Number(stimulus.figureCenterDeg ?? -150);
  const figureCenterDeg = wrapDeg(figureStartDeg + figureDisplacementDeg);
  const screenCenterDeg = stimulus.motion === "constant"
    ? figureCenterDeg
    : wrapDeg(figureStartDeg);
  const figureWidthDeg = Number(stimulus.figureWidthDeg ?? 30);
  const figureHeightDeg = Number(stimulus.figureHeightDeg ?? 120);
  const insideHeight = Math.abs(elevationDeg) <= figureHeightDeg / 2;
  const insideFigure = angularDistanceDeg(azimuthDeg, figureCenterDeg) <= figureWidthDeg / 2 && insideHeight;
  const randomSignal = stimulus.signal === "random";
  const screenWidthDeg = Math.max(
    0,
    Number(stimulus.staticBlackScreenDeg ?? (randomSignal ? DEFAULT_SCREEN_WIDTH_DEG : 0)),
  );
  const insideScreen = randomSignal
    && screenWidthDeg > 0
    && insideHeight
    && angularDistanceDeg(azimuthDeg, screenCenterDeg) <= figureWidthDeg / 2 + screenWidthDeg;

  const stationary = textureValueAt(stimulus, azimuthDeg);
  const ground = textureValueAt(stimulus, azimuthDeg - groundDisplacementDeg);
  let figure = randomSignal ? 1 : textureValueAt(stimulus, azimuthDeg - figureDisplacementDeg);

  if (stimulus.pattern === "tF") {
    const flicker = Math.cos(2 * Math.PI * Number(stimulus.temporalFrequencyHz ?? 2.5) * timeSec);
    figure = 0.5 + (figure - 0.5) * flicker;
  }

  let value = stationary;
  if (stimulus.pattern === "G") {
    value = ground;
  } else if (["FG", "F", "tF"].includes(stimulus.pattern)) {
    const outside = stimulus.pattern === "FG" ? ground : stationary;
    value = insideFigure ? figure : insideScreen ? 0 : outside;
  } else if (["pE", "pEG", "rE", "rEG"].includes(stimulus.pattern)) {
    const boundaryDeg = figureStartDeg + figureDisplacementDeg;
    let unwrappedAzimuthDeg = azimuthDeg;
    for (; unwrappedAzimuthDeg < figureStartDeg - 180; ) unwrappedAzimuthDeg += 360;
    for (; unwrappedAzimuthDeg > figureStartDeg + 360; ) unwrappedAzimuthDeg -= 360;
    const behind = unwrappedAzimuthDeg <= boundaryDeg;
    const moving = stimulus.pattern.startsWith("p") ? behind : !behind;
    const outside = stimulus.pattern.endsWith("G") ? ground : stationary;
    value = moving ? figure : outside;
  }

  const contrast = clip(Number(stimulus.contrast ?? 1), 0, 1);
  return clip(0.5 + (value - 0.5) * contrast, 0, 1);
}

export function filteredLuminanceAt(stimulus, azimuthDeg, elevationDeg, timeSec, tauSec) {
  if (tauSec <= 0) return luminanceAt(stimulus, azimuthDeg, elevationDeg, timeSec);
  const sampleStepSec = tauSec * 0.5;
  let total = 0;
  for (let k = 0; k < LOW_PASS_WEIGHTS.length; k += 1) {
    total += LOW_PASS_WEIGHTS[k] * luminanceAt(
      stimulus,
      azimuthDeg,
      elevationDeg,
      Math.max(0, timeSec - k * sampleStepSec),
    );
  }
  return total / LOW_PASS_WEIGHT_SUM;
}

export function surfaceVelocityAt(stimulus, azimuthDeg, elevationDeg, timeSec) {
  if (["fig3-edge", "fig5-bar"].includes(stimulus.validationProtocol)) {
    return validationFig3DirectionSign(stimulus)
      * Math.max(1e-3, Math.abs(Number(stimulus.validationSpeedDps ?? 30)));
  }
  if (angularDistanceDeg(azimuthDeg, 0) > Number(stimulus.backgroundCoverageDeg ?? 360) / 2) {
    return 0;
  }

  let figureDisplacementDeg = delayedDisplacementAt(stimulus, timeSec, "figure");
  if (stimulus.pattern === "tF") figureDisplacementDeg = 0;
  const figureStartDeg = Number(stimulus.figureCenterDeg ?? -150);
  const figureCenterDeg = wrapDeg(figureStartDeg + figureDisplacementDeg);
  const screenCenterDeg = stimulus.motion === "constant"
    ? figureCenterDeg
    : wrapDeg(figureStartDeg);
  const figureWidthDeg = Number(stimulus.figureWidthDeg ?? 30);
  const figureHeightDeg = Number(stimulus.figureHeightDeg ?? 120);
  const insideHeight = Math.abs(elevationDeg) <= figureHeightDeg / 2;
  const insideFigure = angularDistanceDeg(azimuthDeg, figureCenterDeg) <= figureWidthDeg / 2 && insideHeight;
  const randomSignal = stimulus.signal === "random";
  const screenWidthDeg = Math.max(
    0,
    Number(stimulus.staticBlackScreenDeg ?? (randomSignal ? DEFAULT_SCREEN_WIDTH_DEG : 0)),
  );
  const insideScreen = randomSignal
    && screenWidthDeg > 0
    && insideHeight
    && angularDistanceDeg(azimuthDeg, screenCenterDeg) <= figureWidthDeg / 2 + screenWidthDeg;

  const figureVelocityDps = delayedVelocityAt(stimulus, timeSec, "figure");
  const groundVelocityDps = delayedVelocityAt(stimulus, timeSec, "ground");
  if (stimulus.pattern === "G") return groundVelocityDps;
  if (stimulus.pattern === "tF") return 0;
  if (["FG", "F"].includes(stimulus.pattern)) {
    const outsideVelocityDps = stimulus.pattern === "FG" ? groundVelocityDps : 0;
    return insideFigure || insideScreen ? figureVelocityDps : outsideVelocityDps;
  }
  if (["pE", "pEG", "rE", "rEG"].includes(stimulus.pattern)) {
    const boundaryDeg = figureStartDeg + figureDisplacementDeg;
    let unwrappedAzimuthDeg = azimuthDeg;
    for (; unwrappedAzimuthDeg < figureStartDeg - 180; ) unwrappedAzimuthDeg += 360;
    for (; unwrappedAzimuthDeg > figureStartDeg + 360; ) unwrappedAzimuthDeg -= 360;
    const behind = unwrappedAzimuthDeg <= boundaryDeg;
    const moving = stimulus.pattern.startsWith("p") ? behind : !behind;
    const outsideVelocityDps = stimulus.pattern.endsWith("G") ? groundVelocityDps : 0;
    return moving ? figureVelocityDps : outsideVelocityDps;
  }
  return 0;
}

function isLeftSide(side) {
  const normalized = String(side ?? "").trim().toUpperCase();
  return normalized === "L" || normalized === "LEFT";
}

export function opticFlowSpeedFactor(projectedVelocityDps, settings = {}) {
  const speedDps = Math.abs(Number(projectedVelocityDps));
  if (!Number.isFinite(speedDps) || speedDps <= 1e-9) return 0;
  const nullDirection = projectedVelocityDps < 0;
  const nullLeak = clip(Number(settings.opticFlowNullLeak ?? 0.03), 0, 1);
  if (settings.opticFlowSpeedModel === "directionalLogGaussian") {
    const preferredSpeedDps = Math.max(1e-6, Number(
      nullDirection
        ? settings.opticFlowNullPreferredSpeedDPS ?? 112
        : settings.opticFlowPreferredSpeedDPS ?? 28,
    ));
    const logSigma = Math.max(0.05, Number(
      nullDirection
        ? settings.opticFlowNullLogSpeedSigma ?? 1
        : settings.opticFlowLogSpeedSigma ?? 1,
    ));
    const logRatio = Math.log(speedDps / preferredSpeedDps);
    const tuning = Math.exp(-0.5 * (logRatio / logSigma) ** 2);
    return (nullDirection ? nullLeak : 1) * tuning;
  }
  if (settings.opticFlowSpeedModel === "directionalLowPass") {
    const halfSpeedDps = Math.max(1e-6, Number(
      nullDirection
        ? settings.opticFlowNullSpeedHalfDPS ?? 112
        : settings.opticFlowSpeedHalfDPS ?? 28,
    ));
    const exponent = Math.max(0.05, Number(
      nullDirection
        ? settings.opticFlowNullSpeedExponent ?? 1
        : settings.opticFlowSpeedExponent ?? 1,
    ));
    const tuning = 1 / (1 + (speedDps / halfSpeedDps) ** exponent);
    return (nullDirection ? nullLeak : 1) * tuning;
  }
  if (settings.opticFlowSpeedModel === "preferredLowPassConstantNull") {
    if (nullDirection) return nullLeak;
    const halfSpeedDps = Math.max(
      1e-6,
      Number(settings.opticFlowSpeedHalfDPS ?? 28),
    );
    const exponent = Math.max(0.05, Number(settings.opticFlowSpeedExponent ?? 1));
    return 1 / (1 + (speedDps / halfSpeedDps) ** exponent);
  }
  if ([
    "referenceNormalizedDirectionalLowPass",
    "referenceNormalizedPdLowPassNdHighPass",
    "referenceNormalizedPdLowPassNdLogQuadratic",
    "referenceNormalizedPdLowPassNdSafeLowPass",
    "referenceNormalizedPdLowPassNdBoundedLogQuadratic",
  ].includes(settings.opticFlowSpeedModel)) {
    const referenceSpeedDps = Math.max(
      1e-6,
      Number(settings.opticFlowReferenceSpeedDPS ?? 30),
    );
    const referenceFactor = Math.max(
      0,
      Number(settings.opticFlowReferenceFactor ?? (
        referenceSpeedDps
        / (Math.max(1e-6, Number(settings.opticFlowVelocity50DPS ?? 60)) + referenceSpeedDps)
      )),
    );
    if (
      nullDirection
      && settings.opticFlowSpeedModel === "referenceNormalizedPdLowPassNdLogQuadratic"
    ) {
      const logRatio = Math.log(speedDps / referenceSpeedDps);
      const linear = Number(settings.opticFlowNullLogLinear ?? 0);
      const quadratic = Number(settings.opticFlowNullLogQuadratic ?? 0);
      return referenceFactor
        * nullLeak
        * Math.exp(linear * logRatio + quadratic * logRatio * logRatio);
    }
    if (
      nullDirection
      && settings.opticFlowSpeedModel === "referenceNormalizedPdLowPassNdSafeLowPass"
    ) {
      const onsetHalfSpeedDps = Math.max(
        1e-6,
        Number(settings.opticFlowNullOnsetHalfDPS ?? 3),
      );
      const onsetExponent = Math.max(
        0.05,
        Number(settings.opticFlowNullOnsetExponent ?? 2),
      );
      const halfSpeedDps = Math.max(
        1e-6,
        Number(settings.opticFlowNullSpeedHalfDPS ?? 30),
      );
      const exponent = Math.max(
        0.05,
        Number(settings.opticFlowNullSpeedExponent ?? 1),
      );
      const speedFloor = clip(
        Number(settings.opticFlowNullSpeedFloor ?? 0.9),
        0,
        0.999999,
      );
      const tuningAt = (speed) => {
        const onset = 1 / (1 + (onsetHalfSpeedDps / speed) ** onsetExponent);
        const lowPass = speedFloor + (1 - speedFloor)
          / (1 + (speed / halfSpeedDps) ** exponent);
        return onset * lowPass;
      };
      return referenceFactor
        * nullLeak
        * tuningAt(speedDps)
        / Math.max(1e-12, tuningAt(referenceSpeedDps));
    }
    if (
      nullDirection
      && settings.opticFlowSpeedModel
        === "referenceNormalizedPdLowPassNdBoundedLogQuadratic"
    ) {
      const offsetSpeedDps = Math.max(
        1e-6,
        Number(settings.opticFlowNullLogOffsetDPS ?? 3),
      );
      const onsetHalfSpeedDps = Math.max(
        1e-6,
        Number(settings.opticFlowNullOnsetHalfDPS ?? 3),
      );
      const onsetExponent = Math.max(
        0.05,
        Number(settings.opticFlowNullOnsetExponent ?? 2),
      );
      const factorCap = Math.max(
        1e-6,
        Number(settings.opticFlowNullFactorCap ?? 4),
      );
      const linear = Number(settings.opticFlowNullLogLinear ?? 0);
      const quadratic = Number(settings.opticFlowNullLogQuadratic ?? 0);
      const tuningAt = (speed) => {
        const logRatio = Math.log(
          (speed + offsetSpeedDps) / (referenceSpeedDps + offsetSpeedDps),
        );
        const logRaw = clip(
          linear * logRatio + quadratic * logRatio * logRatio,
          -60,
          60,
        );
        const raw = Math.exp(logRaw);
        const bounded = factorCap * raw / (factorCap + raw);
        const onset = 1 / (1 + (onsetHalfSpeedDps / speed) ** onsetExponent);
        return onset * bounded;
      };
      return referenceFactor
        * nullLeak
        * tuningAt(speedDps)
        / Math.max(1e-12, tuningAt(referenceSpeedDps));
    }
    const halfSpeedDps = Math.max(1e-6, Number(
      nullDirection
        ? settings.opticFlowNullSpeedHalfDPS ?? 112
        : settings.opticFlowSpeedHalfDPS ?? 28,
    ));
    const exponent = Math.max(0.05, Number(
      nullDirection
        ? settings.opticFlowNullSpeedExponent ?? 1
        : settings.opticFlowSpeedExponent ?? 1,
    ));
    const nullHighPass = nullDirection
      && settings.opticFlowSpeedModel === "referenceNormalizedPdLowPassNdHighPass";
    const speedFloor = nullDirection
      ? 0
      : clip(Number(settings.opticFlowSpeedFloor ?? 0), 0, 0.999999);
    const tuning = nullHighPass
      ? speedDps ** exponent / (speedDps ** exponent + halfSpeedDps ** exponent)
      : speedFloor + (1 - speedFloor)
        / (1 + (speedDps / halfSpeedDps) ** exponent);
    const referenceTuning = nullHighPass
      ? referenceSpeedDps ** exponent
        / (referenceSpeedDps ** exponent + halfSpeedDps ** exponent)
      : speedFloor + (1 - speedFloor)
        / (1 + (referenceSpeedDps / halfSpeedDps) ** exponent);
    return referenceFactor
      * (nullDirection ? nullLeak : 1)
      * tuning / Math.max(1e-12, referenceTuning);
  }
  const velocity50Dps = Math.max(1e-6, Number(settings.opticFlowVelocity50DPS ?? 60));
  const saturation = speedDps / (velocity50Dps + speedDps);
  return (nullDirection ? nullLeak : 1) * saturation;
}

export function precomputeOpticFlowCells(cells, targetIndex) {
  const indices = [];
  const azimuthDeg = [];
  const elevationDeg = [];
  const preferredDirectionX = [];
  const missingPremotionRootIds = [];

  for (let index = 0; index < cells.length; index += 1) {
    if (index === targetIndex) continue;
    const cell = cells[index];
    // Use the same unshifted premotion RF center used to place the target
    // stimulus. The target's Mi subfields retain their own internal offsets;
    // the direct optic surrogate must not hide a timing shift in its position.
    if (
      !cell.premotion
      || !Number.isFinite(Number(cell.premotion.azimuth))
      || !Number.isFinite(Number(cell.premotion.elevation))
    ) {
      missingPremotionRootIds.push(String(cell.rootId));
      continue;
    }
    indices.push(index);
    azimuthDeg.push(Number(cell.premotion.azimuth));
    elevationDeg.push(Number(cell.premotion.elevation));
    // Tuning-copy T4b fallback: preserve each cell's measured PD axis instead of
    // inferring direction from hemisphere. The horizontal stimulus is projected
    // onto the normalized premotion offset vector.
    const offsetAzimuth = Number(cell.premotion.offsetAzimuth);
    const offsetElevation = Number(cell.premotion.offsetElevation);
    const offsetNorm = Math.hypot(offsetAzimuth, offsetElevation);
    preferredDirectionX.push(
      Number.isFinite(offsetNorm) && offsetNorm > 1e-9
        ? offsetAzimuth / offsetNorm
        : (isLeftSide(cell.side) ? -1 : 1),
    );
  }

  return {
    indices,
    azimuthDeg,
    elevationDeg,
    preferredDirectionX,
    missingPremotionRootIds,
  };
}

export function createOpticFlowDriveCache(stimulus, arrays, dtSec, settings = {}) {
  const inputTauSec = Math.max(0, Number(settings.inputTauMS ?? 20) / 1000);
  const edgeWindowSec = Math.max(dtSec, Number(settings.opticFlowEdgeWindowMS ?? 20) / 1000);
  const inputStride = inputTauSec > 0 ? inputTauSec * 0.5 / dtSec : 0;
  const edgeStride = edgeWindowSec / dtSec;
  const inputStrideSteps = Math.round(inputStride);
  const edgeWindowSteps = Math.round(edgeStride);
  const aligned = (
    dtSec > 0
    && Math.abs(inputStride - inputStrideSteps) < 1e-9
    && Math.abs(edgeStride - edgeWindowSteps) < 1e-9
  );

  const rfFwhmDeg = Math.max(0, Number(settings.opticFlowRfFwhmDeg ?? 0));
  const rfSigmaDeg = rfFwhmDeg > 0 ? clip(rfFwhmDeg, 1, 30) / 2.3548 : 0;
  const useAnalyticValidationRf = settings.opticFlowUseAnalyticValidationRf !== false;
  const analyticFig3 = useAnalyticValidationRf
    && stimulus.validationProtocol === "fig3-edge" && rfSigmaDeg > 0;
  const analyticFig5 = useAnalyticValidationRf
    && stimulus.validationProtocol === "fig5-bar" && rfSigmaDeg > 0;
  const analyticValidationRf = analyticFig3 || analyticFig5;
  const discreteSamples = gaussianRfSamples(settings);
  const samples = analyticValidationRf
    ? [{ sigmaX: 0, sigmaY: 0, weight: 1 }]
    : rfSigmaDeg > 0
    ? discreteSamples
    : [{ sigmaX: 0, sigmaY: 0, weight: 1 }];
  const spatialWeightSum = samples.reduce((sum, sample) => sum + sample.weight, 0);
  const samplesPerCell = samples.length;
  const sampleCount = arrays.indices.length * samplesPerCell;
  const sampleAzimuthDeg = new Float64Array(sampleCount);
  const sampleElevationDeg = new Float64Array(sampleCount);
  const sampleWeight = new Float64Array(sampleCount);

  for (let cell = 0; cell < arrays.indices.length; cell += 1) {
    for (let sampleIndex = 0; sampleIndex < samplesPerCell; sampleIndex += 1) {
      const sample = samples[sampleIndex];
      const flatIndex = cell * samplesPerCell + sampleIndex;
      sampleAzimuthDeg[flatIndex] = arrays.azimuthDeg[cell] + sample.sigmaX * rfSigmaDeg;
      sampleElevationDeg[flatIndex] = arrays.elevationDeg[cell] + sample.sigmaY * rfSigmaDeg;
      sampleWeight[flatIndex] = sample.weight;
    }
  }

  const lowPassTapCount = inputTauSec > 0 ? LOW_PASS_WEIGHTS.length : 1;
  const maximumHistorySteps = edgeWindowSteps
    + Math.max(0, lowPassTapCount - 1) * inputStrideSteps;
  const frameCount = Math.max(2, maximumHistorySteps + 2);
  const frameTags = new Int32Array(frameCount);
  frameTags.fill(-1);

  return {
    enabled: aligned,
    stimulus,
    arrays,
    dtSec,
    inputStrideSteps,
    edgeWindowSteps,
    lowPassTapCount,
    samplesPerCell,
    sampleAzimuthDeg,
    sampleElevationDeg,
    sampleWeight,
    spatialWeightSum,
    analyticFig3,
    analyticFig5,
    rfFwhmDeg,
    frameCount,
    frameTags,
    frames: new Float64Array(frameCount * sampleCount),
    output: new Float64Array(arrays.indices.length),
    lastStep: -1,
  };
}

function cachedOpticFlowDriveBatch(timeSec, stimulus, dtSec, settings, arrays, cache) {
  if (
    !cache?.enabled
    || cache.stimulus !== stimulus
    || cache.arrays !== arrays
    || Math.abs(cache.dtSec - dtSec) > 1e-12
  ) return null;

  const stepFloat = timeSec / dtSec;
  const step = Math.round(stepFloat);
  if (step < 0 || Math.abs(stepFloat - step) > 1e-8 || step < cache.lastStep) return null;

  const sampleCount = cache.sampleAzimuthDeg.length;
  const populateFrame = (frameStep) => {
    const normalizedStep = Math.max(0, frameStep);
    const slot = normalizedStep % cache.frameCount;
    if (cache.frameTags[slot] === normalizedStep) return slot * sampleCount;
    const frameOffset = slot * sampleCount;
    const frameTimeSec = normalizedStep * dtSec;
    for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
      cache.frames[frameOffset + sampleIndex] = cache.analyticFig3
        ? validationFig3AcceptedLuminance(
          stimulus,
          cache.sampleAzimuthDeg[sampleIndex],
          cache.sampleElevationDeg[sampleIndex],
          frameTimeSec,
          cache.rfFwhmDeg,
        )
        : cache.analyticFig5
        ? validationFig5AcceptedLuminance(
          stimulus,
          cache.sampleAzimuthDeg[sampleIndex],
          cache.sampleElevationDeg[sampleIndex],
          frameTimeSec,
          cache.rfFwhmDeg,
        )
        : luminanceAt(
          stimulus,
          cache.sampleAzimuthDeg[sampleIndex],
          cache.sampleElevationDeg[sampleIndex],
          frameTimeSec,
        );
    }
    cache.frameTags[slot] = normalizedStep;
    return frameOffset;
  };

  const firstMissingStep = cache.lastStep < 0 ? 0 : cache.lastStep + 1;
  for (let frameStep = firstMissingStep; frameStep <= step; frameStep += 1) {
    populateFrame(frameStep);
  }
  cache.lastStep = Math.max(cache.lastStep, step);

  const frameOffset = (frameStep) => populateFrame(Math.max(0, frameStep));
  const nowFrameOffsets = [];
  const beforeFrameOffsets = [];
  for (let tap = 0; tap < cache.lowPassTapCount; tap += 1) {
    const tapOffset = tap * cache.inputStrideSteps;
    nowFrameOffsets.push(frameOffset(step - tapOffset));
    beforeFrameOffsets.push(frameOffset(step - cache.edgeWindowSteps - tapOffset));
  }

  const edgeGain = Math.max(0, Number(settings.opticFlowEdgeGain ?? 8));
  const responseGain = Math.max(0, Number(settings.opticFlowGain ?? 2));
  const temporalWeightSum = cache.lowPassTapCount > 1 ? LOW_PASS_WEIGHT_SUM : 1;

  for (let cell = 0; cell < arrays.indices.length; cell += 1) {
    let weightedNow = 0;
    let weightedBefore = 0;
    const sampleStart = cell * cache.samplesPerCell;
    const sampleEnd = sampleStart + cache.samplesPerCell;
    for (let sampleIndex = sampleStart; sampleIndex < sampleEnd; sampleIndex += 1) {
      let filteredNow = 0;
      let filteredBefore = 0;
      for (let tap = 0; tap < cache.lowPassTapCount; tap += 1) {
        const temporalWeight = cache.lowPassTapCount > 1 ? LOW_PASS_WEIGHTS[tap] : 1;
        filteredNow += temporalWeight * cache.frames[nowFrameOffsets[tap] + sampleIndex];
        filteredBefore += temporalWeight * cache.frames[beforeFrameOffsets[tap] + sampleIndex];
      }
      const spatialWeight = cache.sampleWeight[sampleIndex];
      weightedNow += spatialWeight * filteredNow / temporalWeightSum;
      weightedBefore += spatialWeight * filteredBefore / temporalWeightSum;
    }

    const edgeDrive = Math.min(
      1,
      Math.max(0, weightedNow / cache.spatialWeightSum - weightedBefore / cache.spatialWeightSum) * edgeGain,
    );
    const localVelocityDps = surfaceVelocityAt(
      stimulus,
      arrays.azimuthDeg[cell],
      arrays.elevationDeg[cell],
      timeSec,
    );
    const projectedVelocityDps = localVelocityDps * arrays.preferredDirectionX[cell];
    const motionFactor = opticFlowSpeedFactor(projectedVelocityDps, settings);
    cache.output[cell] = clip(responseGain * edgeDrive * motionFactor, 0, 2);
  }
  return cache.output;
}

export function createOpticFlowTemporalFilter(cellCount, dtSec, settings = {}) {
  const delaySec = Math.max(0, Number(settings.opticFlowDelayMS ?? 68) / 1000);
  const lowPassTauSec = Math.max(0, Number(settings.opticFlowLowPassTauMS ?? 96) / 1000);
  const delaySteps = Math.max(0, Math.round(delaySec / dtSec));
  const frameCount = Math.max(1, delaySteps + 1);
  return {
    cellCount,
    delaySteps,
    frameCount,
    step: 0,
    alpha: lowPassTauSec > 0 ? 1 - Math.exp(-dtSec / lowPassTauSec) : 1,
    frames: new Float64Array(frameCount * cellCount),
    filtered: new Float64Array(cellCount),
  };
}

export function filterOpticFlowDrive(rawDrive, state) {
  if (rawDrive.length !== state.cellCount) {
    throw new Error(`Optic-flow temporal filter expected ${state.cellCount} cells, received ${rawDrive.length}.`);
  }

  const writeFrame = state.step % state.frameCount;
  state.frames.set(rawDrive, writeFrame * state.cellCount);
  const delayedStep = state.step - state.delaySteps;
  const readFrame = delayedStep >= 0 ? delayedStep % state.frameCount : -1;
  const readOffset = readFrame * state.cellCount;

  for (let index = 0; index < state.cellCount; index += 1) {
    const delayed = readFrame >= 0 ? state.frames[readOffset + index] : 0;
    state.filtered[index] += state.alpha * (delayed - state.filtered[index]);
  }
  state.step += 1;
  return state.filtered;
}

export function opticFlowDriveBatch(timeSec, stimulus, dtSec, settings, arrays, cache = null) {
  const cachedOutput = cachedOpticFlowDriveBatch(timeSec, stimulus, dtSec, settings, arrays, cache);
  if (cachedOutput) return cachedOutput;
  const inputTauSec = Number(settings.inputTauMS ?? 20) / 1000;
  const edgeWindowSec = Math.max(dtSec, Number(settings.opticFlowEdgeWindowMS ?? 20) / 1000);
  const edgeGain = Math.max(0, Number(settings.opticFlowEdgeGain ?? 8));
  const responseGain = Math.max(0, Number(settings.opticFlowGain ?? 2));
  const previousTimeSec = Math.max(0, timeSec - edgeWindowSec);
  const output = new Float64Array(arrays.indices.length);

  const rfFwhmDeg = Math.max(0, Number(settings.opticFlowRfFwhmDeg ?? 0));
  const rfSigmaDeg = rfFwhmDeg > 0 ? clip(rfFwhmDeg, 1, 30) / 2.3548 : 0;
  const useAnalyticValidationRf = settings.opticFlowUseAnalyticValidationRf !== false;
  const analyticFig3 = useAnalyticValidationRf
    && stimulus.validationProtocol === "fig3-edge" && rfSigmaDeg > 0;
  const analyticFig5 = useAnalyticValidationRf
    && stimulus.validationProtocol === "fig5-bar" && rfSigmaDeg > 0;
  const samples = gaussianRfSamples(settings);
  const spatialWeightSum = samples.reduce((sum, sample) => sum + sample.weight, 0);

  function driveFromLuminance(now, before, localVelocityDps, preferredDirectionX) {
    const edgeDrive = Math.min(1, Math.max(0, now - before) * edgeGain);
    const projectedVelocityDps = localVelocityDps * preferredDirectionX;
    const motionFactor = opticFlowSpeedFactor(projectedVelocityDps, settings);
    return clip(responseGain * edgeDrive * motionFactor, 0, 2);
  }

  function sampleLuminance(azimuthDeg, elevationDeg, sampleTimeSec) {
    if (analyticFig3 || analyticFig5) {
      const alignedTimeSec = Math.max(0, Math.round(sampleTimeSec / dtSec) * dtSec);
      const acceptedLuminance = analyticFig3
        ? validationFig3AcceptedLuminance
        : validationFig5AcceptedLuminance;
      if (inputTauSec <= 0) {
        return acceptedLuminance(
          stimulus,
          azimuthDeg,
          elevationDeg,
          alignedTimeSec,
          rfFwhmDeg,
        );
      }
      const sampleStepSec = inputTauSec * 0.5;
      let total = 0;
      for (let tap = 0; tap < LOW_PASS_WEIGHTS.length; tap += 1) {
        total += LOW_PASS_WEIGHTS[tap] * acceptedLuminance(
          stimulus,
          azimuthDeg,
          elevationDeg,
          Math.max(
            0,
            Math.round((alignedTimeSec - tap * sampleStepSec) / dtSec) * dtSec,
          ),
          rfFwhmDeg,
        );
      }
      return total / LOW_PASS_WEIGHT_SUM;
    }
    return filteredLuminanceAt(
      stimulus,
      azimuthDeg,
      elevationDeg,
      sampleTimeSec,
      inputTauSec,
    );
  }

  for (let i = 0; i < arrays.indices.length; i += 1) {
    if (rfSigmaDeg <= 0) {
      const centerAzimuthDeg = arrays.azimuthDeg[i];
      const centerElevationDeg = arrays.elevationDeg[i];
      output[i] = driveFromLuminance(
        sampleLuminance(centerAzimuthDeg, centerElevationDeg, timeSec),
        sampleLuminance(centerAzimuthDeg, centerElevationDeg, previousTimeSec),
        surfaceVelocityAt(stimulus, centerAzimuthDeg, centerElevationDeg, timeSec),
        arrays.preferredDirectionX[i],
      );
      continue;
    }
    if (analyticFig3 || analyticFig5) {
      const centerAzimuthDeg = arrays.azimuthDeg[i];
      const centerElevationDeg = arrays.elevationDeg[i];
      output[i] = driveFromLuminance(
        sampleLuminance(centerAzimuthDeg, centerElevationDeg, timeSec),
        sampleLuminance(centerAzimuthDeg, centerElevationDeg, previousTimeSec),
        surfaceVelocityAt(stimulus, centerAzimuthDeg, centerElevationDeg, timeSec),
        arrays.preferredDirectionX[i],
      );
      continue;
    }
    let weightedNow = 0;
    let weightedBefore = 0;
    for (const sample of samples) {
      const sampleAzimuthDeg = arrays.azimuthDeg[i] + sample.sigmaX * rfSigmaDeg;
      const sampleElevationDeg = arrays.elevationDeg[i] + sample.sigmaY * rfSigmaDeg;
      weightedNow += sample.weight * sampleLuminance(
        sampleAzimuthDeg,
        sampleElevationDeg,
        timeSec,
      );
      weightedBefore += sample.weight * sampleLuminance(
        sampleAzimuthDeg,
        sampleElevationDeg,
        previousTimeSec,
      );
    }
    const centerVelocityDps = surfaceVelocityAt(
      stimulus,
      arrays.azimuthDeg[i],
      arrays.elevationDeg[i],
      timeSec,
    );
    output[i] = driveFromLuminance(
      weightedNow / spatialWeightSum,
      weightedBefore / spatialWeightSum,
      centerVelocityDps,
      arrays.preferredDirectionX[i],
    );
  }
  return output;
}

function sigmoid(value) {
  return 1 / (1 + Math.exp(-clip(value, -40, 40)));
}

export function sigmoidMembraneActivity(voltageMv, leakMv = -60) {
  const raw = sigmoid((voltageMv + 45) / 4);
  const resting = sigmoid((leakMv + 45) / 4);
  return clip((raw - resting) / Math.max(1e-9, 1 - resting), 0, 1);
}
