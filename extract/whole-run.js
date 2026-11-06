// Compact copy of one result: the axon, dendrite and soma trace of every T4a over the whole run (rounded to
// 0.001 mV), and the vCH, LPi15 and H2 traces.
//   node whole-run.js <result.json.gz> <out.json.gz> <key>
const fs = require("fs"), zlib = require("zlib");
const [, , gz, out, key] = process.argv;
const r = JSON.parse(zlib.gunzipSync(fs.readFileSync(gz)));
const R3 = (a) => (a ? Array.from(a).map((v) => Math.round(v * 1000) / 1000) : null);
const cells = {};
for (const [root, tr] of Object.entries(r.pointVoltageByRoot)) {
  const reg = (r.regionVoltageByRoot || {})[root];
  cells[root] = { axon: R3(tr), dendrite: reg ? R3(reg.dendrite) : null, soma: reg ? R3(reg.soma) : null };
}
const sides = (r.sides || []).map((sd) => ({ vchSide: sd.vchSide, lpi15Side: sd.lpi15Side, vchMeanMv: R3(sd.vchMeanMv),
  lpi15Mv: R3(sd.lpi15Mv), lpi15ConductanceOnVchNS: R3(sd.lpi15ConductanceOnVchNS),
  t4ToVchConductanceNS: R3(sd.t4ToVchConductanceNS), vchToT4ConductanceNS: R3(sd.vchToT4ConductanceNS) }));
const h2 = (r.h2Sides || []).map((u) => ({ side: u.side, h2MeanMv: R3(u.h2MeanMv),
  h2ToVchConductanceNS: R3(u.h2ToVchConductanceNS), t4bToH2ConductanceNS: R3(u.t4bToH2ConductanceNS) }));
fs.writeFileSync(out, zlib.gzipSync(JSON.stringify({ key, time: R3(r.time), meta: r.meta, cells, sides, h2 })));
console.log("fgc extract " + key + ": " + Object.keys(cells).length + " cells, " + r.time.length + " samples");
