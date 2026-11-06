// Peaks and traces of one result, from the last 2 s of the run (written right after each run on the cluster,
// because the full result is large).
//   node peaks-traces.js <result.json.gz> <peaks.json> <traces.json> <key> <cells.json>
// peaks.json: peak and trough axon voltage of every T4a, its peak dendrite and soma voltage, and the vCH, LPi15
// and H2 peaks. traces.json: the axon, dendrite and soma traces of the T4a listed in cells.json (a JSON list of
// {rootId: ...}), the vCH, LPi15 and H2 traces, and the target cell's traces. Values are rounded to 1e-4.
const fs=require("fs"),zlib=require("zlib");
const [,,gz,peaksOut,tracesOut,key,rfPath]=process.argv;
const rf=new Set(JSON.parse(fs.readFileSync(rfPath,"utf8")).map(c=>c.rootId));
const r=JSON.parse(zlib.gunzipSync(fs.readFileSync(gz)));
const dt=(r.time[1]-r.time[0])*1000, n=Math.round(2000/dt), s=r.time.length-n;
const cut=(a)=>Array.from(a).slice(s).map(v=>Number(Number(v).toFixed(4)));
const mx=(a)=>{let m=-Infinity;for(let i=s;i<a.length;i++)if(a[i]>m)m=a[i];return Number(m.toFixed(4));};
const mn=(a)=>{let m=Infinity;for(let i=s;i<a.length;i++)if(a[i]<m)m=a[i];return Number(m.toFixed(4));};
const axonPeak={},axonTrough={},dendritePeak={},somaPeak={};
for (const [root,tr] of Object.entries(r.pointVoltageByRoot)) { axonPeak[root]=mx(tr); axonTrough[root]=mn(tr); }
const reg=r.regionVoltageByRoot||{};
for (const [root,b] of Object.entries(reg)) { dendritePeak[root]=mx(b.dendrite); somaPeak[root]=mx(b.soma); }
const sides={},h2={};
for (const sd of r.sides||[]) sides[sd.vchSide]={vchMeanMv:cut(sd.vchMeanMv),lpi15Mv:cut(sd.lpi15Mv),
  lpi15Side:sd.lpi15Side,vchPeak:mx(sd.vchMeanMv),vchTrough:mn(sd.vchMeanMv),lpi15Peak:mx(sd.lpi15Mv),
  lpi15ConductanceOnVchNS:cut(sd.lpi15ConductanceOnVchNS),t4ToVchConductanceNS:cut(sd.t4ToVchConductanceNS),
  vchToT4ConductanceNS:cut(sd.vchToT4ConductanceNS)};
for (const u of r.h2Sides||[]) h2[u.side]={h2MeanMv:cut(u.h2MeanMv),h2Peak:mx(u.h2MeanMv),h2Trough:mn(u.h2MeanMv),
  h2ToVchConductanceNS:cut(u.h2ToVchConductanceNS),t4bToH2ConductanceNS:cut(u.t4bToH2ConductanceNS)};
fs.writeFileSync(peaksOut,JSON.stringify({key,phase:r.meta.figurePhaseDeg??null,
  groundHemifield:r.meta.groundHemifield,cellCount:Object.keys(axonPeak).length,
  axonPeak,axonTrough,dendritePeak,somaPeak,
  sidesPeak:Object.fromEntries(Object.entries(sides).map(([k,v])=>[k,{vchPeak:v.vchPeak,vchTrough:v.vchTrough,lpi15Peak:v.lpi15Peak,lpi15Side:v.lpi15Side}])),
  h2Peak:Object.fromEntries(Object.entries(h2).map(([k,v])=>[k,{peak:v.h2Peak,trough:v.h2Trough}]))}));
const traces={};
for (const root of Object.keys(r.pointVoltageByRoot)) {
  if (!rf.has(root)) continue;
  traces[root]={axon:cut(r.pointVoltageByRoot[root])};
  if (reg[root]) { traces[root].dendrite=cut(reg[root].dendrite); traces[root].soma=cut(reg[root].soma); }
}
fs.writeFileSync(tracesOut,JSON.stringify({key,dtMs:dt,cells:traces,sides,h2,
  target:{rootId:r.targetRootId,dendrite:cut(r.targetDendriteMv),soma:cut(r.targetSomaMv),
          axon:cut(r.targetAxonMv),inhibitionNS:cut(r.targetVchFeedbackConductanceNS),
          vchAtContacts:cut(r.targetVchReleaseMeanMv)}}));
console.log("extracted "+key+": "+Object.keys(axonPeak).length+" cells, "+Object.keys(traces).length+" RF traces");
