// Release-site voltage histograms (result.releaseSiteHist) of one result, from the last 2 s of the run.
//   node release-histograms.js <result.json.gz> <out.json> <key>
const fs=require("fs"),zlib=require("zlib");
const [,,gz,out,key]=process.argv;
const r=JSON.parse(zlib.gunzipSync(fs.readFileSync(gz)));
const dt=(r.time[1]-r.time[0])*1000, n=Math.round(2000/dt), s=r.time.length-n;
const h=r.releaseSiteHist;
if(!h||h.frames.length!==r.time.length) throw new Error("frames "+(h&&h.frames.length)+" vs time "+r.time.length);
fs.writeFileSync(out,JSON.stringify({key,dtMs:dt,loMv:h.loMv,binMv:h.binMv,bins:h.bins,groups:h.groups,frames:h.frames.slice(s)}));
console.log("rs "+key+": "+h.groups.length+" groups, "+(h.frames.length-s)+" frames of "+h.frames.length);
