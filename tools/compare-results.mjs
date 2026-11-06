// Compare two results (json.gz) value by value, ignoring `meta` (descriptive strings) and wall-clock fields.
//   node tools/compare-results.mjs a.json.gz b.json.gz
// Prints the number of values compared and how many differ; exit code 1 if any differ.
import fs from "node:fs";
import zlib from "node:zlib";

const [, , fa, fb] = process.argv;
const A = JSON.parse(zlib.gunzipSync(fs.readFileSync(fa)));
const B = JSON.parse(zlib.gunzipSync(fs.readFileSync(fb)));
const SKIP = new Set(["meta"]);
const TIMING = /(elapsed|wallclock|timing|durationMs|startedAt|finishedAt|generatedAt|runtime)/i;
let leaves = 0, diffs = 0, maxAbs = 0;
const report = [];
function walk(a, b, path) {
  if (a === null || b === null || typeof a !== "object" || typeof b !== "object") {
    leaves += 1;
    if (a !== b && !(Number.isNaN(a) && Number.isNaN(b))) {
      diffs += 1;
      if (typeof a === "number" && typeof b === "number") maxAbs = Math.max(maxAbs, Math.abs(a - b));
      if (report.length < 12) report.push(`${path}: ${JSON.stringify(a)?.slice(0, 60)} vs ${JSON.stringify(b)?.slice(0, 60)}`);
    }
    return;
  }
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    if (SKIP.has(k) || TIMING.test(k)) continue;
    if (!(k in a) || !(k in b)) {
      diffs += 1;
      if (report.length < 12) report.push(`${path}.${k}: present only in ${k in a ? "first" : "second"}`);
      continue;
    }
    walk(a[k], b[k], `${path}.${k}`);
  }
}
walk(A, B, "");
console.log(`${leaves} leaves compared, ${diffs} differ, max |difference| ${maxAbs}`);
for (const line of report) console.log("  " + line);
process.exit(diffs ? 1 : 0);
