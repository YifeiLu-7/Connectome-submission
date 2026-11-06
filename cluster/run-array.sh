#!/usr/bin/env bash
#SBATCH --job-name=t4a-fg
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=logs/%x-%A_%a.out
# Run every config listed in $WORK/manifest.txt (one key per line; the config is $WORK/configs/<key>.json) as a
# Slurm job array. Any number of array tasks can run at once:
#   cd $WORK && mkdir -p logs && sbatch --array=1-8 --export=ALL,REPO=...,CIRCUIT=...,SKELETONS=... $REPO/cluster/run-array.sh
# Each task claims one key at a time (mkdir claims/<key> is the lock) and writes results/<key>.json.gz. When no key
# is left, a second pass takes over claims older than 50 min that never finished (preempted tasks). A key is done
# when sums/<key>.sha256 exists; failed keys are marked claims/<key>/failed and not retried.
# Environment:
#   REPO       this repository (required; Slurm runs a copy of this script, so it cannot find the repository itself)
#   CIRCUIT    circuit JSON (required)
#   SKELETONS  directory with one <rootId>.csv skeleton per T4a and T4b (required)
#   WORK       run directory with manifest.txt and configs/ (default: the current directory)
#   NODE       Node.js 22 or newer (default: node)
#   CELLS      optional: a cell list for extract/peaks-traces.js; when set, peaks/<key>.json and traces/<key>.json are
#              written as well
set -uo pipefail
: "${REPO:?set REPO to the repository directory}"
: "${CIRCUIT:?set CIRCUIT to the circuit JSON}"
: "${SKELETONS:?set SKELETONS to the skeleton directory}"
WORK="${WORK:-$PWD}"
NODE="${NODE:-node}"
M="$REPO/model"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
export YIFEI_PARALLEL_THREADS=$THREADS YIFEI_CABLE_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OMP_DYNAMIC=FALSE
mkdir -p "$WORK/results" "$WORK/claims" "$WORK/sums"
[ -n "${CELLS:-}" ] && mkdir -p "$WORK/peaks" "$WORK/traces"

run_one() {
  local K=$1 out="$WORK/results/$1.json.gz"
  echo "START $K $(date +%T) $(hostname)"
  if "$NODE" "$M/run.mjs" --circuit "$CIRCUIT" --worker "$M/sim.worker.js" \
       --native-solver "$M/native-cable-solver.mjs" --native-addon "$M/native-cable-solver.node" \
       --parallel-solver "$M/parallel-cable-pool.mjs" --parallel-premotion "$M/parallel-premotion-pool.mjs" \
       --config "$WORK/configs/$K.json" --skeleton-dir "$SKELETONS" --output "$out" \
     && { [ -z "${CELLS:-}" ] || "$NODE" "$REPO/extract/peaks-traces.js" "$out" \
            "$WORK/peaks/$K.json" "$WORK/traces/$K.json" "$K" "$CELLS"; }; then
    ( cd "$WORK" && sha256sum "results/$K.json.gz" > "sums/$K.sha256" )
    echo "DONE $K $(date +%T)"
  else
    echo "FAIL $K $(date +%T)"
    touch "$WORK/claims/$K/failed"
  fi
}

for PASS in first second; do
  while read -r K; do
    K=${K%$'\r'}; [ -n "$K" ] || continue
    [ -s "$WORK/sums/$K.sha256" ] && continue
    if ! mkdir "$WORK/claims/$K" 2>/dev/null; then
      [ $PASS = second ] || continue
      [ -e "$WORK/claims/$K/failed" ] && continue
      T0=$(awk '{print $4}' "$WORK/claims/$K/owner" 2>/dev/null); T0=${T0:-0}
      [ $(( $(date +%s) - T0 )) -gt 3000 ] || continue
      rm -rf "$WORK/claims/$K"; mkdir "$WORK/claims/$K" 2>/dev/null || continue
      echo "RETAKE $K"
    fi
    echo "${SLURM_JOB_ID:-x} ${SLURM_ARRAY_TASK_ID:-0} $(hostname) $(date +%s)" > "$WORK/claims/$K/owner"
    run_one "$K"
  done < "$WORK/manifest.txt"
done
echo "NOTHING LEFT $(date +%T)"
