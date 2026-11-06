#!/usr/bin/env bash
# Build the native cable solver (native-cable-solver.node) next to this script.
#   bash model/build-native.sh
# Needs g++ (C++17) and the Node.js headers; NODE_INCLUDE defaults to <node prefix>/include/node.
# -ffp-contract=off and -fno-fast-math keep the arithmetic identical to the JavaScript solver.
set -euo pipefail
cd "$(dirname "$0")"
NODE_INCLUDE="${NODE_INCLUDE:-$(cd "$(dirname "$(command -v node)")/.." && pwd)/include/node}"
[ -f "$NODE_INCLUDE/node_api.h" ] || { echo "node_api.h not found in $NODE_INCLUDE (set NODE_INCLUDE)" >&2; exit 1; }
g++ -O3 -std=c++17 -fPIC -shared -pthread -ffp-contract=off -fno-fast-math \
  -I"$NODE_INCLUDE" native-cable-solver.cc -o native-cable-solver.node
echo "built $(pwd)/native-cable-solver.node"
