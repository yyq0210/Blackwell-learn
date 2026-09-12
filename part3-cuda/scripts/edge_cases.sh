#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/edge
while read -r name m n k; do
 echo "EDGE $name $m $n $k"
 timeout 120s "build/$name" "$m" "$n" "$k" 12345 > "results/edge/${name}_${m}_${n}_${k}.json"
done <<'CASES'
v01_single_tile 128 128 64
v02_k_loop 128 128 192
v03_multi_cta 256 384 192
v04_tma 256 384 192
v05_double_buffer 256 384 192
v06_persistent 2048 2048 192
v07_warp_specialized 2048 2048 192
v08_two_cta 4096 2048 192
v09_multi_consumer 8192 4096 192
v09_tuned 8192 4096 192
v09_tuned 512 256 64
CASES
