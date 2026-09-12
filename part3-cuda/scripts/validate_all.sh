#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
versions=(v01_single_tile v02_k_loop v03_multi_cta v04_tma v05_double_buffer v06_persistent v07_warp_specialized v08_two_cta v09_multi_consumer v09_tuned)
mkdir -p results/final
for name in "${versions[@]}"; do
 echo "BUILD $name"
 ./build.sh "$name"
 m=4096; n=4096; k=4096
 if [[ "$name" == v01* ]]; then m=128; n=128; k=64; fi
 if [[ "$name" == v02* ]]; then m=128; n=128; fi
 echo "RUN $name $m $n $k"
 timeout 120s "build/$name" "$m" "$n" "$k" > "results/final/$name.json"
 cat "results/final/$name.json"
done
