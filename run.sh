#!/bin/bash
# ./build/lbmd3q27 \
#   --mode C \
#   --nx 128 --ny 32 --nz 32 \
#   --steps 1000 \
#   --outlet zero-gauge-pressure \
#   --diag-every 100 \
#   --do-not-write-full-volume \
#   --write-cross-sections \
#   --output-every 1000 \
#   --output-dir build/mode_c \
#   --write-node-map

./build/lbmd3q27 \
  --mode B \
  --nx 128 --ny 64 --nz 32 \
  --steps 10000 \
  --diag-every 100 \
  --do-not-write-full-volume \
  --write-cross-sections \
  --output-every 1000 \
  --output-dir build/mode_b \
  --write-node-map
