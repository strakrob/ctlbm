#!/bin/bash
./build/lbmd3q27 \
  --mode C \
  --nx 128 --ny 64 --nz 64 \
  --steps 1000 \
  --tau 0.8 \
  --inlet-profile parabolic \
  --inlet-velocity 0.02 \
  --outlet zero-gauge-pressure \
  --diag-every 100 \
  --do-not-write-full-volume \
  --write-cross-sections \
  --output-every 500 \
  --output-dir build/mode_c \
  --write-node-map

# nvprof --print-gpu-trace --print-api-trace \
#   ./build/lbmd3q27 \
#   --mode A --nx 128 --ny 32 --nz 32 \
#   --steps 500 --tau 0.8 --force-x 1e-6 \
#   --diag-every 0 --output-every 0 --do-not-write-full-volume
