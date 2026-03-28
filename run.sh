#!/bin/bash
./build/lbmd3q27 \
  --mode C \
  --nx 256 --ny 64 --nz 64 \
  --steps 10000 \
  --tau 0.8 \
  --inlet-profile parabolic \
  --inlet-velocity 0.02 \
  --outlet zero-gauge-pressure \
  --diag-every 200 \
  --do-not-write-full-volume \
  --write-cross-sections \
  --output-every 1000 \
  --output-dir build/mode_c \
  --write-node-map