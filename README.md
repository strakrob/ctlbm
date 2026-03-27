# CUDA D3Q27 LBM (SRT) with periodic-shift streaming

This repository contains a CUDA C implementation of a **D3Q27** lattice Boltzmann solver using a **single-relaxation-time (SRT/BGK)** collision model.

## Implemented requirements

- D3Q27 stencil.
- SRT collision operator.
- Poiseuille-type flow in a rectangular duct (periodic in streamwise `x`, no-slip walls at `y/z` boundaries).
- Streaming is implemented with a **periodic shift map** (`sx, sy, sz`) over one distribution array (no ping-pong swap).
- Output snapshots are written in VTK XML ImageData (`.vti`) format.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

The CMake config sets `CMAKE_CUDA_ARCHITECTURES=75` (Turing+) by default.

## Run

```bash
./build/lbmd3q27 [nx ny nz nsteps output_every]
```

Example:

```bash
./build/lbmd3q27 96 40 32 5000 1000
```

This will write files like:

- `duct_step_0001000.vti`
- `duct_step_0002000.vti`
- ...

which can be opened in ParaView.
