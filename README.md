# Periodic-Shift D3Q27 Duct Solver

This repository implements a CUDA C/C++ lattice Boltzmann solver for 3D incompressible rectangular-duct flow with:

- D3Q27 lattice.
- SRT / BGK collision.
- double precision by default, optional single precision via `-DLBM_USE_FLOAT=ON`.
- one distribution-function storage set only.
- periodic-shift streaming with per-population logical offsets.
- `.vti` output for ParaView.
- three selectable streamwise boundary-condition modes.

## What “single-grid periodic-shift” means here

The solver keeps exactly one physical DF array of size `Q * Nx * Ny * Nz`.

It does **not** use:

- ping-pong fields,
- source/destination swaps,
- pull/push double buffering.

Instead, each population `q` carries cumulative logical shifts `sx[q]`, `sy[q]`, `sz[q]`. A logical access to `(q, x, y, z)` is mapped to physical storage by:

```text
physical_x = wrap(x - sx[q], Nx)
physical_y = wrap(y - sy[q], Ny)
physical_z = wrap(z - sz[q], Nz)
```

At every timestep the host updates the shift vectors by the discrete velocity:

```text
sx[q] <- wrap(sx[q] + cx[q], Nx)
sy[q] <- wrap(sy[q] + cy[q], Ny)
sz[q] <- wrap(sz[q] + cz[q], Nz)
```

The collision kernel then:

1. reads the logical field at time `t` via the previous shifts,
2. computes macroscopic fields, equilibrium, BGK collision, and Guo forcing if enabled,
3. writes the post-collision state back through the new shifts.

That write is the streamed state at `t + 1`, still inside the same DF array.

## Source layout

- `src/lbm.cuh`: shared types, D3Q27 constants, indexing helpers, shift-address mapping.
- `src/lbm_kernels.cu`: initialization, macroscopic recovery, BGK collision, Guo forcing, shift-streaming.
- `src/boundary_conditions.cu`: wall bounce-back, pressure inlet/outlet, velocity inlet, outlet treatments.
- `src/vti_writer.cu`: ASCII VTK ImageData output.
- `src/main.cu`: CLI, timestep loop, diagnostics, validation metrics.
- `tools/validate_rect_duct.py`: lightweight helper for summarizing `diagnostics.csv`.

## Boundary-condition modes

The streamwise mode is selected at runtime with `--mode`.

### Mode A

`--mode A`

- periodic in `x`,
- driven by a constant body force `--force-x`,
- y/z walls use explicit bounce-back on a wall-node layer.

### Mode B

`--mode B`

- non-periodic `x`,
- inlet density `--rho-inlet`,
- outlet density `--rho-outlet`,
- the full inlet and outlet boundary nodes are refreshed with a local non-equilibrium extrapolation rule built from the adjacent interior state and the target boundary density.

### Mode C

`--mode C`

- prescribed inlet velocity at `x = 0`,
- inlet profile selected with `--inlet-profile uniform|parabolic`,
- outlet selected with `--outlet extrapolation|zero-gauge-pressure`.

Implemented outlet variants:

- `extrapolation`: incoming outlet populations (`cx < 0`) are copied from the adjacent interior plane.
- `zero-gauge-pressure`: the outlet node is reconstructed with `rho = rho0` and the interior non-equilibrium part.

## Boundary ordering

Each timestep uses this order:

1. collide and stream in place while reading and writing through the current logical shifts,
2. advance the logical shifts on the host,
3. apply wall bounce-back on y/z wall nodes,
4. apply streamwise inlet/outlet reconstruction for Mode B or Mode C,
5. recover `rho`, `ux`, `uy`, `uz` for diagnostics and output when requested.

## Geometry stamping

The node-type map is now built on the host before initialization and uploaded
once to the GPU. This keeps custom geometry setup explicit and lets you stamp
additional primitives into the map without touching the device kernels.

Available host-side helpers in `src/boundary_conditions.cu`:

- `build_default_node_type_map(...)` for the baseline duct walls and x-boundary planes,
- `fill_box(...)`,
- `fill_ball(...)`,
- `fill_plane(...)`,
- `fill_cylinder(...)`.

The intended customization point is `apply_user_node_type_primitives(...)` in
`src/main.cu`. It is empty by default and contains example calls for stamping
extra wall, inlet, or outlet regions into the node map before the equilibrium
field is initialized.

## Build

CUDA 12+ is expected.

```bash
cmake -S . -B build
cmake --build build -j
```

Optional single precision:

```bash
cmake -S . -B build -DLBM_USE_FLOAT=ON
cmake --build build -j
```

The project targets `sm_75` and newer by default.

## Run

Show options:

```bash
./build/lbmd3q27 --help
```

Add `--write-cross-sections` to emit midpoint `xy`, `xz`, and `yz` slice files
alongside the standard 3D snapshot.

### Mode A example

```bash
./build/lbmd3q27 \
  --mode A \
  --nx 96 --ny 32 --nz 32 \
  --steps 8000 \
  --tau 0.8 \
  --force-x 1e-6 \
  --diag-every 200 \
  --output-every 1000 \
  --output-dir build/mode_a
```

### Mode B example

```bash
./build/lbmd3q27 \
  --mode B \
  --nx 128 --ny 32 --nz 32 \
  --steps 10000 \
  --tau 0.8 \
  --rho-inlet 1.001 \
  --rho-outlet 0.999 \
  --diag-every 200 \
  --output-every 1000 \
  --output-dir build/mode_b
```

### Mode C example

```bash
./build/lbmd3q27 \
  --mode C \
  --nx 128 --ny 32 --nz 32 \
  --steps 10000 \
  --tau 0.8 \
  --inlet-profile parabolic \
  --inlet-velocity 0.02 \
  --outlet extrapolation \
  --diag-every 200 \
  --output-every 1000 \
  --output-dir build/mode_c
```

## Output

The solver writes:

- `duct_step_0000000.vti`,
- `duct_step_0001000.vti`,
- ...
- `diagnostics.csv`.

With `--write-cross-sections`, each snapshot also writes:

- `duct_step_0001000_xy.vti` at `z = Nz / 2`,
- `duct_step_0001000_xz.vti` at `y = Ny / 2`,
- `duct_step_0001000_yz.vti` at `x = Nx / 2`.

The `.vti` files contain:

- `rho`,
- `ux`,
- `uy`,
- `uz`,
- `velocity` as a 3-component vector.

Open them directly in ParaView as `ImageData`.

## Validation and diagnostics

Console output and `diagnostics.csv` report:

- persistent memory demand on host, GPU, and combined, both per simulation and per cell,
- total mass,
- mean density,
- bulk velocity on a representative interior plane,
- volumetric flow rate,
- maximum streamwise velocity,
- residual as the change in bulk velocity between samples,
- `L2` error,
- `balance` metric,
- MLUPS throughput statistics.

Validation metric details:

- Mode A: compares the interior streamwise profile against a rectangular-duct analytical series using the imposed body force.
- Mode B: compares the interior streamwise profile against the same analytical series using the density-derived pressure gradient.
- Mode C: compares the inlet plane streamwise velocity against the requested inlet profile.

Balance metric details:

- Mode A: relative total-mass drift against the initial state.
- Mode B / C: relative inlet/outlet flow-rate mismatch.

MLUPS metric details:

- `mlups_current`: throughput over the last measured timestep interval between diagnostic samples,
- `mlups_min`: minimum measured interval throughput so far,
- `mlups_avg`: average throughput based on total measured lattice updates divided by total measured wall time,
- `mlups_max`: maximum measured interval throughput so far.

Summarize the last recorded diagnostic row:

```bash
python3 tools/validate_rect_duct.py build/mode_a/diagnostics.csv
```

## Notes on the wall model

The implementation uses an explicit wall-node layer on the four duct walls in `y` and `z` and performs bounce-back after the streamed logical field has been formed. This keeps the wall treatment compatible with the single-grid periodic-shift storage rule and easy to audit in `src/boundary_conditions.cu`.

## Known practical limits

- This implementation prioritizes clarity and inspectability over aggressive CUDA optimization.
- The pressure and velocity streamwise boundaries use local non-equilibrium extrapolation style reconstruction rather than a more elaborate higher-order formulation.
