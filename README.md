# grr — General Relativistic Ray Tracer

A Rust engine for simulating how light and particles move near black holes, compiled to WebAssembly so it runs in your browser.

**[Try the live demo](#running-the-demo)** — watch photons bend around a black hole in real-time.

## What is this?

Black holes warp spacetime. Light passing near one doesn't travel in straight lines — it curves, spirals, and can even get trapped. This project solves Einstein's equations numerically to compute those paths exactly, then visualizes them in 3D with three.js.

Everything you see in the demo is physically accurate. The math runs in Rust compiled to WebAssembly; the rendering is JavaScript/WebGL.

### What you're seeing in the demo

- **Black sphere** — the event horizon. Nothing escapes from inside this.
- **Cyan ring** — the ISCO (Innermost Stable Circular Orbit) at r = 6M. This is the closest a planet or particle can orbit without spiraling in. In real black holes, the inner edge of the accretion disk sits here.
- **Gold ring** — the photon sphere at r = 3M. Light itself can orbit here, but it's unstable — the slightest nudge sends it inward or outward.
- **Violet rosette** — a precessing orbit. In Newton's gravity, elliptical orbits are closed loops. In Einstein's, they precess — each orbit rotates slightly. This is the same effect that causes Mercury's perihelion advance.
- **Colored rays** — photons fired toward the black hole from far away:
  - **Red** — captured. These photons had too little angular momentum and fell in.
  - **Amber** — near-critical. These just barely miss the photon sphere and loop around multiple times before escaping or falling in.
  - **Teal** — deflected. These pass by and get bent, but escape to infinity.
- **White dots** — tracer particles showing the motion along each path.

### What the sliders do

| Slider | What it controls |
|--------|-----------------|
| **spin a** | Black hole spin (0 = Schwarzschild, non-spinning; 0.98 = near-extremal Kerr). Cranking this up shows frame dragging — spacetime itself gets twisted by the spin. |
| **rays** | Number of photon trajectories. More rays = denser visualization. |
| **max b/M** | Maximum impact parameter. Think of this as how far from center the outermost photon is aimed. |
| **source r/M** | How far away the photons start. |
| **speed** | Animation speed. Set to 0 to freeze. |

## Architecture

```
grr (workspace)
├── core/          Spacetime metrics + Christoffel symbols (Schwarzschild, Kerr)
├── integrator/    ODE solvers (RK4, Dormand-Prince 5(4), Dormand-Prince 8(7))
├── geodesic/      Geodesic equation solver (state, RHS, conserved quantities)
├── tetrad/        ZAMO tetrad construction
├── camera/        Pixel-to-photon initial condition mapping
└── grr-wasm/      WebAssembly wrapper + three.js demo
    ├── src/lib.rs    wasm-bindgen API
    ├── index.html    Interactive 3D visualization
    └── pkg/          Built WASM + JS glue (generated)
```

All five core crates are **pure math** — zero external dependencies, no IO, no threading, no FFI. They compile to `wasm32-unknown-unknown` with no changes.

### WASM API

```typescript
// Trace a geodesic — returns Cartesian trajectory points
trace_schwarzschild(state: Float64Array, lambda_end, tolerance, max_steps, sample_every) → TraceResult
trace_kerr(spin, state: Float64Array, ...) → TraceResult

// Build initial conditions (solves for k^t automatically)
solve_schwarzschild_state(r, theta, phi, kr, ktheta, kphi, is_null) → Float64Array
solve_kerr_state(spin, r, theta, phi, kr, ktheta, kphi, is_null) → Float64Array

// Pre-built initial conditions
circular_orbit_ic(spin, r, prograde) → Float64Array
photon_sphere_ic(spin, prograde) → Float64Array

// Reference values
isco_radius(spin, prograde) → number
horizon_radius(spin) → number
```

## Running the demo

### Prerequisites

- [Rust](https://rustup.rs/)
- wasm32 target: `rustup target add wasm32-unknown-unknown`
- wasm-pack: `cargo install wasm-pack`

### Build and run

```bash
# Build the WASM module
wasm-pack build grr-wasm --target web

# Serve locally (WASM requires HTTP, not file://)
cd grr-wasm && python3 -m http.server 8080

# Open http://localhost:8080
```

### Running Rust tests

```bash
cargo test --workspace
```

## The physics

All quantities use **geometric units**: G = c = M = 1.

| Concept | Value (M=1) | Real-world meaning |
|---------|-------------|-------------------|
| Event horizon | r = 2 (Schwarzschild) | Point of no return |
| Photon sphere | r = 3 | Light can orbit here (unstable) |
| ISCO | r = 6 | Innermost stable orbit for massive particles |
| Critical impact parameter | b = 3√3 ≈ 5.196 | Photons aimed closer than this get captured |

The geodesic equation — the path of a free particle in curved spacetime — is:

```
d²xᵘ/dλ² + Γᵘₐᵦ (dxᵃ/dλ)(dxᵇ/dλ) = 0
```

where Γ are the Christoffel symbols (encoding spacetime curvature) and λ is the affine parameter along the path. This is a system of 8 coupled ODEs, solved with adaptive Dormand-Prince integration.

## Credits

Core GR engine by [cavemanloverboy](https://github.com/cavemanloverboy/grr). WASM bridge, interactive demo, and validation suite by [fwaz](https://github.com/fwazb).
