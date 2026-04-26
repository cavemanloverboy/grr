use wasm_bindgen::prelude::*;
use std::f64::consts::FRAC_PI_2;

use grr_core::math::Vector;
use grr_core::math::field::MetricField;
use grr_core::math::metric::Metric;
use grr_core::math::spacetimes::schwarzchild::Schwarzschild;
use grr_core::math::spacetimes::kerr::KerrBoydLindquist;
use grr_geodesic::state::State;
use grr_integrator::dp54::{Dp54Controller, dp54_step};

// ── Coordinate conversion ──────────────────────────────────────────

fn bl_to_cartesian(r: f64, theta: f64, phi: f64) -> [f64; 3] {
    let st = theta.sin();
    [r * st * phi.cos(), r * st * phi.sin(), r * theta.cos()]
}

// ── k^t solver ─────────────────────────────────────────────────────
//
// Given spatial momentum (k^r, k^θ, k^φ) and position, solve for k^t
// such that g_μν k^μ k^ν = 0 (null) or -1 (timelike).
//
// The metric dot product expands to a quadratic in k^t:
//   g_tt (k^t)² + 2 g_tφ k^t k^φ + [spatial terms] = μ²
//
// We extract metric components via dot products of basis vectors.

fn solve_kt<F: MetricField>(field: &F, x: &Vector, kr: f64, kth: f64, kph: f64, is_null: bool) -> f64 {
    let m = field.metric_at(x);
    let gtt = m.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
    let gtp = m.dot(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 1.0]);
    let grr = m.dot(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]);
    let gthth = m.dot(&[0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 1.0, 0.0]);
    let gpp = m.dot(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]);

    let norm_target = if is_null { 0.0 } else { -1.0 };

    // A kt² + B kt + C = 0
    let a = gtt;
    let b = 2.0 * gtp * kph;
    let c = grr * kr * kr + gthth * kth * kth + gpp * kph * kph - norm_target;

    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return f64::NAN;
    }
    let sd = disc.sqrt();
    let kt1 = (-b + sd) / (2.0 * a);
    let kt2 = (-b - sd) / (2.0 * a);

    // Both should be real. Pick the future-directed (positive) root.
    // If both positive, pick the larger (more energy).
    if kt1 > 0.0 && kt2 > 0.0 {
        kt1.max(kt2)
    } else if kt1 > 0.0 {
        kt1
    } else {
        kt2
    }
}

// ── Generic tracer ─────────────────────────────────────────────────

fn trace_impl<F: MetricField>(
    field: &F,
    r_horizon: f64,
    mut state: State,
    lambda_end: f64,
    ctrl: &Dp54Controller,
    max_steps: usize,
    sample_every: usize,
) -> TraceResult {
    let rhs = |_l: f64, s: &[f64; 8]| State(*s).geodesic_rhs(field).0;

    let mut points = Vec::new();
    let mut lambda = 0.0_f64;
    let mut dl = 0.01_f64;
    let mut k1 = rhs(0.0, &state.0);
    let mut err_prev = 1.0_f64;
    let mut accepted = 0_usize;
    let mut total = 0_usize;
    let mut reason = "complete";

    // Record initial point
    let [_, r, th, ph] = *state.position();
    let [x, y, z] = bl_to_cartesian(r, th, ph);
    points.extend_from_slice(&[x, y, z]);

    while lambda < lambda_end {
        if total >= max_steps {
            reason = "max_steps";
            break;
        }

        let h = dl.min(lambda_end - lambda);
        let (new_state, k7, err) = dp54_step(&rhs, lambda, &state.0, h, k1);
        let en = ctrl.err_norm(&state.0, &new_state, &err);
        total += 1;

        if en <= 1.0 {
            lambda += h;
            state = State(new_state);
            k1 = k7;
            accepted += 1;

            let r = state.position()[1];
            if r < 1.01 * r_horizon {
                reason = "horizon";
                let [_, r, th, ph] = *state.position();
                let [x, y, z] = bl_to_cartesian(r, th, ph);
                points.extend_from_slice(&[x, y, z]);
                break;
            }
            if r > 1e4 {
                reason = "escape";
                break;
            }
            if r.is_nan() {
                reason = "nan";
                break;
            }

            if accepted % sample_every == 0 {
                let [_, r, th, ph] = *state.position();
                let [x, y, z] = bl_to_cartesian(r, th, ph);
                points.extend_from_slice(&[x, y, z]);
            }

            dl = h * ctrl.factor(en, err_prev);
            err_prev = en.max(1e-4);
        } else {
            dl = h * ctrl.factor(en, 1.0);
        }
    }

    // Record final point (unless we already did on horizon hit)
    if reason != "horizon" {
        let [_, r, th, ph] = *state.position();
        let [x, y, z] = bl_to_cartesian(r, th, ph);
        points.extend_from_slice(&[x, y, z]);
    }

    TraceResult {
        points,
        lambda_final: lambda,
        termination: reason.to_string(),
    }
}

// ── Result type ────────────────────────────────────────────────────

#[wasm_bindgen]
pub struct TraceResult {
    points: Vec<f64>,
    lambda_final: f64,
    termination: String,
}

#[wasm_bindgen]
impl TraceResult {
    /// Flat Cartesian coordinates: [x,y,z, x,y,z, ...]. Stride = 3.
    #[wasm_bindgen(getter)]
    pub fn points(&self) -> Vec<f64> {
        self.points.clone()
    }

    /// Number of trajectory sample points.
    #[wasm_bindgen(getter)]
    pub fn num_points(&self) -> u32 {
        (self.points.len() / 3) as u32
    }

    /// Why integration stopped: "complete", "horizon", "escape", "max_steps", "nan".
    #[wasm_bindgen(getter)]
    pub fn termination(&self) -> String {
        self.termination.clone()
    }

    /// Final affine parameter value.
    #[wasm_bindgen(getter)]
    pub fn lambda_final(&self) -> f64 {
        self.lambda_final
    }
}

// ── Schwarzschild geodesics ────────────────────────────────────────

/// Trace a geodesic in Schwarzschild spacetime (M=1).
///
/// `state`: 8-element array [t, r, θ, φ, k^t, k^r, k^θ, k^φ].
/// Returns a TraceResult with Cartesian trajectory points.
#[wasm_bindgen]
pub fn trace_schwarzschild(
    state: &[f64],
    lambda_end: f64,
    tolerance: f64,
    max_steps: u32,
    sample_every: u32,
) -> TraceResult {
    assert!(state.len() >= 8, "state must have 8 elements");
    assert!(tolerance > 0.0, "tolerance must be positive");
    assert!(sample_every > 0, "sample_every must be > 0");
    let field = Schwarzschild;
    let x: Vector = [state[0], state[1], state[2], state[3]];
    let k: Vector = [state[4], state[5], state[6], state[7]];
    let s = State::new(x, k);
    let ctrl = Dp54Controller {
        atol: tolerance,
        rtol: tolerance,
        ..Dp54Controller::default()
    };
    trace_impl(&field, 2.0, s, lambda_end, &ctrl, max_steps as usize, sample_every as usize)
}

/// Build a full 8-component state for Schwarzschild from spatial momentum.
/// Solves for k^t from the null/timelike constraint.
#[wasm_bindgen]
pub fn solve_schwarzschild_state(
    r: f64, theta: f64, phi: f64,
    kr: f64, ktheta: f64, kphi: f64,
    is_null: bool,
) -> Vec<f64> {
    let field = Schwarzschild;
    let x: Vector = [0.0, r, theta, phi];
    let kt = solve_kt(&field, &x, kr, ktheta, kphi, is_null);
    vec![0.0, r, theta, phi, kt, kr, ktheta, kphi]
}

// ── Kerr geodesics ─────────────────────────────────────────────────

/// Trace a geodesic in Kerr spacetime (M=1, spin parameter `a`).
///
/// `state`: 8-element array [t, r, θ, φ, k^t, k^r, k^θ, k^φ].
#[wasm_bindgen]
pub fn trace_kerr(
    spin: f64,
    state: &[f64],
    lambda_end: f64,
    tolerance: f64,
    max_steps: u32,
    sample_every: u32,
) -> TraceResult {
    assert!(state.len() >= 8, "state must have 8 elements");
    assert!(tolerance > 0.0, "tolerance must be positive");
    assert!(sample_every > 0, "sample_every must be > 0");
    assert!(spin.abs() < 1.0, "spin must satisfy |a| < 1");
    let field = KerrBoydLindquist::new(spin);
    let r_plus = 1.0 + (1.0 - spin * spin).sqrt();
    let x: Vector = [state[0], state[1], state[2], state[3]];
    let k: Vector = [state[4], state[5], state[6], state[7]];
    let s = State::new(x, k);
    let ctrl = Dp54Controller {
        atol: tolerance,
        rtol: tolerance,
        ..Dp54Controller::default()
    };
    trace_impl(&field, r_plus, s, lambda_end, &ctrl, max_steps as usize, sample_every as usize)
}

/// Build a full 8-component state for Kerr from spatial momentum.
/// Solves for k^t from the null/timelike constraint.
#[wasm_bindgen]
pub fn solve_kerr_state(
    spin: f64,
    r: f64, theta: f64, phi: f64,
    kr: f64, ktheta: f64, kphi: f64,
    is_null: bool,
) -> Vec<f64> {
    let field = KerrBoydLindquist::new(spin);
    let x: Vector = [0.0, r, theta, phi];
    let kt = solve_kt(&field, &x, kr, ktheta, kphi, is_null);
    vec![0.0, r, theta, phi, kt, kr, ktheta, kphi]
}

// ── Initial condition helpers ──────────────────────────────────────

/// Initial conditions for a timelike circular equatorial orbit in Kerr.
/// Works for Schwarzschild too (pass spin=0).
/// Returns 8-element state: [t, r, θ, φ, u^t, u^r, u^θ, u^φ].
#[wasm_bindgen]
pub fn circular_orbit_ic(spin: f64, r: f64, prograde: bool) -> Vec<f64> {
    let sign: f64 = if prograde { 1.0 } else { -1.0 };
    let omega = sign / (r.powf(1.5) + spin * sign);

    let field = KerrBoydLindquist::new(spin);
    let x: Vector = [0.0, r, FRAC_PI_2, 0.0];
    let m = field.metric_at(&x);

    let gtt = m.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]);
    let gtp = m.dot(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 1.0]);
    let gpp = m.dot(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]);

    // g_μν u^μ u^ν = -1 with u^φ = Ω u^t
    // (g_tt + 2 g_tφ Ω + g_φφ Ω²)(u^t)² = -1
    let ut = (1.0 / -(gtt + 2.0 * gtp * omega + gpp * omega * omega)).sqrt();
    let uphi = omega * ut;

    vec![0.0, r, FRAC_PI_2, 0.0, ut, 0.0, 0.0, uphi]
}

/// Initial conditions for a null circular orbit at the photon sphere.
/// Returns 8-element state: [t, r, θ, φ, k^t, k^r, k^θ, k^φ].
#[wasm_bindgen]
pub fn photon_sphere_ic(spin: f64, prograde: bool) -> Vec<f64> {
    // Photon sphere radius in Kerr (M=1):
    //   prograde:   r = 2(1 + cos[2/3 · arccos(-a)])
    //   retrograde: r = 2(1 + cos[2/3 · arccos(a)])
    let arg = if prograde { -spin } else { spin };
    let r = 2.0 * (1.0 + ((2.0 / 3.0) * arg.acos()).cos());

    let kphi = if prograde { 1.0 } else { -1.0 };

    let field = KerrBoydLindquist::new(spin);
    let x: Vector = [0.0, r, FRAC_PI_2, 0.0];
    let kt = solve_kt(&field, &x, 0.0, 0.0, kphi, true);

    vec![0.0, r, FRAC_PI_2, 0.0, kt, 0.0, 0.0, kphi]
}

/// ISCO radius for Kerr spacetime (M=1).
#[wasm_bindgen]
pub fn isco_radius(spin: f64, prograde: bool) -> f64 {
    let a = spin;
    let z1 = 1.0 + (1.0 - a * a).cbrt() * ((1.0 + a).cbrt() + (1.0 - a).cbrt());
    let z2 = (3.0 * a * a + z1 * z1).sqrt();
    if prograde {
        3.0 + z2 - ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
    } else {
        3.0 + z2 + ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
    }
}

/// Kerr outer horizon radius: r+ = 1 + sqrt(1 - a²).
#[wasm_bindgen]
pub fn horizon_radius(spin: f64) -> f64 {
    1.0 + (1.0 - spin * spin).sqrt()
}

// ── Smoke test ─────────────────────────────────────────────────────

#[wasm_bindgen]
pub fn hello_grr() -> String {
    "grr wasm online".to_string()
}
