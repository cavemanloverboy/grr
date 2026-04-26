//! TODO (feat): need to add conserved quantity monitoring

use std::f64::consts::{FRAC_PI_2, PI};

use grr_core::math::field::MetricField;
use grr_integrator::dp54::{Dp54Controller, dp54_step};

use crate::state::State;

/// BoydLindquist goes crazy ar r=r_plus so we gotta stop before then
pub const TERMINATION_RADIUS_FACTOR: f64 = 1.01;

pub struct GeodesicConfig {
    /// outer horizon radius (depends on metric). p
    /// photon terminates when r < TERMINATION_RADIUS_FACTOR * r_plus.
    pub r_plus: f64,
    /// Camera radius. Photon escapes when r > 2 * r_cam AND k^r > 0.
    pub r_cam: f64,
    /// Disk inner edge (typically r_ISCO).
    pub r_in: f64,
    /// Disk outer edge.
    pub r_out: f64,
    /// Step count safety cap.
    pub max_steps: usize,
}

#[derive(Debug)]
pub struct GeodesicResult {
    pub final_state: State,
    pub final_lambda: f64,
    pub termination: TerminationReason,
    pub n_accepted: usize,
    pub n_rejected: usize,
}

#[derive(Debug, PartialEq, Eq)]
pub enum TerminationReason {
    /// Triggers:
    ///  1. r drops below TERMINATION_RADIUS_FACTOR * r_plus (outer horizon).
    ///
    /// Note: 1% buffer keeps us in regular coordinates — at r = r_plus
    /// exactly, the BL metric is singular and the integrator stalls or NaNs.
    ///
    /// Action: pixel becomes black.
    HorizonEvent,

    /// Triggers:
    ///  1. r > 2 * r_cam AND
    ///  2. k^r > 0 (radially outgoing).
    ///
    /// Note: the k^r > 0 check matters — a photon at large r moving inward
    /// can still get captured. Once past the camera and outbound, asymptotic
    /// flatness guarantees no return.
    ///
    /// Action: pixel gets background sky.
    EscapeEvent,

    /// Triggers:
    ///  1. θ crosses π/2 (signed change between steps) AND
    ///  2. r ∈ [r_in, r_out] at the crossing
    ///
    /// Action: bisect to find λ where θ = π/2 to |Δθ| < 1e-8 — localizes
    /// (r_hit, φ_hit) precisely, which the final image depends on. Pixel
    /// gets disk emission, redshift-corrected.
    EquatorialCrossingEvent,

    // TODO (remove). adding this in case we need it for debugging
    MaxStepsEvent,
}

impl TerminationReason {
    /// returns (reason, lambda_offset_from_step_start, exact_state).
    /// for horizon/escape, offset = dl (end of step) and state = new step.
    /// for disk crossing, offset and state come from sub-step bisection.
    #[inline(always)]
    fn check<F: Fn(f64, &[f64; 8]) -> [f64; 8]>(
        cfg: &GeodesicConfig,
        rhs: &F,
        state: &[f64; 8],
        old_state: &[f64; 8],
        k1_old: &[f64; 8],
        dl: f64,
    ) -> Option<(TerminationReason, f64, [f64; 8])> {
        let [_, r, th, _, _, kr, _, _] = *state;
        let old_th = old_state[2];

        // check HorizonEvent
        // 1. r drops below TERMINATION_RADIUS_FACTOR * r_plus (outer horizon).
        let horizon_event = r < TERMINATION_RADIUS_FACTOR * cfg.r_plus;
        if horizon_event {
            return Some((TerminationReason::HorizonEvent, dl, *state));
        }

        // check EscapeEvent
        // 1. r > 2 * r_cam AND
        // 2. k^r > 0 (radially outgoing).
        let escape_event = (r > 2.0 * cfg.r_cam) && (kr > 0.0);
        if escape_event {
            return Some((TerminationReason::EscapeEvent, dl, *state));
        }

        // check EquatorialCrossingEvent
        // 1. θ crosses π/2 (signed change between steps) AND
        // 2. r ∈ [r_in, r_out] at the crossing
        let target = FRAC_PI_2;
        let th_crossed = (old_th - target) * (th - target) < 0.0;
        let within_ring = (r >= cfg.r_in) && r <= cfg.r_out;
        let equatorial_crossing_event = th_crossed && within_ring;
        if equatorial_crossing_event {
            // bisect on λ in [0, dl] (relative to old_state) to find exact crossing.
            let (lambda_offset, exact_state) =
                bisect_disk_crossing(rhs, old_state, k1_old, state, dl);
            return Some((
                TerminationReason::EquatorialCrossingEvent,
                lambda_offset,
                exact_state,
            ));
        }

        None
    }
}

pub fn integrate_geodesic_dp54<F: MetricField>(
    field: &F,
    mut state: State,
    mut dl: f64,
    ctrl: &Dp54Controller,
    cfg: &GeodesicConfig,
) -> GeodesicResult {
    let integrand = geodesic_rhs_closure(field);
    let mut n_accepted = 0;
    let mut n_rejected = 0;

    // affine parameter starts at 0; only differences matter for geodesics.
    let mut lambda = 0.0;

    let mut k1 = integrand(lambda, &state.0);
    let mut err_prev: f64 = 1.0;

    // never step more than 5% of cam distance
    let max_dl = 0.05 * cfg.r_cam;
    loop {
        dl = dl.min(max_dl);

        // TODO (remove): debug
        if n_accepted + n_rejected >= cfg.max_steps {
            return GeodesicResult {
                final_state: state,
                final_lambda: lambda,
                termination: TerminationReason::MaxStepsEvent,
                n_accepted,
                n_rejected,
            };
        }

        let (new_state, k7, err) = dp54_step(&integrand, lambda, &state.0, dl, k1);
        let err_norm = ctrl.err_norm(&state.0, &new_state, &err);

        if err_norm <= 1.0 {
            // accept
            n_accepted += 1;

            if let Some((reason, lambda_offset, exact_state)) =
                TerminationReason::check(cfg, &integrand, &new_state, &state.0, &k1, dl)
            {
                return GeodesicResult {
                    final_state: State(exact_state),
                    final_lambda: lambda + lambda_offset,
                    termination: reason,
                    n_accepted,
                    n_rejected,
                };
            }

            // commit and continue
            lambda += dl;
            state = State(new_state);
            k1 = k7;
            dl *= ctrl.factor(err_norm, err_prev);
            err_prev = err_norm.max(1e-4);
        } else {
            // reject: don't advance, don't consume k7, don't update err_prev.
            // use I-style shrink (set beta=0 effectively by passing err_prev=1).
            n_rejected += 1;
            dl *= ctrl.factor(err_norm, 1.0);
        }
    }
}

#[inline(always)]
fn geodesic_rhs_closure<'a, F: MetricField>(
    field: &'a F,
) -> impl Fn(f64, &[f64; 8]) -> [f64; 8] + 'a {
    move |_t, state| State(*state).geodesic_rhs(field).0
}

/// bisect on λ in [0, dl_step] starting from state_old to find where
/// θ = π/2 exactly. returns (λ_offset, state_at_crossing).
///
/// each iteration takes a fresh dp54 step from state_old by dl_mid.
/// recomputing from state_old (rather than chaining) keeps integration
/// error from accumulating across the bisection.
fn bisect_disk_crossing<F: Fn(f64, &[f64; 8]) -> [f64; 8]>(
    rhs: &F,
    state_old: &[f64; 8],
    k1_old: &[f64; 8],
    state_new: &[f64; 8],
    dl_step: f64,
) -> (f64, [f64; 8]) {
    let target = PI / 2.0;
    let theta_old_offset = state_old[2] - target;

    let mut dl_lo = 0.0;
    let mut dl_hi = dl_step;
    let mut state_at_hi = *state_new;

    for _ in 0..40 {
        let dl_mid = 0.5 * (dl_lo + dl_hi);
        let (state_mid, _, _) = dp54_step(rhs, 0.0, state_old, dl_mid, *k1_old);
        let theta_mid_offset = state_mid[2] - target;

        if theta_old_offset * theta_mid_offset < 0.0 {
            // crossing is in [dl_lo, dl_mid]
            dl_hi = dl_mid;
            state_at_hi = state_mid;
        } else {
            // crossing is in [dl_mid, dl_hi]
            dl_lo = dl_mid;
        }

        if (dl_hi - dl_lo) < 1e-12 {
            return (dl_hi, state_at_hi);
        }
    }

    (dl_hi, state_at_hi)
}
