//! TODO (feat): need to add conserved quantity monitoring

use std::f64::consts::PI;

use grr_core::math::field::MetricField;
use grr_integrator::dp54::{Dp54Controller, dp54_step};

use crate::state::State;

/// BoydLindquist goes crazy ar r=r_plus so we gotta stop before then
const TERMINATION_RADIUS: f64 = 1.01;

pub struct GeodesicConfig {
    /// outer horizon radius (depends on metric). p
    /// photon terminates when r < TERMINATION_RADIUS * r_plus.
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

pub struct GeodesicResult {
    pub final_state: State,
    pub final_lambda: f64,
    pub termination: TerminationReason,
    pub n_accepted: usize,
    pub n_rejected: usize,
}

pub enum TerminationReason {
    /// Triggers:
    ///  1. r drops below TERMINATION_RADIUS * r_plus (outer horizon).
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
    #[inline(always)]
    fn check(
        cfg: &GeodesicConfig,
        state: &[f64; 8],
        old_state: &[f64; 8],
    ) -> Option<TerminationReason> {
        let [_, r, th, _, _, kr, _, _] = state;
        let old_th = old_state[2];

        // check HorizonEvent
        // 1. r drops below TERMINATION_RADIUS * r_plus (outer horizon).
        let horizon_event = *r < TERMINATION_RADIUS * cfg.r_plus;
        if horizon_event {
            return Some(TerminationReason::HorizonEvent);
        }

        // check EscapeEvent
        // 1. r > 2 * r_cam AND
        // 2. k^r > 0 (radially outgoing).
        let escape_event = (*r > 2.0 * cfg.r_cam) && (*kr > 0.0);
        if escape_event {
            return Some(TerminationReason::EscapeEvent);
        }

        // check EquatorialCrossingEvent
        // 1. θ crosses π/2 (signed change between steps) AND
        // 2. r ∈ [r_in, r_out] at the crossing
        let th_crossed = (old_th - PI / 2.0) * (th - PI / 2.0) < 0.0;
        let within_ring = (*r >= cfg.r_in) && *r <= cfg.r_out;
        let equatorial_crossing_event = th_crossed && within_ring;
        if equatorial_crossing_event {
            // TODO: bisect to find exact λ_crossing
            return Some(TerminationReason::EquatorialCrossingEvent);
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
            lambda += dl;
            state = State(new_state);

            if let Some(reason) = TerminationReason::check(cfg, &new_state, &state.0) {
                return GeodesicResult {
                    final_state: state,
                    final_lambda: lambda,
                    termination: reason,
                    n_accepted,
                    n_rejected,
                };
            }
            // commit and continue
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
