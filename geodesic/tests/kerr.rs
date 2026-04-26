use std::f64::consts::FRAC_PI_2;

use grr_core::math::spacetimes::kerr::KerrBoydLindquist;
use grr_geodesic::{
    integrator::{
        GeodesicConfig, TERMINATION_RADIUS_FACTOR, TerminationReason, integrate_geodesic_dp54,
    },
    state::State,
};
use grr_integrator::dp54::Dp54Controller;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

#[test]
#[rustfmt::skip]
fn test_kerr_intersect_horizon() {
    // start at (r=10, θ=π/2, φ=0) going radially inward
    let a = 0.98; /* nearly maximally spinning kerr */
    let r = 10.0;
    let theta: f64 = FRAC_PI_2;
    let sigma = r*r + a*a * theta.cos().powi(2);
    let delta = r*r - 2.0*r + a*a;
    let kt_sq = sigma * sigma / (delta * (sigma - 2.0 * r));
    let kt = kt_sq.sqrt();
    let k = [kt, -1.0, 0.0, 0.0];
    let x = [0.0, 10.0, FRAC_PI_2, 0.0];
    let state = State::new(x, k);

    let field = KerrBoydLindquist::new(a);
    let ctrl = Dp54Controller {
        atol: 1e-10,
        rtol: 1e-10,
        ..Default::default()
    };
    let r_plus = 1.0 + (1.0 - a * a).sqrt();
    let cfg = GeodesicConfig {
        r_plus,
        r_cam: 20.0,
        r_in: 0.0,
        r_out: 0.0,
        max_steps: 1_000_000,
    };

    // initially null
    let norm_sq = state.norm_squared(&field);
    assert!(norm_sq < 1e-6, "not initially null");

    let dl = 0.01;
    let result = integrate_geodesic_dp54(&field, state, dl, &ctrl, &cfg);

    let final_r = result.final_state.0[1];

    // reason == horizon event
    assert_eq!(result.termination, TerminationReason::HorizonEvent, "did not intersect horizon: {result:?}");

    // r is between horizon and our "≈horizon"
    assert!(final_r >= cfg.r_plus && final_r <= cfg.r_plus * TERMINATION_RADIUS_FACTOR);

    // geodesic still null
    let norm_sq = result.final_state.norm_squared(&field);
    assert!(norm_sq < 1e-6, "geodesic not null {norm_sq}");

    // energy conserved
    let orig_energy = state.energy(&field);
    let final_energy = result.final_state.energy(&field);
    assert!(approx_eq(orig_energy, final_energy, 1e-8), "energy not conserved");
}
