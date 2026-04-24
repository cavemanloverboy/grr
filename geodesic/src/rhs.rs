use grr_core::math::{christoffel::Christoffels, field::MetricField};

use crate::state::State;

impl State {
    #[inline(always)]
    pub fn geodesic_rhs<F: MetricField>(&self, field: &F) -> State {
        let x = self.position();
        let k = self.momentum();
        let accel = field.christoffels_at(x).geodesic_accel(k);
        State::new(*k, accel)
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use grr_core::math::{
        metric::Metric,
        spacetimes::{kerr::KerrBoydLindquist, minkowski::Minkowski, schwarzchild::Schwarzschild},
    };

    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn rhs_trivial_in_flat_space_free_motion() {
        // In Minkowski-spherical at (r=10, θ=π/2, φ=0), a particle at rest:
        // k = (1, 0, 0, 0) is timelike (g_tt · 1² = -1).
        // Every nonzero Minkowski-spherical Christoffel has a spatial lower index,
        // so Γ^μ_αβ k^α k^β = 0. RHS should be (k, 0).
        let field = Minkowski;
        let x = [0.0, 10.0, PI / 2.0, 0.0];
        let k = [1.0, 0.0, 0.0, 0.0];
        let state = State::new(x, k);
        let rhs = state.geodesic_rhs(&field);
        assert_eq!(rhs.position(), state.momentum());
        assert_eq!(rhs.momentum(), &[0.0; 4]);
    }

    #[test]
    fn photon_sphere_equatorial_orbit_accel_vanishes() {
        // Schwarzschild photon sphere at r=3. consider equatorial orbit
        let field = Schwarzschild;
        let r = 3.0;
        let k_phi = 1.0_f64;
        let k_t = 27.0_f64.sqrt();
        let x = [0.0, r, FRAC_PI_2, 0.0];
        let k = [k_t, 0.0, 0.0, k_phi];
        let state = State::new(x, k);

        // Make sure this is null geodesic
        let k_sq = field.metric_at(&x).dot(&k, &k);
        assert!(k_sq.abs() < 1e-12, "initial k not null: k·k = {}", k_sq);

        // calculate k, accel
        let rhs = state.geodesic_rhs(&field);

        // no acceleration. linear t(tau), phi(tau)
        let accel = rhs.momentum();
        let zero_accel = accel.into_iter().all(|&a| approx_eq(a, 0.0, 1e-12));
        assert!(zero_accel, "nonzero accel for equatorial orbit: {accel:?}")
    }

    #[test]
    fn kerr_retrograde_photon_sphere_equatorial_orbit_accel_vanishes() {
        // extremal Kerr (a=1): retrograde equatorial photon sphere at r=4.
        //
        // a tangential null geodesic in the equatorial plane is a circular
        // (unstable) orbit. the nontrivial check is accel^r, which requires
        //   Γ^r_tt (k^t)² + 2 Γ^r_tφ k^t k^φ + Γ^r_φφ (k^φ)² = 0
        // to cancel. the t-φ cross term is the frame-dragging contribution —
        // absent in Schwarzschild (this is what makes it further than r=3).

        let a = 1.0;
        let r = 4.0; // retrograde photon sphere at a=1
        let field = KerrBoydLindquist::new(a);
        let x = [0.0, r, FRAC_PI_2, 0.0];

        // build null tangential k: k = (k^t, 0, 0, k^φ) with k·k = 0.
        // for retrograde (k^φ < 0 for prograde observer; signs depend on convention).
        // solve the quadratic in (k^t / k^φ) from the null condition:
        //   g_tt (k^t)² + 2 g_tφ k^t k^φ + g_φφ (k^φ)² = 0
        //
        // at r=4, θ=π/2, a=1
        //   Σ = r² + a²cos²θ = 16
        //   Δ = r² - 2r + a² = 16 - 8 + 1 = 9
        //   g_tt = -(1 - 2r/Σ) = -(1 - 8/16) = -0.5
        //   g_tφ = -2ar sin²θ / Σ = -8/16 = -0.5
        //   g_φφ = (r² + a² + 2a²r sin²θ/Σ) sin²θ = (16 + 1 + 8/16) = 17.5
        //   null: -0.5 (k^t)² - k^t k^φ + 17.5 (k^φ)² = 0
        //   let ρ = k^t / k^φ:  -0.5 ρ² - ρ + 17.5 = 0  =>  ρ² + 2ρ - 35 = 0
        //   ρ = (-2 ± √(4+140))/2 = (-2 ± 12)/2 = 5 or -7
        //
        //   prograde (Ω > 0 w.r.t. static observer): ρ = 5  (k^t = 5, k^φ = 1)
        //   retrograde: ρ = -7  (k^t = 7, k^φ = -1), equivalently (k^t = -7, k^φ = 1)
        //
        // we want a future-directed photon (k^t > 0) and retrograde motion,
        // so k^t = 7, k^φ = -1.
        let k = [7.0, 0.0, 0.0, -1.0];

        // make sure this is null geodesic
        let k_sq = field.metric_at(&x).dot(&k, &k);
        assert!(k_sq.abs() < 1e-12, "initial k not null: k·k = {k_sq}");

        // calculate k, accel
        let state = State::new(x, k);
        let rhs = state.geodesic_rhs(&field);
        let accel = rhs.momentum();

        // no acceleration. linear t(tau), phi(tau)
        let zero_accel = accel.iter().all(|&a| approx_eq(a, 0.0, 1e-12));
        assert!(zero_accel, "nonzero accel at retro eq orbit: {accel:?}");
    }
}
