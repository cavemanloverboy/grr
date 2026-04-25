pub fn harmonic_oscillator(_t: f64, &[x, v]: &[f64; 2]) -> [f64; 2] {
    [v, -x]
}

pub fn kepler_2d(_t: f64, &[x, y, vx, vy]: &[f64; 4]) -> [f64; 4] {
    let r2 = x * x + y * y;
    let r3 = r2 * r2.sqrt(); // r³
    [vx, vy, -x / r3, -y / r3]
}

pub fn kepler_energy([x, y, vx, vy]: [f64; 4]) -> f64 {
    let r = (x * x + y * y).sqrt();
    0.5 * (vx * vx + vy * vy) - 1.0 / r
}

pub fn kepler_angular_momentum([x, y, vx, vy]: [f64; 4]) -> f64 {
    x * vy - y * vx
}

fn approx_eq<const N: usize>(a: [f64; N], b: [f64; N], tol: f64) -> bool {
    let mut diff = [0.0; N];
    for i in 0..N {
        diff[i] = (b[i] - a[i]).abs();
    }

    diff.iter().all(|&d| d < tol)
}

mod rk4 {
    use std::f64::consts::PI;

    use grr_integrator::rk4::rk4_integrator;

    use super::*;

    #[test]
    fn test_rk4_harmonic_oscillator() {
        // Starts at rest at x=1
        let t = 0.0;
        let state = [1.0, 0.0];
        let t_end = 100.0 * (2.0 * PI);
        let dt = 0.01;

        // Integrate via rk4
        let integrand = harmonic_oscillator;
        let end_state = rk4_integrator(integrand, state, t, t_end, dt);

        // Assert this is near the original state
        let near_orig_state = approx_eq(state, end_state, 1e-6);
        assert!(near_orig_state, "{state:?}!={end_state:?}");
    }

    #[test]
    fn test_rk4_kepler_circular_orbit() {
        // Starts at x=1, going in +y direction
        let t = 0.0;
        let state = [1.0, 0.0, 0.0, 1.0];
        let t_end = 100.0 * (2.0 * PI);
        let dt = 0.01;

        // Integrate via rk4
        let integrand = kepler_2d;
        let end_state = rk4_integrator(integrand, state, t, t_end, dt);

        // Assert this is near the original state
        let near_orig_state = approx_eq(state, end_state, 1e-6);
        assert!(near_orig_state, "{state:?}!={end_state:?}");

        let e0 = kepler_energy(state);
        let e1 = kepler_energy(end_state);
        let l0 = kepler_angular_momentum(state);
        let l1 = kepler_angular_momentum(end_state);
        let [x, y, _, _] = end_state;
        let r = (x * x + y * y).sqrt();

        // Tolerances chosen ~2 orders looser than rtol to account for
        // accumulation over ~10^4-10^5 steps.
        assert!(
            (e1 - e0).abs() < 1e-8,
            "energy not conserved: {e0} -> {e1}, drift = {:e}",
            e1 - e0
        );
        assert!(
            (l1 - l0).abs() < 1e-8,
            "L not conserved: {l0} -> {l1}, drift = {:e}",
            l1 - l0
        );
        assert!((r - 1.0).abs() < 1e-7, "radius drifted: {r}");
    }
}

mod dp54 {
    use std::f64::consts::PI;

    use grr_integrator::dp54::{Dp54Controller, dp54_integrator};

    use super::*;

    #[test]
    fn test_dp54_harmonic_oscillator() {
        // Starts at rest at x=1
        let t = 0.0;
        let state = [1.0, 0.0];
        let t_end = 100.0 * (2.0 * PI);
        let dt = 0.01;

        // Integrate via dp54
        let integrand = harmonic_oscillator;
        let ctrl = Dp54Controller {
            atol: 1e-10,
            rtol: 1e-10,
            safety: 0.5,
            ..Default::default()
        };
        let end_state = dp54_integrator(integrand, state, t, t_end, dt, &ctrl);

        // Assert this is near the original state
        let near_orig_state = approx_eq(state, end_state, 1e-6);
        assert!(near_orig_state, "{state:?}!={end_state:?}");
    }

    #[test]
    fn test_dp54_kepler_circular_orbit() {
        // Starts at x=1, going in +y direction
        let t = 0.0;
        let state = [1.0, 0.0, 0.0, 1.0];
        let t_end = 100.0 * (2.0 * PI);
        let dt = 0.01;

        // Integrate via dp54
        let integrand = kepler_2d;
        let ctrl = Dp54Controller {
            atol: 1e-10,
            rtol: 1e-10,
            safety: 0.5,
            ..Default::default()
        };
        let end_state = dp54_integrator(integrand, state, t, t_end, dt, &ctrl);

        // Assert this is near the original state
        let near_orig_state = approx_eq(state, end_state, 1e-6);
        assert!(near_orig_state, "{state:?}!={end_state:?}");

        let e0 = kepler_energy(state);
        let e1 = kepler_energy(end_state);
        let l0 = kepler_angular_momentum(state);
        let l1 = kepler_angular_momentum(end_state);
        let [x, y, _, _] = end_state;
        let r = (x * x + y * y).sqrt();

        // Tolerances chosen ~2 orders looser than rtol to account for
        // accumulation over ~10^4-10^5 steps.
        assert!(
            (e1 - e0).abs() < 1e-8,
            "energy not conserved: {e0} -> {e1}, drift = {:e}",
            e1 - e0
        );
        assert!(
            (l1 - l0).abs() < 1e-8,
            "L not conserved: {l0} -> {l1}, drift = {:e}",
            l1 - l0
        );
        assert!((r - 1.0).abs() < 1e-7, "radius drifted: {r}");
    }
}
