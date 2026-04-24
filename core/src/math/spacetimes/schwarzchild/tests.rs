use super::*;
use crate::math::metric::Metric;
use std::{
    array,
    f64::consts::{FRAC_PI_2, FRAC_PI_4},
};

const TOL: f64 = 1e-12;
const FD_TOL: f64 = 1e-6;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

/// Extract the metric as a dense 4x4 matrix by probing with basis vectors.
fn dense_metric(x: &Vector) -> [[f64; 4]; 4] {
    let schwarzs_metric = Schwarzschild;
    let schwarzs_g = schwarzs_metric.metric_at(x);
    array::from_fn(|mu| {
        array::from_fn(|nu| {
            // 1.0 if same idx, 0.0 otherise
            let e_mu = array::from_fn(|mu_| (mu == mu_) as u64 as f64);
            let e_nu = array::from_fn(|nu_| (nu == nu_) as u64 as f64);
            schwarzs_g.dot(&e_mu, &e_nu)
        })
    })
}

/// Extract the inverse metric as dense 4x4 via raise on basis vectors.
fn dense_metric_inverse(x: &Vector) -> [[f64; 4]; 4] {
    let schwarzs_g = Schwarzschild.metric_at(x);
    array::from_fn(|mu| {
        // e_mu = delta_mu_mu_
        let e_mu = array::from_fn(|mu_| (mu == mu_) as u64 as f64);
        let raised = schwarzs_g.raise(&e_mu);
        array::from_fn(|nu| raised[nu])
    })
}

/// Numerical ∂_alpha g_{mu nu} via central differences.
fn dense_metric_finite_difference(x: &Vector, alpha: usize) -> [[f64; 4]; 4] {
    // step size
    let h = 1e-5;

    // Calculate x_plus and x_minus
    let mut x_plus = x.clone();
    x_plus[alpha] += h;
    let mut x_minus = x.clone();
    x_minus[alpha] -= h;

    // Now calculate g_plus and g_minus
    let g_plus: [[f64; 4]; 4] = dense_metric(&x_plus);
    let g_minus: [[f64; 4]; 4] = dense_metric(&x_minus);

    let symmetric_diff = |mu: usize, nu: usize| (g_plus[mu][nu] - g_minus[mu][nu]) / (2.0 * h);
    array::from_fn(|mu| array::from_fn(|nu| symmetric_diff(mu, nu)))
}

/// FD Christoffel: Γ^μ_{αβ} = ½ g^{μσ} (∂_α g_{σβ} + ∂_β g_{σα} - ∂_σ g_{αβ}).
fn christoffel_finite_difference(x: &Vector) -> [[[f64; 4]; 4]; 4] {
    let g_inv = dense_metric_inverse(x);
    let dg: [[[f64; 4]; 4]; 4] = std::array::from_fn(|a| dense_metric_finite_difference(x, a));
    let mut gamma = [[[0.0; 4]; 4]; 4];
    for mu in 0..4 {
        for alpha in 0..4 {
            for beta in 0..4 {
                let mut s = 0.0;
                for sigma in 0..4 {
                    s += g_inv[mu][sigma]
                        * (dg[alpha][sigma][beta] + dg[beta][sigma][alpha]
                            - dg[sigma][alpha][beta]);
                }
                gamma[mu][alpha][beta] = 0.5 * s;
            }
        }
    }
    gamma
}

/// Expand the specialized struct into a dense [mu][alpha][beta] layout.
fn to_dense(c: &SchwarzschildChristoffels) -> [[[f64; 4]; 4]; 4] {
    let mut g = [[[0.0; 4]; 4]; 4];
    // Γ^t_{tr} = Γ^t_{rt}
    g[0][0][1] = c.gamma_t_tr;
    g[0][1][0] = c.gamma_t_tr;
    // Γ^r_{tt}, Γ^r_{rr}, Γ^r_{θθ}, Γ^r_{φφ}
    g[1][0][0] = c.gamma_r_tt;
    g[1][1][1] = c.gamma_r_rr;
    g[1][2][2] = c.gamma_r_thth;
    g[1][3][3] = c.gamma_r_phph;
    // Γ^θ_{rθ} = Γ^θ_{θr}, Γ^θ_{φφ}
    g[2][1][2] = c.gamma_th_rth;
    g[2][2][1] = c.gamma_th_rth;
    g[2][3][3] = c.gamma_th_phph;
    // Γ^φ_{rφ} = Γ^φ_{φr}, Γ^φ_{θφ} = Γ^φ_{φθ}
    g[3][1][3] = c.gamma_ph_rph;
    g[3][3][1] = c.gamma_ph_rph;
    g[3][2][3] = c.gamma_ph_thph;
    g[3][3][2] = c.gamma_ph_thph;
    g
}

fn dense_geodesic_accel(gamma: &[[[f64; 4]; 4]; 4], k: &Vector) -> Vector {
    let mut out = [0.0; 4];
    for mu in 0..4 {
        let mut s = 0.0;
        for a in 0..4 {
            for b in 0..4 {
                s += gamma[mu][a][b] * k[a] * k[b];
            }
        }
        out[mu] = -s;
    }
    out
}

// ---------- Layer 1: metric values ----------

#[test]
fn metric_component_values_equatorial() {
    // r=10, θ=π/2: g_tt = -0.8, g_rr = 1.25, g_θθ = g_φφ = 100
    let g = Schwarzschild.metric_at(&[0.0, 10.0, FRAC_PI_2, 0.0]);
    assert!(approx_eq(
        g.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]),
        -0.8,
        TOL
    ));
    assert!(approx_eq(
        g.dot(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]),
        1.25,
        TOL
    ));
    assert!(approx_eq(
        g.dot(&[0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 1.0, 0.0]),
        100.0,
        TOL
    ));
    assert!(approx_eq(
        g.dot(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]),
        100.0,
        TOL
    ));
}

#[test]
fn metric_off_equator() {
    // g_φφ = r² sin²θ; at r=20, θ=π/4: 400 * 0.5 = 200
    let g = Schwarzschild.metric_at(&[0.0, 20.0, FRAC_PI_4, 0.0]);
    assert!(approx_eq(
        g.dot(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]),
        200.0,
        TOL
    ));
}

#[test]
fn metric_asymptotic_flatness() {
    // Large r: g_tt → -1, g_rr → 1
    let g = Schwarzschild.metric_at(&[0.0, 1e8, 1.0, 0.0]);
    assert!(approx_eq(
        g.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]),
        -1.0,
        1e-7
    ));
    assert!(approx_eq(
        g.dot(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]),
        1.0,
        1e-7
    ));
}

#[test]
fn christoffel_components_at_r10_equatorial() {
    // Hand-computed values at r=10, θ=π/2:
    //   Γ^t_tr    = 1/(10·8) = 0.0125
    //   Γ^r_tt    = 8/1000  = 0.008
    //   Γ^r_rr    = -0.0125
    //   Γ^r_θθ    = -8
    //   Γ^r_φφ    = -8 · sin²(π/2) = -8
    //   Γ^θ_rθ    = 0.1
    //   Γ^θ_φφ    = -sin(π/2)·cos(π/2) = 0
    //   Γ^φ_rφ    = 0.1
    //   Γ^φ_θφ    = cos(π/2)/sin(π/2) = 0
    let c = Schwarzschild.christoffels_at(&[0.0, 10.0, FRAC_PI_2, 0.0]);
    assert!(approx_eq(c.gamma_t_tr, 0.0125, TOL));
    assert!(approx_eq(c.gamma_r_tt, 0.008, TOL));
    assert!(approx_eq(c.gamma_r_rr, -0.0125, TOL));
    assert!(approx_eq(c.gamma_r_thth, -8.0, TOL));
    assert!(approx_eq(c.gamma_r_phph, -8.0, TOL));
    assert!(approx_eq(c.gamma_th_rth, 0.1, TOL));
    assert!(approx_eq(c.gamma_th_phph, 0.0, TOL));
    assert!(approx_eq(c.gamma_ph_rph, 0.1, TOL));
    assert!(approx_eq(c.gamma_ph_thph, 0.0, TOL));
}

#[test]
fn christoffel_components_at_r10_off_equator() {
    // r=10, θ=π/4: sin²θ = 0.5, sinθ cosθ = 0.5, cot θ = 1
    let c = Schwarzschild.christoffels_at(&[0.0, 10.0, FRAC_PI_4, 0.0]);
    assert!(approx_eq(c.gamma_r_phph, -4.0, TOL)); // -8 * 0.5
    assert!(approx_eq(c.gamma_th_phph, -0.5, TOL));
    assert!(approx_eq(c.gamma_ph_thph, 1.0, TOL));
}

#[test]
fn christoffel_symmetry_relations() {
    // Γ^r_rr ≡ -Γ^t_tr and Γ^θ_rθ ≡ Γ^φ_rφ at every point.
    // If this ever fails, a future dedup of these fields would be unsafe.
    for &(r, th) in &[(5.0, 0.7), (10.0, 1.2), (50.0, 2.1), (100.0, 1.57)] {
        let c = Schwarzschild.christoffels_at(&[0.0, r, th, 0.0]);
        assert!(
            approx_eq(c.gamma_r_rr, -c.gamma_t_tr, TOL),
            "Γ^r_rr ≠ -Γ^t_tr at r={r}, θ={th}"
        );
        assert!(
            approx_eq(c.gamma_th_rth, c.gamma_ph_rph, TOL),
            "Γ^θ_rθ ≠ Γ^φ_rφ at r={r}, θ={th}"
        );
    }
}

#[test]
fn christoffel_closed_form_matches_finite_difference() {
    let test_points = [
        [0.0, 10.0, 1.2, 0.5],
        [0.0, 5.0, 0.8, 2.1],
        [0.0, 50.0, FRAC_PI_2, 0.0],
        [0.0, 3.5, 0.4, -1.7],
    ];

    for x in test_points {
        let cf = to_dense(&Schwarzschild.christoffels_at(&x));
        let fd = christoffel_finite_difference(&x);

        for mu in 0..4 {
            for a in 0..4 {
                for b in 0..4 {
                    let diff = (cf[mu][a][b] - fd[mu][a][b]).abs();
                    assert!(
                        diff < FD_TOL,
                        "mismatch at x={x:?} (μ,α,β)=({mu},{a},{b}): CF={} FD={} diff={}",
                        cf[mu][a][b],
                        fd[mu][a][b],
                        diff
                    );
                }
            }
        }
    }
}

#[test]
fn christoffel_lower_index_symmetry() {
    // Γ^μ_{αβ} == Γ^μ_{βα} after expansion
    let dense = to_dense(&Schwarzschild.christoffels_at(&[0.0, 7.0, 0.9, 0.3]));
    for mu in 0..4 {
        for a in 0..4 {
            for b in 0..4 {
                assert_eq!(
                    dense[mu][a][b], dense[mu][b][a],
                    "symmetry violation at (μ,α,β)=({mu},{a},{b})"
                );
            }
        }
    }
}

#[test]
fn specialized_accel_matches_dense() {
    let x = [0.0, 8.0, 1.0, 0.4];
    let c_struct = Schwarzschild.christoffels_at(&x);
    let c_dense = to_dense(&c_struct);

    let ks: [Vector; 5] = [
        [1.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.3, 0.2],
        [1.2, -0.3, 0.01, 0.05],
        [0.7, 0.4, -0.2, -0.1],
    ];

    for k in ks {
        let a_spec = c_struct.geodesic_accel(&k);
        let a_dense = dense_geodesic_accel(&c_dense, &k);
        for i in 0..4 {
            assert!(
                (a_spec[i] - a_dense[i]).abs() < 1e-13,
                "mismatch at k={k:?} component {i}: spec={} dense={}",
                a_spec[i],
                a_dense[i]
            );
        }
    }
}

#[test]
fn static_particle_accel_is_radial_inward() {
    // Static test particle at r=10, θ=π/2: k = (k^t, 0, 0, 0) with
    // g_tt (k^t)² = -1  =>  k^t = 1/√0.8
    // Only nonzero contribution: a^r = -Γ^r_tt (k^t)² = -0.008/0.8 = -0.01
    let x = [0.0, 10.0, FRAC_PI_2, 0.0];
    let kt = (1.0_f64 / 0.8).sqrt();
    let k = [kt, 0.0, 0.0, 0.0];
    let a = Schwarzschild.christoffels_at(&x).geodesic_accel(&k);
    assert!(approx_eq(a[0], 0.0, TOL));
    assert!(approx_eq(a[1], -0.01, TOL));
    assert!(approx_eq(a[2], 0.0, TOL));
    assert!(approx_eq(a[3], 0.0, TOL));
}
