use super::*;
use crate::math::metric::Metric;
use std::f64::consts::FRAC_PI_2;

const TOL: f64 = 1e-12;
const FD_TOL: f64 = 1e-5;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol
}

fn dense_metric(field: &KerrBoydLindquist, x: &Vector) -> [[f64; 4]; 4] {
    let g = field.metric_at(x);
    let mut m = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            let mut e_mu = [0.0; 4];
            e_mu[mu] = 1.0;
            let mut e_nu = [0.0; 4];
            e_nu[nu] = 1.0;
            m[mu][nu] = g.dot(&e_mu, &e_nu);
        }
    }
    m
}

fn dense_metric_inverse(field: &KerrBoydLindquist, x: &Vector) -> [[f64; 4]; 4] {
    let g = field.metric_at(x);
    let mut m_inv = [[0.0; 4]; 4];
    for mu in 0..4 {
        let mut e_mu = [0.0; 4];
        e_mu[mu] = 1.0;
        let raised = g.raise(&e_mu);
        for nu in 0..4 {
            m_inv[nu][mu] = raised[nu];
        }
    }
    m_inv
}

fn dense_metric_finite_diff(field: &KerrBoydLindquist, x: &Vector, alpha: usize) -> [[f64; 4]; 4] {
    let h = 1e-5;
    let mut x_plus = *x;
    x_plus[alpha] += h;
    let mut x_minus = *x;
    x_minus[alpha] -= h;
    let gp = dense_metric(field, &x_plus);
    let gm = dense_metric(field, &x_minus);
    let mut out = [[0.0; 4]; 4];
    for mu in 0..4 {
        for nu in 0..4 {
            out[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * h);
        }
    }
    out
}

fn christoffel_finite_diff(field: &KerrBoydLindquist, x: &Vector) -> [[[f64; 4]; 4]; 4] {
    let g_inv = dense_metric_inverse(field, x);
    let dg: [[[f64; 4]; 4]; 4] = std::array::from_fn(|a| dense_metric_finite_diff(field, x, a));
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

/// Expand the specialized struct into the dense [μ][α][β] layout for comparison.
fn to_dense(c: &KerrChristoffels) -> [[[f64; 4]; 4]; 4] {
    let mut g = [[[0.0; 4]; 4]; 4];
    // t row
    g[0][0][1] = c.t_tr;
    g[0][1][0] = c.t_tr;
    g[0][0][2] = c.t_tth;
    g[0][2][0] = c.t_tth;
    g[0][1][3] = c.t_rph;
    g[0][3][1] = c.t_rph;
    g[0][2][3] = c.t_thph;
    g[0][3][2] = c.t_thph;
    // r row
    g[1][0][0] = c.r_tt;
    g[1][0][3] = c.r_tph;
    g[1][3][0] = c.r_tph;
    g[1][1][1] = c.r_rr;
    g[1][1][2] = c.r_rth;
    g[1][2][1] = c.r_rth;
    g[1][2][2] = c.r_thth;
    g[1][3][3] = c.r_phph;
    // θ row
    g[2][0][0] = c.th_tt;
    g[2][0][3] = c.th_tph;
    g[2][3][0] = c.th_tph;
    g[2][1][1] = c.th_rr;
    g[2][1][2] = c.th_rth;
    g[2][2][1] = c.th_rth;
    g[2][2][2] = c.th_thth;
    g[2][3][3] = c.th_phph;
    // φ row
    g[3][0][1] = c.ph_tr;
    g[3][1][0] = c.ph_tr;
    g[3][0][2] = c.ph_tth;
    g[3][2][0] = c.ph_tth;
    g[3][1][3] = c.ph_rph;
    g[3][3][1] = c.ph_rph;
    g[3][2][3] = c.ph_thph;
    g[3][3][2] = c.ph_thph;
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

#[test]
fn metric_reduces_to_schwarzschild_at_a_zero() {
    // With a=0, g_tφ should vanish and diagonals should match Schwarzschild values.
    let field = KerrBoydLindquist::new(0.0);
    let x = [0.0, 10.0, FRAC_PI_2, 0.0];
    let g = field.metric_at(&x);
    // g_tt = -(1 - 2/r) = -0.8
    assert!(approx_eq(
        g.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]),
        -0.8,
        TOL
    ));
    // g_rr = 1/0.8 = 1.25
    assert!(approx_eq(
        g.dot(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]),
        1.25,
        TOL
    ));
    // g_θθ = r² = 100
    assert!(approx_eq(
        g.dot(&[0.0, 0.0, 1.0, 0.0], &[0.0, 0.0, 1.0, 0.0]),
        100.0,
        TOL
    ));
    // g_φφ = r² sin²θ = 100
    assert!(approx_eq(
        g.dot(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0, 1.0]),
        100.0,
        TOL
    ));
    // g_tφ = 0 (cross term)
    assert!(approx_eq(
        g.dot(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 1.0]),
        0.0,
        TOL
    ));
}

#[test]
fn metric_frame_dragging_is_nonzero_for_spin() {
    // g_tφ ≠ 0 when a ≠ 0 (off equator would also work; equatorial is strongest)
    let field = KerrBoydLindquist::new(0.9);
    let x = [0.0, 5.0, FRAC_PI_2, 0.0];
    let g = field.metric_at(&x);
    let g_tphi = g.dot(&[1.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 1.0]);
    assert!(
        g_tphi.abs() > 0.1,
        "expected substantial frame dragging, got g_tφ = {g_tphi}"
    );
}

#[test]
fn metric_asymptotic_flatness() {
    // At very large r, Kerr approaches Minkowski in spherical coordinates.
    let field = KerrBoydLindquist::new(0.9);
    let x = [0.0, 1e8, 1.0, 0.0];
    let g = field.metric_at(&x);
    assert!(approx_eq(
        g.dot(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0, 0.0, 0.0]),
        -1.0,
        1e-6
    ));
    assert!(approx_eq(
        g.dot(&[0.0, 1.0, 0.0, 0.0], &[0.0, 1.0, 0.0, 0.0]),
        1.0,
        1e-6
    ));
}

#[test]
fn christoffel_closed_form_matches_fd() {
    let cases: &[(f64, Vector)] = &[
        (0.0, [0.0, 10.0, 1.2, 0.5]), // Schwarzschild limit
        (0.5, [0.0, 8.0, 1.0, 0.3]),
        (0.9, [0.0, 5.0, 0.7, 2.1]),
        (0.998, [0.0, 4.0, 1.1, -1.2]), // near-extremal
        (0.5, [0.0, 50.0, FRAC_PI_2, 0.0]),
    ];

    for (a_spin, x) in cases {
        let field = KerrBoydLindquist::new(*a_spin);
        let cf = to_dense(&field.christoffels_at(x));
        let fd = christoffel_finite_diff(&field, x);

        for mu in 0..4 {
            for alpha in 0..4 {
                for beta in 0..4 {
                    let diff = (cf[mu][alpha][beta] - fd[mu][alpha][beta]).abs();
                    let scale = cf[mu][alpha][beta]
                        .abs()
                        .max(fd[mu][alpha][beta].abs())
                        .max(1.0);
                    assert!(
                        diff / scale < FD_TOL,
                        "mismatch at a={a_spin}, x={x:?}, (μ,α,β)=({mu},{alpha},{beta}): \
                             CF={} FD={} rel_diff={}",
                        cf[mu][alpha][beta],
                        fd[mu][alpha][beta],
                        diff / scale
                    );
                }
            }
        }
    }
}

#[test]
fn christoffel_lower_index_symmetry() {
    // Γ^μ_{αβ} == Γ^μ_{βα} after expansion — catches bugs in to_dense().
    let field = KerrBoydLindquist::new(0.7);
    let dense = to_dense(&field.christoffels_at(&[0.0, 7.0, 0.9, 0.3]));
    for mu in 0..4 {
        for alpha in 0..4 {
            for beta in 0..4 {
                assert_eq!(
                    dense[mu][alpha][beta], dense[mu][beta][alpha],
                    "symmetry violation at (μ,α,β)=({mu},{alpha},{beta})"
                );
            }
        }
    }
}

#[test]
fn specialized_accel_matches_dense() {
    let field = KerrBoydLindquist::new(0.8);
    let x = [0.0, 6.0, 1.1, 0.4];
    let c_struct = field.christoffels_at(&x);
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
                (a_spec[i] - a_dense[i]).abs() < 1e-12,
                "mismatch at k={k:?} component {i}: spec={} dense={}",
                a_spec[i],
                a_dense[i]
            );
        }
    }
}

#[test]
fn kerr_at_a_zero_matches_schwarzschild_components() {
    // At a=0, Kerr should reduce to Schwarzschild. Spot-check via FD.
    let field = KerrBoydLindquist::new(0.0);
    let x = [0.0, 10.0, FRAC_PI_2, 0.0];
    let cf = to_dense(&field.christoffels_at(&x));

    // Known Schwarzschild values at r=10, θ=π/2:
    //   Γ^t_tr = 1/(r(r-2)) = 1/80 = 0.0125
    //   Γ^r_tt = (r-2)/r³ = 8/1000 = 0.008
    //   Γ^r_rr = -0.0125
    //   Γ^r_θθ = -(r-2) = -8
    //   Γ^r_φφ = -(r-2)sin²θ = -8
    //   Γ^θ_rθ = 1/r = 0.1
    //   Γ^φ_rφ = 1/r = 0.1
    //   Γ^θ_φφ = -sinθ cosθ = 0 (at θ=π/2)
    //   Γ^φ_θφ = cot θ = 0 (at θ=π/2)
    assert!(approx_eq(cf[0][0][1], 0.0125, TOL), "Γ^t_tr");
    assert!(approx_eq(cf[1][0][0], 0.008, TOL), "Γ^r_tt");
    assert!(approx_eq(cf[1][1][1], -0.0125, TOL), "Γ^r_rr");
    assert!(approx_eq(cf[1][2][2], -8.0, TOL), "Γ^r_θθ");
    assert!(approx_eq(cf[1][3][3], -8.0, TOL), "Γ^r_φφ");
    assert!(approx_eq(cf[2][1][2], 0.1, TOL), "Γ^θ_rθ");
    assert!(approx_eq(cf[3][1][3], 0.1, TOL), "Γ^φ_rφ");
    assert!(approx_eq(cf[2][3][3], 0.0, TOL), "Γ^θ_φφ");
    assert!(approx_eq(cf[3][2][3], 0.0, TOL), "Γ^φ_θφ");
}

#[test]
fn circular_equatorial_photon_orbit_at_correct_radius() {
    // For a=0 Schwarzschild, there is an unstable circular photon orbit at r=3.
    // A photon placed at r=3, θ=π/2 with purely tangential momentum should have
    // zero radial acceleration (dk^r/dλ = 0) for a moment.
    //
    // For a null geodesic in the equatorial plane: k = (k^t, 0, 0, k^φ)
    // with null condition g_tt (k^t)² + g_φφ (k^φ)² = 0.
    // At r=3: g_tt = -1/3, g_φφ = 9. So (k^t)² = 27 (k^φ)², take k^φ = 1, k^t = √27.
    //
    // accel^r should be ≈ 0.
    let field = KerrBoydLindquist::new(0.0);
    let x = [0.0, 3.0, FRAC_PI_2, 0.0];
    let k_phi = 1.0_f64;
    let k_t = 27.0_f64.sqrt();
    let k = [k_t, 0.0, 0.0, k_phi];
    let a = field.christoffels_at(&x).geodesic_accel(&k);
    assert!(
        a[1].abs() < 1e-10,
        "expected zero radial accel at photon sphere, got {}",
        a[1]
    );
}
