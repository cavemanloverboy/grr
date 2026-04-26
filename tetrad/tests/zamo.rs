use core::f64::consts::FRAC_PI_2;
use grr_core::math::{
    Vector,
    field::MetricField,
    metric::Metric,
    spacetimes::{kerr::KerrBoydLindquist, schwarzchild::Schwarzschild},
};
use grr_tetrad::{Tetrad, zamo_tetrad};

const TOL: f64 = 1e-12;

/// core orthonormality check: g(e_(a), e_(b)) = η_(a)(b).
/// diagonal: -1 for a=b=0, +1 for a=b in {1,2,3}, 0 off-diagonal.
fn assert_orthonormal<F: MetricField>(field: &F, x: &Vector, t: &Tetrad, tol: f64) {
    let g = field.metric_at(x);
    let eta = [
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for a in 0..4 {
        for b in 0..4 {
            let inner = g.dot(&t.legs[a], &t.legs[b]);
            assert!(
                (inner - eta[a][b]).abs() < tol,
                "g(e_({a}), e_({b})) = {inner}, expected {} (off by {})",
                eta[a][b],
                inner - eta[a][b]
            );
        }
    }
}

#[test]
fn orthonormal_schwarzschild() {
    // a=0 reduces to a static observer. test at several radii and angles.
    let field = Schwarzschild;
    let cases: &[(f64, f64)] = &[
        (10.0, 1.0),
        (100.0, FRAC_PI_2),
        (1000.0, 0.5),
        (3.5, 0.7), // close to horizon
        (50.0, 2.4),
    ];
    for &(r, th) in cases {
        let x = [0.0, r, th, 0.0];
        let t = zamo_tetrad(&field, &x);
        assert_orthonormal(&field, &x, &t, TOL);
    }
}

#[test]
fn orthonormal_kerr() {
    // stress-test across spin values and spacetime points.
    let cases: &[(f64, f64, f64)] = &[
        (0.1, 10.0, 1.0),
        (0.5, 5.0, FRAC_PI_2),
        (0.9, 8.0, 0.7),
        (0.998, 20.0, 1.4),
        (0.5, 100.0, 0.3),
        (0.7, 15.0, FRAC_PI_2),
    ];
    for &(a_spin, r, th) in cases {
        let field = KerrBoydLindquist::new(a_spin);
        let x = [0.0, r, th, 0.0];
        let t = zamo_tetrad(&field, &x);
        assert_orthonormal(&field, &x, &t, TOL);
    }
}

#[test]
fn schwarzschild_zamo_has_zero_omega() {
    // schwarzschild has g_tφ = 0, so ω = 0 and the timelike leg has no φ-component.
    let field = Schwarzschild;
    let x = [0.0, 10.0, 1.0, 0.0];
    let t = zamo_tetrad(&field, &x);
    assert!(
        t.legs[0][3].abs() < 1e-15,
        "schwarzschild zamo should have e_(0)^φ = 0, got {}",
        t.legs[0][3]
    );
}

#[test]
fn kerr_zamo_has_nonzero_omega() {
    // a > 0 means frame dragging. e_(0) should have a nonzero φ-component.
    let field = KerrBoydLindquist::new(0.9);
    let x = [0.0, 5.0, FRAC_PI_2, 0.0];
    let t = zamo_tetrad(&field, &x);
    assert!(
        t.legs[0][3].abs() > 1e-3,
        "kerr a=0.9 zamo should have substantial frame dragging, got e_(0)^φ = {}",
        t.legs[0][3]
    );
}

#[test]
fn zamo_omega_falls_off_at_large_r() {
    // frame-dragging falls as 1/r³. at r = 10⁴ M with a = 1, ω should be tiny.
    let field = KerrBoydLindquist::new(1.0);
    let x = [0.0, 1e4, 1.0, 0.0];
    let t = zamo_tetrad(&field, &x);
    // e_(0) = (1/N, 0, 0, ω/N), so ω = e_(0)[3] / e_(0)[0]
    let omega = t.legs[0][3] / t.legs[0][0];
    assert!(
        omega.abs() < 1e-11,
        "zamo at large r should have ω ≈ 0, got {omega}"
    );
}

#[test]
fn schwarzschild_lapse_at_r10_equals_sqrt_zero_point_eight() {
    // at r=10, schwarzschild static observer has N = √(1 - 2/10) = √0.8.
    // the timelike leg is e_(0) = (1/N, 0, 0, 0), so e_(0)[0] = 1/N.
    let field = Schwarzschild;
    let x = [0.0, 10.0, 1.0, 0.0];
    let t = zamo_tetrad(&field, &x);
    let inv_n = t.legs[0][0];
    let n = inv_n.recip();
    assert!(
        (n - 0.8_f64.sqrt()).abs() < 1e-14,
        "expected N = √0.8 ≈ {}, got {n}",
        0.8_f64.sqrt()
    );
}

// asymptotic sanity check
#[test]
fn at_large_r_zamo_is_minkowski_spherical_orthonormal_frame() {
    // far from the hole, zamo legs approach the natural orthonormal frame
    // in spherical minkowski:
    //   e_(0) → (1, 0, 0, 0)
    //   e_(1) → (0, 1, 0, 0)
    //   e_(2) → (0, 0, 1/r, 0)
    //   e_(3) → (0, 0, 0, 1/(r sin θ))
    let field = KerrBoydLindquist::new(0.5);
    let r = 1e6;
    let th = 1.0;
    let x = [0.0, r, th, 0.0];
    let t = zamo_tetrad(&field, &x);

    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1.0);

    assert!(rel(t.legs[0][0], 1.0) < 1e-5);
    assert!(rel(t.legs[1][1], 1.0) < 1e-5);
    assert!(rel(t.legs[2][2], 1.0 / r) < 1e-5);
    assert!(rel(t.legs[3][3], 1.0 / (r * th.sin())) < 1e-5);
}
