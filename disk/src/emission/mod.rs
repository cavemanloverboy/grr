//! Page-Thorne (1974) thin-disk flux profile.
//! reference: Page & Thorne, ApJ 191, 499 (1974), equations 15k–15n.
//!
//! G = c = M = 1, M_dot = 1.
//!
//! flux profile is
//! F(r) = M_dot * f(x, a*) / (4 pi M^2 x^2)
//!
//! with x = sqrt(r/M), a* = a/M. Setting M = M_dot = 1: F = f / (4 pi x^2).

use super::r_isco;
use core::f64::consts::PI;

/// three roots of the cubic x^3 - 3x + 2*a_star = 0.
/// page/thorne p.502 (lower half)
///
/// using the trigonometric solution with phi = (1/3) * acos(a_star):
/// x1 =  2 * cos(phi - pi/3)
/// x2 =  2 * cos(phi + pi/3)
/// x3 = -2 * cos(phi)
fn cubic_roots(a_star: f64) -> (f64, f64, f64) {
    let phi = a_star.acos() / 3.0;
    let x1 = 2.0 * (phi - PI / 3.0).cos();
    let x2 = 2.0 * (phi + PI / 3.0).cos();
    let x3 = -2.0 * phi.cos();
    (x1, x2, x3)
}

/// page-thorne 1974 flux function f(x, a*). equation 15n.
///
/// f = (3 / 2M) * 1/(x^2 * (x^3 - 3x + 2*a_*))
///     * [ x - x0 - (3/2)*a_* * ln(x/x0)
///         - c1 * ln((x - x1)/(x0 - x1))
///         - c2 * ln((x - x2)/(x0 - x2))
///         - c3 * ln((x - x3)/(x0 - x3)) ]
///
/// with the log-coefficient prefactors:
///
///   c1 = 3*(x1 - a*)^2 / (x1 * (x1 - x2) * (x1 - x3))
///   c2 = 3*(x2 - a*)^2 / (x2 * (x2 - x1) * (x2 - x3))
///   c3 = 3*(x3 - a*)^2 / (x3 * (x3 - x1) * (x3 - x2))
///
/// where x0 = sqrt(r_isco/M) and (x1, x2, x3) are roots of the cubic
/// from page-thorne eq. 15l (see `cubic_roots`).
///
/// returns 0 inside ISCO (x <= x0) since circular geodesic motion is
/// unstable there and the disk plunges.
///
/// (recall we are doing M=1 :D)
pub fn flux_f(x: f64, a_star: f64) -> f64 {
    let r_isco = r_isco(a_star);
    let x0 = r_isco.prograde.sqrt();
    if x <= x0 {
        // there is nothing within isco
        return 0.0;
    }
    let (x1, x2, x3) = cubic_roots(a_star);

    let c1 = 3.0 * (x1 - a_star).powi(2) / (x1 * (x1 - x2) * (x1 - x3));
    let c2 = 3.0 * (x2 - a_star).powi(2) / (x2 * (x2 - x1) * (x2 - x3));
    let c3 = 3.0 * (x3 - a_star).powi(2) / (x3 * (x3 - x1) * (x3 - x2));

    let bracket = (x - x0)
        - 1.5 * a_star * (x / x0).ln()
        - c1 * ((x - x1) / (x0 - x1)).ln()
        - c2 * ((x - x2) / (x0 - x2)).ln()
        - c3 * ((x - x3) / (x0 - x3)).ln();

    let prefactor = 1.5 / (x * x * (x.powi(3) - 3.0 * x + 2.0 * a_star));
    prefactor * bracket
}

/// time-averaged radiative flux per unit proper area, geometrized units.
///
/// F(r) = M_dot * f(x, a*) / (4 pi M^2 x^2)
///      = f(x, a*) / (4 pi x^2)        (with M = M_dot = 1)
///
/// where x = sqrt(r/M). this is page-thorne eq. 11b combined with the
/// kerr metric component e^(nu+psi+mu) = r = M*x^2 from eq. 15d.
///
/// units: in code units this returns flux scaled to the dimensionless
/// disk model. for physical units multiply by M_dot * c^6 / (G^2 M^3).
pub fn flux(r: f64, a_star: f64) -> f64 {
    let x = r.sqrt();
    flux_f(x, a_star) / (4.0 * PI * x * x)
}

/// bolometric specific intensity emitted by the disk in its rest frame.
/// for a lambertian (isotropic) thin disk, I_em = F / pi.
pub fn bolometric_intensity(r: f64, a_star: f64) -> f64 {
    flux(r, a_star) / PI
}
