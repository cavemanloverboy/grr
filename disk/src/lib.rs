use grr_core::math::{Vector, field::MetricField, metric::Metric};

pub struct Disk {
    pub r_in: f64,
    pub r_out: f64,
    pub a: f64,
}

#[repr(C)]
pub struct ISCO {
    pub prograde: f64,
    pub retrograde: f64,
}

/// for kerr with spin a, r_isco given by:
///
/// z1 = 1 + (1 - a^2)^(1/3) * [(1 + a)^(1/3) + (1 - a)^(1/3)]
/// z2 = sqrt(3*a² + z1^2)
/// r_isco = 3 + z2 +/- sqrt((3 - z1)*(3 + z1 + 2*z2))
pub fn r_isco(a: f64) -> ISCO {
    let a2 = a * a;
    let z1 =
        1.0 + (1.0 - a2).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
    let z2 = (3.0 * a2 + z1 * z1).sqrt();

    let plus_minus_term = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt();
    let prograde = 3.0 + z2 - plus_minus_term;
    let retrograde = 3.0 + z2 + plus_minus_term;

    ISCO {
        prograde,
        retrograde,
    }
}

fn keplerian_omega(r: f64, a: f64) -> f64 {
    1.0 / (r.powf(1.5) + a)
}

pub fn fluid_4velocity<F: MetricField>(field: &F, r: f64, a: f64) -> Vector {
    let omega = keplerian_omega(r, a);
    let x = [0.0, r, core::f64::consts::FRAC_PI_2, 0.0];
    let metric = field.metric_at(&x);
    let unnorm = [1.0, 0.0, 0.0, omega];
    let norm_sq = metric.dot(&unnorm, &unnorm);
    debug_assert!(
        norm_sq < 0.0,
        "fluid 4-velocity not timelike at r={r}, a={a}: u*u_unnorm = {norm_sq}"
    );
    let u_t = (-norm_sq).sqrt().recip();
    [u_t, 0.0, 0.0, omega * u_t]
}

/// computes g = ν_obs / ν_em.
/// the observed value is ν = -k_μ u^μ, so
///
/// g = ν_obs / ν_em
///   = (-k_μ u^μ_obs) / (-k_μ u^μ_em)
// /  = (k_μ u^μ_obs) / (k_μ u^μ_em)
pub fn redshift_factor<F: MetricField>(
    field: &F,
    x_hit: &Vector,
    k_hit: &Vector,
    u_emitter: &Vector,
    u_observer: &Vector,
) -> f64 {
    let metric = field.metric_at(x_hit);
    let k_dot_u_em = metric.dot(k_hit, u_emitter);
    let k_dot_u_obs = metric.dot(k_hit, u_observer);
    k_dot_u_obs / k_dot_u_em
}
