use crate::math::{Vector, christoffel::Christoffels, field::MetricField, metric::DiagonalMetric};

#[cfg(test)]
mod tests;

pub struct Schwarzschild;

impl MetricField for Schwarzschild {
    type LocalMetric = DiagonalMetric;
    type LocalChristoffels = SchwarzschildChristoffels;

    fn metric_at(&self, x: &crate::math::Vector) -> DiagonalMetric {
        let r = x[1];
        let th = x[2];
        let f = 1.0 - 2.0 / r;
        let sin_t = th.sin();
        DiagonalMetric::new([-f, 1.0 / f, r * r, r * r * sin_t * sin_t])
    }

    /// Γ^t_tr = 1 / (r(r - 2))                      [and Γ^t_rt by symmetry]
    /// Γ^r_tt = (r - 2) / r³
    /// Γ^r_rr = -1 / (r(r - 2))
    /// Γ^r_θθ = -(r - 2)
    /// Γ^r_φφ = -(r - 2) sin²θ
    /// Γ^θ_rθ = 1/r                                 [and Γ^θ_θr by symmetry]
    /// Γ^θ_φφ = -sinθ cosθ
    /// Γ^φ_rφ = 1/r                                 [and Γ^φ_φr by symmetry]
    /// Γ^φ_θφ = cosθ/sinθ = cot θ                   [and Γ^φ_φθ by symmetry]
    fn christoffels_at(&self, x: &Vector) -> SchwarzschildChristoffels {
        let r = x[1];
        let th = x[2];
        let (sin_th, cos_th) = th.sin_cos();

        let r_minus_2 = r - 2.0;
        let inv_r = 1.0 / r;
        let inv_r_rm2 = 1.0 / (r * r_minus_2);

        SchwarzschildChristoffels {
            gamma_t_tr: inv_r_rm2,
            gamma_r_tt: r_minus_2 / (r * r * r),
            gamma_r_rr: -inv_r_rm2,
            gamma_r_thth: -r_minus_2,
            gamma_r_phph: -r_minus_2 * sin_th * sin_th,
            gamma_th_rth: inv_r,
            gamma_th_phph: -sin_th * cos_th,
            gamma_ph_rph: inv_r,
            gamma_ph_thph: cos_th / sin_th,
        }
    }
}

/// TODO (perf): this is just over one cache line. there are some
/// related components, e.g.
///
/// Γ^t_tr == -Γ^r_rr
/// Γ^θ_rθ == Γ^φ_rφ
///
/// we can exploit this to get it <=1 cache line.
pub struct SchwarzschildChristoffels {
    gamma_t_tr: f64,    // Γ^t_tr
    gamma_r_tt: f64,    // Γ^r_tt
    gamma_r_rr: f64,    // Γ^r_rr
    gamma_r_thth: f64,  // Γ^r_θθ
    gamma_r_phph: f64,  // Γ^r_φφ
    gamma_th_rth: f64,  // Γ^θ_rθ
    gamma_th_phph: f64, // Γ^θ_φφ
    gamma_ph_rph: f64,  // Γ^φ_rφ
    gamma_ph_thph: f64, // Γ^φ_θφ
}

impl Christoffels for SchwarzschildChristoffels {
    #[inline(always)]
    fn geodesic_accel(&self, k: &Vector) -> Vector {
        let [kt, kr, kth, kph] = k;
        // Each Γ with unequal lower indices contributes 2× (from αβ + βα)
        [
            -2.0 * self.gamma_t_tr * kt * kr,
            -(self.gamma_r_tt * kt * kt
                + self.gamma_r_rr * kr * kr
                + self.gamma_r_thth * kth * kth
                + self.gamma_r_phph * kph * kph),
            -(2.0 * self.gamma_th_rth * kr * kth + self.gamma_th_phph * kph * kph),
            -2.0 * (self.gamma_ph_rph * kr * kph + self.gamma_ph_thph * kth * kph),
        ]
    }
}
