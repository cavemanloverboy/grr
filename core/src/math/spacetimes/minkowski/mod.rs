use crate::math::{Vector, christoffel::Christoffels, field::MetricField, metric::DiagonalMetric};

#[cfg(test)]
mod tests {}

pub struct Minkowski;

impl MetricField for Minkowski {
    type LocalMetric = DiagonalMetric;
    type LocalChristoffels = MinkowskiChristoffels;

    fn metric_at(&self, _x: &Vector) -> Self::LocalMetric {
        DiagonalMetric::new([-1.0, 1.0, 1.0, 1.0])
    }

    fn christoffels_at(&self, x: &Vector) -> Self::LocalChristoffels {
        let r = x[1];
        let th = x[2];

        let (sin_th, cos_th) = th.sin_cos();
        let r_inv = 1.0 / r;

        // Γ^r_θθ  = -r
        // Γ^r_φφ  = -r sin²θ
        // Γ^θ_rθ  = Γ^θ_θr  = 1/r
        // Γ^θ_φφ  = -sinθ cosθ
        // Γ^φ_rφ  = Γ^φ_φr  = 1/r
        // Γ^φ_θφ  = Γ^φ_φθ  = cosθ/sinθ

        MinkowskiChristoffels {
            r_th_th: -r,
            r_phi_phi: -r * sin_th * sin_th,
            th_r_th: r_inv,
            th_phi_phi: -sin_th * cos_th,
            phi_r_phi: r_inv,
            phi_th_phi: cos_th / sin_th,
        }
    }
}

pub struct MinkowskiChristoffels {
    r_th_th: f64,
    r_phi_phi: f64,
    th_r_th: f64,
    th_phi_phi: f64,
    phi_r_phi: f64,
    phi_th_phi: f64,
}

impl Christoffels for MinkowskiChristoffels {
    fn geodesic_accel(&self, k: &Vector) -> Vector {
        let [_kt, kr, kth, kph] = *k;
        [
            // accel^t = 0 (no Γ^t_αβ components)
            0.0,
            // accel^r: Γ^r_θθ kθ² + Γ^r_φφ kφ²
            -(self.r_th_th * kth * kth + self.r_phi_phi * kph * kph),
            // accel^θ: 2 Γ^θ_rθ kr kθ + Γ^θ_φφ kφ²
            -(2.0 * self.th_r_th * kr * kth + self.th_phi_phi * kph * kph),
            // accel^φ: 2 Γ^φ_rφ kr kφ + 2 Γ^φ_θφ kθ kφ
            -2.0 * (self.phi_r_phi * kr * kph + self.phi_th_phi * kth * kph),
        ]
    }
}
