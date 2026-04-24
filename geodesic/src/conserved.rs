use grr_core::math::{field::MetricField, metric::Metric};

use crate::state::State;

impl State {
    /// k_t = g_tμ k^μ. For stationary spacetimes (Schwarzschild, Kerr), -k_t
    /// is the conserved energy E along a geodesic.
    #[inline(always)]
    pub fn energy<F: MetricField>(&self, field: &F) -> f64 {
        let x = self.position();
        let k = self.momentum();
        -field.metric_at(&x).lower(&k)[0]
    }

    /// k_φ = g_φμ k^μ. Conserved axial angular momentum for axisymmetric spacetimes.
    #[inline(always)]
    pub fn angular_momentum_z<F: MetricField>(&self, field: &F) -> f64 {
        let x = self.position();
        let k = self.momentum();
        field.metric_at(&x).lower(&k)[3]
    }

    /// g_μν k^μ k^ν. Should be 0 for photons, -1 for massive particles.
    /// Monitor drift as a numerical health signal during integration.
    #[inline(always)]
    pub fn norm_squared<F: MetricField>(&self, field: &F) -> f64 {
        let x = self.position();
        let k = self.momentum();
        field.metric_at(&x).dot(&k, &k)
    }
}
