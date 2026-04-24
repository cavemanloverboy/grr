use crate::math::Vector;

pub trait Metric {
    fn dot(&self, a: &Vector, b: &Vector) -> f64;
    fn lower(&self, v: &Vector) -> Vector;
    fn raise(&self, v: &Vector) -> Vector;
}

pub struct DiagonalMetric([f64; 4]);

impl DiagonalMetric {
    #[inline(always)]
    pub fn new(diag: [f64; 4]) -> Self {
        DiagonalMetric(diag)
    }
}

impl Metric for DiagonalMetric {
    #[inline(always)]
    fn dot(&self, a: &Vector, b: &Vector) -> f64 {
        a[0] * b[0] * self.0[0]
            + a[1] * b[1] * self.0[1]
            + a[2] * b[2] * self.0[2]
            + a[3] * b[3] * self.0[3]
    }

    #[inline(always)]
    fn lower(&self, v: &Vector) -> Vector {
        [
            v[0] * self.0[0],
            v[1] * self.0[1],
            v[2] * self.0[2],
            v[3] * self.0[3],
        ]
    }

    #[inline(always)]
    fn raise(&self, v: &Vector) -> Vector {
        [
            v[0] / self.0[0],
            v[1] / self.0[1],
            v[2] / self.0[2],
            v[3] / self.0[3],
        ]
    }
}

pub struct TPhiMetric([f64; 5]);

impl TPhiMetric {
    #[inline(always)]
    pub fn new(g: [f64; 5]) -> TPhiMetric {
        TPhiMetric(g)
    }
}

impl Metric for TPhiMetric {
    #[inline(always)]
    fn dot(&self, a: &Vector, b: &Vector) -> f64 {
        let [g_tt, g_rr, g_th, g_ph, g_tp] = self.0;
        g_tt * a[0] * b[0]
            + g_rr * a[1] * b[1]
            + g_th * a[2] * b[2]
            + g_ph * a[3] * b[3]
            + g_tp * (a[0] * b[3] + a[3] * b[0])
    }

    #[inline(always)]
    fn lower(&self, v: &Vector) -> Vector {
        let [g_tt, g_rr, g_th, g_ph, g_tp] = self.0;
        [
            g_tt * v[0] + g_tp * v[3],
            g_rr * v[1],
            g_th * v[2],
            g_tp * v[0] + g_ph * v[3],
        ]
    }

    // TODO (perf): this inv_det may be no bueno
    #[inline(always)]
    fn raise(&self, v: &Vector) -> Vector {
        let [g_tt, g_rr, g_th, g_ph, g_tp] = self.0;
        // Invert the 2x2 (t, φ) block:
        //   [ g_tt  g_tφ ] ^-1    1  [  g_φφ  -g_tφ ]
        //   [ g_tφ  g_φφ ]     = --- [ -g_tφ   g_tt ]
        //                         D
        // where D = g_tt · g_φφ - g_tφ²
        let det_tp = g_tt * g_ph - g_tp * g_tp;
        let inv_det = 1.0 / det_tp;
        [
            (g_ph * v[0] - g_tp * v[3]) * inv_det,
            v[1] / g_rr,
            v[2] / g_th,
            (-g_tp * v[0] + g_tt * v[3]) * inv_det,
        ]
    }
}
