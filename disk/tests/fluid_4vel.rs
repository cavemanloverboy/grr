use std::f64::consts::FRAC_PI_2;

use grr_core::math::{field::MetricField, metric::Metric, spacetimes::kerr::KerrBoydLindquist};

use grr_disk::{fluid_4velocity, r_isco};

#[test]
fn fluid_4velocity_is_timelike_unit_normalized() {
    for a in [0.0, 0.5, 0.9, 0.998] {
        let field = KerrBoydLindquist::new(a);
        let r_in = r_isco(a).prograde;
        for r in [r_in + 0.01, r_in + 1.0, 5.0_f64.max(r_in + 0.5), 10.0, 20.0] {
            let u = fluid_4velocity(&field, r, a);
            let x = [0.0, r, FRAC_PI_2, 0.0];
            let norm = field.metric_at(&x).dot(&u, &u);
            assert!((norm + 1.0).abs() < 1e-12, "u·u at a={a}, r={r}: {norm}");
        }
    }
}
