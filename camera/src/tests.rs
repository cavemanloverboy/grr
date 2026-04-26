use grr_core::math::{field::MetricField, metric::Metric, spacetimes::schwarzchild::Schwarzschild};
use grr_tetrad::zamo_tetrad;

use super::*;

#[test]
fn pixel_to_sky_angles_center_pixel_is_zero() {
    // 5x5 grid. pixel (2, 2) sits exactly at the image.
    // angles should be zero.
    let cam = Camera {
        width: 5,
        height: 5,
        field_of_view: 0.2,
        r_cam: 100.0,
        th_cam: 1.4,
    };
    let (a, b) = cam.pixel_to_sky_angles(2, 2);
    assert!(a.abs() < 1e-15, "{a}");
    assert!(b.abs() < 1e-15, "{b}");
}

#[test]
fn tetrad_direction_is_always_null() {
    let cam = Camera {
        width: 1920,
        height: 1080,
        field_of_view: 0.5,
        r_cam: 100.0,
        th_cam: 1.4,
    };
    for i in 0..1920 {
        for j in 0..1080 {
            let n = cam.tetrad_frame_direction(i, j);
            let norm = -n[0] * n[0] + n[1] * n[1] + n[2] * n[2] + n[3] * n[3];
            assert!(
                norm.abs() < 1e-15,
                "n at ({i}, {j}) not null: η*n*n = {norm}"
            );
        }
    }
}

#[test]
fn coordinate_momentum_is_null_in_curved_spacetime() {
    let r_cam = 1000.0;
    let th_cam = 85.0_f64.to_radians();
    let cam = Camera::new(r_cam, th_cam);
    let field = Schwarzschild;
    let tetrad = zamo_tetrad(&field, &[0.0, r_cam, th_cam, 0.0]);
    for j in (0..cam.height as usize).step_by(50) {
        for i in (0..cam.width as usize).step_by(50) {
            let (x, k) = cam.pixel_to_initial_state(&tetrad, i, j);
            let norm = field.metric_at(&x).dot(&k, &k);
            assert!(norm.abs() < 1e-10, "k not null at ({i}, {j}): k*k = {norm}");
        }
    }
}
