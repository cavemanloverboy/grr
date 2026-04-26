use grr_core::math::{metric::Metric, spacetimes::schwarzchild::Schwarzschild};

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
    let cam = Camera::new(1000.0, 85.0_f64.to_radians());
    let field = Schwarzschild;
    for j in (0..cam.height as usize).step_by(50) {
        for i in (0..cam.width as usize).step_by(50) {
            let (x, k) = cam.pixel_to_initial_state(&field, i, j);
            let norm = field.metric_at(&x).dot(&k, &k);
            assert!(norm.abs() < 1e-10, "k not null at ({i}, {j}): k*k = {norm}");
        }
    }
}
