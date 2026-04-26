//! camera. sets up a grid of pixels and transforms that
//! to a grid of initial conditions for photons that ended up there.
//! from here, we can simulate photon trajectory backwards and then
//! choose pixel color based on that
use std::f64::consts::FRAC_PI_4;

use grr_core::math::{Vector, field::MetricField};
use grr_tetrad::zamo_tetrad;

pub struct Camera {
    r_cam: f64,
    th_cam: f64,
    field_of_view: f64,
    width: u16,
    height: u16,
}

impl Camera {
    pub fn new(r_cam: f64, th_cam: f64) -> Camera {
        Camera {
            r_cam,
            th_cam,
            field_of_view: FRAC_PI_4,
            width: 1920,
            height: 1080,
        }
    }

    /// update position of the camera
    pub fn update_pos(&mut self, r_cam: f64, th_cam: f64) {
        self.r_cam = r_cam;
        self.th_cam = th_cam;
    }

    /// radians
    pub fn update_fov(&mut self, fov: f64) {
        self.field_of_view = fov;
    }

    // pixel width, height
    pub fn update_image_res(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }

    pub fn pixel_to_initial_state<F: MetricField>(
        &self,
        field: &F,
        i: usize,
        j: usize,
    ) -> (Vector, Vector) {
        let x_cam = [0.0, self.r_cam, self.th_cam, 0.0];
        let k = self.pixel_to_coordinate_momentum(field, i, j);
        (x_cam, k)
    }

    /// sky-plane angles (α , β) in radians for pixel (i, j).
    /// α  is the horizontal offset, β the vertical, both measured from
    /// the optical axis. (0, 0) is the image center.
    ///
    /// α = (i - width/2 + 0.5) / width * fov
    /// β = -(j - height/2 + 0.5) / height * fov * (height / width)
    fn pixel_to_sky_angles(&self, i: usize, j: usize) -> (f64, f64) {
        let w = self.width as f64;
        let h = self.height as f64;
        let aspect = h / w;
        let alpha = (i as f64 - w / 2.0 + 0.5) / w * self.field_of_view;
        let beta = -(j as f64 - h / 2.0 + 0.5) / h * self.field_of_view * aspect;
        (alpha, beta)
    }

    /// photon direction in the camera's tetrad frame, given sky-plane angles.
    /// returns n^(a) where a indexes (time, radial, polar, azimuthal).
    /// the result is null in the tetrad-frame minkowski metric.
    fn tetrad_frame_direction(&self, i: usize, j: usize) -> [f64; 4] {
        let (alpha, beta) = self.pixel_to_sky_angles(i, j);
        let (sin_a, cos_a) = alpha.sin_cos();
        let (sin_b, cos_b) = beta.sin_cos();
        [-1.0, cos_a * cos_b, sin_b, sin_a * cos_b]
    }

    fn pixel_to_coordinate_momentum<F: MetricField>(
        &self,
        field: &F,
        i: usize,
        j: usize,
    ) -> Vector {
        let x_cam = [0.0, self.r_cam, self.th_cam, 0.0];
        let tetrad = zamo_tetrad(field, &x_cam);
        let n = self.tetrad_frame_direction(i, j);
        let mut k = [0.0; 4];
        for a in 0..4 {
            for mu in 0..4 {
                k[mu] += n[a] * tetrad.legs[a][mu];
            }
        }
        k
    }
}

#[cfg(test)]
mod tests {
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
}
