use std::sync::atomic::{AtomicUsize, Ordering};

use grr_camera::Camera;
use grr_core::math::field::MetricField;
use grr_disk::{Disk, emission::bolometric_intensity, fluid_4velocity, redshift_factor};
use grr_geodesic::{
    integrator::{GeodesicConfig, TerminationReason, integrate_geodesic_dp54},
    state::State,
};
use grr_integrator::dp54::Dp54Controller;
use grr_tetrad::{Tetrad, zamo_tetrad};

pub struct RenderInputs<'a, F: MetricField> {
    pub camera: &'a Camera,
    pub disk: &'a Disk,
    pub field: &'a F,
    pub a_star: f64,
}

impl<'a, F: MetricField> RenderInputs<'a, F> {
    /// build the geodesic integration config implied by these render inputs.
    pub fn geodesic_config(&self) -> GeodesicConfig {
        GeodesicConfig {
            r_plus: r_plus(self.a_star),
            r_cam: self.camera.r_cam,
            r_in: self.disk.r_in,
            r_out: self.disk.r_out,
            max_steps: 100_000,
        }
    }
}
// TODO (cleanup): move this into spacetimes probably
pub fn r_plus(a_star: f64) -> f64 {
    1.0 + (1.0 - a_star * a_star).sqrt()
}

pub struct RenderedImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f64>,
}

fn render_pixel<F: MetricField>(
    inputs: &RenderInputs<F>,
    camera_tetrad: &Tetrad,
    ctrl: &Dp54Controller,
    cfg: &GeodesicConfig,
    i: usize,
    j: usize,
) -> f64 {
    // generate initial conditions for photon at this pixel
    let (x_cam, k_cam) = inputs.camera.pixel_to_initial_state(inputs.field, i, j);
    let initial_state = State::new(x_cam, k_cam);

    // integrate geodesic
    let dl = 1e-5;
    let result = integrate_geodesic_dp54(inputs.field, initial_state, dl, &ctrl, &cfg);

    // color pixel based on outcome
    match result.termination {
        // dark
        TerminationReason::HorizonEvent => 0.0,

        // TODO: some background image
        TerminationReason::EscapeEvent => 0.0,

        // bolometric intensity + redshift for thin disk model
        TerminationReason::EquatorialCrossingEvent => {
            let r_hit = result.final_state.position()[1];
            let u_em = fluid_4velocity(inputs.field, r_hit, inputs.a_star);
            let u_obs = camera_tetrad.legs[0];
            let x_hit = result.final_state.position();
            let k_hit = result.final_state.momentum();
            let g = redshift_factor(inputs.field, &x_hit, &k_hit, &u_em, &u_obs);
            let i_em = bolometric_intensity(r_hit, inputs.a_star);
            g.powi(4) * i_em
        }

        TerminationReason::MaxStepsEvent => panic!("todo remove debug"),
    }
}

pub fn render<F: MetricField + Sync>(inputs: &RenderInputs<F>) -> RenderedImage {
    let n = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let (w, h) = inputs.camera.image_dims();
    let camera_tetrad = zamo_tetrad(inputs.field, &inputs.camera.position());
    let ctrl = Dp54Controller {
        atol: 1e-10,
        rtol: 1e-10,
        safety: 0.25,
        ..Default::default()
    };
    let cfg = inputs.geodesic_config();

    let pixel_atomic: AtomicUsize = AtomicUsize::new(0);

    let mut pixels: Vec<f64> = Vec::with_capacity(w * h);
    let pixel_ptr = pixels.as_mut_ptr() as usize;

    std::thread::scope(|s| {
        for _ in 0..n {
            s.spawn(|| {
                loop {
                    // Pick a pixel
                    let pixel = pixel_atomic.fetch_add(1, Ordering::Relaxed);
                    if pixel >= w * h {
                        break;
                    }
                    let i = pixel % w;
                    let j = pixel / w;

                    let color = render_pixel(inputs, &camera_tetrad, &ctrl, &cfg, i, j);
                    unsafe {
                        *(pixel_ptr as *mut f64).add(pixel) = color;
                    }
                }
            });
        }
    });

    unsafe {
        // SAFETY: every index in 0..w*h was written by exactly one thread.
        pixels.set_len(w * h);
    }

    RenderedImage {
        width: w,
        height: h,
        pixels,
    }
}
