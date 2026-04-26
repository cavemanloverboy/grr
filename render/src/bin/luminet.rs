// reproduce a luminet 1979-style image of a thin schwarzschild disk.

use std::error::Error;
use std::time::Instant;

use grr_camera::Camera;
use grr_core::math::spacetimes::schwarzchild::Schwarzschild;
use grr_disk::{Disk, r_isco};
use grr_png::save_grayscale;
use grr_render::{RenderInputs, RenderedImage, r_plus, render};

fn main() -> Result<(), Box<dyn Error>> {
    let r_cam = 60f64;
    let th_cam = 80f64.to_radians();
    let mut camera = Camera::new(r_cam, th_cam);
    camera.update_image_res(3840, 2160);

    let a = 0.0;
    let r_isco = r_isco(a);
    let r_out = 20.0;
    let disk = Disk {
        r_in: r_isco.retrograde,
        r_out,
        a,
    };

    let r_plus = r_plus(a);
    println!(
        "r_plus = {:.3}; r_in {:.3}; r_out {:.3}; r_cam {:.3}",
        r_plus, r_isco.retrograde, disk.r_out, r_cam
    );

    let field = Schwarzschild;

    let inputs = RenderInputs {
        camera: &camera,
        disk: &disk,
        field: &field,
        a_star: 0.0,
    };

    let (w, h) = camera.image_dims();
    println!("rendering {}x{} schwarzschild disk...", w, h);

    let start = Instant::now();
    let image = render(&inputs);
    let elapsed = start.elapsed();
    println!(
        "  {:.2}s ({:.1} kpix/s)",
        elapsed.as_secs_f64(),
        (w * h) as f64 / elapsed.as_secs_f64() / 1000.0
    );

    let log_grayscale = pixels_to_grayscale_bytes(&image, 1.0);
    save_grayscale(
        &log_grayscale,
        image.width as u32,
        image.height as u32,
        "luminet.png",
    )?;
    println!("saved luminet.png");

    Ok(())
}

fn pixels_to_grayscale_bytes(image: &RenderedImage, exposure: f64) -> Vec<u8> {
    // log scale to compress the dynamic range of disk emission
    let logged: Vec<f64> = image
        .pixels
        .iter()
        .map(|&v| if v > 0.0 { (1.0 + v).ln() } else { 0.0 })
        .collect();
    let max = logged.iter().cloned().fold(0.0_f64, f64::max);
    let scale = exposure / max.max(1e-30);

    logged
        .iter()
        .map(|&v| {
            let mapped = (v * scale).clamp(0.0, 1.0);
            let gamma = mapped.powf(1.0 / 2.2);
            (gamma * 255.0) as u8
        })
        .collect()
}
