use std::f64::consts::PI;

use grr_geodesic::state::State;

#[test]
#[rustfmt::skip]
fn test_minkowski_straight_down() {
    // if we start at (r=10, θ=π/4, φ=0)
    // going straight downward (kt = sqrt(50.5), -1/sqrt(2), 1/sqrt(2), 0)
    // then we should reach (r=10, θ=3π/4, φ=0) after lambda=?
    let x = [0.0, 10.0, PI / 4.0, 0.0];
    let k = [50.5f64.sqrt(), -1.0 / 2.0f64.sqrt(), 1.0 / 2.0f64.sqrt(), 0.0];
    let _state = State::new(x, k);

    // TODO (feat): ok i need to change the geodesic integrator api bc can't really do minkowski rn
}

#[test]
#[rustfmt::skip]
fn test_minkowski_cross_r0() {
    // start at (r=10, θ=π/4, φ=0) going straight inward (kr=-1, -1, 0, 0)
    // TODO (feat): ok i need to change the geodesic integrator api bc can't really do minkowski rn
}
