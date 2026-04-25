use core::array;

pub fn rk4_integrator<const N: usize>(
    integrand: impl Fn(f64, &[f64; N]) -> [f64; N],
    mut state: [f64; N],
    mut time: f64,
    t_end: f64,
    mut dt: f64,
) -> [f64; N] {
    while time < t_end {
        dt = dt.min(t_end - time);
        state = rk4(&integrand, time, &state, dt);
        time += dt;
    }
    state
}

/// this function does a single step of the rk4 algorithm.
#[rustfmt::skip]
pub fn rk4<F: Fn(f64, &[f64; N]) -> [f64; N], const N: usize>(
    // function that evaluates derivative
    f: &F,
    // current time
    tn: f64,
    // current function value
    yn: &[f64; N],
    // timestep
    h: f64,
) -> [f64; N]
{
    let k1 = f(tn, yn);
    let k2 = f(tn + h / 2.0, &array::from_fn(|i| yn[i] + h * k1[i] / 2.0));
    let k3 = f(tn + h / 2.0, &array::from_fn(|i| yn[i] + h * k2[i] / 2.0));
    let k4 = f(tn + h,       &array::from_fn(|i| yn[i] + h * k3[i]));

    array::from_fn(|i| yn[i] + h * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0)
}
