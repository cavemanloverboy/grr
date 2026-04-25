use core::array;

pub fn dp54_integrator<const N: usize>(
    integrand: impl Fn(f64, &[f64; N]) -> [f64; N],
    mut state: [f64; N],
    mut time: f64,
    t_end: f64,
    mut dt: f64,
    ctrl: &Dp54Controller,
) -> [f64; N] {
    let mut k1 = integrand(time, &state);
    let mut err_prev: f64 = 1.0;

    while time < t_end {
        dt = dt.min(t_end - time);

        let (new_state, k7, err) = dp54_step(&integrand, time, &state, dt, k1);
        let err_norm = ctrl.err_norm(&state, &new_state, &err);

        if err_norm <= 1.0 {
            // Accept
            time += dt;
            state = new_state;
            k1 = k7;
            dt *= ctrl.factor(err_norm, err_prev);
            err_prev = err_norm.max(1e-4);
        } else {
            // Reject: don't advance, don't consume k7, don't update err_prev.
            // Use I-style shrink (set beta=0 effectively by passing err_prev=1).
            dt *= ctrl.factor(err_norm, 1.0);
        }
    }
    state
}

// TODO (perf): reuse arrays
#[rustfmt::skip]
pub fn dp54_step<F: Fn(f64, &[f64; N]) -> [f64; N], const N: usize>(
    f: &F,
    t: f64,
    y: &[f64; N],
    h: f64,
    k1: [f64; N],
) -> ([f64; N], [f64; N], [f64; N])
{
    let k2 = f(t + h * 1.0 / 5.0,  &array::from_fn(|i| y[i] + h * (1.0 / 5.0) * k1[i]));
    let k3 = f(t + h * 3.0 / 10.0, &array::from_fn(|i| y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i])));
    let k4 = f(t + h * 4.0 / 5.0,  &array::from_fn(|i| y[i] + h * (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i] + 32.0 / 9.0 * k3[i])));
    let k5 = f(t + h * 8.0 / 9.0,  &array::from_fn(|i| y[i] + h * (19372.0 / 6561.0 * k1[i] - 25360.0 / 2187.0 * k2[i] + 64448.0 / 6561.0 * k3[i] - 212.0 / 729.0 * k4[i])));
    let k6 = f(t + h,              &array::from_fn(|i| y[i] + h * (9017.0 / 3168.0 * k1[i] - 355.0 / 33.0 * k2[i] + 46732.0 / 5247.0 * k3[i] + 49.0 / 176.0 * k4[i] - 5103.0 / 18656.0 * k5[i])));

    // 5th order solution
    let y5 = array::from_fn(|i| y[i] + h * (35.0 / 384.0 * k1[i] + 500.0 / 1113.0 * k3[i] + 125.0 / 192.0 * k4[i] - 2187.0 / 6784.0 * k5[i] + 11.0 / 84.0 * k6[i]));

    // k7 (needed only for error estimate)
    let k7 = f(t + h, &y5);

    // 4th order solution
    let y4: [f64; N] = array::from_fn(|i| y[i] + h * (5179.0 / 57600.0 * k1[i] + 7571.0 / 16695.0 * k3[i] + 393.0 / 640.0 * k4[i] - 92097.0 / 339200.0 * k5[i] + 187.0 / 2100.0 * k6[i] + 1.0 / 40.0 * k7[i]));

    let err =  array::from_fn(|i| (y5[i] - y4[i]).abs());

    (y5, k7, err)
}

#[derive(Debug, Clone, Copy)]
pub struct Dp54Controller {
    pub atol: f64,
    pub rtol: f64,
    pub safety: f64,
    pub min_factor: f64,
    pub max_factor: f64,
    /// PI controller exponents. For pure I-control, set `beta = 0.0`.
    /// Hairer recommends alpha ≈ 0.7/p, beta ≈ 0.4/p for p = 5.
    pub alpha: f64,
    pub beta: f64,
}

impl Default for Dp54Controller {
    fn default() -> Self {
        Self {
            atol: 1e-6,
            rtol: 1e-6,
            safety: 0.9,
            min_factor: 0.2,
            max_factor: 5.0,
            alpha: 0.7 / 5.0,
            beta: 0.4 / 5.0,
        }
    }
}

impl Dp54Controller {
    /// Pure I-controller (no PI smoothing). Useful baseline / for comparison.
    pub fn i_controller(atol: f64, rtol: f64) -> Self {
        Self {
            atol,
            rtol,
            alpha: 1.0 / 5.0,
            beta: 0.0,
            ..Self::default()
        }
    }

    /// Scaled RMS error norm. ≤ 1 means accept.
    fn err_norm<const N: usize>(&self, y: &[f64; N], y_new: &[f64; N], err: &[f64; N]) -> f64 {
        let mut sumsq = 0.0;
        for i in 0..N {
            let sc = self.atol + self.rtol * y[i].abs().max(y_new[i].abs());
            let r = err[i] / sc;
            sumsq += r * r;
        }
        (sumsq / N as f64).sqrt()
    }

    /// Returns the multiplicative factor to apply to `dt`, clamped.
    fn factor(&self, err_norm: f64, err_prev: f64) -> f64 {
        let raw = if err_norm == 0.0 {
            self.max_factor
        } else {
            self.safety * err_norm.powf(-self.alpha) * err_prev.powf(self.beta)
        };
        raw.clamp(self.min_factor, self.max_factor)
    }
}
