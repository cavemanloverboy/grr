//! Generic ODE integrators.
//!
//! Three integrators, each a free function:
//! - [`rk4_step`] / [`rk4_integrate`] — fixed-step RK4. Debugging oracle.
//! - [`dp5_step`] / [`dp5_integrate`] — adaptive Dormand-Prince 5(4). Default.
//! - [`dp87_step`] / [`dp87_integrate`] — adaptive Dormand-Prince 8(7). High Resolution.
//!
//! State is a fixed-size array [f64; N].
//! Specify a problem with a closure of type `Fn(f64, &[f64; N]) -> [f64; N]`.
//!
//! Example: for some y'(t) = t

pub mod dp54;
pub mod rk4;
