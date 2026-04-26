pub struct Disk {
    pub r_in: f64,
    pub r_out: f64,
    pub a: f64,
}

#[repr(C)]
pub struct ISCO {
    prograde: f64,
    retrograde: f64,
}

/// for kerr with spin a, r_isco given by:
///
/// z1 = 1 + (1 - a^2)^(1/3) * [(1 + a)^(1/3) + (1 - a)^(1/3)]
/// z2 = sqrt(3*a² + z1^2)
/// r_isco = 3 + z2 +/- sqrt((3 - z1)*(3 + z1 + 2*z2))
pub fn r_isco(a: f64) -> ISCO {
    let a2 = a * a;
    let z1 =
        1.0 + (1.0 - a2).powf(1.0 / 3.0) * ((1.0 + a).powf(1.0 / 3.0) + (1.0 - a).powf(1.0 / 3.0));
    let z2 = (3.0 * a2 + z1 * z1).sqrt();

    let plus_minus_term = ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt();
    let prograde = 3.0 + z2 - plus_minus_term;
    let retrograde = 3.0 + z2 + plus_minus_term;

    ISCO {
        prograde,
        retrograde,
    }
}
