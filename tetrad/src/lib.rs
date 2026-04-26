use grr_core::math::{Vector, field::MetricField, metric::Metric};

/// orthonormal tetrad at an event in spacetime.
/// `legs[a]` is the a-th tetrad vector e_(a)^μ in coordinate basis.
/// `legs[0]` is timelike (the observer's 4-velocity, normalized to -1).
/// `legs[1..4]` are spacelike, mutually orthogonal, normalized to +1.
#[derive(Clone, Copy, Debug)]
pub struct Tetrad {
    pub legs: [Vector; 4],
}

/// zero-angular-momentum observer (ZAMO) tetrad at event `x`.
/// (requires θ ≠ 0, π (avoid spin axis) and r outside the horizon.)
#[rustfmt::skip]
pub fn zamo_tetrad<F: MetricField>(field: &F, x: &Vector) -> Tetrad {
    let g = field.metric_at(x);

    // probe metric components via dot products on coordinate basis vectors.
    let e_t   = [1.0, 0.0, 0.0, 0.0];
    let e_r   = [0.0, 1.0, 0.0, 0.0];
    let e_th  = [0.0, 0.0, 1.0, 0.0];
    let e_phi = [0.0, 0.0, 0.0, 1.0];

    let g_tt    = g.dot(&e_t,   &e_t);
    let g_rr    = g.dot(&e_r,   &e_r);
    let g_thth  = g.dot(&e_th,  &e_th);
    let g_phph  = g.dot(&e_phi, &e_phi);
    let g_tphi  = g.dot(&e_t,   &e_phi);

    // frame-dragging angular velocity ω = -g_tφ / g_φφ. for diagonal metrics
    // (schwarzschild, minkowski) g_tφ = 0 and ω = 0, recovering static observer
    let omega = -g_tphi / g_phph;

    // lapse N² = -g_tt - 2ω·g_tφ + ω²·g_φφ
    //          = -g_tt + g_tφ²/g_φφ
    let n_sq = -g_tt + g_tphi * g_tphi / g_phph;
    assert!(n_sq > 0.0, "zamo needs N² > 0. if we hit this we may be inside ergosphere");
    let inv_n = n_sq.sqrt().recip();

    Tetrad {
        legs: [
            // timelike:  e_(0) = (1, 0,       0,       ω) / N
            // radial:    e_(1) = (0, 1/√g_rr, 0,       0)
            // polar:     e_(2) = (0, 0,       1/√g_θθ, 0)
            // azimuthal: e_(3) = (0, 0,       0,       1/√g_φφ)
            [inv_n, 0.0,                 0.0,                   omega * inv_n        ],
            [0.0,   g_rr.sqrt().recip(), 0.0,                   0.0                  ],
            [0.0,   0.0,                 g_thth.sqrt().recip(), 0.0                  ],
            [0.0,   0.0,                 0.0,                   g_phph.sqrt().recip()],
        ],
    }
}
