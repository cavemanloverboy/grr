//! this implementation is heavily influenced by scipy's implementation.
//!
//! tableau values are copy-pasted from scipy, and err5/err3 convention is kept.
//!
//! not that for some reason this dp87 method is sometimes called DOP853...
use std::array;

use tableau::*;

#[derive(Debug, Clone, Copy)]
pub struct Dp87Controller {
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

impl Default for Dp87Controller {
    fn default() -> Self {
        Self {
            atol: 1e-6,
            rtol: 1e-6,
            safety: 0.9,
            min_factor: 0.333,
            max_factor: 6.0,
            // For DOP87: order p = 8, so alpha = 0.7/p, beta = 0.4/p
            alpha: 0.7 / 8.0,
            beta: 0.4 / 8.0,
        }
    }
}

impl Dp87Controller {
    /// Scaled error norm for DOP87. Returns a scalar where ≤ 1 means accept.
    ///
    /// Unlike DP5(4), where the embedded error `y5 - y4` already has units of `y`
    /// (an O(h^5) quantity), DOP87 stores the two error estimators as weighted
    /// sums of stage derivatives:
    ///   err5[i] = Σ E5[s] * k[s][i]
    ///   err3[i] = Σ E3[s] * k[s][i]
    /// These have units of `[y]/[t]`, so a factor of `h` is needed to compare
    /// them to the tolerance scale `sc = atol + rtol * max(|y|, |y_new|)`.
    ///
    /// The norm is Hairer's blended formula (Solving ODEs I, eq IV.5.20):
    ///   err_norm = |h| * ||err5||² / sqrt((||err5||² + 0.01·||err3||²) · N)
    /// where ||·|| is the scaled 2-norm Σ (e[i]/sc[i])².
    ///
    /// In the typical regime where `err3` is small relative to `err5`, this
    /// reduces to `|h| * ||err5|| / sqrt(N)` — RMS error scaled by step size.
    /// When `err3` grows (signaling near-discontinuities or stiffness), the
    /// denominator inflates and shrinks `err_norm`, damping the controller's
    /// reaction to avoid over-aggressive step rejection in pathological regions.
    ///
    /// We keep SciPy's convention of `err5`/`err3` returned from the step
    /// *without* the `h` factor; this function applies it. The `h` does not
    /// factor out cleanly under any rebracketing, so the controller is the
    /// right place for it.
    fn err_norm<const N: usize>(
        &self,
        h: f64,
        y: &[f64; N],
        y_new: &[f64; N],
        err5: &[f64; N],
        err3: &[f64; N],
    ) -> f64 {
        let mut e5_sq = 0.0;
        let mut e3_sq = 0.0;
        for i in 0..N {
            let sc = self.atol + self.rtol * y[i].abs().max(y_new[i].abs());
            let r5 = err5[i] / sc;
            let r3 = err3[i] / sc;
            e5_sq += r5 * r5;
            e3_sq += r3 * r3;
        }
        if e5_sq == 0.0 && e3_sq == 0.0 {
            return 0.0;
        }
        h.abs() * e5_sq / ((e5_sq + 0.01 * e3_sq) * N as f64).sqrt()
    }

    /// Pure I-controller. Useful baseline / for comparison.
    pub fn i_controller(atol: f64, rtol: f64) -> Self {
        Self {
            atol,
            rtol,
            alpha: 1.0 / 8.0,
            beta: 0.0,
            ..Self::default()
        }
    }

    /// Multiplicative factor to apply to `dt`, clamped to [min_factor, max_factor].
    fn factor(&self, err_norm: f64, err_prev: f64) -> f64 {
        let raw = if err_norm == 0.0 {
            self.max_factor
        } else {
            self.safety * err_norm.powf(-self.alpha) * err_prev.powf(self.beta)
        };
        raw.clamp(self.min_factor, self.max_factor)
    }
}

pub fn dp87_integrator<const N: usize>(
    integrand: impl Fn(f64, &[f64; N]) -> [f64; N],
    mut state: [f64; N],
    mut time: f64,
    t_end: f64,
    mut dt: f64,
    ctrl: &Dp87Controller,
) -> [f64; N] {
    let mut k1 = integrand(time, &state);
    let mut err_prev: f64 = 1.0;

    while time < t_end {
        dt = dt.min(t_end - time);

        let (new_state, k7, err5, err3) = dp87_step(&integrand, time, &state, dt, k1);
        let err_norm = ctrl.err_norm(dt, &state, &new_state, &err5, &err3);

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

#[rustfmt::skip]
pub fn dp87_step<F: Fn(f64, &[f64; N]) -> [f64; N], const N: usize>(
    f: &F,
    t: f64,
    y: &[f64; N],
    h: f64,
    k1: [f64; N],
) -> ([f64; N], [f64; N], [f64; N], [f64; N])
{
    let k2  = f(t + C[1]  * h, &array::from_fn(|i| y[i] + h * (A[1][0] * k1[i])));
    let k3  = f(t + C[2]  * h, &array::from_fn(|i| y[i] + h * (A[2][0] * k1[i] + A[2][1] * k2[i])));
    let k4  = f(t + C[3]  * h, &array::from_fn(|i| y[i] + h * (A[3][0] * k1[i] + A[3][2] * k3[i])));
    let k5  = f(t + C[4]  * h, &array::from_fn(|i| y[i] + h * (A[4][0] * k1[i] + A[4][2] * k3[i] + A[4][3] * k4[i])));
    let k6  = f(t + C[5]  * h, &array::from_fn(|i| y[i] + h * (A[5][0] * k1[i] + A[5][3] * k4[i] + A[5][4] * k5[i])));
    let k7  = f(t + C[6]  * h, &array::from_fn(|i| y[i] + h * (A[6][0] * k1[i] + A[6][3] * k4[i] + A[6][4] * k5[i] + A[6][5] * k6[i])));
    let k8  = f(t + C[7]  * h, &array::from_fn(|i| y[i] + h * (A[7][0] * k1[i] + A[7][3] * k4[i] + A[7][4] * k5[i] + A[7][5] * k6[i] + A[7][6] * k7[i])));
    let k9  = f(t + C[8]  * h, &array::from_fn(|i| y[i] + h * (A[8][0] * k1[i] + A[8][3] * k4[i] + A[8][4] * k5[i] + A[8][5] * k6[i] + A[8][6] * k7[i] + A[8][7] * k8[i])));
    let k10 = f(t + C[9]  * h, &array::from_fn(|i| y[i] + h * (A[9][0] * k1[i] + A[9][3] * k4[i] + A[9][4] * k5[i] + A[9][5] * k6[i] + A[9][6] * k7[i] + A[9][7] * k8[i] + A[9][8] * k9[i])));
    let k11 = f(t + C[10] * h, &array::from_fn(|i| y[i] + h * (A[10][0] * k1[i] + A[10][3] * k4[i] + A[10][4] * k5[i] + A[10][5] * k6[i] + A[10][6] * k7[i] + A[10][7] * k8[i] + A[10][8] * k9[i] + A[10][9] * k10[i])));
    let k12 = f(t + C[11] * h, &array::from_fn(|i| y[i] + h * (A[11][0] * k1[i] + A[11][3] * k4[i] + A[11][4] * k5[i] + A[11][5] * k6[i] + A[11][6] * k7[i] + A[11][7] * k8[i] + A[11][8] * k9[i] + A[11][9] * k10[i] + A[11][10] * k11[i])));

    // 8th-order solution
    let y_new: [f64; N] = array::from_fn(|i| y[i] + h * (B[0] * k1[i] + B[5] * k6[i] + B[6] * k7[i] + B[7] * k8[i] + B[8] * k9[i] + B[9] * k10[i] + B[10] * k11[i] + B[11] * k12[i]));

    // FSAL stage: k13 = f(t+h, y_new). Used for error estimate AND as next step's k1.
    let k13 = f(t + h, &y_new);

    // Error estimators: weighted sums of stage derivatives. Both E5 and E3
    // have nonzero entries only at indices {0, 5, 6, 7, 8, 9, 10, 11};
    // index 12 (k13) is zero in both, so k13 isn't used here (only for FSAL).
    let err5: [f64; N] = array::from_fn(|i| {
        E5[0]  * k1[i]
      + E5[5]  * k6[i]
      + E5[6]  * k7[i]
      + E5[7]  * k8[i]
      + E5[8]  * k9[i]
      + E5[9]  * k10[i]
      + E5[10] * k11[i]
      + E5[11] * k12[i]
    });

    let err3: [f64; N] = array::from_fn(|i| {
        E3[0]  * k1[i]
      + E3[5]  * k6[i]
      + E3[6]  * k7[i]
      + E3[7]  * k8[i]
      + E3[8]  * k9[i]
      + E3[9]  * k10[i]
      + E3[10] * k11[i]
      + E3[11] * k12[i]
    });

    (y_new, k13, err5, err3)
}

#[rustfmt::skip]
mod tableau {
    pub const N_STAGES: usize = 12;
    pub const N_STAGES_EXTENDED: usize = 16;
    #[allow(unused)] /* incase we ever want dense output */
    pub const INTERPOLATOR_POWER: usize = 7;

    pub const C: [f64; N_STAGES_EXTENDED] = [
        0.0,
        0.526001519587677318785587544488e-01,
        0.789002279381515978178381316732e-01,
        0.118350341907227396726757197510,
        0.281649658092772603273242802490,
        0.333333333333333333333333333333,
        0.25,
        0.307692307692307692307692307692,
        0.651282051282051282051282051282,
        0.6,
        0.857142857142857142857142857142,
        1.0,
        1.0,
        0.1,
        0.2,
        0.777777777777777777777777777778
    ];

    pub const A: [[f64; N_STAGES_EXTENDED]; N_STAGES_EXTENDED] = const {
        let mut a = [[0.0; N_STAGES_EXTENDED]; N_STAGES_EXTENDED];
        a[1][0] = 5.26001519587677318785587544488e-2;

        a[2][0] = 1.97250569845378994544595329183e-2;
        a[2][1] = 5.91751709536136983633785987549e-2;

        a[3][0] = 2.95875854768068491816892993775e-2;
        a[3][2] = 8.87627564304205475450678981324e-2;

        a[4][0] = 2.41365134159266685502369798665e-1;
        a[4][2] = -8.84549479328286085344864962717e-1;
        a[4][3] = 9.24834003261792003115737966543e-1;

        a[5][0] = 3.7037037037037037037037037037e-2;
        a[5][3] = 1.70828608729473871279604482173e-1;
        a[5][4] = 1.25467687566822425016691814123e-1;

        a[6][0] = 3.7109375e-2;
        a[6][3] = 1.70252211019544039314978060272e-1;
        a[6][4] = 6.02165389804559606850219397283e-2;
        a[6][5] = -1.7578125e-2;

        a[7][0] = 3.70920001185047927108779319836e-2;
        a[7][3] = 1.70383925712239993810214054705e-1;
        a[7][4] = 1.07262030446373284651809199168e-1;
        a[7][5] = -1.53194377486244017527936158236e-2;
        a[7][6] = 8.27378916381402288758473766002e-3;

        a[8][0] = 6.24110958716075717114429577812e-1;
        a[8][3] = -3.36089262944694129406857109825;
        a[8][4] = -8.68219346841726006818189891453e-1;
        a[8][5] = 2.75920996994467083049415600797e1;
        a[8][6] = 2.01540675504778934086186788979e1;
        a[8][7] = -4.34898841810699588477366255144e1;

        a[9][0] = 4.77662536438264365890433908527e-1;
        a[9][3] = -2.48811461997166764192642586468;
        a[9][4] = -5.90290826836842996371446475743e-1;
        a[9][5] = 2.12300514481811942347288949897e1;
        a[9][6] = 1.52792336328824235832596922938e1;
        a[9][7] = -3.32882109689848629194453265587e1;
        a[9][8] = -2.03312017085086261358222928593e-2;

        a[10][0] = -9.3714243008598732571704021658e-1;
        a[10][3] = 5.18637242884406370830023853209;
        a[10][4] = 1.09143734899672957818500254654;
        a[10][5] = -8.14978701074692612513997267357;
        a[10][6] = -1.85200656599969598641566180701e1;
        a[10][7] = 2.27394870993505042818970056734e1;
        a[10][8] = 2.49360555267965238987089396762;
        a[10][9] = -3.0467644718982195003823669022;

        a[11][0] = 2.27331014751653820792359768449;
        a[11][3] = -1.05344954667372501984066689879e1;
        a[11][4] = -2.00087205822486249909675718444;
        a[11][5] = -1.79589318631187989172765950534e1;
        a[11][6] = 2.79488845294199600508499808837e1;
        a[11][7] = -2.85899827713502369474065508674;
        a[11][8] = -8.87285693353062954433549289258;
        a[11][9] = 1.23605671757943030647266201528e1;
        a[11][10] = 6.43392746015763530355970484046e-1;

        a[12][0] = 5.42937341165687622380535766363e-2;
        a[12][5] = 4.45031289275240888144113950566;
        a[12][6] = 1.89151789931450038304281599044;
        a[12][7] = -5.8012039600105847814672114227;
        a[12][8] = 3.1116436695781989440891606237e-1;
        a[12][9] = -1.52160949662516078556178806805e-1;
        a[12][10] = 2.01365400804030348374776537501e-1;
        a[12][11] = 4.47106157277725905176885569043e-2;

        a[13][0] = 5.61675022830479523392909219681e-2;
        a[13][6] = 2.53500210216624811088794765333e-1;
        a[13][7] = -2.46239037470802489917441475441e-1;
        a[13][8] = -1.24191423263816360469010140626e-1;
        a[13][9] = 1.5329179827876569731206322685e-1;
        a[13][10] = 8.20105229563468988491666602057e-3;
        a[13][11] = 7.56789766054569976138603589584e-3;
        a[13][12] = -8.298e-3;

        a[14][0] = 3.18346481635021405060768473261e-2;
        a[14][5] = 2.83009096723667755288322961402e-2;
        a[14][6] = 5.35419883074385676223797384372e-2;
        a[14][7] = -5.49237485713909884646569340306e-2;
        a[14][10] = -1.08347328697249322858509316994e-4;
        a[14][11] = 3.82571090835658412954920192323e-4;
        a[14][12] = -3.40465008687404560802977114492e-4;
        a[14][13] = 1.41312443674632500278074618366e-1;

        a[15][0] = -4.28896301583791923408573538692e-1;
        a[15][5] = -4.69762141536116384314449447206;
        a[15][6] = 7.68342119606259904184240953878;
        a[15][7] = 4.06898981839711007970213554331;
        a[15][8] = 3.56727187455281109270669543021e-1;
        a[15][12] = -1.39902416515901462129418009734e-3;
        a[15][13] = 2.9475147891527723389556272149;
        a[15][14] = -9.15095847217987001081870187138;
        a
    };


    pub const B: [f64; N_STAGES] = const {
        let mut b = [0.0; N_STAGES];
        let mut i = 0;
        while i < N_STAGES {
            b[i] = A[N_STAGES][i];
            i += 1;
        }
        b
    };


    pub const E3: [f64; N_STAGES+1] = const {
        let mut e3 = [0.0; N_STAGES+1];
        let mut i = 0;
        while i < N_STAGES {
            e3[i] = B[i];
            i += 1;
        }
        e3[0] -= 0.244094488188976377952755905512;
        e3[8] -= 0.733846688281611857341361741547;
        e3[11] -= 0.220588235294117647058823529412e-1;
        e3
    };

    pub const E5: [f64; N_STAGES+1] = const {
        let mut e5 = [0.0; N_STAGES+1];
        e5[0] = 0.1312004499419488073250102996e-1;
        e5[5] = -0.1225156446376204440720569753e+1;
        e5[6] = -0.4957589496572501915214079952;
        e5[7] = 0.1664377182454986536961530415e+1;
        e5[8] = -0.3503288487499736816886487290;
        e5[9] = 0.3341791187130174790297318841;
        e5[10] = 0.8192320648511571246570742613e-1;
        e5[11] = -0.2235530786388629525884427845e-1;
        e5
    };

    #[allow(unused)] /* incase we ever want dense output */
    pub const D: [[f64; N_STAGES_EXTENDED]; INTERPOLATOR_POWER-3] = const {
        let mut d = [[0.0; N_STAGES_EXTENDED]; INTERPOLATOR_POWER-3];

        d[0][0] = -0.84289382761090128651353491142e+1;
        d[0][5] = 0.56671495351937776962531783590;
        d[0][6] = -0.30689499459498916912797304727e+1;
        d[0][7] = 0.23846676565120698287728149680e+1;
        d[0][8] = 0.21170345824450282767155149946e+1;
        d[0][9] = -0.87139158377797299206789907490;
        d[0][10] = 0.22404374302607882758541771650e+1;
        d[0][11] = 0.63157877876946881815570249290;
        d[0][12] = -0.88990336451333310820698117400e-1;
        d[0][13] = 0.18148505520854727256656404962e+2;
        d[0][14] = -0.91946323924783554000451984436e+1;
        d[0][15] = -0.44360363875948939664310572000e+1;

        d[1][0] = 0.10427508642579134603413151009e+2;
        d[1][5] = 0.24228349177525818288430175319e+3;
        d[1][6] = 0.16520045171727028198505394887e+3;
        d[1][7] = -0.37454675472269020279518312152e+3;
        d[1][8] = -0.22113666853125306036270938578e+2;
        d[1][9] = 0.77334326684722638389603898808e+1;
        d[1][10] = -0.30674084731089398182061213626e+2;
        d[1][11] = -0.93321305264302278729567221706e+1;
        d[1][12] = 0.15697238121770843886131091075e+2;
        d[1][13] = -0.31139403219565177677282850411e+2;
        d[1][14] = -0.93529243588444783865713862664e+1;
        d[1][15] = 0.35816841486394083752465898540e+2;

        d[2][0] = 0.19985053242002433820987653617e+2;
        d[2][5] = -0.38703730874935176555105901742e+3;
        d[2][6] = -0.18917813819516756882830838328e+3;
        d[2][7] = 0.52780815920542364900561016686e+3;
        d[2][8] = -0.11573902539959630126141871134e+2;
        d[2][9] = 0.68812326946963000169666922661e+1;
        d[2][10] = -0.10006050966910838403183860980e+1;
        d[2][11] = 0.77771377980534432092869265740;
        d[2][12] = -0.27782057523535084065932004339e+1;
        d[2][13] = -0.60196695231264120758267380846e+2;
        d[2][14] = 0.84320405506677161018159903784e+2;
        d[2][15] = 0.11992291136182789328035130030e+2;

        d[3][0] = -0.25693933462703749003312586129e+2;
        d[3][5] = -0.15418974869023643374053993627e+3;
        d[3][6] = -0.23152937917604549567536039109e+3;
        d[3][7] = 0.35763911791061412378285349910e+3;
        d[3][8] = 0.93405324183624310003907691704e+2;
        d[3][9] = -0.37458323136451633156875139351e+2;
        d[3][10] = 0.10409964950896230045147246184e+3;
        d[3][11] = 0.29840293426660503123344363579e+2;
        d[3][12] = -0.43533456590011143754432175058e+2;
        d[3][13] = 0.96324553959188282948394950600e+2;
        d[3][14] = -0.39177261675615439165231486172e+2;
        d[3][15] = -0.14972683625798562581422125276e+3;
        d
    };
}
