use grr_core::math::Vector;

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct State(pub [f64; 8]);

impl State {
    #[inline(always)]
    pub fn new(x: Vector, k: Vector) -> State {
        State([x[0], x[1], x[2], x[3], k[0], k[1], k[2], k[3]])
    }

    #[inline(always)]
    pub fn position(&self) -> &Vector {
        // SAFETY: accessing 0..4 in len=8 array
        unsafe { &*self.0.as_ptr().cast() }
    }

    #[inline(always)]
    pub fn momentum(&self) -> &Vector {
        // SAFETY: accessing 4..7 in len=8 array
        unsafe { &*self.0.as_ptr().add(4).cast() }
    }
}
