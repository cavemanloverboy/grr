use crate::math::Vector;

pub trait Christoffels {
    fn geodesic_accel(&self, k: &Vector) -> Vector;
}

// TODO (cleanup)
// // Γ^μ_αβ with α ≤ β packed, since symmetric in lower indices.
// // 4 × 10 = 40 entries.
// pub struct Christoffels([[f64; 10]; 4]);
//
// impl Index<(usize, usize, usize)> for Christoffels {
//     type Output = f64;
//     fn index(&self, (mu, alpha, beta): (usize, usize, usize)) -> &Self::Output {
//         let (a, b) = (alpha.min(beta), alpha.max(beta));
//         // Pack (a,b) with a<=b into 0..10 via triangular indexing
//         let idx = a * (7 - a) / 2 + b;
//         &self.0[mu][idx]
//     }
// }
//
// impl Christoffels {
//     /// Compute -Γ^μ_αβ k^α k^β for the geodesic RHS.
//     #[inline(always)]
//     pub fn contract_geodesic(&self, k: &Vector) -> Vector {
//         let mut result = [0.0; 4];
//         for mu in 0..4 {
//             let mut acc = 0.0;
//             for alpha in 0..4 {
//                 for beta in 0..4 {
//                     acc += self[(mu, alpha, beta)] * k[alpha] * k[beta];
//                 }
//             }
//             result[mu] = -acc;
//         }
//         result
//     }
// }
