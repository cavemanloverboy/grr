use crate::math::{Vector, christoffel::Christoffels, metric::Metric};

pub trait MetricField {
    type LocalMetric: Metric;
    type LocalChristoffels: Christoffels;
    fn metric_at(&self, x: &Vector) -> Self::LocalMetric;
    fn christoffels_at(&self, x: &Vector) -> Self::LocalChristoffels;
}
