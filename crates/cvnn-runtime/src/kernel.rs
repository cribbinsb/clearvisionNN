//! The filters a sender may reduce its picture with: the four the training
//! data was reduced with (Pillow's `BOX`, `BILINEAR`, `BICUBIC` and `LANCZOS`),
//! each as its weight at a distance and how far it reaches.

use std::f64::consts::PI;

/// Pillow's cubic: the Keys kernel with this sharpness.
const CUBIC_A: f64 = -0.5;
/// Lobes of the Lanczos window either side of the centre.
const LANCZOS_LOBES: f64 = 3.0;

/// A reduction filter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kernel {
    /// The mean of the samples an output sample covers.
    Area,
    /// A triangle.
    Bilinear,
    /// The Keys cubic.
    Bicubic,
    /// A sinc windowed by a wider sinc.
    Lanczos,
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

impl Kernel {
    /// The kernel called `name`, as the training configuration names them.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "area" => Some(Self::Area),
            "bilinear" => Some(Self::Bilinear),
            "bicubic" => Some(Self::Bicubic),
            "lanczos" => Some(Self::Lanczos),
            _ => None,
        }
    }

    /// How far the kernel reaches either side of its centre, at unit scale.
    #[must_use]
    pub fn support(self) -> f64 {
        match self {
            Self::Area => 0.5,
            Self::Bilinear => 1.0,
            Self::Bicubic => 2.0,
            Self::Lanczos => LANCZOS_LOBES,
        }
    }

    /// The kernel's weight at `distance` from its centre, at unit scale.
    #[must_use]
    pub fn weight(self, distance: f64) -> f64 {
        let x = distance.abs();
        match self {
            // Half-open, as Pillow has it: a sample on the boundary belongs to one side only.
            Self::Area => f64::from(u8::from(distance > -0.5 && distance <= 0.5)),
            Self::Bilinear => (1.0 - x).max(0.0),
            Self::Bicubic if x < 1.0 => ((CUBIC_A + 2.0) * x - (CUBIC_A + 3.0)) * x * x + 1.0,
            Self::Bicubic if x < 2.0 => (((x - 5.0) * x + 8.0) * x - 4.0) * CUBIC_A,
            Self::Lanczos if x < LANCZOS_LOBES => sinc(x) * sinc(x / LANCZOS_LOBES),
            Self::Bicubic | Self::Lanczos => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_one_at_its_centre_and_nothing_beyond_its_reach() {
        for kernel in [
            Kernel::Area,
            Kernel::Bilinear,
            Kernel::Bicubic,
            Kernel::Lanczos,
        ] {
            assert!((kernel.weight(0.0) - 1.0).abs() < 1e-12);
            assert!(kernel.weight(kernel.support() + 0.01).abs() < 1e-12);
        }
    }

    #[test]
    fn names_are_the_training_configuration_s() {
        assert_eq!(Kernel::from_name("lanczos"), Some(Kernel::Lanczos));
        assert_eq!(Kernel::from_name("nearest"), None);
    }
}
