//! The released 8-bit models, built into the program that uses them: each as
//! the words of its integer network (what `cvnn_runtime::integer::IntegerModel`
//! reads) and its contract (the JSON the exporters write beside it).
//!
//! These weights are licensed for non-commercial use (CC BY-NC 4.0), unlike
//! the code of the other crates: some of the video they were trained on is.

#![forbid(unsafe_code)]

const WORD_BYTES: usize = 4;

/// A rung of the size ladder that has a released model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Size {
    /// 16 channels by 6 layers.
    N,
    /// 24 channels by 10 layers.
    S,
    /// 32 channels by 15 layers.
    M,
    /// 40 channels by 15 layers.
    L,
    /// 64 channels by 17 layers.
    Xl,
}

impl Size {
    /// Every size, smallest first.
    pub const ALL: [Self; 5] = [Self::N, Self::S, Self::M, Self::L, Self::Xl];

    /// The size's name, as the contracts spell it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::N => "n",
            Self::S => "s",
            Self::M => "m",
            Self::L => "l",
            Self::Xl => "xl",
        }
    }

    /// The model file's bytes: little-endian 32-bit words.
    #[must_use]
    pub fn bytes(self) -> &'static [u8] {
        match self {
            Self::N => include_bytes!("../models/n.i8"),
            Self::S => include_bytes!("../models/s.i8"),
            Self::M => include_bytes!("../models/m.i8"),
            Self::L => include_bytes!("../models/l.i8"),
            Self::Xl => include_bytes!("../models/xl.i8"),
        }
    }

    /// The model's words, as the integer network's reader takes them.
    #[must_use]
    pub fn words(self) -> Vec<i32> {
        let (words, _) = self.bytes().as_chunks::<WORD_BYTES>();
        words.iter().map(|word| i32::from_le_bytes(*word)).collect()
    }

    /// The model's contract, as JSON: what the network expects of its input.
    #[must_use]
    pub fn contract(self) -> &'static str {
        match self {
            Self::N => include_str!("../models/n.json"),
            Self::S => include_str!("../models/s.json"),
            Self::M => include_str!("../models/m.json"),
            Self::L => include_str!("../models/l.json"),
            Self::Xl => include_str!("../models/xl.json"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cvnn_runtime::integer::IntegerModel;

    #[test]
    fn every_model_is_one_the_runtime_reads() {
        for size in Size::ALL {
            let model = IntegerModel::from_words(&size.words());
            assert!(model.is_ok(), "{}: {:?}", size.name(), model.err());
            assert!(
                size.contract()
                    .contains(&format!("\"size\": \"{}\"", size.name()))
            );
        }
    }
}
