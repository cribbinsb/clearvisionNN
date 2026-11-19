//! An integer model file: a header, the packed weights and the
//! requantisation constants of each layer, and the tanh table. All of it is
//! 32-bit little-endian words (docs/INTEGER.md gives the layout).

use cvnn_common::{Error, Result};

use super::layers::{Conv, SkipAdd};
use super::{
    HEADER_WORDS, MAGIC, MULTIPLIER_BITS, PICTURE_OUTPUTS, SCHEMA_VERSION, STATE_BITS,
    WORD_CHANNELS,
};

/// Taps of a 3x3 kernel.
pub(super) const TAPS: usize = 9;
/// Largest trunk width, block count, halo and state size a file may claim:
/// far above any trained model, low enough that a corrupt file cannot exhaust memory.
const MAX_CHANNELS: usize = 256;
const MAX_BLOCKS: usize = 64;
const MAX_INPUT_WORDS: usize = 64;

/// A network in integer form, as exported.
#[derive(Debug, Clone)]
pub struct IntegerModel {
    /// Channels of the float input tensor the planes are taken from.
    pub(super) input_channels: usize,
    /// Leading input words holding unsigned planes; the rest hold signed ones.
    pub(super) unsigned_words: usize,
    /// For each input byte, the float input channel it holds (`None`: padding).
    pub(super) slots: Vec<Option<usize>>,
    pub(super) stem: Conv,
    pub(super) body: Vec<Conv>,
    pub(super) skip: SkipAdd,
    pub(super) head: Conv,
    pub(super) state_outputs: usize,
    /// Magnitude of the stored state for each magnitude of the tanh input.
    pub(super) tanh: Vec<u8>,
    halo: usize,
    tile: usize,
}

fn invalid<T>(why: &str) -> Result<T> {
    Err(Error::InvalidSpec(format!("integer model: {why}")))
}

/// Reads a file's words in order.
pub(super) struct Reader<'a> {
    words: &'a [i32],
    at: usize,
}

impl Reader<'_> {
    pub(super) fn take(&mut self, count: usize) -> Result<&[i32]> {
        let end = self
            .at
            .checked_add(count)
            .filter(|&end| end <= self.words.len());
        let Some(end) = end else {
            return invalid("the file is shorter than its header says");
        };
        let taken = &self.words[self.at..end];
        self.at = end;
        Ok(taken)
    }

    /// `count` values, each within `range`.
    pub(super) fn bounded(
        &mut self,
        count: usize,
        range: core::ops::Range<i32>,
    ) -> Result<Vec<i32>> {
        let values = self.take(count)?;
        if values.iter().any(|value| !range.contains(value)) {
            return invalid("a requantisation constant is out of range");
        }
        Ok(values.to_vec())
    }
}

fn dimension(word: i32, most: usize, what: &str) -> Result<usize> {
    match usize::try_from(word) {
        Ok(value) if value <= most => Ok(value),
        _ => invalid(&format!("{what} is out of range")),
    }
}

impl IntegerModel {
    /// The model in `words`, the content of a model file.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSpec`] if the file is not an integer model of this
    /// schema version, or is inconsistent or truncated.
    pub fn from_words(words: &[i32]) -> Result<Self> {
        let mut reader = Reader { words, at: 0 };
        let header = reader.take(HEADER_WORDS)?.to_vec();
        if header[0] != MAGIC {
            return invalid("not an integer model file");
        }
        if header[1] != SCHEMA_VERSION {
            return invalid(&format!(
                "schema {}, this runtime reads {SCHEMA_VERSION}",
                header[1]
            ));
        }
        let input_channels = dimension(header[2], MAX_CHANNELS, "input channels")?;
        let unsigned_words = dimension(header[3], MAX_INPUT_WORDS, "unsigned input words")?;
        let signed_words = dimension(header[4], MAX_INPUT_WORDS, "signed input words")?;
        let channels = dimension(header[5], MAX_CHANNELS, "trunk channels")?;
        let blocks = dimension(header[6], MAX_BLOCKS, "blocks")?;
        let pictures = dimension(header[7], PICTURE_OUTPUTS, "picture outputs")?;
        let state_outputs = dimension(header[8], MAX_CHANNELS, "state outputs")?;
        let halo = dimension(header[9], MAX_BLOCKS + 2, "halo")?;
        let tile = dimension(header[10], 1 << 16, "tile")?;
        let table = dimension(header[11], 1 << 16, "tanh table length")?;
        if state_outputs > 0 && header[12] != STATE_BITS {
            return invalid(&format!(
                "{}-bit state, this runtime carries {STATE_BITS}",
                header[12]
            ));
        }
        let shape_ok = channels > 0
            && channels.is_multiple_of(WORD_CHANNELS)
            && state_outputs.is_multiple_of(WORD_CHANNELS)
            && table.is_multiple_of(WORD_CHANNELS)
            && pictures == PICTURE_OUTPUTS
            && blocks > 0
            && unsigned_words > 0
            && (state_outputs == 0) == (table == 0);
        if !shape_ok {
            return invalid("inconsistent header");
        }
        let input_words = unsigned_words + signed_words;
        let slots = reader
            .take(input_words * WORD_CHANNELS)?
            .iter()
            .map(|&slot| match usize::try_from(slot) {
                Ok(channel) if channel < input_channels => Ok(Some(channel)),
                Err(_) if slot == -1 => Ok(None),
                _ => invalid("an input slot names a channel the input lacks"),
            })
            .collect::<Result<Vec<_>>>()?;
        let stem = Conv::read(&mut reader, input_words * WORD_CHANNELS, channels, false)?;
        let body = (0..blocks)
            .map(|_| Conv::read(&mut reader, channels, channels, true))
            .collect::<Result<Vec<_>>>()?;
        let skip = SkipAdd::read(&mut reader)?;
        let head = Conv::read(
            &mut reader,
            channels,
            PICTURE_OUTPUTS + state_outputs,
            false,
        )?;
        let tanh = reader
            .take(table / WORD_CHANNELS)?
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect();
        if reader.at != words.len() {
            return invalid("the file is longer than its header says");
        }
        Ok(Self {
            input_channels,
            unsigned_words,
            slots,
            stem,
            body,
            skip,
            head,
            state_outputs,
            tanh,
            halo,
            tile,
        })
    }

    /// Channels of the float input tensor the packed planes are taken from.
    #[must_use]
    pub fn input_channels(&self) -> usize {
        self.input_channels
    }

    /// Words per pixel of the packed input.
    #[must_use]
    pub fn input_words(&self) -> usize {
        self.slots.len() / WORD_CHANNELS
    }

    /// Words per pixel of the packed output: the picture outputs, then the state.
    #[must_use]
    pub fn output_words(&self) -> usize {
        super::PICTURE_WORDS + self.state_outputs / WORD_CHANNELS
    }

    /// State channels the network returns (0 for a per-frame model).
    #[must_use]
    pub fn state_outputs(&self) -> usize {
        self.state_outputs
    }

    /// The tile side the model is deployed with (0: the whole frame as one tile).
    #[must_use]
    pub fn tile(&self) -> usize {
        self.tile
    }

    /// Pixels of real neighbourhood each tile is given beyond its own.
    #[must_use]
    pub fn halo(&self) -> usize {
        self.halo
    }
}

/// The largest multiplier a file may hold.
pub(super) const MULTIPLIER_RANGE: core::ops::Range<i32> = 0..1 << MULTIPLIER_BITS;
