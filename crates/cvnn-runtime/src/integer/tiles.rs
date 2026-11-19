//! How a frame is cut into tiles, and what each layer computes for a tile.
//!
//! A tile is given a halo of real neighbouring input. Layer by layer the
//! computed region shrinks back to the tile, so that after the halo's worth
//! of layers nothing depends on pixels the tile was not given; layers before
//! that see zeros beyond the region the layer below computed. With a halo of
//! at least the layer count less one this equals the untiled network.

/// A rectangle of pixels, `x0..x1` by `y0..y1`, in frame coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Rect {
    pub(super) x0: isize,
    pub(super) y0: isize,
    pub(super) x1: isize,
    pub(super) y1: isize,
}

impl Rect {
    pub(super) fn contains(&self, x: isize, y: isize) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }

    /// Every pixel, row by row.
    pub(super) fn pixels(self) -> impl Iterator<Item = (isize, isize)> {
        (self.y0..self.y1).flat_map(move |y| (self.x0..self.x1).map(move |x| (x, y)))
    }
}

/// The tiling of one frame size for one model.
#[derive(Debug, Clone, Copy)]
pub(super) struct Tiling {
    frame: Rect,
    /// Width and height of a tile.
    tile: (isize, isize),
    halo: isize,
    layers: isize,
}

impl Tiling {
    /// `tile` is the side of a tile; 0 makes the whole frame one tile, which needs no halo.
    pub(super) fn new(
        width: usize,
        height: usize,
        tile: usize,
        halo: usize,
        layers: usize,
    ) -> Self {
        let (width, height) = (width.cast_signed(), height.cast_signed());
        let whole = tile == 0;
        let side = tile.cast_signed();
        Self {
            frame: Rect {
                x0: 0,
                y0: 0,
                x1: width,
                y1: height,
            },
            tile: if whole { (width, height) } else { (side, side) },
            halo: if whole { 0 } else { halo.cast_signed() },
            layers: layers.cast_signed(),
        }
    }

    /// The tiles themselves, row by row.
    pub(super) fn cores(self) -> impl Iterator<Item = Rect> {
        let (width, height) = self.tile;
        let count = |frame: isize, tile: isize| {
            frame
                .cast_unsigned()
                .div_ceil(tile.cast_unsigned())
                .cast_signed()
        };
        let (columns, rows) = (count(self.frame.x1, width), count(self.frame.y1, height));
        (0..rows).flat_map(move |row| {
            (0..columns).map(move |column| {
                self.clip(Rect {
                    x0: column * width,
                    y0: row * height,
                    x1: (column + 1) * width,
                    y1: (row + 1) * height,
                })
            })
        })
    }

    /// What `layer` (0: the stem) computes for the tile `core`.
    pub(super) fn region(&self, core: Rect, layer: usize) -> Rect {
        let reach = self.halo.min(self.layers - 1 - layer.cast_signed());
        self.clip(Rect {
            x0: core.x0 - reach,
            y0: core.y0 - reach,
            x1: core.x1 + reach,
            y1: core.y1 + reach,
        })
    }

    /// The whole frame.
    pub(super) fn frame(&self) -> Rect {
        self.frame
    }

    /// Top-left corner and width of the buffer holding a tile's activations.
    pub(super) fn window(&self, core: Rect) -> ((isize, isize), usize) {
        let width = (self.tile.0 + 2 * self.halo).cast_unsigned();
        ((core.x0 - self.halo, core.y0 - self.halo), width)
    }

    /// Pixels of a tile's activation buffer.
    pub(super) fn window_pixels(&self) -> usize {
        ((self.tile.0 + 2 * self.halo) * (self.tile.1 + 2 * self.halo)).cast_unsigned()
    }

    fn clip(&self, rect: Rect) -> Rect {
        Rect {
            x0: rect.x0.max(0),
            y0: rect.y0.max(0),
            x1: rect.x1.min(self.frame.x1),
            y1: rect.y1.min(self.frame.y1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regions_shrink_to_the_tile_and_stop_at_the_frame() {
        let tiling = Tiling::new(100, 60, 32, 6, 12);
        let cores: Vec<Rect> = tiling.cores().collect();
        assert_eq!(cores.len(), 4 * 2);
        let last = cores[7];
        assert_eq!((last.x0, last.y0, last.x1, last.y1), (96, 32, 100, 60));
        let inner = cores[1];
        assert_eq!(tiling.region(inner, 11), inner);
        assert_eq!(tiling.region(inner, 10).x0, inner.x0 - 1);
        let stem = tiling.region(inner, 0);
        assert_eq!((stem.x0, stem.y0, stem.y1), (inner.x0 - 6, 0, inner.y1 + 6));
    }

    #[test]
    fn tile_zero_is_the_whole_frame_without_a_halo() {
        let tiling = Tiling::new(100, 60, 0, 6, 12);
        let cores: Vec<Rect> = tiling.cores().collect();
        assert_eq!(cores, vec![tiling.frame()]);
        assert_eq!(tiling.region(cores[0], 0), tiling.frame());
    }
}
