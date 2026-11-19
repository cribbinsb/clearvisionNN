//! A row-major grid with one cell per analysis block of a frame.

/// Side of an analysis block in luma samples. Motion, match error and
/// stillness are all reported on this grid.
pub const BLOCK_SIZE: usize = 8;

/// One value per [`BLOCK_SIZE`] block, row-major. Blocks on the right and
/// bottom edges may overhang the frame.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockGrid<T> {
    cols: usize,
    rows: usize,
    cells: Vec<T>,
}

impl<T: Clone> BlockGrid<T> {
    /// A grid covering a `width` x `height` frame, every cell set to `value`.
    #[must_use]
    pub fn for_frame(width: usize, height: usize, value: T) -> Self {
        let (cols, rows) = (width.div_ceil(BLOCK_SIZE), height.div_ceil(BLOCK_SIZE));
        Self {
            cols,
            rows,
            cells: vec![value; cols * rows],
        }
    }
}

impl<T> BlockGrid<T> {
    /// Blocks per row.
    #[must_use]
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Rows of blocks.
    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// All cells, row-major.
    #[must_use]
    pub fn cells(&self) -> &[T] {
        &self.cells
    }

    /// All cells, row-major, writable.
    pub fn cells_mut(&mut self) -> &mut [T] {
        &mut self.cells
    }

    /// The (column, row) of every cell, in the order of [`Self::cells`].
    pub fn positions(&self) -> impl Iterator<Item = (usize, usize)> + use<T> {
        let cols = self.cols;
        (0..self.rows).flat_map(move |row| (0..cols).map(move |col| (col, row)))
    }
}
