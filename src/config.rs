pub const MAX_ITERATIONS: u64 = 1 << 24;
pub const INITIAL_WIDTH: usize = 0x800;
pub const INITIAL_HEIGHT: usize = 0x400;

#[derive(PartialEq, Clone, Copy)]
pub enum ViewMode {
    Brot,
    FFT,
}
