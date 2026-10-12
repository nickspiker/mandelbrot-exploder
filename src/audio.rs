use rodio::{OutputStream, Sink, Source};
use std::time::Duration;

pub struct AudioPlayer {
    _stream: OutputStream,
    sink: Sink,
}

impl AudioPlayer {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let (stream, stream_handle) = OutputStream::try_default()?;
        let sink = Sink::try_new(&stream_handle)?;
        Ok(Self {
            _stream: stream,
            sink,
        })
    }

    pub fn play_stereo_samples(&self, left: Vec<f32>, right: Vec<f32>, sample_rate: u32) {
        if left.is_empty() || right.is_empty() {
            return;
        }

        // Interleave left and right channels
        let mut stereo_samples = Vec::with_capacity(left.len() * 2);
        for i in 0..left.len().min(right.len()) {
            stereo_samples.push(left[i]);
            stereo_samples.push(right[i]);
        }

        let source = StereoBuffer {
            samples: stereo_samples,
            sample_rate,
            position: 0,
        };

        self.sink.append(source);
    }
}

struct StereoBuffer {
    samples: Vec<f32>,
    sample_rate: u32,
    position: usize,
}

impl Iterator for StereoBuffer {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.samples.len() {
            None
        } else {
            let sample = self.samples[self.position];
            self.position += 1;
            Some(sample)
        }
    }
}

impl Source for StereoBuffer {
    fn current_frame_len(&self) -> Option<usize> {
        Some(self.samples.len() - self.position)
    }

    fn channels(&self) -> u16 {
        2 // Stereo
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        Some(Duration::from_secs_f64(
            (self.samples.len() / 2) as f64 / self.sample_rate as f64,
        ))
    }
}
