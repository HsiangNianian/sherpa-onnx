//! Streaming speaker diarization with a Sortformer model, e.g., Nemotron-3-Diarization.
//!
//! Feed audio as it arrives; each speaker's activity in every 10 ms frame comes back once the
//! chunk holding the frame and its right context have been heard.
//!
//! ```no_run
//! use sherpa_onnx::{OnlineSpeakerDiarization, OnlineSpeakerDiarizationConfig};
//!
//! let config = OnlineSpeakerDiarizationConfig {
//!     model: Some("./sherpa-onnx-nemotron-3-diarization/model.int8.onnx".into()),
//!     // NVIDIA's 1.04 s setting
//!     chunk_length: 9,
//!     chunk_right_context: 4,
//!     ..Default::default()
//! };
//! let mut sd = OnlineSpeakerDiarization::create(&config).expect("create");
//! let samples = vec![0.0f32; 16000];
//! sd.accept_waveform(&samples);
//! let frames = sd.frames();
//! for (i, row) in frames.probs.chunks(frames.num_speakers).enumerate() {
//!     let _active: Vec<usize> = (0..frames.num_speakers).filter(|&s| row[s] > 0.5).collect();
//!     let _at = (frames.first_frame + i) as f32 * sd.frame_shift();
//! }
//! ```

use crate::utils::to_c_ptr;
use sherpa_onnx_sys as sys;
use std::ffi::CString;

/// Configuration for [`OnlineSpeakerDiarization`].
#[derive(Clone, Debug)]
pub struct OnlineSpeakerDiarizationConfig {
    /// Sortformer model path.
    pub model: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    pub provider: Option<String>,
    /// Encoder frames (80 ms) per step; 0 for the model's offline setting, about 30 s of latency.
    /// NVIDIA's low-latency settings: 9 and 4 (1.04 s), 6 and 2 (0.64 s), 3 and 1 (0.32 s).
    pub chunk_length: i32,
    /// Encoder frames of right context the model looks ahead at.
    pub chunk_right_context: i32,
    /// FIFO queue length; 0 for the model's own.
    pub fifo_length: i32,
    /// Speaker cache update period; 0 for the model's own.
    pub speaker_cache_update_period: i32,
}

impl Default for OnlineSpeakerDiarizationConfig {
    fn default() -> Self {
        Self {
            model: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".into()),
            chunk_length: 0,
            chunk_right_context: 0,
            fifo_length: 0,
            speaker_cache_update_period: 0,
        }
    }
}

impl OnlineSpeakerDiarizationConfig {
    fn to_sys(&self, cstrings: &mut Vec<CString>) -> sys::OnlineSpeakerDiarizationConfig {
        sys::OnlineSpeakerDiarizationConfig {
            model: to_c_ptr(&self.model, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
            chunk_length: self.chunk_length,
            chunk_right_context: self.chunk_right_context,
            fifo_length: self.fifo_length,
            speaker_cache_update_period: self.speaker_cache_update_period,
        }
    }
}

/// Speaker probabilities of the frames computed since the last call to
/// [`OnlineSpeakerDiarization::frames`].
#[derive(Clone, Debug, Default)]
pub struct OnlineSpeakerDiarizationFrames {
    /// Index of the first frame, counting from the start of the stream.
    pub first_frame: usize,
    /// Number of speaker slots: the length of each row of `probs`.
    pub num_speakers: usize,
    /// Row-major (frames, num_speakers) activity probabilities.
    pub probs: Vec<f32>,
}

impl OnlineSpeakerDiarizationFrames {
    /// Number of frames.
    pub fn len(&self) -> usize {
        if self.num_speakers == 0 { 0 } else { self.probs.len() / self.num_speakers }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Streaming speaker diarizer.
pub struct OnlineSpeakerDiarization {
    ptr: *const sys::OnlineSpeakerDiarization,
}

// SAFETY: one object is used from one thread at a time; the methods that change it take &mut self.
unsafe impl Send for OnlineSpeakerDiarization {}

impl OnlineSpeakerDiarization {
    /// Create a diarizer from `config`, or None if the model cannot be loaded.
    pub fn create(config: &OnlineSpeakerDiarizationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOnlineSpeakerDiarization(&sys_config) };
        if ptr.is_null() { None } else { Some(Self { ptr }) }
    }

    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationGetSampleRate(self.ptr) }
    }

    pub fn num_speakers(&self) -> i32 {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationGetNumSpeakers(self.ptr) }
    }

    /// Seconds per output frame.
    pub fn frame_shift(&self) -> f32 {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationGetFrameShift(self.ptr) }
    }

    /// Seconds of audio after a frame that must arrive before the frame comes back.
    pub fn latency(&self) -> f32 {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationGetLatency(self.ptr) }
    }

    /// Feed mono samples in [-1, 1] at [`Self::sample_rate`].
    pub fn accept_waveform(&mut self, samples: &[f32]) {
        unsafe {
            sys::SherpaOnnxOnlineSpeakerDiarizationAcceptWaveform(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
            )
        }
    }

    /// No more audio: process the rest of the stream.
    pub fn input_finished(&mut self) {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationInputFinished(self.ptr) }
    }

    /// Forget the stream; the next audio starts a new one, with new speakers.
    pub fn reset(&mut self) {
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationReset(self.ptr) }
    }

    /// Take the frames computed since the last call.
    pub fn frames(&mut self) -> OnlineSpeakerDiarizationFrames {
        let p = unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationGetFrames(self.ptr) };
        if p.is_null() {
            return OnlineSpeakerDiarizationFrames::default();
        }
        let frames = unsafe { &*p };
        let n = (frames.num_frames.max(0) as usize) * (frames.num_speakers.max(0) as usize);
        let probs = if n == 0 || frames.probs.is_null() {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(frames.probs, n) }.to_vec()
        };
        let ans = OnlineSpeakerDiarizationFrames {
            first_frame: frames.first_frame.max(0) as usize,
            num_speakers: frames.num_speakers.max(0) as usize,
            probs,
        };
        unsafe { sys::SherpaOnnxOnlineSpeakerDiarizationDestroyFrames(p) };
        ans
    }
}

impl Drop for OnlineSpeakerDiarization {
    fn drop(&mut self) {
        unsafe { sys::SherpaOnnxDestroyOnlineSpeakerDiarization(self.ptr) }
    }
}
