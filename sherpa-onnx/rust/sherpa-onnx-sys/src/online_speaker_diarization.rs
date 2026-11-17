#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_float};

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OnlineSpeakerDiarizationConfig {
    pub model: *const c_char,
    pub num_threads: i32,
    pub debug: i32,
    pub provider: *const c_char,
    pub chunk_length: i32,
    pub chunk_right_context: i32,
    pub fifo_length: i32,
    pub speaker_cache_update_period: i32,
}

#[repr(C)]
pub struct OnlineSpeakerDiarization {
    _private: [u8; 0],
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OnlineSpeakerDiarizationFrames {
    pub first_frame: i32,
    pub num_frames: i32,
    pub num_speakers: i32,
    pub probs: *const c_float,
}

extern "C" {
    pub fn SherpaOnnxCreateOnlineSpeakerDiarization(
        config: *const OnlineSpeakerDiarizationConfig,
    ) -> *const OnlineSpeakerDiarization;

    pub fn SherpaOnnxDestroyOnlineSpeakerDiarization(sd: *const OnlineSpeakerDiarization);

    pub fn SherpaOnnxOnlineSpeakerDiarizationGetSampleRate(
        sd: *const OnlineSpeakerDiarization,
    ) -> i32;

    pub fn SherpaOnnxOnlineSpeakerDiarizationGetNumSpeakers(
        sd: *const OnlineSpeakerDiarization,
    ) -> i32;

    pub fn SherpaOnnxOnlineSpeakerDiarizationGetFrameShift(
        sd: *const OnlineSpeakerDiarization,
    ) -> c_float;

    pub fn SherpaOnnxOnlineSpeakerDiarizationGetLatency(
        sd: *const OnlineSpeakerDiarization,
    ) -> c_float;

    pub fn SherpaOnnxOnlineSpeakerDiarizationAcceptWaveform(
        sd: *const OnlineSpeakerDiarization,
        samples: *const c_float,
        n: i32,
    );

    pub fn SherpaOnnxOnlineSpeakerDiarizationInputFinished(sd: *const OnlineSpeakerDiarization);

    pub fn SherpaOnnxOnlineSpeakerDiarizationReset(sd: *const OnlineSpeakerDiarization);

    pub fn SherpaOnnxOnlineSpeakerDiarizationGetFrames(
        sd: *const OnlineSpeakerDiarization,
    ) -> *const OnlineSpeakerDiarizationFrames;

    pub fn SherpaOnnxOnlineSpeakerDiarizationDestroyFrames(
        frames: *const OnlineSpeakerDiarizationFrames,
    );
}
