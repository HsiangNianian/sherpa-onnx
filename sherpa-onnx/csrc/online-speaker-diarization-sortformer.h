// sherpa-onnx/csrc/online-speaker-diarization-sortformer.h
//
// Copyright (c)  2026  boardcast contributors

#ifndef SHERPA_ONNX_CSRC_ONLINE_SPEAKER_DIARIZATION_SORTFORMER_H_
#define SHERPA_ONNX_CSRC_ONLINE_SPEAKER_DIARIZATION_SORTFORMER_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"

namespace sherpa_onnx {

struct OnlineSpeakerDiarizationSortformerConfig {
  // The model is segmentation.sortformer.model; num_threads, provider and
  // debug are used as for offline diarization.
  OfflineSpeakerSegmentationModelConfig segmentation;

  // Encoder frames (80 ms each) emitted per step, and the frames after them
  // the model looks ahead at. Together they set the latency. 0 for the
  // model's own (offline) values, which give the same result as
  // OfflineSpeakerDiarization, at about 30 s of latency.
  //
  // NVIDIA's low-latency settings for Nemotron-3-Diarization, with a FIFO
  // queue of 264 frames and a speaker cache update period of 222:
  //   1.04 s: chunk_length 9, chunk_right_context 4
  //   0.64 s: chunk_length 6, chunk_right_context 2
  //   0.32 s: chunk_length 3, chunk_right_context 1
  int32_t chunk_length = 0;
  int32_t chunk_right_context = 0;

  // The FIFO queue length and the speaker cache update period, in encoder
  // frames. 0 for the model's own: its low-latency values if chunk_length is
  // set and the model has them, its offline values otherwise.
  int32_t fifo_length = 0;
  int32_t speaker_cache_update_period = 0;

  OnlineSpeakerDiarizationSortformerConfig() = default;

  bool Validate() const;

  std::string ToString() const;
};

// Speaker diarization of a stream of audio with the streaming Sortformer,
// e.g., https://huggingface.co/nvidia/Nemotron-3-Diarization
//
// Feed audio as it arrives with AcceptWaveform(). Every time a chunk and its
// right context are available, the chunk is processed and the activity
// probability of each speaker in each of its 10 ms frames becomes available
// from GetFrames(). Speakers keep their IDs for the whole stream; IDs are
// the model's speaker slots, in the order of first arrival.
//
// With chunk_length and chunk_right_context left at 0, the probabilities are
// the same as those OfflineSpeakerDiarization computes for the whole audio.
class OnlineSpeakerDiarizationSortformer {
 public:
  explicit OnlineSpeakerDiarizationSortformer(
      const OnlineSpeakerDiarizationSortformerConfig &config);

  ~OnlineSpeakerDiarizationSortformer();

  int32_t SampleRate() const;

  // Number of speaker slots, i.e., the number of columns of GetFrames()
  int32_t NumSpeakers() const;

  // Duration of one output frame, in seconds
  float FrameShift() const;

  // Seconds of audio after a frame that must arrive before the frame is
  // returned
  float Latency() const;

  // samples are in the range [-1, 1], at SampleRate()
  void AcceptWaveform(const float *samples, int32_t n);

  // No more audio: process what is left, including the last partial chunk
  void InputFinished();

  /* Return the speaker probabilities of the frames computed since the last
   * call, as a row-major matrix of shape (num_frames, NumSpeakers()).
   *
   * @param first_frame On return, the index of the first returned frame,
   *                    counting from the start of the stream.
   */
  std::vector<float> GetFrames(int32_t *first_frame);

  // Forget the stream: the next audio starts a new one, with new speakers
  void Reset();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_SPEAKER_DIARIZATION_SORTFORMER_H_
