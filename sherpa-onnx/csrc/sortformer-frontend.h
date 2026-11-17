// sherpa-onnx/csrc/sortformer-frontend.h
//
// Copyright (c)  2026  Silvio Tomatis

#ifndef SHERPA_ONNX_CSRC_SORTFORMER_FRONTEND_H_
#define SHERPA_ONNX_CSRC_SORTFORMER_FRONTEND_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kaldi-native-fbank/csrc/online-feature.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model-meta-data.h"

namespace sherpa_onnx {

// The log-mel frontend of the Sortformer diarization models. It is shared by
// the offline diarization, which sees the whole waveform, and the online
// one, which keeps only the part of the waveform it still needs.
class SortformerFrontend {
 public:
  SortformerFrontend(const OfflineSpeakerSegmentationSortformerModelMetaData &meta,
                     bool debug)
      : meta_(meta) {
    knf::FbankOptions &opts = opts_;
    opts.frame_opts.samp_freq = meta.sample_rate;
    opts.frame_opts.frame_length_ms =
        1000.0f * meta.win_length / meta.sample_rate;
    opts.frame_opts.frame_shift_ms =
        1000.0f * meta.hop_length / meta.sample_rate;
    opts.frame_opts.dither = 0;
    opts.frame_opts.remove_dc_offset = false;
    // We apply the preemphasis over the whole waveform in Compute()
    opts.frame_opts.preemph_coeff = 0;
    // A symmetric Hann window, like torch.hann_window(periodic=False)
    opts.frame_opts.window_type = "hanning";
    opts.frame_opts.snip_edges = true;
    opts.frame_opts.round_to_power_of_two = true;
    opts.mel_opts.num_bins = meta.num_mel_bins;
    opts.mel_opts.low_freq = 0;
    opts.mel_opts.high_freq = 0;  // Nyquist
    opts.mel_opts.is_librosa = true;
    opts.use_power = true;
    opts.use_log_fbank = false;

    // kaldi-native-fbank derives the FFT size from the window length.
    // Reject an incompatible frontend rather than silently changing it.
    if (opts.frame_opts.PaddedWindowSize() != meta.n_fft ||
        opts.frame_opts.WindowSize() != meta.win_length ||
        opts.frame_opts.WindowShift() != meta.hop_length ||
        meta.win_length % 2 != 0) {
      SHERPA_ONNX_LOGE("Unsupported Sortformer frontend geometry");
      SHERPA_ONNX_EXIT(-1);
    }

    if (debug) {
      SHERPA_ONNX_LOGE("%s", opts.ToString().c_str());
    }
  }

  // The first waveform sample that frame `frame` needs, counting the sample
  // before it that the preemphasis reads.
  int64_t FirstSampleOf(int32_t frame) const {
    return std::max<int64_t>(
        0, static_cast<int64_t>(frame) * meta_.hop_length - meta_.win_length / 2 - 1);
  }

  // The number of waveform samples needed before frame `frame` can be
  // computed in full.
  int64_t SamplesNeededFor(int32_t frame) const {
    return static_cast<int64_t>(frame) * meta_.hop_length - meta_.win_length / 2 +
           meta_.win_length;
  }

  /* Compute num_out frames of log-mel features starting at frame start.
   *
   * It matches NeMo's AudioToMelSpectrogramPreprocessor without
   * normalization: preemphasis over the whole waveform, then
   * torch.stft(center=True, pad_mode="constant"). A win_length window centered
   * in n_fft gives the same power spectrum as a win_length window starting
   * win_length / 2 samples before the frame center.
   *
   * `audio` holds samples [audio_begin, audio_end) of the waveform. Samples
   * after audio_end are zero, as are those before 0. It must hold every sample
   * from FirstSampleOf(start) that is known.
   *
   * Frames at and after num_frames are zero, like the padding of the last
   * encoder frame in the reference implementation.
   */
  std::vector<float> Compute(const float *audio, int64_t audio_begin,
                             int64_t audio_end, int32_t start, int32_t num_out,
                             int32_t num_frames) const {
    const int32_t hop = meta_.hop_length;
    const int32_t win = meta_.win_length;
    const int32_t num_bins = meta_.num_mel_bins;
    const float preemph = meta_.preemphasis;

    std::vector<float> ans(static_cast<int64_t>(num_out) * num_bins);

    int32_t num_valid = std::min(num_out, num_frames - start);
    if (num_valid <= 0) {
      return ans;
    }

    auto at = [&](int64_t t) { return audio[t - audio_begin]; };

    // Samples [begin, begin + len) of the preemphasized waveform, with zeros
    // outside [0, audio_end)
    int64_t begin = static_cast<int64_t>(start) * hop - win / 2;
    int64_t len = static_cast<int64_t>(num_valid - 1) * hop + win;
    std::vector<float> samples(len);
    for (int64_t i = 0; i != len; ++i) {
      int64_t t = begin + i;
      if (t < 0 || t >= audio_end) {
        continue;
      }
      samples[i] = t == 0 ? at(0) : at(t) - preemph * at(t - 1);
    }

    knf::OnlineFbank fbank(opts_);
    fbank.AcceptWaveform(meta_.sample_rate, samples.data(),
                         static_cast<int32_t>(samples.size()));
    fbank.InputFinished();

    if (fbank.NumFramesReady() < num_valid) {
      SHERPA_ONNX_LOGE("Expect %d feature frames. Got: %d", num_valid,
                       fbank.NumFramesReady());
      SHERPA_ONNX_EXIT(-1);
    }

    // torch.log(mel_spec + 2**-24)
    constexpr float kLogGuard = 5.960464477539063e-08f;
    for (int32_t i = 0; i != num_valid; ++i) {
      const float *f = fbank.GetFrame(i);
      float *dst = ans.data() + static_cast<int64_t>(i) * num_bins;
      for (int32_t j = 0; j != num_bins; ++j) {
        dst[j] = std::log(f[j] + kLogGuard);
      }
    }

    return ans;
  }

 private:
  OfflineSpeakerSegmentationSortformerModelMetaData meta_;
  knf::FbankOptions opts_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SORTFORMER_FRONTEND_H_
