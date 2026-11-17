// sherpa-onnx/csrc/online-speaker-diarization-sortformer.cc
//
// Copyright (c)  2026  boardcast contributors

#include "sherpa-onnx/csrc/online-speaker-diarization-sortformer.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <utility>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-sortformer-model.h"
#include "sherpa-onnx/csrc/sortformer-frontend.h"
#include "sherpa-onnx/csrc/sortformer-speaker-cache.h"

namespace sherpa_onnx {

bool OnlineSpeakerDiarizationSortformerConfig::Validate() const {
  if (segmentation.sortformer.model.empty()) {
    SHERPA_ONNX_LOGE("Please provide a Sortformer model");
    return false;
  }

  if (!FileExists(segmentation.sortformer.model)) {
    SHERPA_ONNX_LOGE("Sortformer model '%s' does not exist",
                     segmentation.sortformer.model.c_str());
    return false;
  }

  if (chunk_length < 0 || chunk_right_context < 0 || fifo_length < 0 ||
      speaker_cache_update_period < 0) {
    SHERPA_ONNX_LOGE(
        "chunk_length, chunk_right_context, fifo_length and "
        "speaker_cache_update_period must not be negative");
    return false;
  }

  if (chunk_length == 0 && chunk_right_context > 0) {
    SHERPA_ONNX_LOGE("chunk_right_context needs a chunk_length");
    return false;
  }

  return true;
}

std::string OnlineSpeakerDiarizationSortformerConfig::ToString() const {
  std::ostringstream os;

  os << "OnlineSpeakerDiarizationSortformerConfig(";
  os << "segmentation=" << segmentation.ToString() << ", ";
  os << "chunk_length=" << chunk_length << ", ";
  os << "chunk_right_context=" << chunk_right_context << ", ";
  os << "fifo_length=" << fifo_length << ", ";
  os << "speaker_cache_update_period=" << speaker_cache_update_period << ")";

  return os.str();
}

namespace {

// The model's geometry with the configured chunk, right context, FIFO length
// and speaker cache update period
OfflineSpeakerSegmentationSortformerModelMetaData StreamingMetaData(
    OfflineSpeakerSegmentationSortformerModelMetaData meta,
    const OnlineSpeakerDiarizationSortformerConfig &config) {
  bool low_latency = config.chunk_length > 0;
  if (low_latency) {
    meta.chunk_length = config.chunk_length;
    meta.chunk_right_context = config.chunk_right_context;
  }

  if (config.fifo_length > 0) {
    meta.cache.fifo_length = config.fifo_length;
  } else if (low_latency && meta.streaming_fifo_length > 0) {
    meta.cache.fifo_length = meta.streaming_fifo_length;
  }

  if (config.speaker_cache_update_period > 0) {
    meta.cache.speaker_cache_update_period = config.speaker_cache_update_period;
  } else if (low_latency && meta.streaming_speaker_cache_update_period > 0) {
    meta.cache.speaker_cache_update_period =
        meta.streaming_speaker_cache_update_period;
  }

  return meta;
}

}  // namespace

class OnlineSpeakerDiarizationSortformer::Impl {
 public:
  explicit Impl(const OnlineSpeakerDiarizationSortformerConfig &config)
      : config_(config),
        model_(config_.segmentation),
        meta_(StreamingMetaData(model_.GetModelMetaData(), config_)),
        frontend_(meta_, config_.segmentation.debug),
        cache_(std::make_unique<SortformerSpeakerCache>(meta_.cache)) {
    if (config_.segmentation.debug) {
      SHERPA_ONNX_LOGE("chunk_length=%d, chunk_right_context=%d, fifo_length=%d, "
                       "speaker_cache_update_period=%d, latency=%.2f s",
                       meta_.chunk_length, meta_.chunk_right_context,
                       meta_.cache.fifo_length,
                       meta_.cache.speaker_cache_update_period, Latency());
    }
  }

  int32_t SampleRate() const { return meta_.sample_rate; }

  int32_t NumSpeakers() const { return meta_.cache.num_speakers; }

  float FrameShift() const {
    return static_cast<float>(meta_.hop_length) / meta_.sample_rate;
  }

  float Latency() const {
    return static_cast<float>(meta_.chunk_length + meta_.chunk_right_context) *
           meta_.cache.subsampling_factor * meta_.hop_length /
           meta_.sample_rate;
  }

  void AcceptWaveform(const float *samples, int32_t n) {
    if (finished_) {
      SHERPA_ONNX_LOGE("AcceptWaveform() after InputFinished() is ignored");
      return;
    }
    if (!samples || n <= 0) {
      return;
    }

    buffer_.insert(buffer_.end(), samples, samples + n);
    num_samples_ += n;

    // Process every chunk whose frames and right context are complete. Such
    // a chunk is the same as the one the offline diarization processes,
    // which sees the whole waveform.
    const int32_t factor = meta_.cache.subsampling_factor;
    while (true) {
      int32_t start = next_chunk_ * meta_.chunk_length;
      int32_t end = start + meta_.chunk_length;
      int32_t stop = end + meta_.chunk_right_context;

      int32_t num_frames = static_cast<int32_t>(num_samples_ / meta_.hop_length);
      if (num_samples_ < frontend_.SamplesNeededFor(stop * factor - 1) ||
          num_frames < stop * factor) {
        break;
      }

      ProcessChunk(start, end, stop, num_frames);
    }
  }

  void InputFinished() {
    if (finished_) {
      return;
    }
    finished_ = true;

    // The rest, chunked exactly as the offline diarization does it
    const int32_t factor = meta_.cache.subsampling_factor;
    int32_t num_frames = static_cast<int32_t>(num_samples_ / meta_.hop_length);
    if (num_frames == 0) {
      return;
    }

    // The centered STFT has an extra, masked feature frame. Keep its
    // zero-padded embedding: it contributes to the output convolution.
    int32_t num_embeds = num_frames / factor + 1;
    int32_t num_chunks =
        (num_embeds + meta_.chunk_length - 1) / meta_.chunk_length;

    for (; next_chunk_ < num_chunks;) {
      int32_t start = next_chunk_ * meta_.chunk_length;
      int32_t end = std::min(start + meta_.chunk_length, num_embeds);
      int32_t stop = std::min(end + meta_.chunk_right_context, num_embeds);
      ProcessChunk(start, end, stop, num_frames);
    }
  }

  std::vector<float> GetFrames(int32_t *first_frame) {
    if (first_frame) {
      *first_frame = pending_first_frame_;
    }
    pending_first_frame_ += static_cast<int32_t>(pending_.size()) / NumSpeakers();
    return std::exchange(pending_, {});
  }

  void Reset() {
    cache_ = std::make_unique<SortformerSpeakerCache>(meta_.cache);
    buffer_.clear();
    buffer_begin_ = 0;
    num_samples_ = 0;
    next_chunk_ = 0;
    pending_.clear();
    pending_first_frame_ = 0;
    finished_ = false;
  }

 private:
  // Process the encoder frames [start, end) with the right context up to
  // stop. num_frames is the number of feature frames known so far, or of the
  // whole stream once it is finished.
  void ProcessChunk(int32_t start, int32_t end, int32_t stop,
                    int32_t num_frames) {
    const int32_t num_speakers = meta_.cache.num_speakers;
    const int32_t factor = meta_.cache.subsampling_factor;
    const int32_t hidden_size = meta_.cache.hidden_size;
    int32_t num_chunk_frames = end - start;
    int32_t num_step_frames = stop - start;

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<float> features = frontend_.Compute(
        buffer_.data(), buffer_begin_, num_samples_, start * factor,
        num_step_frames * factor, num_frames);

    std::array<int64_t, 3> features_shape{1, num_step_frames * factor,
                                          meta_.num_mel_bins};
    Ort::Value features_tensor = Ort::Value::CreateTensor(
        memory_info, features.data(), features.size(), features_shape.data(),
        features_shape.size());

    int32_t num_cached = cache_->NumCachedFrames();
    std::array<int64_t, 3> cached_shape{1, num_cached, hidden_size};
    Ort::Value cached_tensor = Ort::Value::CreateTensor<float>(
        model_.Allocator(), cached_shape.data(), cached_shape.size());
    if (num_cached > 0) {
      std::vector<float> cached = cache_->GetEmbeds();
      std::copy(cached.begin(), cached.end(),
                cached_tensor.GetTensorMutableData<float>());
    }

    auto [probs, chunk_embeds] =
        model_.Forward(std::move(features_tensor), std::move(cached_tensor),
                       std::min(stop * factor, num_frames) - start * factor);

    const float *p = probs.GetTensorData<float>();
    int32_t num_input_frames = num_cached + num_step_frames;

    // Keep the probabilities of the chunk frames, without the cached frames
    // and the right context
    int32_t offset = start * factor;
    int32_t count = std::min(num_chunk_frames * factor, num_frames - offset);
    if (count > 0) {
      const float *src =
          p + static_cast<int64_t>(num_cached) * factor * num_speakers;
      pending_.insert(pending_.end(), src,
                      src + static_cast<int64_t>(count) * num_speakers);
    }

    cache_->Update(chunk_embeds.GetTensorData<float>(), num_chunk_frames, p,
                   num_input_frames);

    ++next_chunk_;

    // Drop the samples no later chunk reads
    int64_t keep_from = frontend_.FirstSampleOf(end * factor);
    if (keep_from > buffer_begin_) {
      int64_t drop = std::min<int64_t>(keep_from - buffer_begin_,
                                       static_cast<int64_t>(buffer_.size()));
      buffer_.erase(buffer_.begin(), buffer_.begin() + drop);
      buffer_begin_ += drop;
    }
  }

 private:
  OnlineSpeakerDiarizationSortformerConfig config_;
  OfflineSpeakerSegmentationSortformerModel model_;
  OfflineSpeakerSegmentationSortformerModelMetaData meta_;
  SortformerFrontend frontend_;
  std::unique_ptr<SortformerSpeakerCache> cache_;

  // Samples [buffer_begin_, num_samples_) of the stream
  std::vector<float> buffer_;
  int64_t buffer_begin_ = 0;
  int64_t num_samples_ = 0;

  int32_t next_chunk_ = 0;

  // Speaker probabilities not yet returned by GetFrames()
  std::vector<float> pending_;
  int32_t pending_first_frame_ = 0;

  bool finished_ = false;
};

OnlineSpeakerDiarizationSortformer::OnlineSpeakerDiarizationSortformer(
    const OnlineSpeakerDiarizationSortformerConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OnlineSpeakerDiarizationSortformer::~OnlineSpeakerDiarizationSortformer() =
    default;

int32_t OnlineSpeakerDiarizationSortformer::SampleRate() const {
  return impl_->SampleRate();
}

int32_t OnlineSpeakerDiarizationSortformer::NumSpeakers() const {
  return impl_->NumSpeakers();
}

float OnlineSpeakerDiarizationSortformer::FrameShift() const {
  return impl_->FrameShift();
}

float OnlineSpeakerDiarizationSortformer::Latency() const {
  return impl_->Latency();
}

void OnlineSpeakerDiarizationSortformer::AcceptWaveform(const float *samples,
                                                        int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

void OnlineSpeakerDiarizationSortformer::InputFinished() {
  impl_->InputFinished();
}

std::vector<float> OnlineSpeakerDiarizationSortformer::GetFrames(
    int32_t *first_frame) {
  return impl_->GetFrames(first_frame);
}

void OnlineSpeakerDiarizationSortformer::Reset() { impl_->Reset(); }

}  // namespace sherpa_onnx
