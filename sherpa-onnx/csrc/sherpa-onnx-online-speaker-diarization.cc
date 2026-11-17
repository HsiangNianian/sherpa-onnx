// sherpa-onnx/csrc/sherpa-onnx-online-speaker-diarization.cc
//
// Copyright (c)  2026  boardcast contributors

#include <algorithm>
#include <chrono>  // NOLINT
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-speaker-diarization-sortformer-impl.h"
#include "sherpa-onnx/csrc/online-speaker-diarization-sortformer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Streaming speaker diarization with Nemotron-3-Diarization (Sortformer).

The wave file is fed in pieces, as if it were being recorded, and each
speaker's turns are printed once the stream is over.

  ./bin/sherpa-onnx-online-speaker-diarization \
    --sortformer-model=./sherpa-onnx-nemotron-3-diarization/model.int8.onnx \
    --chunk-length=9 --chunk-right-context=4 \
    ./0-four-speakers-zh.wav

--chunk-length and --chunk-right-context are in 80 ms frames. NVIDIA's
low-latency settings are 9 and 4 (1.04 s), 6 and 2 (0.64 s), and 3 and 1
(0.32 s). Leave them at 0 for the model's offline settings, which give the
same result as sherpa-onnx-offline-speaker-diarization.

--compare-offline also runs the offline diarization on the whole file and
prints how the two differ.
)usage";

  sherpa_onnx::OnlineSpeakerDiarizationSortformerConfig config;
  float threshold = 0.5f;
  int32_t piece_ms = 100;
  bool compare_offline = false;

  sherpa_onnx::ParseOptions po(kUsageMessage);
  po.Register("sortformer-model", &config.segmentation.sortformer.model,
              "Path to the Sortformer ONNX model");
  po.Register("num-threads", &config.segmentation.num_threads,
              "Number of threads");
  po.Register("provider", &config.segmentation.provider,
              "Execution provider, e.g., cpu");
  po.Register("debug", &config.segmentation.debug, "Print debug information");
  po.Register("chunk-length", &config.chunk_length,
              "Encoder frames (80 ms) per step. 0 for the model's own");
  po.Register("chunk-right-context", &config.chunk_right_context,
              "Encoder frames of right context");
  po.Register("fifo-length", &config.fifo_length,
              "FIFO queue length. 0 for the model's own");
  po.Register("speaker-cache-update-period",
              &config.speaker_cache_update_period,
              "Speaker cache update period. 0 for the model's own");
  po.Register("threshold", &threshold,
              "A speaker is active in a frame above this probability");
  po.Register("piece-ms", &piece_ms, "Milliseconds of audio fed at a time");
  po.Register("compare-offline", &compare_offline,
              "Also run the offline diarization and compare");
  po.Read(argc, argv);

  if (!config.Validate() || po.NumArgs() != 1 || piece_ms <= 0) {
    po.PrintUsage();
    return -1;
  }

  auto load_start = std::chrono::steady_clock::now();
  sherpa_onnx::OnlineSpeakerDiarizationSortformer sd(config);
  double load_s = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - load_start)
                      .count();

  int32_t sample_rate = 0;
  bool is_ok = false;
  std::vector<float> samples = sherpa_onnx::ReadWave(po.GetArg(1), &sample_rate, &is_ok);
  if (!is_ok || sample_rate != sd.SampleRate()) {
    std::cerr << "Expect a " << sd.SampleRate() << " Hz wave file\n";
    return -1;
  }

  const int32_t num_speakers = sd.NumSpeakers();
  const int32_t piece = sample_rate * piece_ms / 1000;
  std::vector<float> probs;

  // The longest time a piece of audio waits for its probabilities, measured
  // from when the piece is fed: the model's latency plus the compute time
  double worst_step_s = 0;
  auto start = std::chrono::steady_clock::now();
  for (int32_t i = 0; i < static_cast<int32_t>(samples.size()); i += piece) {
    int32_t n = std::min(piece, static_cast<int32_t>(samples.size()) - i);
    auto t = std::chrono::steady_clock::now();
    sd.AcceptWaveform(samples.data() + i, n);
    worst_step_s = std::max(
        worst_step_s,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t)
            .count());
    int32_t first = 0;
    std::vector<float> frames = sd.GetFrames(&first);
    probs.insert(probs.end(), frames.begin(), frames.end());
  }
  sd.InputFinished();
  int32_t first = 0;
  std::vector<float> frames = sd.GetFrames(&first);
  probs.insert(probs.end(), frames.begin(), frames.end());

  double elapsed_s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - start)
                         .count();
  double audio_s = static_cast<double>(samples.size()) / sample_rate;
  int32_t num_frames = static_cast<int32_t>(probs.size()) / num_speakers;

  fprintf(stdout,
          "audio %.1f s, %d frames; load %.2f s; compute %.2f s; RTF %.4f; "
          "model latency %.2f s; slowest step %.3f s\n",
          audio_s, num_frames, load_s, elapsed_s, elapsed_s / audio_s,
          sd.Latency(), worst_step_s);

  // Each speaker's turns, as runs of active frames
  float shift = sd.FrameShift();
  for (int32_t s = 0; s != num_speakers; ++s) {
    int32_t begin = -1;
    int32_t turns = 0;
    double total = 0;
    for (int32_t f = 0; f <= num_frames; ++f) {
      bool active = f < num_frames && probs[f * num_speakers + s] > threshold;
      if (active && begin < 0) {
        begin = f;
      } else if (!active && begin >= 0) {
        ++turns;
        total += (f - begin) * shift;
        begin = -1;
      }
    }
    if (turns > 0) {
      fprintf(stdout, "speaker %d: %d turns, %.1f s\n", s, turns, total);
    }
  }

  if (compare_offline) {
    sherpa_onnx::OfflineSpeakerDiarizationConfig offline_config;
    offline_config.segmentation = config.segmentation;
    sherpa_onnx::OfflineSpeakerDiarizationSortformerImpl offline(offline_config);
    std::vector<float> reference =
        offline.ComputeSpeakerProbs(samples.data(), samples.size());

    if (reference.size() != probs.size()) {
      fprintf(stdout, "offline gives %zu values, streaming %zu\n",
              reference.size(), probs.size());
      return 1;
    }

    float max_diff = 0;
    int64_t agree = 0;
    for (size_t i = 0; i != probs.size(); ++i) {
      max_diff = std::max(max_diff, std::abs(probs[i] - reference[i]));
      agree += (probs[i] > threshold) == (reference[i] > threshold);
    }
    fprintf(stdout,
            "against offline: max probability difference %.3g; speaker "
            "decisions agree on %.2f%% of frame-speaker pairs\n",
            max_diff, 100.0 * agree / probs.size());
  }

  return 0;
}
