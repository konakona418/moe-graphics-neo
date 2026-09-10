// Audio engine smoke test: init the command-queue engine, create sources,
// load real Ogg Vorbis data (static + streamed), play/pause/stop, then tear
// everything down. No audible verification; the point is the queue machinery,
// decode paths, AL lifecycle and teardown staying crash-free. Skips gracefully
// when no OpenAL device exists.

#include <Neo/Audio.hpp>
#include <Neo/OggProvider.hpp>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <thread>

#include <glm/glm.hpp>

namespace {
    std::vector<uint8_t> ReadFile(const char* path) {
        std::ifstream file(path, std::ios::binary);
        return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), {});
    }
}// namespace

int main() {
    const std::vector<uint8_t> ogg = ReadFile(MOE_SOURCE_DIR "/test/assets/tone.ogg");
    if (ogg.empty()) {
        std::fprintf(stderr, "[audio-smoke] FAILED: cannot read test/assets/tone.ogg\n");
        return 1;
    }

    moe::neo::Audio audio;

    std::string error;
    if (!audio.Init(error)) {
        std::printf("[audio-smoke] no OpenAL device, skipping: %s\n", error.c_str());
        return 0;
    }
    std::printf("[audio-smoke] engine initialized\n");

    const moe::neo::AudioSourceHandle staticSource = audio.CreateSource();
    if (!staticSource.IsValid()) {
        std::fprintf(stderr, "[audio-smoke] FAILED: create static source\n");
        return 1;
    }
    const moe::neo::AudioSourceHandle streamSource = audio.CreateSource();
    if (!streamSource.IsValid()) {
        std::fprintf(stderr, "[audio-smoke] FAILED: create stream source\n");
        return 1;
    }

    // Caller owns the ogg bytes; they must outlive the providers (i.e. until
    // the sources are destroyed / the engine is torn down).
    audio.LoadSource(staticSource,
            std::make_unique<moe::neo::OggProvider>(ogg.data(), ogg.size(),
                    moe::neo::OggProvider::Mode::kStatic),
            false);
    audio.LoadSource(streamSource,
            std::make_unique<moe::neo::OggProvider>(ogg.data(), ogg.size(),
                    moe::neo::OggProvider::Mode::kStreamed),
            true);

    audio.SetListenerPosition(glm::vec3(0.0f, 0.0f, 0.0f));
    audio.SetSourcePosition(staticSource, glm::vec3(1.0f, 0.0f, 2.0f));

    audio.Play(staticSource);
    audio.Play(streamSource);
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    audio.Pause(staticSource);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    audio.Play(staticSource); // resume

    audio.Stop(streamSource);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    audio.DestroySource(staticSource);
    audio.DestroySource(streamSource);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    audio.Destroy();
    std::printf("[audio-smoke] PASS\n");
    return 0;
}
