// Audio engine smoke test: init the command-queue engine, create sources,

#include <Core/Error.hpp>
// load real Ogg Vorbis data (static + streamed), play/pause/stop, then tear
// everything down. No audible verification; the point is the queue machinery,
// decode paths, AL lifecycle and teardown staying crash-free. Skips gracefully
// when no OpenAL device exists.

#include <Core/FileIo.hpp>
#include <Neo/Audio.hpp>
#include <Neo/OggProvider.hpp>

#include <chrono>
#include <cstdio>
#include <thread>

#include <glm/glm.hpp>

int main() {
    std::vector<uint8_t> ogg;
    if (!moe::ReadFileBytes(MOE_SOURCE_DIR "/test/assets/tone.ogg", ogg)) {
        std::fprintf(stderr, "[audio-smoke] FAILED: %s\n", moe::Error::Get().c_str());
        return 1;
    }

    moe::neo::Audio audio;

    if (!audio.Init()) {
        std::printf("[audio-smoke] no OpenAL device, skipping: %s\n", moe::Error::Get().c_str());
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
