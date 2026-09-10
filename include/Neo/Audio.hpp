#pragma once

#include <Neo/Cache.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace moe::neo {
    class AudioSource; // internal (AL-backed), owned by the Audio engine

    // Stable handle to an audio source (slot index + generation, see Cache).
    using AudioSourceHandle = Handle<AudioSource>;

    // Decoded audio data feed. Implementations are constructed by the caller
    // (decoding is allowed to block there), then handed to Audio::LoadSource,
    // which transfers ownership to the audio thread.
    class AudioDataProvider {
    public:
        virtual ~AudioDataProvider() = default;

        // Streaming sources feed the engine in chunks; static ones are fully
        // decoded up front.
        virtual bool IsStreaming() const = 0;

        virtual uint32_t GetSampleRate() const = 0;
        virtual uint8_t GetChannelCount() const = 0;

        // Static: fills the whole decoded PCM (16-bit signed interleaved).
        virtual bool LoadStatic(std::vector<uint8_t>& outPcm) = 0;

        // Streaming: fills the next chunk (16-bit signed interleaved); returns
        // the number of bytes written, 0 at EOF.
        virtual size_t StreamNextChunk(std::vector<uint8_t>& outPcm) = 0;

        virtual void SeekToStart() = 0;
    };

    // Command-queue audio engine on a dedicated thread (one OpenAL context,
    // current only on that thread). All AL work happens there; every public
    // method is thread-safe and submits a command. CreateSource is the only
    // blocking call (it must return a handle).
    class Audio {
    public:
        Audio();
        ~Audio(); // leak trap: aborts if initialized but not Destroyed

        Audio(const Audio&) = delete;
        Audio& operator=(const Audio&) = delete;

        // Opens the OpenAL device on the audio thread and starts its loop.
        // Idempotent; fails cleanly when no device is available.
        bool Init(std::string& error);

        // Stops the audio thread and tears down OpenAL. Idempotent.
        void Destroy();

        // ---- sources ----
        AudioSourceHandle CreateSource(); // blocking (returns once created)
        void DestroySource(AudioSourceHandle source);

        // Loads audio data into the source (replaces the previous provider).
        // Ownership of the provider moves to the audio thread.
        void LoadSource(AudioSourceHandle source,
                std::unique_ptr<AudioDataProvider> provider, bool loop);

        void Play(AudioSourceHandle source);
        void Pause(AudioSourceHandle source);
        void Stop(AudioSourceHandle source);

        void SetSourcePosition(AudioSourceHandle source, const glm::vec3& position);
        void DisableSourceAttenuation(AudioSourceHandle source);

        // ---- listener ----
        void SetListenerPosition(const glm::vec3& position);
        void SetListenerVelocity(const glm::vec3& velocity);
        void SetListenerOrientation(const glm::vec3& forward, const glm::vec3& up);
        void SetListenerGain(float gain);

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
