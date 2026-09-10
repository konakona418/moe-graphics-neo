#pragma once

#include <Neo/Audio.hpp>

#include <cstdint>
#include <memory>

namespace moe::neo {
    // Ogg Vorbis audio data provider. Decoding happens at construction on the
    // caller's thread (static mode fully decodes; streamed mode opens the
    // vorbis stream for the audio thread to pull chunks from). The caller
    // owns the ogg bytes for the provider's lifetime.
    class OggProvider : public AudioDataProvider {
    public:
        enum class Mode {
            kStatic,   // fully decoded at construction (stb_vorbis)
            kStreamed, // chunked decoding on the audio thread (vorbisfile)
        };

        OggProvider(const uint8_t* oggData, size_t size, Mode mode);
        ~OggProvider() override;

        OggProvider(const OggProvider&) = delete;
        OggProvider& operator=(const OggProvider&) = delete;

        bool IsStreaming() const override;
        uint32_t GetSampleRate() const override;
        uint8_t GetChannelCount() const override;

        bool LoadStatic(std::vector<uint8_t>& outPcm) override;
        size_t StreamNextChunk(std::vector<uint8_t>& outPcm) override;
        void SeekToStart() override;

    private:
        struct Impl;
        std::unique_ptr<Impl> mImpl;
    };
}// namespace moe::neo
