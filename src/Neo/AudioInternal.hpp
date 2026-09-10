#pragma once

#include <Neo/Audio.hpp>

#include <AL/al.h>
#include <AL/alc.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace moe::neo {
    // ---- AL-side audio objects (all touched only on the audio thread) ----

    struct AudioBuffer {
        ALuint mId{0};
    };

    struct AudioSource {
        static constexpr uint32_t kMaxStreamingBuffers = 8;

        ALuint mAlId{0};
        std::unique_ptr<AudioDataProvider> mProvider;
        bool mLoop{false};
        bool mIsPlaying{false};

        // Ring of AL buffers for streaming sources (created at load, reused
        // round-robin: unqueue -> refill -> requeue). Static sources use slot 0
        // attached via AL_BUFFER.
        AudioBuffer mBuffers[kMaxStreamingBuffers];
        uint32_t mBufferCount{0}; // created AL buffers
        uint32_t mQueuedHead{0};  // ring position of the oldest queued buffer
        uint32_t mQueuedCount{0}; // currently queued on the source

        // Decode scratch (owned by the audio thread).
        std::vector<uint8_t> mScratch;

        bool IsStreaming() const {
            return mProvider != nullptr && mProvider->IsStreaming();
        }
    };

    // ---- command queue (fixed ring, mutex-guarded; no heap) ----

    struct AudioCommand {
        enum class Type : uint8_t {
            kCreateSource,
            kDestroySource,
            kLoadSource,
            kPlay,
            kPause,
            kStop,
            kSetPosition,
            kDisableAttenuation,
            kSetListenerPosition,
            kSetListenerVelocity,
            kSetListenerOrientation,
            kSetListenerGain,
        };

        Type mType{Type::kCreateSource};
        AudioSourceHandle mSource{};
        std::unique_ptr<AudioDataProvider> mProvider; // kLoadSource
        bool mLoop{false};
        glm::vec3 mVec{};
        glm::vec3 mVec2{};
        float mFloat{0.0f};
        // kCreateSource: set by the audio thread before flagging mDone
        AudioSourceHandle* mOutHandle{nullptr};
        std::atomic_flag* mDone{nullptr};
    };

    static constexpr uint32_t kCommandCapacity = 256;
}// namespace moe::neo
