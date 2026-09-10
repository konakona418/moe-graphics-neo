#include "AudioInternal.hpp"

#include <Core/Logger.hpp>

#include <thread>

namespace moe::neo {
    namespace {
        constexpr char kOpenAlDeviceName[] = "OpenAL Soft";
        constexpr auto kTickDuration = std::chrono::milliseconds(5);

        const char* GetAlErrorString(ALenum err) {
            switch (err) {
                case AL_NO_ERROR: return "no error";
                case AL_INVALID_NAME: return "invalid name";
                case AL_INVALID_ENUM: return "invalid enum";
                case AL_INVALID_VALUE: return "invalid value";
                case AL_INVALID_OPERATION: return "invalid operation";
                case AL_OUT_OF_MEMORY: return "out of memory";
                default: return "unknown";
            }
        }

        bool CheckAlError(const char* what) {
            const ALenum err = alGetError();
            if (err != AL_NO_ERROR) {
                moe::Logger::error("[neo] OpenAL {}: {}", what, GetAlErrorString(err));
                return false;
            }
            return true;
        }

        // 1 = mono, 2 = stereo (16-bit). Anything else is unsupported.
        bool FormatFromChannels(uint8_t channels, ALenum* outFormat) {
            switch (channels) {
                case 1: *outFormat = AL_FORMAT_MONO16; return true;
                case 2: *outFormat = AL_FORMAT_STEREO16; return true;
                default: return false;
            }
        }

        // Unqueues the oldest queued buffer, decodes the next chunk into it and
        // requeues. Returns false when the stream ended (buffer retired).
        bool RefillStreamBuffer(AudioSource& source, ALenum format, uint32_t sampleRate) {
            const ALuint bufferId = source.mBuffers[source.mQueuedHead].mId;
            ALuint unqueued = bufferId;
            alSourceUnqueueBuffers(source.mAlId, 1, &unqueued);
            source.mQueuedHead = (source.mQueuedHead + 1) % source.mBufferCount;
            source.mQueuedCount--;

            size_t bytes = source.mProvider->StreamNextChunk(source.mScratch);
            if (bytes == 0 && source.mLoop) {
                source.mProvider->SeekToStart();
                bytes = source.mProvider->StreamNextChunk(source.mScratch);
            }
            if (bytes == 0) {
                return false; // EOF
            }

            alBufferData(bufferId, format, source.mScratch.data(),
                    static_cast<ALsizei>(bytes), static_cast<ALsizei>(sampleRate));
            alSourceQueueBuffers(source.mAlId, 1, &bufferId);
            source.mQueuedCount++;
            return true;
        }

        void ReleaseSourceAl(AudioSource& source) {
            if (source.mAlId != 0) {
                alSourceStop(source.mAlId);
                if (source.mQueuedCount > 0) {
                    ALuint ids[AudioSource::kMaxStreamingBuffers];
                    for (uint32_t i = 0; i < source.mQueuedCount; ++i) {
                        ids[i] = source.mBuffers[(source.mQueuedHead + i) % source.mBufferCount].mId;
                    }
                    alSourceUnqueueBuffers(source.mAlId, source.mQueuedCount, ids);
                    source.mQueuedCount = 0;
                }
                alDeleteSources(1, &source.mAlId);
                source.mAlId = 0;
            }
            if (source.mBufferCount > 0) {
                for (uint32_t i = 0; i < source.mBufferCount; ++i) {
                    alDeleteBuffers(1, &source.mBuffers[i].mId);
                    source.mBuffers[i].mId = 0;
                }
                source.mBufferCount = 0;
            }
            source.mQueuedHead = 0;
        }

        void LoadStaticData(AudioSource& source) {
            if (!source.mProvider->LoadStatic(source.mScratch) || source.mScratch.empty()) {
                moe::Logger::error("[neo] audio: failed to decode static data");
                return;
            }
            ALenum format = 0;
            if (!FormatFromChannels(source.mProvider->GetChannelCount(), &format)) {
                moe::Logger::error("[neo] audio: unsupported channel count {}",
                        source.mProvider->GetChannelCount());
                return;
            }

            alGenBuffers(1, &source.mBuffers[0].mId);
            alBufferData(source.mBuffers[0].mId, format, source.mScratch.data(),
                    static_cast<ALsizei>(source.mScratch.size()),
                    static_cast<ALsizei>(source.mProvider->GetSampleRate()));
            if (CheckAlError("buffer data")) {
                source.mBufferCount = 1;
                alSourcei(source.mAlId, AL_BUFFER, static_cast<ALint>(source.mBuffers[0].mId));
            }
        }

        void LoadStreamingBuffers(AudioSource& source) {
            ALenum format = 0;
            if (!FormatFromChannels(source.mProvider->GetChannelCount(), &format)) {
                moe::Logger::error("[neo] audio: unsupported channel count {}",
                        source.mProvider->GetChannelCount());
                return;
            }
            const ALsizei sampleRate = static_cast<ALsizei>(source.mProvider->GetSampleRate());

            // Prime the ring with as many chunks as the stream has.
            for (uint32_t i = 0; i < AudioSource::kMaxStreamingBuffers; ++i) {
                const size_t bytes = source.mProvider->StreamNextChunk(source.mScratch);
                if (bytes == 0) {
                    break;
                }
                alGenBuffers(1, &source.mBuffers[i].mId);
                alBufferData(source.mBuffers[i].mId, format, source.mScratch.data(),
                        static_cast<ALsizei>(bytes), sampleRate);
                alSourceQueueBuffers(source.mAlId, 1, &source.mBuffers[i].mId);
                source.mBufferCount = i + 1;
                source.mQueuedCount = i + 1;
            }
            if (source.mBufferCount == 0) {
                moe::Logger::error("[neo] audio: stream contains no data");
            }
        }
    }// namespace

    struct Audio::Impl {
        // ---- OpenAL (owned by the audio thread during its life) ----
        ALCdevice* mDevice{nullptr};
        ALCcontext* mContext{nullptr};

        // ---- command ring (mutex-guarded, fixed capacity, no heap) ----
        AudioCommand mCommands[kCommandCapacity];
        uint32_t mWriteIndex{0};
        uint32_t mReadIndex{0};
        std::mutex mQueueMutex;

        // ---- sources ----
        Cache<AudioSource> mSources;
        std::vector<AudioSourceHandle> mAliveSources;

        // ---- thread ----
        std::thread mAudioThread;
        std::atomic<bool> mRunning{false};
        bool mInitialized{false};

        bool PushCommand(AudioCommand& cmd) {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            const uint32_t next = (mWriteIndex + 1) % kCommandCapacity;
            if (next == mReadIndex) {
                moe::Logger::error("[neo] audio: command queue full, dropping command");
                return false;
            }
            mCommands[mWriteIndex] = std::move(cmd);
            mWriteIndex = next;
            return true;
        }

        void ExecuteCommand(AudioCommand& cmd) {
            AudioSource* source = mSources.Get(cmd.mSource);
            switch (cmd.mType) {
                case AudioCommand::Type::kCreateSource: {
                    AudioSource newSource;
                    alGenSources(1, &newSource.mAlId);
                    if (newSource.mAlId != 0) {
                        const AudioSourceHandle handle = mSources.Add(std::move(newSource));
                        mAliveSources.push_back(handle);
                        *cmd.mOutHandle = handle;
                    } else {
                        moe::Logger::error("[neo] audio: alGenSources failed");
                    }
                    if (cmd.mDone != nullptr) {
                        cmd.mDone->test_and_set();
                    }
                    return;
                }
                case AudioCommand::Type::kDestroySource: {
                    if (source == nullptr) {
                        return;
                    }
                    ReleaseSourceAl(*source);
                    source->mProvider.reset();
                    mSources.Remove(cmd.mSource);
                    for (size_t i = 0; i < mAliveSources.size(); ++i) {
                        if (mAliveSources[i].mIndex == cmd.mSource.mIndex
                                && mAliveSources[i].mGeneration == cmd.mSource.mGeneration) {
                            mAliveSources.erase(mAliveSources.begin() + static_cast<ptrdiff_t>(i));
                            break;
                        }
                    }
                    return;
                }
                case AudioCommand::Type::kLoadSource: {
                    if (source == nullptr) {
                        return;
                    }
                    ReleaseSourceAl(*source); // unload the previous data
                    source->mProvider = std::move(cmd.mProvider);
                    source->mLoop = cmd.mLoop;
                    source->mIsPlaying = false;
                    if (source->IsStreaming()) {
                        LoadStreamingBuffers(*source);
                    } else {
                        LoadStaticData(*source);
                    }
                    return;
                }
                case AudioCommand::Type::kPlay: {
                    if (source == nullptr) {
                        return;
                    }
                    source->mIsPlaying = true;
                    alSourcePlay(source->mAlId);
                    return;
                }
                case AudioCommand::Type::kPause: {
                    if (source == nullptr) {
                        return;
                    }
                    source->mIsPlaying = false;
                    alSourcePause(source->mAlId);
                    return;
                }
                case AudioCommand::Type::kStop: {
                    if (source == nullptr) {
                        return;
                    }
                    source->mIsPlaying = false;
                    alSourceStop(source->mAlId);
                    return;
                }
                case AudioCommand::Type::kSetPosition: {
                    if (source == nullptr) {
                        return;
                    }
                    alSource3f(source->mAlId, AL_POSITION, cmd.mVec.x, cmd.mVec.y, cmd.mVec.z);
                    return;
                }
                case AudioCommand::Type::kDisableAttenuation: {
                    if (source == nullptr) {
                        return;
                    }
                    alSourcef(source->mAlId, AL_ROLLOFF_FACTOR, 0.0f);
                    return;
                }
                case AudioCommand::Type::kSetListenerPosition:
                    alListener3f(AL_POSITION, cmd.mVec.x, cmd.mVec.y, cmd.mVec.z);
                    return;
                case AudioCommand::Type::kSetListenerVelocity:
                    alListener3f(AL_VELOCITY, cmd.mVec.x, cmd.mVec.y, cmd.mVec.z);
                    return;
                case AudioCommand::Type::kSetListenerOrientation: {
                    // at (forward), up
                    const float orientation[6] = {
                            cmd.mVec.x, cmd.mVec.y, cmd.mVec.z,
                            cmd.mVec2.x, cmd.mVec2.y, cmd.mVec2.z,
                    };
                    alListenerfv(AL_ORIENTATION, orientation);
                    return;
                }
                case AudioCommand::Type::kSetListenerGain:
                    alListenerf(AL_GAIN, cmd.mFloat);
                    return;
            }
        }

        void HandleCommands() {
            std::lock_guard<std::mutex> lock(mQueueMutex);
            while (mReadIndex != mWriteIndex) {
                AudioCommand& cmd = mCommands[mReadIndex];
                ExecuteCommand(cmd);
                cmd.mProvider.reset(); // release payloads as we go
                mReadIndex = (mReadIndex + 1) % kCommandCapacity;
            }
        }

        void UpdateSources() {
            for (const AudioSourceHandle handle : mAliveSources) {
                AudioSource* source = mSources.Get(handle);
                if (source == nullptr || !source->IsStreaming()) {
                    continue;
                }
                ALenum format = 0;
                if (!FormatFromChannels(source->mProvider->GetChannelCount(), &format)) {
                    continue;
                }
                const uint32_t sampleRate = source->mProvider->GetSampleRate();

                ALint processed = 0;
                alGetSourcei(source->mAlId, AL_BUFFERS_PROCESSED, &processed);
                while (processed-- > 0) {
                    RefillStreamBuffer(*source, format, sampleRate);
                }

                ALint queued = 0;
                ALint state = 0;
                alGetSourcei(source->mAlId, AL_BUFFERS_QUEUED, &queued);
                alGetSourcei(source->mAlId, AL_SOURCE_STATE, &state);
                if (state != AL_PLAYING && queued > 0 && source->mIsPlaying) {
                    alSourcePlay(source->mAlId);
                }
            }
        }

        void MainAudioLoop() {
            auto lastTick = std::chrono::steady_clock::now();
            while (mRunning.load()) {
                HandleCommands();
                UpdateSources();

                const auto now = std::chrono::steady_clock::now();
                const auto elapsed = now - lastTick;
                if (elapsed < kTickDuration) {
                    std::this_thread::sleep_for(kTickDuration - elapsed);
                } else {
                    moe::Logger::warn("[neo] audio: main loop is running behind");
                }
                lastTick = std::chrono::steady_clock::now();
            }
            HandleCommands(); // final drain before exit
        }
    };

    Audio::Audio() = default;

    Audio::~Audio() {
        if (mImpl != nullptr) {
            std::fprintf(stderr, "[neo] Audio leaked: Destroy() not called\n");
            std::abort();
        }
    }

    bool Audio::Init(std::string& error) {
        if (mImpl != nullptr) {
            return true; // already initialized
        }
        mImpl = std::make_unique<Impl>();

        std::atomic_flag initDone = ATOMIC_FLAG_INIT;
        bool initOk = false;
        std::string initError;

        mImpl->mRunning = true;
        mImpl->mAudioThread = std::thread([&]() {
            moe::Logger::setThreadName("Audio");

            mImpl->mDevice = alcOpenDevice(kOpenAlDeviceName);
            if (mImpl->mDevice == nullptr) {
                initError = "no OpenAL device available";
                initDone.test_and_set();
                return;
            }
            moe::Logger::info("[neo] audio: opened device '{}'",
                    alcGetString(mImpl->mDevice, ALC_DEVICE_SPECIFIER));

            mImpl->mContext = alcCreateContext(mImpl->mDevice, nullptr);
            if (mImpl->mContext == nullptr || !alcMakeContextCurrent(mImpl->mContext)) {
                initError = "failed to create/make current the OpenAL context";
                initDone.test_and_set();
                return;
            }

            // listener defaults (identity orientation, no attenuation bias)
            alListener3f(AL_POSITION, 0.0f, 0.0f, 0.0f);
            alListener3f(AL_VELOCITY, 0.0f, 0.0f, 0.0f);
            const float orientation[6] = {0.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f};
            alListenerfv(AL_ORIENTATION, orientation);
            alListenerf(AL_GAIN, 1.0f);

            if (!CheckAlError("listener defaults")) {
                initError = "OpenAL error during listener setup";
                initDone.test_and_set();
                return;
            }

            initOk = true;
            initDone.test_and_set();

            mImpl->MainAudioLoop();
            alcMakeContextCurrent(nullptr);
        });

        while (!initDone.test()) {
            std::this_thread::yield();
        }

        if (!initOk) {
            mImpl->mRunning = false;
            if (mImpl->mAudioThread.joinable()) {
                mImpl->mAudioThread.join();
            }
            error = std::move(initError);
            mImpl.reset();
            return false;
        }

        mImpl->mInitialized = true;
        return true;
    }

    void Audio::Destroy() {
        if (mImpl == nullptr) {
            return;
        }
        mImpl->mRunning = false;
        if (mImpl->mAudioThread.joinable()) {
            mImpl->mAudioThread.join();
        }
        if (mImpl->mContext != nullptr) {
            alcMakeContextCurrent(nullptr);
            alcDestroyContext(mImpl->mContext);
            mImpl->mContext = nullptr;
        }
        if (mImpl->mDevice != nullptr) {
            alcCloseDevice(mImpl->mDevice);
            mImpl->mDevice = nullptr;
        }
        mImpl->mInitialized = false;
        mImpl.reset();
    }

    AudioSourceHandle Audio::CreateSource() {
        if (mImpl == nullptr) {
            return {};
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kCreateSource;
        std::atomic_flag done = ATOMIC_FLAG_INIT;
        AudioSourceHandle handle{};
        cmd.mOutHandle = &handle;
        cmd.mDone = &done;
        if (!mImpl->PushCommand(cmd)) {
            return {};
        }
        while (!done.test()) {
            std::this_thread::yield();
        }
        return handle;
    }

    void Audio::DestroySource(AudioSourceHandle source) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kDestroySource;
        cmd.mSource = source;
        mImpl->PushCommand(cmd);
    }

    void Audio::LoadSource(AudioSourceHandle source,
            std::unique_ptr<AudioDataProvider> provider, bool loop) {
        if (mImpl == nullptr || !source.IsValid() || provider == nullptr) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kLoadSource;
        cmd.mSource = source;
        cmd.mProvider = std::move(provider);
        cmd.mLoop = loop;
        mImpl->PushCommand(cmd);
    }

    void Audio::Play(AudioSourceHandle source) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kPlay;
        cmd.mSource = source;
        mImpl->PushCommand(cmd);
    }

    void Audio::Pause(AudioSourceHandle source) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kPause;
        cmd.mSource = source;
        mImpl->PushCommand(cmd);
    }

    void Audio::Stop(AudioSourceHandle source) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kStop;
        cmd.mSource = source;
        mImpl->PushCommand(cmd);
    }

    void Audio::SetSourcePosition(AudioSourceHandle source, const glm::vec3& position) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kSetPosition;
        cmd.mSource = source;
        cmd.mVec = position;
        mImpl->PushCommand(cmd);
    }

    void Audio::DisableSourceAttenuation(AudioSourceHandle source) {
        if (mImpl == nullptr || !source.IsValid()) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kDisableAttenuation;
        cmd.mSource = source;
        mImpl->PushCommand(cmd);
    }

    void Audio::SetListenerPosition(const glm::vec3& position) {
        if (mImpl == nullptr) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kSetListenerPosition;
        cmd.mVec = position;
        mImpl->PushCommand(cmd);
    }

    void Audio::SetListenerVelocity(const glm::vec3& velocity) {
        if (mImpl == nullptr) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kSetListenerVelocity;
        cmd.mVec = velocity;
        mImpl->PushCommand(cmd);
    }

    void Audio::SetListenerOrientation(const glm::vec3& forward, const glm::vec3& up) {
        if (mImpl == nullptr) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kSetListenerOrientation;
        cmd.mVec = forward;
        cmd.mVec2 = up;
        mImpl->PushCommand(cmd);
    }

    void Audio::SetListenerGain(float gain) {
        if (mImpl == nullptr) {
            return;
        }
        AudioCommand cmd{};
        cmd.mType = AudioCommand::Type::kSetListenerGain;
        cmd.mFloat = gain;
        mImpl->PushCommand(cmd);
    }
}// namespace moe::neo
