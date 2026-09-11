#include <Neo/OggProvider.hpp>
#include <Core/Profile.hpp>

#include <Core/Logger.hpp>

#include <cstdlib>

#include <vorbis/vorbisfile.h>

#define STB_VORBIS_IMPLEMENTATION
#include <stb_vorbis.c>

namespace moe::neo {
    namespace {
        constexpr size_t kDefaultChunkSize = 1024 * 16;
    }// namespace

    struct OggProvider::Impl {
        Mode mMode{Mode::kStatic};
        uint32_t mSampleRate{0};
        uint8_t mChannels{0};
        bool mValid{false};

        // bytes handed in by the caller; must outlive the provider
        const uint8_t* mData{nullptr};
        size_t mSize{0};

        // ---- kStatic (decoded at construction) ----
        std::vector<uint8_t> mDecoded;

        // ---- kStreamed ----
        OggVorbis_File mOggFile{};
        size_t mReadOffset{0};
        size_t mChunkSize{kDefaultChunkSize};

        size_t ReadFromMemory(void* dst, size_t size, size_t count) {
            const size_t requested = size * count;
            const size_t available = mSize - mReadOffset;
            const size_t actual = requested < available ? requested : available;
            if (actual > 0) {
                std::memcpy(dst, mData + mReadOffset, actual);
                mReadOffset += actual;
            }
            return actual / size; // item count, fread semantics
        }

        int SeekMemory(ogg_int64_t offset, int whence) {
            ogg_int64_t base = 0;
            switch (whence) {
                case SEEK_SET: base = 0; break;
                case SEEK_CUR: base = static_cast<ogg_int64_t>(mReadOffset); break;
                case SEEK_END: base = static_cast<ogg_int64_t>(mSize); break;
                default: return -1;
            }
            const ogg_int64_t target = base + offset;
            if (target < 0 || static_cast<uint64_t>(target) > mSize) {
                return -1;
            }
            mReadOffset = static_cast<size_t>(target);
            return 0;
        }

        static size_t ReadFunc(void* ptr, size_t size, size_t nmemb, void* datasource) {
            return static_cast<Impl*>(datasource)->ReadFromMemory(ptr, size, nmemb);
        }

        static int SeekFunc(void* datasource, ogg_int64_t offset, int whence) {
            return static_cast<Impl*>(datasource)->SeekMemory(offset, whence);
        }

        static int CloseFunc(void*) {
            return 0;
        }

        static long TellFunc(void* datasource) {
            return static_cast<long>(static_cast<Impl*>(datasource)->mReadOffset);
        }

        // Decodes one chunk (or the whole file in static mode) into out.
        size_t DecodeInto(std::vector<uint8_t>& out) {
            if (mMode == Mode::kStatic) {
                out = mDecoded;
                return out.size();
            }

            out.resize(mChunkSize);
            size_t total = 0;
            while (total < mChunkSize) {
                const long n = ov_read(&mOggFile,
                        reinterpret_cast<char*>(out.data() + total),
                        static_cast<int>(mChunkSize - total),
                        0, // little endian
                        2, // 16 bits per sample
                        1, // signed
                        nullptr);
                if (n < 0) {
                    moe::Logger::Error("[neo] audio: error reading Ogg Vorbis stream");
                    return 0;
                }
                if (n == 0) {
                    break; // EOF
                }
                total += static_cast<size_t>(n);
            }
            out.resize(total);
            return total;
        }
    };

    OggProvider::OggProvider(const uint8_t* oggData, size_t size, Mode mode)
        : mImpl(std::make_unique<Impl>()) {
        MOE_PROFILE_ZONE();
        mImpl->mMode = mode;
        mImpl->mData = oggData;
        mImpl->mSize = size;

        if (mode == Mode::kStatic) {
            int channels = 0;
            int sampleRate = 0;
            short* decoded = nullptr;
            const int sampleCount = stb_vorbis_decode_memory(
                    reinterpret_cast<const unsigned char*>(oggData), static_cast<int>(size),
                    &channels, &sampleRate, &decoded);
            if (sampleCount < 0) {
                moe::Logger::Error("[neo] audio: failed to decode static Ogg Vorbis data");
                return;
            }
            const size_t totalBytes = static_cast<size_t>(sampleCount)
                    * sizeof(short) * static_cast<size_t>(channels);
            mImpl->mDecoded.assign(reinterpret_cast<const uint8_t*>(decoded),
                    reinterpret_cast<const uint8_t*>(decoded) + totalBytes);
            std::free(decoded);

            mImpl->mSampleRate = static_cast<uint32_t>(sampleRate);
            mImpl->mChannels = static_cast<uint8_t>(channels);
            mImpl->mValid = true;
        } else {
            ov_callbacks callbacks{};
            callbacks.read_func = &Impl::ReadFunc;
            callbacks.seek_func = &Impl::SeekFunc;
            callbacks.close_func = &Impl::CloseFunc;
            callbacks.tell_func = &Impl::TellFunc;

            if (ov_open_callbacks(mImpl.get(), &mImpl->mOggFile, nullptr, 0, callbacks) < 0) {
                moe::Logger::Error("[neo] audio: failed to open streamed Ogg Vorbis data");
                return;
            }
            const vorbis_info* info = ov_info(&mImpl->mOggFile, -1);
            mImpl->mSampleRate = static_cast<uint32_t>(info->rate);
            mImpl->mChannels = static_cast<uint8_t>(info->channels);
            mImpl->mValid = true;
        }
    }

    OggProvider::~OggProvider() {
        if (mImpl->mValid && mImpl->mMode == Mode::kStreamed) {
            ov_clear(&mImpl->mOggFile);
        }
    }

    bool OggProvider::IsStreaming() const {
        return mImpl->mMode == Mode::kStreamed;
    }

    uint32_t OggProvider::GetSampleRate() const {
        return mImpl->mSampleRate;
    }

    uint8_t OggProvider::GetChannelCount() const {
        return mImpl->mChannels;
    }

    bool OggProvider::LoadStatic(std::vector<uint8_t>& outPcm) {
        MOE_PROFILE_ZONE();
        if (!mImpl->mValid || mImpl->mMode != Mode::kStatic) {
            moe::Logger::Error("[neo] audio: static load requested on a non-static provider");
            return false;
        }
        outPcm = mImpl->mDecoded;
        return !outPcm.empty();
    }

    size_t OggProvider::StreamNextChunk(std::vector<uint8_t>& outPcm) {
        MOE_PROFILE_ZONE();
        if (!mImpl->mValid || mImpl->mMode != Mode::kStreamed) {
            return 0;
        }
        return mImpl->DecodeInto(outPcm);
    }

    void OggProvider::SeekToStart() {
        if (!mImpl->mValid || mImpl->mMode != Mode::kStreamed) {
            return;
        }
        if (ov_pcm_seek(&mImpl->mOggFile, 0) != 0) {
            moe::Logger::Error("[neo] audio: failed to seek stream to start");
        }
    }
}// namespace moe::neo
