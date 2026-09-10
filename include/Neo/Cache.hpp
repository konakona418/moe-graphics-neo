#pragma once

#include <Core/Pool.hpp>

#include <cstdint>
#include <new>
#include <utility>

namespace moe::neo {
    inline constexpr uint32_t kInvalidIndex = 0xFFFFFFFF;

    // Lightweight typed handle: a slot index + a generation counter. A stale
    // handle (slot reused or released) carries an old generation and fails
    // validation, preventing ABA-style misuse. Not reference counted.
    template<typename T>
    struct Handle {
        uint32_t mIndex{kInvalidIndex};
        uint32_t mGeneration{0};

        bool IsValid() const {
            return mIndex != kInvalidIndex;
        }
    };

    // Slot-based cache owning values, addressed by Handle<T>. Explicit
    // Add/Remove; the cache never frees on its own. Removing a slot bumps its
    // generation so outstanding handles to it become invalid. Values are
    // destroyed on Remove/Clear and in the destructor. Backed by a Pool:
    // slots never move, so values are never relocated. The generation
    // bookkeeping persists across slot reuse (only the value sub-object is
    // constructed/destructed), so every reuse invalidates old handles.
    template<typename T>
    class Cache {
    public:
        Cache() = default;

        ~Cache() {
            Clear();
        }

        Cache(const Cache&) = delete;
        Cache& operator=(const Cache&) = delete;

        Handle<T> Add(T&& value) {
            const uint32_t index = mSlots.Acquire();
            Slot* slot = mSlots.Get(index);
            slot->mGeneration += 1;
            new (&slot->mValue) T(std::move(value)); // slot bytes were zeroed
            slot->mOccupied = true;
            ++mCount;
            return Handle<T>{index, slot->mGeneration};
        }

        // Constructs T in a fresh slot from the given arguments. Used for
        // values that are not movable (RHI handles, shader programs).
        template<typename... Args>
        Handle<T> Emplace(Args&&... args) {
            const uint32_t index = mSlots.Acquire();
            Slot* slot = mSlots.Get(index);
            slot->mGeneration += 1;
            new (&slot->mValue) T(std::forward<Args>(args)...);
            slot->mOccupied = true;
            ++mCount;
            return Handle<T>{index, slot->mGeneration};
        }

        void Remove(Handle<T> handle) {
            Slot* slot = Resolve(handle);
            if (slot == nullptr) {
                return;
            }
            slot->mGeneration += 1; // invalidate outstanding handles first
            slot->mValue.~T();
            slot->mOccupied = false;
            mSlots.Release(handle.mIndex);
            --mCount;
        }

        T* Get(Handle<T> handle) {
            Slot* slot = Resolve(handle);
            return slot != nullptr ? &slot->mValue : nullptr;
        }

        const T* Get(Handle<T> handle) const {
            const Slot* slot = Resolve(handle);
            return slot != nullptr ? &slot->mValue : nullptr;
        }

        void Clear() {
            for (uint32_t i = 0; i < mSlots.GetCapacity(); ++i) {
                Slot* slot = mSlots.Get(i);
                if (slot->mOccupied) {
                    slot->mValue.~T();
                    slot->mOccupied = false;
                }
            }
            mSlots.Reset();
            mCount = 0;
        }

        size_t GetCount() const {
            return mCount;
        }

        // Calls fn(T&) for every live value (e.g. explicit teardown of
        // GPU-backed values before Clear()).
        template<typename F>
        void ForEach(F&& fn) {
            for (uint32_t i = 0; i < mSlots.GetCapacity(); ++i) {
                Slot* slot = mSlots.Get(i);
                if (slot->mOccupied) {
                    fn(slot->mValue);
                }
            }
        }

    private:
        // Only the mValue sub-object is a live C++ object: mGeneration and
        // mOccupied are plain bookkeeping bytes (zeroed when a Pool block is
        // created, then persisted across slot reuse).
        struct Slot {
            T mValue{};
            uint32_t mGeneration{0};
            bool mOccupied{false};
        };

        Slot* Resolve(Handle<T> handle) {
            if (!handle.IsValid() || handle.mIndex >= mSlots.GetCapacity()) {
                return nullptr;
            }
            Slot* slot = mSlots.Get(handle.mIndex);
            if (!slot->mOccupied || slot->mGeneration != handle.mGeneration) {
                return nullptr;
            }
            return slot;
        }

        const Slot* Resolve(Handle<T> handle) const {
            if (!handle.IsValid() || handle.mIndex >= mSlots.GetCapacity()) {
                return nullptr;
            }
            const Slot* slot = mSlots.Get(handle.mIndex);
            if (!slot->mOccupied || slot->mGeneration != handle.mGeneration) {
                return nullptr;
            }
            return slot;
        }

        Pool<Slot> mSlots;
        size_t mCount{0};
    };
}// namespace moe::neo
