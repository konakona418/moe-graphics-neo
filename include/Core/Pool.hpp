#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>

namespace moe {
    // Fixed-slot object pool with a free list. Acquire returns a slot index
    // that is reused after Release. Not thread-safe. Fits handle-backed caches
    // (slot reuse without per-object allocation).
    //
    // Storage is a chain of fixed blocks (kSlotsPerBlock slots each): slot
    // indices never move, so constructed objects are never relocated. Values
    // are not destroyed here — callers must pair Construct with Destruct
    // (same contract as the std::vector-backed version).
    template<typename T>
    class Pool {
    public:
        static constexpr size_t kSlotsPerBlock = 64;

        explicit Pool(size_t capacityHint = 64) {
            while (mCapacity < capacityHint) {
                AddBlock();
            }
        }

        ~Pool() {
            FreeBlocks();
        }

        Pool(const Pool&) = delete;
        Pool& operator=(const Pool&) = delete;

        // Returns an index of an unused slot; T is not constructed until
        // Construct(index). The slot remains reserved until Release(index).
        uint32_t Acquire() {
            if (mFreeCount == 0) {
                AddBlock();
            }
            return mFreeStack[--mFreeCount];
        }

        void Release(uint32_t index) {
            if (mFreeCount == mFreeCapacity) {
                GrowFreeStack();
            }
            mFreeStack[mFreeCount++] = index;
        }

        // Constructs T in the slot (must not already be constructed).
        template<typename... Args>
        T* Construct(uint32_t index, Args&&... args) {
            return new (GetSlot(index)) T(std::forward<Args>(args)...);
        }

        void Destruct(uint32_t index) {
            Get(index)->~T();
        }

        T* Get(uint32_t index) {
            return static_cast<T*>(GetSlot(index));
        }

        const T* Get(uint32_t index) const {
            return static_cast<const T*>(GetSlot(index));
        }

        size_t GetCapacity() const {
            return mCapacity;
        }

        // Rebuilds the free stack to cover every slot. The caller must have
        // Destruct'ed every constructed value first.
        void Reset() {
            mFreeCount = 0;
            for (uint32_t i = 0; i < mCapacity; ++i) {
                Release(i);
            }
        }

    private:
        struct alignas(std::max_align_t) Block {
            Block* mNext;
        };

        void* GetSlot(uint32_t index) const {
            const Block* block = mBlocks[index / kSlotsPerBlock];
            return reinterpret_cast<uint8_t*>(const_cast<Block*>(block)) + sizeof(Block)
                    + (index % kSlotsPerBlock) * sizeof(T);
        }

        void AddBlock() {
            void* raw = std::malloc(sizeof(Block) + kSlotsPerBlock * sizeof(T));
            if (raw == nullptr) {
                std::abort(); // no-silent-failure
            }
            // zeroed so never-constructed slots read as empty state
            std::memset(raw, 0, sizeof(Block) + kSlotsPerBlock * sizeof(T));

            auto* block = static_cast<Block*>(raw);
            block->mNext = mHead;
            mHead = block;

            Block** grownBlocks = static_cast<Block**>(
                    std::realloc(mBlocks, (mBlockCount + 1) * sizeof(Block*)));
            if (grownBlocks == nullptr) {
                std::abort();
            }
            mBlocks = grownBlocks;
            mBlocks[mBlockCount++] = block;

            if (mFreeCount + kSlotsPerBlock > mFreeCapacity) {
                GrowFreeStack();
            }
            for (uint32_t i = 0; i < kSlotsPerBlock; ++i) {
                mFreeStack[mFreeCount + i] = static_cast<uint32_t>(mCapacity + i);
            }
            mFreeCount += kSlotsPerBlock;
            mCapacity += kSlotsPerBlock;
        }

        void GrowFreeStack() {
            const size_t newCapacity = mFreeCapacity != 0 ? mFreeCapacity * 2 : kSlotsPerBlock;
            uint32_t* grown = static_cast<uint32_t*>(
                    std::realloc(mFreeStack, newCapacity * sizeof(uint32_t)));
            if (grown == nullptr) {
                std::abort();
            }
            mFreeStack = grown;
            mFreeCapacity = newCapacity;
        }

        void FreeBlocks() {
            Block* block = mHead;
            while (block != nullptr) {
                Block* next = block->mNext;
                std::free(block);
                block = next;
            }
            std::free(mBlocks);
            std::free(mFreeStack);
        }

        Block* mHead{nullptr};
        Block** mBlocks{nullptr};
        size_t mBlockCount{0};
        uint32_t* mFreeStack{nullptr};
        size_t mFreeCount{0};
        size_t mFreeCapacity{0};
        size_t mCapacity{0};
    };
}// namespace moe
