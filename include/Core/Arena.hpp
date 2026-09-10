#pragma once

#include <cstddef>
#include <cstdint>
#include <new>

namespace moe {
    // Bump (arena) allocator. Allocations come from fixed-size blocks and are
    // freed all at once by Reset(). Not thread-safe. Used for transient data
    // (scene loading, per-frame scratch) where individual frees never happen.
    // Blocks are linked through an intrusive header: no separate bookkeeping
    // arrays, no per-allocation metadata.
    class Arena {
    public:
        explicit Arena(size_t blockSize = 64 * 1024)
            : mBlockSize(blockSize > sizeof(Block) ? blockSize : sizeof(Block) + 64) {}

        ~Arena() {
            FreeBlocks();
        }

        Arena(const Arena&) = delete;
        Arena& operator=(const Arena&) = delete;

        void* Allocate(size_t size, size_t align = alignof(std::max_align_t)) {
            while (mCurrent == nullptr || AlignUp(mOffset, align) + size > mPayloadSize) {
                AddBlock();
            }
            const uintptr_t offset = AlignUp(mOffset, align);
            void* ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(mCurrent) + offset);
            mOffset = offset + size;
            return ptr;
        }

        template<typename T>
        T* Allocate(size_t count = 1) {
            return static_cast<T*>(Allocate(sizeof(T) * count, alignof(T)));
        }

        // Frees all blocks; subsequent allocations allocate a fresh block.
        void Reset() {
            FreeBlocks();
            mCurrent = nullptr;
            mOffset = 0;
        }

        size_t GetUsedBytes() const {
            return mOffset;
        }

    private:
        struct Block {
            Block* mNext;
        };

        static uintptr_t AlignUp(uintptr_t value, size_t align) {
            const uintptr_t mask = static_cast<uintptr_t>(align) - 1;
            return (value + mask) & ~mask;
        }

        void AddBlock() {
            void* raw = ::operator new(mBlockSize);
            auto* block = static_cast<Block*>(raw);
            block->mNext = mHead;
            mHead = block;
            mCurrent = reinterpret_cast<uint8_t*>(block) + sizeof(Block);
            mPayloadSize = mBlockSize - sizeof(Block);
            mOffset = 0;
        }

        void FreeBlocks() {
            Block* block = mHead;
            while (block != nullptr) {
                Block* next = block->mNext;
                ::operator delete(block);
                block = next;
            }
            mHead = nullptr;
        }

        const size_t mBlockSize;
        Block* mHead{nullptr};
        uint8_t* mCurrent{nullptr};
        size_t mOffset{0};
        size_t mPayloadSize{0};
    };
}// namespace moe
