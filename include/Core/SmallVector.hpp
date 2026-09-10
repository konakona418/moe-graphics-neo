#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <utility>

namespace moe {
    // Small-buffer-optimized vector: stores up to N elements inline (no heap
    // allocation), spills to the heap beyond that. Lean subset of std::vector.
    template<typename T, size_t N>
    class SmallVector {
    public:
        static_assert(N > 0, "SmallVector requires N > 0");

        SmallVector() = default;

        ~SmallVector() {
            clear();
            if (mHeap != nullptr) {
                ::operator delete(mHeap);
            }
        }

        SmallVector(const SmallVector& other) {
            reserve(other.mSize);
            for (size_t i = 0; i < other.mSize; ++i) {
                emplace_back(other[i]);
            }
        }

        SmallVector& operator=(const SmallVector& other) {
            if (this != &other) {
                clear();
                reserve(other.mSize);
                for (size_t i = 0; i < other.mSize; ++i) {
                    emplace_back(other[i]);
                }
            }
            return *this;
        }

        SmallVector(SmallVector&& other) noexcept {
            MoveFrom(std::move(other));
        }

        SmallVector& operator=(SmallVector&& other) noexcept {
            if (this != &other) {
                clear();
                if (mHeap != nullptr) {
                    ::operator delete(mHeap);
                    mHeap = nullptr;
                }
                MoveFrom(std::move(other));
            }
            return *this;
        }

        template<typename... Args>
        T& emplace_back(Args&&... args) {
            if (mSize >= Capacity()) {
                Grow();
            }
            T* slot = Data() + mSize;
            new (slot) T(std::forward<Args>(args)...);
            ++mSize;
            return *slot;
        }

        void push_back(const T& value) {
            emplace_back(value);
        }

        void push_back(T&& value) {
            emplace_back(std::move(value));
        }

        void pop_back() {
            if (mSize > 0) {
                Data()[--mSize].~T();
            }
        }

        void clear() {
            for (size_t i = 0; i < mSize; ++i) {
                Data()[i].~T();
            }
            mSize = 0;
        }

        void reserve(size_t capacity) {
            if (capacity <= Capacity()) {
                return;
            }
            GrowTo(capacity);
        }

        T& operator[](size_t index) {
            return Data()[index];
        }

        const T& operator[](size_t index) const {
            return Data()[index];
        }

        T& back() {
            return Data()[mSize - 1];
        }

        const T& back() const {
            return Data()[mSize - 1];
        }

        T* data() {
            return Data();
        }

        const T* data() const {
            return Data();
        }

        T* begin() {
            return Data();
        }

        T* end() {
            return Data() + mSize;
        }

        const T* begin() const {
            return Data();
        }

        const T* end() const {
            return Data() + mSize;
        }

        size_t size() const {
            return mSize;
        }

        bool empty() const {
            return mSize == 0;
        }

        size_t capacity() const {
            return Capacity();
        }

    private:
        size_t Capacity() const {
            return mHeap != nullptr ? mHeapCapacity : N;
        }

        T* Data() {
            return mHeap != nullptr ? mHeap : reinterpret_cast<T*>(mInline);
        }

        const T* Data() const {
            return mHeap != nullptr ? mHeap : reinterpret_cast<const T*>(mInline);
        }

        void MoveFrom(SmallVector&& other) noexcept {
            if (other.mSize <= N) {
                // move inline elements one by one
                for (size_t i = 0; i < other.mSize; ++i) {
                    emplace_back(std::move(other[i]));
                }
                other.clear();
            } else {
                // take the heap
                mHeap = other.mHeap;
                mHeapCapacity = other.mHeapCapacity;
                mSize = other.mSize;
                other.mHeap = nullptr;
                other.mHeapCapacity = 0;
                other.mSize = 0;
            }
        }

        void Grow() {
            GrowTo(Capacity() * 2);
        }

        void GrowTo(size_t newCapacity) {
            T* newStorage = static_cast<T*>(::operator new(newCapacity * sizeof(T)));
            for (size_t i = 0; i < mSize; ++i) {
                new (&newStorage[i]) T(std::move(Data()[i]));
                Data()[i].~T();
            }
            if (mHeap != nullptr) {
                ::operator delete(mHeap);
            }
            mHeap = newStorage;
            mHeapCapacity = newCapacity;
        }

        alignas(T) uint8_t mInline[N * sizeof(T)];
        T* mHeap{nullptr};
        size_t mHeapCapacity{0};
        size_t mSize{0};
    };
}// namespace moe