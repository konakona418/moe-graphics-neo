#include <Core/Arena.hpp>
#include <Core/Pool.hpp>
#include <Core/SmallVector.hpp>

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            std::fprintf(stderr, "DataStruct FAILED: %s (%d)\n", \
                    #cond, __LINE__);                            \
            return EXIT_FAILURE;                                 \
        }                                                        \
    } while (false)

struct Blob {
    int a;
    float b;
};

int main() {
    // ---- Arena ----
    {
        moe::Arena arena(256);
        int* a = arena.Allocate<int>();
        *a = 7;
        char* bytes = static_cast<char*>(arena.Allocate(100, alignof(char)));
        bytes[0] = 'x';
        bytes[99] = 'y';
        Blob* blob = arena.Allocate<Blob>();
        blob->a = 1;
        blob->b = 2.5f;
        CHECK(*a == 7);
        CHECK(bytes[0] == 'x' && bytes[99] == 'y');
        CHECK(blob->a == 1 && blob->b == 2.5f);
        arena.Reset();
        int* b = arena.Allocate<int>();
        *b = 9; // fresh block, previous data gone
        CHECK(*b == 9);
    }

    // ---- Pool ----
    {
        moe::Pool<int> pool(4);
        const uint32_t s0 = pool.Acquire();
        CHECK(s0 < pool.GetCapacity());
        int* v0 = pool.Construct<int>(s0, 42);
        CHECK(*v0 == 42);
        const uint32_t s1 = pool.Acquire();
        const uint32_t s2 = pool.Acquire();
        CHECK(s1 != s0 && s2 != s0 && s1 != s2);
        pool.Destruct(s0);
        pool.Release(s0);
        // the released slot is reused by the next Acquire
        const uint32_t s0b = pool.Acquire();
        CHECK(s0b == s0);
        CHECK(pool.GetCapacity() >= 4);
    }

    // ---- SmallVector ----
    {
        moe::SmallVector<int, 4> vec;
        CHECK(vec.empty());
        vec.push_back(10);
        vec.push_back(20);
        vec.push_back(30);
        vec.push_back(40);
        CHECK(vec.size() == 4);
        CHECK(vec.capacity() == 4); // still inline
        vec.push_back(50); // spills to heap
        CHECK(vec.size() == 5);
        CHECK(vec.capacity() == 8);
        CHECK(vec[0] == 10 && vec[4] == 50);
        int sum = 0;
        for (int value : vec) {
            sum += value;
        }
        CHECK(sum == 150);
        vec.pop_back();
        CHECK(vec.size() == 4 && vec[3] == 40);
        vec.clear();
        CHECK(vec.empty());

        moe::SmallVector<Blob, 2> blobs;
        blobs.emplace_back(Blob{1, 2.0f});
        blobs.emplace_back(Blob{3, 4.0f});
        blobs.emplace_back(Blob{5, 6.0f}); // heap
        CHECK(blobs.size() == 3);
        CHECK(blobs[2].a == 5 && blobs[2].b == 6.0f);

        // move semantics
        moe::SmallVector<int, 4> moved = std::move(vec);
        CHECK(moved.empty() && vec.empty());
    }

    std::printf("DataStruct tests passed.\n");
    return EXIT_SUCCESS;
}