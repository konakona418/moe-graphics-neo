#include <Neo/Caches.hpp>

#include <cstdio>
#include <cstdlib>

#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            std::fprintf(stderr, "NeoCache FAILED: %s (%d)\n",   \
                    #cond, __LINE__);                            \
            return EXIT_FAILURE;                                 \
        }                                                        \
    } while (false)

int main() {
    moe::neo::MeshHandle defaultHandle;
    CHECK(!defaultHandle.IsValid());

    moe::neo::MeshCache cache;

    moe::neo::Mesh quad;
    quad.mName = "quad";
    const auto h1 = cache.Add(std::move(quad));
    CHECK(h1.IsValid());
    CHECK(cache.GetCount() == 1);
    CHECK(cache.Get(h1) != nullptr);
    CHECK(cache.Get(h1)->mName == "quad");

    const auto h2 = cache.Add(moe::neo::Mesh{});
    CHECK(h2.IsValid());
    CHECK(h2.mIndex != h1.mIndex);
    CHECK(cache.GetCount() == 2);

    // remove invalidates outstanding handles
    cache.Remove(h1);
    CHECK(cache.Get(h1) == nullptr);
    CHECK(cache.GetCount() == 1);

    // slot reuse bumps the generation, so the old handle stays stale
    const auto h3 = cache.Add(moe::neo::Mesh{});
    CHECK(h3.mIndex == h1.mIndex);
    CHECK(h3.mGeneration != h1.mGeneration);
    CHECK(cache.Get(h1) == nullptr);
    CHECK(cache.Get(h3) != nullptr);

    // unknown index / double-remove are safe no-ops
    moe::neo::MeshHandle bogus{};
    bogus.mIndex = 9999;
    CHECK(cache.Get(bogus) == nullptr);
    cache.Remove(h3);
    cache.Remove(h3);
    CHECK(cache.GetCount() == 1);

    // distinct handle types stay distinct (compile-time type safety)
    moe::neo::TextureCache textureCache;
    const auto textureHandle = textureCache.Add(moe::neo::Texture{});
    CHECK(textureCache.Get(textureHandle) != nullptr);
    // (no way to pass a TextureHandle where a MeshHandle is expected)

    cache.Clear();
    CHECK(cache.GetCount() == 0);
    CHECK(cache.Get(h2) == nullptr);

    std::printf("NeoCache tests passed.\n");
    return EXIT_SUCCESS;
}