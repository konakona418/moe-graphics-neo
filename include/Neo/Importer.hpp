#pragma once

#include "Neo/Scene.hpp"

#include <string>

namespace moe::neo {
    // glTF importer: parses a .gltf/.glb into the format-agnostic Scene model.
    // Faithful to glTF: preserves empty nodes, the full node hierarchy, TRS
    // transforms, meshes, materials, skeletons and animation clips. Images are
    // referenced but pixel decoding is deferred (Texture entries carry names;
    // channels == 0 means undecoded).
    bool ImportGltf(const char* path, Scene& outScene);
}// namespace moe::neo