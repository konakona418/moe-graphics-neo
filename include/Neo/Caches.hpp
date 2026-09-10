#pragma once

#include "Neo/Animation.hpp"
#include "Neo/Cache.hpp"
#include "Neo/Material.hpp"
#include "Neo/Mesh.hpp"
#include "Neo/Texture.hpp"

namespace moe::neo {
    using MeshHandle = Handle<Mesh>;
    using MaterialHandle = Handle<Material>;
    using TextureHandle = Handle<Texture>;
    using AnimationHandle = Handle<AnimationClip>;

    using MeshCache = Cache<Mesh>;
    using MaterialCache = Cache<Material>;
    using TextureCache = Cache<Texture>;
    using AnimationCache = Cache<AnimationClip>;
}// namespace moe::neo