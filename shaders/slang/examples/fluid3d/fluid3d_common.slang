// Shared constants and helpers for the 3D GPU fluid solver.
//
// Port of the 01_4_Fluid_3D compute kernels from unity-compute-shaders. The
// grid size is baked in as a compile-time constant so the kernels need no size
// uniform. The obstacle is the boundary padding (5 voxels on every side).

static const int kGridSize = 96;
static const float kGridSizeF = 96.0;
static const float kPadding = 5.0;

// Neighbour offsets along +x/-x (R/L), +y/-y (T/B), +z/-z (F/K).
static const int3 kOffT = int3(0, 1, 0);
static const int3 kOffB = int3(0, -1, 0);
static const int3 kOffL = int3(-1, 0, 0);
static const int3 kOffR = int3(1, 0, 0);
static const int3 kOffF = int3(0, 0, 1);
static const int3 kOffK = int3(0, 0, -1);

int3 ClampCoord(int3 id) {
    return clamp(id, int3(0, 0, 0), int3(kGridSize - 1, kGridSize - 1, kGridSize - 1));
}

// 1 = fluid, 0 = obstacle (solid walls around the volume).
float Obstacle(int3 id) {
    if (id.x < kPadding || id.y < kPadding || id.z < kPadding
            || id.x > kGridSizeF - kPadding || id.y > kGridSizeF - kPadding
            || id.z > kGridSizeF - kPadding) {
        return 0.0;
    }
    return 1.0;
}

float ObstacleNeighbor(int3 id, int3 off) {
    return Obstacle(ClampCoord(id + off));
}

float3 BlendColor(float3 a, float3 b, float factor) {
    return lerp(a, b, factor);
}
