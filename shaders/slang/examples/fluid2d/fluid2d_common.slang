// Shared constants and helpers for the 2D GPU fluid solver.
//
// Port of the 01_3_Fluid_2D compute kernels from unity-compute-shaders
// (itself based on Scrawk/GPU-GEMS-2D-Fluid-Simulation). The grid size is baked
// in as a compile-time constant so the kernels need no size uniform.

static const int kGridSize = 512;
static const float kGridSizeF = 512.0;

// Neighbour offsets: top, bottom, left, right.
static const int2 kOffT = int2(0, 1);
static const int2 kOffB = int2(0, -1);
static const int2 kOffL = int2(-1, 0);
static const int2 kOffR = int2(1, 0);

// Clamped neighbour lookup: the solver samples one texel outside the grid, and
// storage-image reads past the edge are not safe to rely on.
int2 ClampCoord(int2 id) {
    return clamp(id, int2(0, 0), int2(kGridSize - 1, kGridSize - 1));
}

// Distance from p to the segment ab.
float SegmentDist(float2 p, float2 a, float2 b) {
    float2 pa = p - a;
    float2 ba = b - a;
    float h = saturate(dot(pa, ba) / max(dot(ba, ba), 1e-6));
    return length(pa - ba * h);
}

// Soft stroke / disc coverage, used to hand-letter the obstacle.
float Stroke(float2 p, float2 a, float2 b, float hw) {
    return 1.0 - smoothstep(hw, hw * 1.6, SegmentDist(p, a, b));
}

float Disc(float2 p, float2 c, float r) {
    return 1.0 - smoothstep(r, r * 1.6, length(p - c));
}

// The reference's Obstacle.png is the word "Fluid" hand-drawn in black on a
// white field (black = obstacle). We reproduce it as stroke geometry.
float InkMask(float2 uv) {
    const float hw = 0.0095;
    float ink = 0.0;

    // F
    ink = max(ink, Stroke(uv, float2(0.195, 0.405), float2(0.225, 0.605), hw));
    ink = max(ink, Stroke(uv, float2(0.150, 0.445), float2(0.292, 0.415), hw));
    ink = max(ink, Stroke(uv, float2(0.215, 0.512), float2(0.302, 0.497), hw));

    // l
    ink = max(ink, Stroke(uv, float2(0.356, 0.430), float2(0.346, 0.602), hw));

    // u
    ink = max(ink, Stroke(uv, float2(0.430, 0.485), float2(0.425, 0.575), hw));
    ink = max(ink, Stroke(uv, float2(0.425, 0.575), float2(0.500, 0.590), hw));
    ink = max(ink, Stroke(uv, float2(0.506, 0.480), float2(0.500, 0.590), hw));

    // i
    ink = max(ink, Disc(uv, float2(0.575, 0.455), 0.011));
    ink = max(ink, Stroke(uv, float2(0.580, 0.500), float2(0.565, 0.607), hw));

    // d
    ink = max(ink, Stroke(uv, float2(0.748, 0.418), float2(0.736, 0.602), hw));
    ink = max(ink,
            Disc(uv, float2(0.685, 0.545), 0.050) - Disc(uv, float2(0.685, 0.545), 0.038));

    return saturate(ink);
}

// 1 = fluid, 0 = obstacle. The outer two texels are a solid wall (the
// reference obstacle texture has a black border).
float Obstacle(int2 id) {
    if (id.x < 2 || id.y < 2 || id.x >= kGridSize - 2 || id.y >= kGridSize - 2) {
        return 0.0;
    }
    float2 uv = (float2(id) + 0.5) / kGridSizeF;
    return 1.0 - InkMask(uv);
}

float ObstacleNeighbor(int2 id, int2 off) {
    return Obstacle(ClampCoord(id + off));
}

float3 BlendColor(float3 a, float3 b, float factor) {
    return lerp(a, b, factor);
}
