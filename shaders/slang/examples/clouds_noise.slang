// [moe("compute")]

// Generates a 3D cloud-density volume: FBM value noise for the cloud shape
// (with vertical falloff) + Worley detail for erosion. Written into a 3D
// storage image, regenerated every frame so the cloud field can drift.

[[vk::binding(0, 0)]] RWTexture3D<float4> noiseTex;

struct PushConstants {
    uint size;    // volume edge length (one dimension, cube)
    float time;   // seconds, drives the drift
    float seed;   // noise seed offset
    float padding;
};

[[vk::push_constant]] PushConstants pcs;

static const int kShapeOctaves = 5;
static const int kDetailOctaves = 4;

float hash(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float valueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(
        lerp(lerp(hash(i + float3(0, 0, 0)), hash(i + float3(1, 0, 0)), f.x),
             lerp(hash(i + float3(0, 1, 0)), hash(i + float3(1, 1, 0)), f.x), f.y),
        lerp(lerp(hash(i + float3(0, 0, 1)), hash(i + float3(1, 0, 1)), f.x),
             lerp(hash(i + float3(0, 1, 1)), hash(i + float3(1, 1, 1)), f.x), f.y),
        f.z);
}

float fbm(float3 p) {
    float v = 0.0;
    float amp = 0.5;
    for (int i = 0; i < kShapeOctaves; ++i) {
        v += amp * valueNoise(p);
        p *= 2.03;
        amp *= 0.5;
    }
    return v;
}

float worley(float3 p) {
    float3 cell = floor(p);
    float minDist = 8.0;
    for (int z = -1; z <= 1; ++z)
        for (int y = -1; y <= 1; ++y)
            for (int x = -1; x <= 1; ++x) {
                float3 c = cell + float3(float(x), float(y), float(z));
                float3 point = c + hash(c);
                float d = length(p - point);
                minDist = min(minDist, d);
            }
    return minDist;
}

[shader("compute")]
[numthreads(4, 4, 4)]
void computeMain(uint3 id : SV_DispatchThreadID) {
    float size = float(pcs.size);
    float3 uv = (float3(id) + 0.5) / size;
    float3 p = uv * 10.0 + float3(pcs.seed, 0.0, pcs.seed * 0.7);

    // slow horizontal drift over time
    float3 shapeP = p + float3(pcs.time * 0.015, 0.0, pcs.time * 0.01);
    float shape = fbm(shapeP);

    // denser in a vertical band: thinner near the bottom, capped near the top
    float height = uv.y;
    float falloff = smoothstep(0.05, 0.3, height) * (1.0 - smoothstep(0.45, 0.9, height));
    float density = saturate(smoothstep(0.45, 0.62, shape)) * falloff;

    // fade the cloud field out near the horizontal box walls, so the volume
    // boundary is not visible as a hard edge when the camera orbits
    float edgeFade = smoothstep(0.0, 0.15, uv.x) * smoothstep(0.0, 0.15, 1.0 - uv.x)
            * smoothstep(0.0, 0.15, uv.z) * smoothstep(0.0, 0.15, 1.0 - uv.z);
    density *= edgeFade;

    // Worley detail erodes holes into the cloud
    float detail = 1.0 - smoothstep(0.0, 0.55, worley(p * 2.5 + pcs.seed));
    float cloud = saturate(density * (1.0 - 0.75 * detail));

    noiseTex[id] = float4(cloud, detail, density, 1.0);
}
