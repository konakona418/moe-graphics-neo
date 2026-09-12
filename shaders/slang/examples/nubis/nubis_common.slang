// Shared procedural helpers for the Nubis3-style volumetric cloud demo.
// Include-only file: no // [moe(...)] metadata, so it is not compiled alone.

#define MOE_PI 3.14159265359

float Hash31(float3 p) {
    p = frac(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return frac((p.x + p.y) * p.z);
}

float3 Hash33(float3 p) {
    p = float3(dot(p, float3(127.1, 311.7, 74.7)),
               dot(p, float3(269.5, 183.3, 246.1)),
               dot(p, float3(113.5, 271.9, 124.6)));
    return frac(sin(p) * 43758.5453);
}

float ValueNoise(float3 p) {
    float3 i = floor(p);
    float3 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = Hash31(i + float3(0.0, 0.0, 0.0));
    float n100 = Hash31(i + float3(1.0, 0.0, 0.0));
    float n010 = Hash31(i + float3(0.0, 1.0, 0.0));
    float n110 = Hash31(i + float3(1.0, 1.0, 0.0));
    float n001 = Hash31(i + float3(0.0, 0.0, 1.0));
    float n101 = Hash31(i + float3(1.0, 0.0, 1.0));
    float n011 = Hash31(i + float3(0.0, 1.0, 1.0));
    float n111 = Hash31(i + float3(1.0, 1.0, 1.0));
    return lerp(
        lerp(lerp(n000, n100, f.x), lerp(n010, n110, f.x), f.y),
        lerp(lerp(n001, n101, f.x), lerp(n011, n111, f.x), f.y),
        f.z);
}

float Fbm(float3 p, int octaves) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < octaves; ++i) {
        v += a * ValueNoise(p);
        p = p * 2.03 + 17.0;
        a *= 0.5;
    }
    return v;
}

// Worley F1 distance, normalised to roughly [0,1].
float Worley(float3 p) {
    float3 cell = floor(p);
    float minDist = 8.0;
    for (int z = -1; z <= 1; ++z) {
        for (int y = -1; y <= 1; ++y) {
            for (int x = -1; x <= 1; ++x) {
                float3 c = cell + float3(float(x), float(y), float(z));
                float3 pt = c + Hash33(c);
                minDist = min(minDist, length(p - pt));
            }
        }
    }
    return min(minDist, 1.0);
}

float SdSphere(float3 p, float3 c, float r) {
    return length(p - c) - r;
}

float SMin(float a, float b, float k) {
    float h = saturate(0.5 + 0.5 * (b - a) / k);
    return lerp(b, a, h) - k * h * (1.0 - h);
}

float Remap(float v, float a, float b, float c, float d) {
    return c + (v - a) / (b - a) * (d - c);
}

// Derived from Set-Range: erodes (positive oldMin) or inflates (negative).
float ValueErosion(float v, float oldMin) {
    return saturate((v - oldMin) / (1.0 - oldMin));
}

float HgPhase(float cosAngle, float g) {
    float g2 = g * g;
    return (1.0 - g2)
            / (4.0 * MOE_PI * pow(max(1.0 + g2 - 2.0 * g * cosAngle, 1e-4), 1.5));
}

float3 Uncharted2Tonemap(float3 w) {
    float a = 0.15;
    float b = 0.50;
    float c = 0.10;
    float d = 0.20;
    float e = 0.02;
    float f = 0.30;
    return ((w * (a * w + c * b) + d * e) / (w * (a * w + b) + d * f)) - e / f;
}
