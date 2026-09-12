// [moe("vertex", "fragment")]

// Raymarches the 3D fluid density volume and composites it front-to-back over
// a dark background (the reference's VolumeShader, adapted to a fullscreen
// pass with an explicit camera basis). Push constants are packed into float4s
// so the name-addressed graphics path resolves few fields.

struct PushConstants {
    float4 mCameraPosTan;  // xyz = camera position, w = tan(half fov)
    float4 mForwardAspect; // xyz = forward, w = aspect
    float4 mRightOpacity;  // xyz = right, w = density opacity
    float4 mUpExposure;    // xyz = up, w = exposure
    float4 mBoxMinSteps;   // xyz = box min, w = step count
    float4 mBoxMaxPad;     // xyz = box max
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture3D<float4> volumeTex;
[[vk::binding(1, 0)]] SamplerState linearSampler;

struct VertexOutput {
    float4 mPosition : SV_Position;
    float2 mUv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.mPosition = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.mUv = pos;
    return output;
}

bool IntersectBox(float3 ro, float3 rd, float3 bmin, float3 bmax, out float t0, out float t1) {
    float3 inv = 1.0 / rd;
    float3 ta = (bmin - ro) * inv;
    float3 tb = (bmax - ro) * inv;
    float3 tmin = min(ta, tb);
    float3 tmax = max(ta, tb);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float3 cameraPos = pcs.mCameraPosTan.xyz;
    float tanHalfFov = pcs.mCameraPosTan.w;
    float3 forward = pcs.mForwardAspect.xyz;
    float aspect = pcs.mForwardAspect.w;
    float3 right = pcs.mRightOpacity.xyz;
    float opacity = pcs.mRightOpacity.w;
    float3 up = pcs.mUpExposure.xyz;
    float exposure = pcs.mUpExposure.w;
    float3 boxMin = pcs.mBoxMinSteps.xyz;
    float stepsF = pcs.mBoxMinSteps.w;
    float3 boxMax = pcs.mBoxMaxPad.xyz;

    float2 ndc = input.mUv * 2.0 - 1.0;
    float3 rayDir = normalize(forward + right * (ndc.x * tanHalfFov * aspect)
            - up * (ndc.y * tanHalfFov));

    float t0;
    float t1;
    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    int steps = int(max(stepsF, 1.0));
    if (IntersectBox(cameraPos, rayDir, boxMin, boxMax, t0, t1)) {
        float t = max(t0, 0.0);
        float stepLen = (t1 - t) / float(steps);
        float3 extent = boxMax - boxMin;
        for (int i = 0; i < steps; ++i) {
            float3 world = cameraPos + rayDir * (t + (float(i) + 0.5) * stepLen);
            float3 uv = (world - boxMin) / extent;
            float4 s = volumeTex.SampleLevel(linearSampler, uv, 0.0);
            float a = saturate(s.a * opacity);
            accum.rgb += (1.0 - accum.a) * a * s.rgb;
            accum.a += (1.0 - accum.a) * a;
        }
    }

    // Faint vertical background so the volume silhouette reads.
    float3 bg = lerp(float3(0.02, 0.025, 0.04), float3(0.01, 0.012, 0.02),
            saturate(input.mUv.y));
    float3 color = bg + accum.rgb * exposure;
    return float4(color, 1.0);
}
