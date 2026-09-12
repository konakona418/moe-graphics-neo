// [moe("vertex", "fragment")]

// Fullscreen raymarch: reconstructs the view ray per pixel from a camera
// basis in push constants (no projection matrix needed), marches the cloud
// volume, and composites over a gradient sky with a simple ground plane.

struct PushConstants {
    float3 cameraPos;
    float3 forward;
    float3 right;
    float3 up;
    float3 sunDir;
    float tanHalfFov;
    float aspect;
    float time;
    float3 boxMin;
    float3 boxMax;
};

[[vk::push_constant]] PushConstants pcs;

[[vk::binding(0, 0)]] Texture3D<float4> noiseTex;
[[vk::binding(1, 0)]] SamplerState linearSampler;

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VertexOutput vertexMain(uint vertexId : SV_VulkanVertexId) {
    VertexOutput output;
    float2 pos = float2(float((vertexId << 1) & 2), float(vertexId & 2));
    output.position = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    output.uv = pos;
    return output;
}

float densityAt(float3 worldPos) {
    float3 uv = (worldPos - pcs.boxMin) / (pcs.boxMax - pcs.boxMin);
    if (any(uv < 0.0) || any(uv > 1.0)) {
        return 0.0;
    }
    float4 n = noiseTex.SampleLevel(linearSampler, uv, 0.0);
    return n.x;
}

[shader("fragment")]
float4 fragmentMain(VertexOutput input) : SV_Target {
    float2 ndc = input.uv * 2.0 - 1.0;
    float3 rayDir = normalize(pcs.forward
            + pcs.right * (ndc.x * pcs.tanHalfFov * pcs.aspect)
            - pcs.up * (ndc.y * pcs.tanHalfFov));

    float3 sunColor = float3(1.0, 0.95, 0.85);
    float3 skyHorizon = float3(0.72, 0.80, 0.86);
    float3 skyZenith = float3(0.35, 0.55, 0.85);
    float3 sky = lerp(skyHorizon, skyZenith, pow(max(rayDir.y, 0.0), 0.6));
    float sunDisk = pow(max(dot(rayDir, pcs.sunDir), 0.0), 500.0);
    sky += sunColor * sunDisk * 1.5;

    float3 color = sky;
    float transmittance = 1.0;

    // simple ground plane at y = 0
    if (rayDir.y < -0.0001) {
        float gt = -pcs.cameraPos.y / rayDir.y;
        float3 hit = pcs.cameraPos + rayDir * gt;
        float dist = length(hit - pcs.cameraPos);
        float haze = 1.0 - exp(-dist * 0.02);
        float3 ground = float3(0.35, 0.28, 0.20);
        color = lerp(ground, skyHorizon, haze);
        float3 sunFlat = normalize(float3(pcs.sunDir.x, 0.0, pcs.sunDir.z));
        float3 hitFlat = normalize(float3(hit.x, 0.0, hit.z));
        color += sunColor * pow(max(dot(sunFlat, hitFlat), 0.0), 8.0) * 0.15;
    }

    // march through the cloud box
    const int kSteps = 80;
    const int kSunSteps = 12;
    float marchLen = length(pcs.boxMax - pcs.boxMin) * 1.6;
    float stepLen = marchLen / float(kSteps);
    float3 step = rayDir * stepLen;
    float3 t = pcs.cameraPos + rayDir * 0.1;
    float3 lightAccum = float3(0.0);
    for (int i = 0; i < kSteps; ++i) {
        float d = densityAt(t);
        if (d > 0.002) {
            // single scattering: optical depth toward the sun (same step size
            // as the main ray so it resolves the cloud layer)
            float sunDepth = 0.0;
            float3 sp = t;
            float3 sunStep = pcs.sunDir * stepLen;
            for (int s = 0; s < kSunSteps; ++s) {
                sunDepth += densityAt(sp) * stepLen;
                sp += sunStep;
            }
            float sunTransmittance = exp(-sunDepth * 0.9);
            // broad phase: lit from all angles, brighter toward the sun
            float phase = 0.35 + 0.65 * pow(max(dot(rayDir, pcs.sunDir), 0.0), 4.0);
            lightAccum += sunColor * sunTransmittance * phase * d * transmittance * 1.0;
            // gentle ambient keeps shadows blue-ish, not black
            lightAccum += skyZenith * d * transmittance * 0.12;
            transmittance *= exp(-d * 0.3);
            if (transmittance < 0.02) {
                break;
            }
        }
        t += step;
    }
    color = color * transmittance + lightAccum;
    return float4(color, 1.0);
}
