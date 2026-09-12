// [moe("compute")]

// Writes a time-varying waveform into a storage buffer. Runs on the compute
// queue (a family separate from graphics) while the display pass samples the
// same buffer on the graphics queue.

[[vk::binding(0, 0)]]
RWStructuredBuffer<float> pattern;

struct PushConstants {
    float mTime;
};

[[vk::push_constant]]
PushConstants pcs;

[shader("compute")]
[numthreads(64, 1, 1)]
void computeMain(uint3 globalId : SV_DispatchThreadID) {
    const uint i = globalId.x;
    const float x = float(i) / 1024.0;
    pattern[i] = 0.5 + 0.5 * sin(x * 50.0 + pcs.mTime);
}
