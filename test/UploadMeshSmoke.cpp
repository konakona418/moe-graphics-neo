#include <Neo/Uploader.hpp>

#include <RHI/CommandList.hpp>
#include <RHI/PipelineCache.hpp>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
    bool ReadBack(moe::rhi::Device& device, const moe::rhi::Buffer& src, moe::rhi::Buffer& outReadback,
            moe::rhi::CommandList& outCmd, std::string& error) {
        moe::rhi::BufferCreateInfo readbackInfo{};
        readbackInfo.mSize = src.GetSize();
        readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
        readbackInfo.mCpuVisible = true;
        if (!device.CreateBuffer(readbackInfo, outReadback)) {
            error = device.GetLastError();
            return false;
        }
        if (!device.CreateCommandList(outCmd)) {
            error = device.GetLastError();
            return false;
        }
        outCmd.Begin();
        outCmd.CopyBuffer(src, outReadback, src.GetSize());
        outCmd.End();
        return device.Submit(outCmd, true);
    }
}// namespace

int main() {
    std::string error;

    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    moe::neo::Uploader uploader;
    moe::neo::UploadedMesh gpu;
    moe::rhi::Buffer vertexReadback;
    moe::rhi::Buffer indexReadback;
    moe::rhi::CommandList vertexCmd;
    moe::rhi::CommandList indexCmd;

    moe::neo::Mesh mesh;
    moe::neo::MeshPrimitive prim;

    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        error = device.GetLastError();
        goto cleanup;
    }
    if (!uploader.Init(device, error)) {
        goto cleanup;
    }

    // CPU quad: 2 triangles, positions + normals + uvs
    mesh.mName = "quad";
    prim.mPositions = {
            {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}};
    prim.mNormals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
    prim.mUv0 = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
    prim.mIndices = {0, 1, 2, 1, 3, 2};
    mesh.mPrimitives.push_back(std::move(prim));

    if (!uploader.UploadMesh(mesh, gpu, error)) {
        goto cleanup;
    }
    if (gpu.mVertexCount != 4 || gpu.mIndexCount != 6 || gpu.mVertexStride != 32) {
        error = "uploaded mesh metadata mismatch";
        goto cleanup;
    }

    // read back the vertex buffer and verify interleaved layout
    if (!ReadBack(device, gpu.mVertexBuffer, vertexReadback, vertexCmd, error)) {
        goto cleanup;
    }
    {
        const auto* data = static_cast<const uint8_t*>(vertexReadback.Map());
        if (data == nullptr) {
            error = "failed to map vertex readback";
            goto cleanup;
        }
        for (uint32_t i = 0; i < 4; ++i) {
            glm::vec3 pos, nrm;
            glm::vec2 uv;
            std::memcpy(&pos, data + i * 32 + 0, sizeof(glm::vec3));
            std::memcpy(&nrm, data + i * 32 + 12, sizeof(glm::vec3));
            std::memcpy(&uv, data + i * 32 + 24, sizeof(glm::vec2));
            if (pos != mesh.mPrimitives[0].mPositions[i]
                    || nrm != mesh.mPrimitives[0].mNormals[i]
                    || uv != mesh.mPrimitives[0].mUv0[i]) {
                error = "vertex data mismatch at vertex " + std::to_string(i);
                vertexReadback.Unmap();
                goto cleanup;
            }
        }
        vertexReadback.Unmap();
    }

    // read back the index buffer
    if (!ReadBack(device, gpu.mIndexBuffer, indexReadback, indexCmd, error)) {
        goto cleanup;
    }
    {
        const auto* data = static_cast<const uint8_t*>(indexReadback.Map());
        if (data == nullptr) {
            error = "failed to map index readback";
            goto cleanup;
        }
        uint32_t idx[6];
        std::memcpy(idx, data, sizeof(idx));
        for (uint32_t i = 0; i < 6; ++i) {
            if (idx[i] != mesh.mPrimitives[0].mIndices[i]) {
                error = "index data mismatch at " + std::to_string(i);
                indexReadback.Unmap();
                goto cleanup;
            }
        }
        indexReadback.Unmap();
    }

    std::printf("UploadMesh smoke passed.\n");

cleanup:
    indexCmd.Destroy();
    vertexCmd.Destroy();
    indexReadback.Destroy();
    vertexReadback.Destroy();
    gpu.Destroy();
    cache.Destroy();
    device.Destroy();

    if (!error.empty()) {
        std::fprintf(stderr, "UploadMesh smoke FAILED: %s\n", error.c_str());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}