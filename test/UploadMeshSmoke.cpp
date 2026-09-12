#include <Neo/TransferManager.hpp>

#include <Core/Defer.hpp>
#include "TestSupport.hpp"
#include <Core/Error.hpp>
#include <RHI/CommandList.hpp>
#include <RHI/PipelineCache.hpp>

#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
    bool ReadBack(moe::rhi::Device& device, const moe::rhi::Buffer& src, moe::rhi::Buffer& outReadback,
            moe::rhi::CommandList& outCmd) {
        moe::rhi::BufferCreateInfo readbackInfo{};
        readbackInfo.mSize = src.GetSize();
        readbackInfo.mUsage = moe::rhi::BufferUsage::kTransferDst;
        readbackInfo.mCpuVisible = true;
        if (!device.CreateBuffer(readbackInfo, outReadback)) {
            return false;
        }
        if (!device.CreateCommandList(outCmd)) {
            return false;
        }
        outCmd.Begin();
        outCmd.CopyBuffer(src, outReadback, src.GetSize());
        outCmd.End();
        return device.Submit(outCmd, true);
    }
}// namespace

int main() {
    constexpr const char* kTestName = "UploadMesh smoke";
    moe::rhi::Device device;
    moe::rhi::DefaultPipelineCache cache;
    moe::rhi::DeviceCreateInfo deviceInfo{};
    deviceInfo.mPipelineCache = &cache;
    deviceInfo.mEnableValidation = true;

    moe::Scheduler scheduler;
    moe::neo::TransferManager uploader;
    moe::neo::UploadedMesh gpu;
    moe::rhi::Buffer vertexReadback;
    moe::rhi::Buffer indexReadback;
    moe::rhi::CommandList vertexCmd;
    moe::rhi::CommandList indexCmd;

    moe::neo::Mesh mesh;
    moe::neo::MeshPrimitive prim;


    moe::Defer cleanup([&] {
        indexCmd.Destroy();
        vertexCmd.Destroy();
        indexReadback.Destroy();
        vertexReadback.Destroy();
        gpu.Destroy();
        uploader.Shutdown();
        scheduler.Shutdown();
        cache.Destroy();
        device.Destroy();
    });
    if (!moe::rhi::Device::Create(deviceInfo, device)) {
        return moe::test::Fail(kTestName);
    }
    if (!scheduler.Init(2) || !uploader.Init(device, scheduler)) {
        return moe::test::Fail(kTestName);
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

    if (!uploader.UploadMesh(mesh, gpu)) {
        return moe::test::Fail(kTestName);
    }
    if (gpu.mVertexCount != 4 || gpu.mIndexCount != 6 || gpu.mVertexStride != 32) {
        return moe::test::Fail(kTestName, "uploaded mesh metadata mismatch");
    }

    // read back the vertex buffer and verify interleaved layout
    if (!ReadBack(device, gpu.mVertexBuffer, vertexReadback, vertexCmd)) {
        return moe::test::Fail(kTestName);
    }
    {
        const auto* data = static_cast<const uint8_t*>(vertexReadback.Map());
        if (data == nullptr) {
            return moe::test::Fail(kTestName, "failed to map vertex readback");
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
                vertexReadback.Unmap();
                return moe::test::Fail(kTestName);
            }
        }
        vertexReadback.Unmap();
    }

    // read back the index buffer
    if (!ReadBack(device, gpu.mIndexBuffer, indexReadback, indexCmd)) {
        return moe::test::Fail(kTestName);
    }
    {
        const auto* data = static_cast<const uint8_t*>(indexReadback.Map());
        if (data == nullptr) {
            return moe::test::Fail(kTestName, "failed to map index readback");
        }
        uint32_t idx[6];
        std::memcpy(idx, data, sizeof(idx));
        for (uint32_t i = 0; i < 6; ++i) {
            if (idx[i] != mesh.mPrimitives[0].mIndices[i]) {
                indexReadback.Unmap();
                return moe::test::Fail(kTestName);
            }
        }
        indexReadback.Unmap();
    }

    std::printf("UploadMesh smoke passed.\n");

    return EXIT_SUCCESS;
}
