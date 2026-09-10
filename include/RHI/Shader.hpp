#pragma once

#include "RHI/RHICommon.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace moe::rhi {
    struct ShaderImpl;
    struct ShaderProgramImpl;
    class DefaultPipelineCache;

    // A single compiled shader stage (.spv) plus its reflection data. The
    // shader remembers its source path so a PipelineCache can reload it.
    class Shader {
    public:
        Shader();
        ~Shader();

        Shader(const Shader&) = delete;
        Shader& operator=(const Shader&) = delete;

        // Reads and reflects the .spv file. Returns false on failure; the
        // reason is available via moe::Error::Get().
        bool Load(const char* spvPath, ShaderStage stage);

        // Re-reads the .spv file and re-reflects (used by hot reload).
        bool Reload();

        const std::string& GetPath() const;
        ShaderStage GetStage() const;
        const ShaderReflection& GetReflection() const;
        // The raw .spv bytes (used by the pipeline cache).
        const std::vector<char>& GetCode() const;
        // Stable hash of the .spv content; changes when the file changes.
        uint64_t GetContentHash() const;

    private:
        friend class DefaultPipelineCache;

        std::unique_ptr<ShaderImpl> mImpl;
    };

    // A bundle of shader stages that forms one pipeline program (e.g. VS+FS,
    // or a single compute stage). Shaders must outlive the program.
    class ShaderProgram {
    public:
        ShaderProgram();
        ~ShaderProgram();

        ShaderProgram(const ShaderProgram&) = delete;
        ShaderProgram& operator=(const ShaderProgram&) = delete;

        bool AddShader(const Shader& shader);
        const Shader* GetStage(ShaderStage stage) const;
        bool HasStage(ShaderStage stage) const;
        uint32_t GetStageCount() const;
        // Combines all stage content hashes; changes on any reload.
        uint64_t GetContentHash() const;

    private:
        std::unique_ptr<ShaderProgramImpl> mImpl;
    };
}// namespace moe::rhi