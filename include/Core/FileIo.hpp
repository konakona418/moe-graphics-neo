#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe {
    // Whole-file IO helpers shared by the engine (shader binaries, assets in
    // tests). Failures are recorded in moe::Error; the output is untouched on
    // failure.
    bool ReadFileBytes(const char* path, std::vector<uint8_t>& out);
    bool ReadFileText(const char* path, std::string& out);
}// namespace moe
