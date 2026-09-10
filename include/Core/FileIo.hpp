#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace moe {
    // Whole-file IO helpers shared by the engine (shader binaries, assets in
    // tests). Errors are reported through `error`; the output is untouched on
    // failure.
    bool ReadFileBytes(const char* path, std::vector<uint8_t>& out, std::string& error);
    bool ReadFileText(const char* path, std::string& out, std::string& error);
}// namespace moe
