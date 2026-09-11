#include "Core/FileIo.hpp"

#include "Core/Error.hpp"
#include <Core/Profile.hpp>

#include <fstream>
#include <iterator>
#include <utility>

namespace moe {
    bool ReadFileBytes(const char* path, std::vector<uint8_t>& out) {
        MOE_PROFILE_ZONE();
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            return Fail(std::string("failed to open file: ") + path);
        }
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(static_cast<size_t>(size));
        if (size > 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) {
            return Fail(std::string("failed to read file: ") + path);
        }
        out = std::move(data);
        return true;
    }

    bool ReadFileText(const char* path, std::string& out) {
        MOE_PROFILE_ZONE();
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            return Fail(std::string("failed to open file: ") + path);
        }
        std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) {
            return Fail(std::string("failed to read file: ") + path);
        }
        out = std::move(data);
        return true;
    }
}// namespace moe
