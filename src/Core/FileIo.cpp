#include "Core/FileIo.hpp"

#include <fstream>
#include <iterator>

namespace moe {
    bool ReadFileBytes(const char* path, std::vector<uint8_t>& out, std::string& error) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            error = std::string("failed to open file: ") + path;
            return false;
        }
        const std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<uint8_t> data(static_cast<size_t>(size));
        if (size > 0 && !file.read(reinterpret_cast<char*>(data.data()), size)) {
            error = std::string("failed to read file: ") + path;
            return false;
        }
        out = std::move(data);
        return true;
    }

    bool ReadFileText(const char* path, std::string& out, std::string& error) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            error = std::string("failed to open file: ") + path;
            return false;
        }
        std::string data((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        if (file.bad()) {
            error = std::string("failed to read file: ") + path;
            return false;
        }
        out = std::move(data);
        return true;
    }
}// namespace moe
