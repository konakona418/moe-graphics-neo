#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/fancy.hpp"
#include "hako.hpp"

constexpr static char TOOL_NAME[] = "hako-ify";
constexpr static char TOOL_VERSION[] = "0.1.0";
constexpr static char USAGE_MESSAGE[] =
        "Usage: hako-ify <input_directory> <output_file>\n"
        "\n"
        "Arguments:\n"
        "  <input_directory>   The directory containing resources to package.\n"
        "  <output_file>       The output file to create.\n"
        "\n"
        "Example:\n"
        "  hako-ify ./assets resources.pkg\n";

std::string replaceAllReversedSlashes(const std::string& input) {
    std::string output = input;
    size_t pos = 0;
    while ((pos = output.find('\\', pos)) != std::string::npos) {
        output.replace(pos, 1, "/");
        pos += 1;
    }
    return output;
}

int readFile(std::string_view filePath, std::vector<uint8_t>* outData) {
    std::ifstream inFile(filePath.data(), std::ios::binary);
    if (!inFile.is_open()) {
        return -1;
    }

    inFile.seekg(0, std::ios::end);
    size_t fileSize = static_cast<size_t>(inFile.tellg());
    inFile.seekg(0, std::ios::beg);

    if (outData == nullptr) {
        inFile.close();
        return static_cast<int>(fileSize);
    }

    outData->resize(fileSize);

    inFile.read(reinterpret_cast<char*>(outData->data()), fileSize);
    inFile.close();

    return fileSize;
}

std::string prettifyFileSize(size_t sizeInBytes) {
    constexpr size_t KB = 1024;
    constexpr size_t MB = 1024 * KB;
    constexpr size_t GB = 1024 * MB;

    char buffer[32];
    if (sizeInBytes >= GB) {
        std::snprintf(buffer, sizeof(buffer), "%.2f GB", static_cast<float>(sizeInBytes) / GB);
    } else if (sizeInBytes >= MB) {
        std::snprintf(buffer, sizeof(buffer), "%.2f MB", static_cast<float>(sizeInBytes) / MB);
    } else if (sizeInBytes >= KB) {
        std::snprintf(buffer, sizeof(buffer), "%.2f KB", static_cast<float>(sizeInBytes) / KB);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%zu bytes", sizeInBytes);
    }
    return std::string(buffer);
}

using HakoifierMetadataEntry = hako::MetadataEntry;

// Metadata format:
// [u16 numEntries LE][entries...]
// Metadata entry format:
// [u16 pathLen LE][padded path bytes][u32 offset LE][u32 size LE]
struct Hakoifier {
public:
    static bool runPackaging(std::string_view inputDir, std::string_view outputFile) {
        bool err;
        auto resources = enumerateResources(inputDir, &err);
        if (err) {
            std::cerr << fancy::colors::RED << "Failed to enumerate resources in directory: "
                      << inputDir << fancy::colors::RESET << std::endl;
            return false;
        }

        if (resources.empty()) {
            std::cerr << fancy::colors::YELLOW << "No resources found in directory: "
                      << inputDir << fancy::colors::RESET << std::endl;
            return true;
        }

        if (resources.size() > std::numeric_limits<uint16_t>::max()) {
            std::cerr << fancy::colors::RED << "Too many resources to package (max "
                      << std::numeric_limits<uint16_t>::max() << "): "
                      << resources.size() << fancy::colors::RESET << std::endl;
            return false;
        }

        size_t currentOffset = 0;
        std::vector<HakoifierMetadataEntry> metadataEntries;

        std::ofstream outFile(outputFile.data(), std::ios::binary);
        std::ofstream outMetaData(std::string(outputFile) + ".meta",
                                  std::ios::binary);

        if (!outFile.is_open()) {
            std::cerr << fancy::colors::RED << "Failed to open output file: "
                      << outputFile << fancy::colors::RESET << std::endl;
            return false;
        }

        if (!outMetaData.is_open()) {
            std::cerr << fancy::colors::RED << "Failed to open output metadata file: " << outputFile
                      << ".meta" << fancy::colors::RESET << std::endl;
            return false;
        }

        int count = 0;
        int total = static_cast<int>(resources.size());
        std::cout << fancy::colors::CYAN << "Packaging " << total << " resources..."
                  << fancy::colors::RESET << std::endl;

        for (const auto& resourcePath: resources) {
            std::ifstream inFile(resourcePath, std::ios::binary);
            if (!inFile.is_open()) {
                std::cerr << fancy::colors::RED << "Failed to open resource file: " << resourcePath
                          << fancy::colors::RESET << std::endl;
                return false;
            }

            inFile.seekg(0, std::ios::end);
            size_t fileSize = static_cast<size_t>(inFile.tellg());
            inFile.seekg(0, std::ios::beg);

            std::vector<char> buffer(fileSize);
            inFile.read(buffer.data(), fileSize);
            inFile.close();

            auto path = replaceAllReversedSlashes(
                    std::filesystem::relative(
                            resourcePath,
                            inputDir)
                            .string());

            fancy::progressBar(
                    static_cast<float>(++count) / total,
                    fancy::colors::YELLOW + "Packing: " + path +
                            fancy::colors::RESET +
                            " (" + prettifyFileSize(fileSize) + ")");

            outFile.write(buffer.data(), fileSize);

            HakoifierMetadataEntry entry = HakoifierMetadataEntry::from(
                    path,
                    currentOffset,
                    fileSize);
            metadataEntries.push_back(entry);

            currentOffset += fileSize;
        }
        std::cout << std::endl
                  << fancy::colors::CYAN
                  << "Packaging complete. Writing metadata..."
                  << fancy::colors::RESET << std::endl;

        writeU16(outMetaData, metadataEntries.size());
        for (const auto& entry: metadataEntries) {
            auto bytes = entry.toBytes();
            outMetaData.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }

        outFile.close();
        outMetaData.close();

        return true;
    }

private:
    static void writeU16(std::ofstream& outFile, uint16_t value) {
        outFile.put(static_cast<char>(value & 0xFF));
        outFile.put(static_cast<char>((value >> 8) & 0xFF));
    }

    static std::vector<std::string> enumerateResources(
            std::string_view inputDir, bool* err = nullptr) {
        std::filesystem::path inputPath{inputDir};
        std::vector<std::string> resources;

        if (!std::filesystem::exists(inputPath)) {
            if (err) *err = true;
            return resources;
        }

        for (const auto& entry: std::filesystem::recursive_directory_iterator(inputPath)) {
            if (entry.is_regular_file()) {
                resources.push_back(entry.path().string());
            }
        }
        return resources;
    }
};

int runVerify(int argc, char** argv) {
    std::string_view hakoFile = argv[2];
    std::cout << "🔍 ";

    std::cout << fancy::colors::MAGENTA
              << "Verifying metadata from '" << hakoFile
              << "'...\n"
              << fancy::colors::RESET;

    std::vector<uint8_t> buffer;
    if (readFile(std::string(hakoFile) + ".meta", &buffer) == -1) {
        std::cerr << fancy::colors::RED << "Failed to open metadata file: "
                  << hakoFile << ".meta" << fancy::colors::RESET << std::endl;
        return 1;
    }

    bool err;
    hako::Metadata metadata = hako::Metadata::fromBytes(
            buffer.data(),
            buffer.size(),
            &err);

    if (err) {
        std::cerr << fancy::colors::RED << "Failed to parse metadata from file: "
                  << hakoFile << ".meta" << fancy::colors::RESET << '\n';
        return 1;
    }

    std::cout << fancy::colors::GREEN
              << "Metadata verification complete. Found "
              << metadata.entries.size() << " entries.\n"
              << fancy::colors::RESET;

    size_t totalSize = 0;
    for (const auto& entry: metadata.entries) {
        totalSize += entry.size;
        std::cout
                << fancy::colors::YELLOW
                << " + "
                << fancy::colors::MAGENTA
                << entry.resourcePathAligned.substr(0, entry.pathLength) << '\n'
                << fancy::colors::YELLOW
                << " | Offset: "
                << fancy::colors::CYAN
                << entry.offset
                << " (~" + prettifyFileSize(entry.offset) + ")"
                << '\n'
                << fancy::colors::YELLOW
                << " | Size: "
                << fancy::colors::CYAN
                << prettifyFileSize(entry.size)
                << fancy::colors::RESET
                << '\n';
    }

    if (readFile(hakoFile, nullptr) != totalSize) {
        std::cerr << fancy::colors::RED
                  << "Total size mismatch! Expected "
                  << prettifyFileSize(totalSize)
                  << " but file size is different.\n"
                  << fancy::colors::RESET;
        return 1;
    } else {
        std::cout << fancy::colors::YELLOW
                  << " + Total size verified: "
                  << fancy::colors::CYAN
                  << prettifyFileSize(totalSize) << '\n'
                  << fancy::colors::RESET;
    }

    std::cout
            << fancy::colors::GREEN
            << "All tasks completed successfully\n"
            << fancy::colors::RESET;

    return 0;
}

int runPackage(int argc, char** argv) {
    std::string_view inputDir = argv[1];
    std::string_view outputFile = argv[2];

    std::cout << fancy::colors::MAGENTA
              << "📦 Packaging resources from '" << inputDir
              << "' into '" << outputFile
              << "'...\n"
              << fancy::colors::RESET;

    if (!Hakoifier::runPackaging(inputDir, outputFile)) {
        std::cerr << fancy::colors::RED << "Packaging failed.\n"
                  << fancy::colors::RESET;
        return 1;
    }

    std::cout << fancy::colors::GREEN << "All tasks completed successfully.\n"
              << fancy::colors::RESET;
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "📦 ";
        fancy::printRainbow(
                std::string(TOOL_NAME) + " v" + std::string(TOOL_VERSION) +
                std::string(" - A resource packaging tool for moe-graphics\n"));
        std::cout << fancy::colors::CYAN << USAGE_MESSAGE << fancy::colors::RESET;
        return 1;
    }

    std::cout << "📦 ";
    fancy::printRainbow(
            std::string(TOOL_NAME) + " v" + std::string(TOOL_VERSION) +
            std::string(" - A resource packaging tool for moe-graphics\n"));

    if (std::string_view(argv[1]) == "--verify" ||
        std::string_view(argv[1]) == "-v") {
        return runVerify(argc, argv);
    }

    return runPackage(argc, argv);
}