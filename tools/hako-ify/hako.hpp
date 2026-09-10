#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace hako {
    struct MetadataEntry {
        uint16_t pathLength;
        std::string resourcePathAligned;
        uint32_t offset;
        uint32_t size;

        static MetadataEntry from(
                const std::string& path,
                uint32_t offset, uint32_t size,
                bool* err = nullptr) {
            MetadataEntry entry;
            if (path.size() > std::numeric_limits<uint16_t>::max()) {
                if (err) *err = true;
                return entry;
            }

            size_t pathSize = path.size();
            size_t padded = ((pathSize + 3) / 4) * 4;
            entry.resourcePathAligned.assign(padded, '\0');
            std::memcpy(&entry.resourcePathAligned[0], path.data(), pathSize);

            entry.pathLength = static_cast<uint16_t>(padded);
            entry.offset = offset;
            entry.size = size;
            return entry;
        }

        std::vector<uint8_t> toBytes() const {
            // format: [u16 pathLen little-endian][padded path bytes][u32 offset LE][u32 size LE]
            std::vector<uint8_t> bytes;
            bytes.reserve(2 + resourcePathAligned.size() + 4 + 4);

            uint16_t pl = pathLength;
            bytes.push_back(static_cast<uint8_t>(pl & 0xFF));
            bytes.push_back(static_cast<uint8_t>((pl >> 8) & 0xFF));

            bytes.insert(bytes.end(), resourcePathAligned.begin(), resourcePathAligned.end());

            uint32_t off = offset;
            for (size_t i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>((off >> (i * 8)) & 0xFF));

            uint32_t sz = size;
            for (size_t i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>((sz >> (i * 8)) & 0xFF));

            return bytes;
        }

        static MetadataEntry fromBytes(
                const uint8_t* data,
                size_t dataSize, size_t& outConsumed,
                bool* err = nullptr) {
            outConsumed = 0;
            if (dataSize < 2) {
                if (err) *err = true;
                return {};
            }

            uint16_t pl = static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
            outConsumed = 2;

            size_t padded = ((static_cast<size_t>(pl) + 3) / 4) * 4;
            if (dataSize < outConsumed + padded + 4 + 4) {
                if (err) *err = true;
                return {};
            };

            MetadataEntry entry;
            entry.pathLength = pl;
            entry.resourcePathAligned.assign(reinterpret_cast<const char*>(data + outConsumed), padded);
            outConsumed += padded;

            uint32_t off = 0;
            for (size_t i = 0; i < 4; ++i) off |= static_cast<uint32_t>(data[outConsumed + i]) << (i * 8);
            outConsumed += 4;

            uint32_t sz = 0;
            for (size_t i = 0; i < 4; ++i) sz |= static_cast<uint32_t>(data[outConsumed + i]) << (i * 8);
            outConsumed += 4;
            entry.offset = off;
            entry.size = sz;
            return entry;
        }
    };

    struct Metadata {
        std::vector<MetadataEntry> entries;

        static Metadata fromBytes(
                const uint8_t* data,
                size_t dataSize,
                bool* err = nullptr) {
            Metadata metadata;
            size_t offset = 0;

            if (dataSize < 2) {
                if (err) *err = true;
                return metadata;
            }

            uint16_t numEntries =
                    static_cast<uint16_t>(data[0]) |
                    (static_cast<uint16_t>(data[1]) << 8);

            offset += 2;

            metadata.entries.reserve(numEntries);
            for (uint16_t i = 0; i < numEntries; ++i) {
                size_t consumed = 0;
                MetadataEntry entry = MetadataEntry::fromBytes(data + offset, dataSize - offset, consumed, err);
                if (err && *err) {
                    return metadata;
                }
                metadata.entries.push_back(entry);
                offset += consumed;
            }

            return metadata;
        }
    };
}// namespace hako
