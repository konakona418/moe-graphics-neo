#pragma once

#include <Core/Error.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace moe::test {
    // Reports "<testName> FAILED: <message>" and returns EXIT_FAILURE. Smoke
    // tests call this from main after registering their teardown Defer guard,
    // so early returns stay clean.
    inline int Fail(const char* testName, const std::string& message) {
        std::fprintf(stderr, "%s FAILED: %s\n", testName, message.c_str());
        return EXIT_FAILURE;
    }

    // Failure carrying the engine's last error (moe::Error).
    inline int Fail(const char* testName) {
        return Fail(testName, moe::Error::Get());
    }
}// namespace moe::test
