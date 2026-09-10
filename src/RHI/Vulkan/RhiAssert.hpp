#pragma once

#include <cstdio>
#include <cstdlib>

// RHI uses no exceptions: on an invariant violation (e.g. an enum value we
// have no mapping for), abort with a message instead of silently failing.
#define MOE_RHI_ASSERT(cond, msg)                                                       \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::fprintf(stderr, "[moe-rhi] assert failed: %s (%s:%d)\n",               \
                    msg, __FILE__, __LINE__);                                           \
            std::abort();                                                               \
        }                                                                               \
    } while (false)