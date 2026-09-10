#pragma once

#include <utility>

namespace moe {
    // Scope guard: runs the callable when this object goes out of scope
    // (normal exit or early return). Declare guards in the order you want them
    // to run LAST; destruction runs in reverse declaration order, so declare a
    // resource's guard AFTER its dependencies' guards.
    template<typename F>
    class Defer {
    public:
        explicit Defer(F fn)
            : mFn(std::move(fn)) {}

        ~Defer() {
            mFn();
        }

        Defer(const Defer&) = delete;
        Defer& operator=(const Defer&) = delete;

    private:
        F mFn;
    };
}// namespace moe