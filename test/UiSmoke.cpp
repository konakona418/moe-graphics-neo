// UI smoke test: exercises the GPU-free half of the system (flatten, layout,
// hit-testing, two-pass dispatch, event list). No device or renderer is
// created — BeginFrame/EndFrame never touch the GPU; only Render does.

#include <UI/Ui.hpp>

#include <Core/Error.hpp>

#include "TestSupport.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

#define CHECK(cond)                                        \
    do {                                                   \
        if (!(cond)) {                                     \
            std::fprintf(stderr, "UI FAILED: %s (%d)\n",   \
                    #cond, __LINE__);                      \
            return EXIT_FAILURE;                           \
        }                                                  \
    } while (false)

namespace {
    using namespace moe::ui;

    UiFrameDesc MakeFrame(uint32_t width, uint32_t height) {
        UiFrameDesc desc;
        desc.mWidth = width;
        desc.mHeight = height;
        desc.mScale = 1.0f;
        return desc;
    }

    // One pointer step: press, then release, over the given point.
    void Click(Ui& ui, const Element& root, UiFrameDesc desc, const glm::vec2& point,
            std::vector<UiEvent>& lastEvents) {
        desc.mInput.mPointer = point;
        desc.mInput.mPrimaryDown = true;
        desc.mInput.mPrimaryPressed = true;
        ui.BeginFrame(desc);
        ui.EndFrame(root);

        desc.mInput.mPrimaryDown = false;
        desc.mInput.mPrimaryPressed = false;
        desc.mInput.mPrimaryReleased = true;
        ui.BeginFrame(desc);
        ui.EndFrame(root);
        lastEvents = ui.GetEvents();
    }
}// namespace

int main() {
    constexpr const char* kTestName = "UI smoke";
    Ui ui;

    int clicksA = 0;
    int clicksB = 0;

    // Row of two fixed-size buttons with a gap: A at x in [0,10], B at [15,25].
    const Element row = Row(
            {
                    Button("a", [&] { ++clicksA; }, "action.a")
                            .SetSize(Size::Fixed(10.0f), Size::Fixed(10.0f)),
                    Button("b", [&] { ++clicksB; }, "action.b")
                            .SetSize(Size::Fixed(10.0f), Size::Fixed(10.0f)),
            },
            Style{.mGap = 5.0f});

    UiFrameDesc desc = MakeFrame(64, 64);
    std::vector<UiEvent> events;

    // Establish a baseline frame (no pointer).
    ui.BeginFrame(desc);
    ui.EndFrame(row);
    CHECK(ui.GetEvents().empty());

    // Click the second button (x = 17 is inside B).
    Click(ui, row, desc, {17.0f, 5.0f}, events);
    CHECK(clicksA == 0);
    CHECK(clicksB == 1);
    {
        bool clickedB = false;
        for (const UiEvent& event : events) {
            if (event.mType == UiEventType::kClick) {
                clickedB = event.mAction == "action.b";
            }
        }
        CHECK(clickedB);
    }

    // Clicking the gap (x = 12) hits nothing.
    Click(ui, row, desc, {12.0f, 5.0f}, events);
    CHECK(clicksA == 0 && clicksB == 1);
    for (const UiEvent& event : events) {
        CHECK(event.mType != UiEventType::kClick);
    }

    // Hover enter then leave produces the matching events.
    desc.mInput.mPointer = {5.0f, 5.0f};
    ui.BeginFrame(desc);
    ui.EndFrame(row);
    bool hoverEnter = false;
    for (const UiEvent& event : ui.GetEvents()) {
        hoverEnter = hoverEnter || event.mType == UiEventType::kHoverEnter;
    }
    CHECK(hoverEnter);

    desc.mInput.mPointer = {40.0f, 40.0f};
    ui.BeginFrame(desc);
    ui.EndFrame(row);
    bool hoverLeave = false;
    for (const UiEvent& event : ui.GetEvents()) {
        hoverLeave = hoverLeave || event.mType == UiEventType::kHoverLeave;
    }
    CHECK(hoverLeave);

    // Align centers a fixed button in a 64x64 stack: rect [27,37] on both axes.
    const Element centered = Stack({
            Align(Button("c", [&] { ++clicksA; }, "action.c")
                            .SetSize(Size::Fixed(10.0f), Size::Fixed(10.0f)),
                    Alignment::kCenter),
    });
    Click(ui, centered, desc, {32.0f, 32.0f}, events);
    CHECK(clicksA == 1);

    // A point outside the centered button does nothing.
    Click(ui, centered, desc, {5.0f, 5.0f}, events);
    CHECK(clicksA == 1);

    // Captured input (e.g. ImGui owns the pointer) suppresses interaction.
    desc.mInput.mCaptured = true;
    desc.mInput.mPointer = {32.0f, 32.0f};
    desc.mInput.mPrimaryDown = true;
    desc.mInput.mPrimaryPressed = true;
    ui.BeginFrame(desc);
    ui.EndFrame(centered);
    desc.mInput.mPrimaryDown = false;
    desc.mInput.mPrimaryPressed = false;
    desc.mInput.mPrimaryReleased = true;
    ui.BeginFrame(desc);
    ui.EndFrame(centered);
    CHECK(clicksA == 1);

    std::printf("UI smoke passed.\n");
    return EXIT_SUCCESS;
}
