#pragma once

// Instrumentation seam over the Tracy client (vendors/tracy). Engine code uses
// only MOE_PROFILE_* so it never includes Tracy directly; when
// MOE_ENABLE_TRACY=OFF every macro below collapses to a no-op, which keeps the
// calls in hot paths free.

#if defined(TRACY_ENABLE)
#include <tracy/Tracy.hpp>

#define MOE_PROFILE_FRAME() FrameMark
#define MOE_PROFILE_FRAME_NAMED(name) FrameMarkNamed(name)
#define MOE_PROFILE_ZONE() ZoneScoped
#define MOE_PROFILE_ZONE_NAMED(name) ZoneScopedN(name)
#define MOE_PROFILE_ZONE_DYNAMIC(name) ZoneTransientN(___tracy_scoped_zone, name, true)
#define MOE_PROFILE_ZONE_COLOR(color) ZoneScopedC(color)
#define MOE_PROFILE_ZONE_TEXT(text, size) ZoneText(text, size)
#define MOE_PROFILE_ZONE_VALUE(value) ZoneValue(value)
#define MOE_PROFILE_THREAD(name) ::tracy::SetThreadName(name)
#define MOE_PROFILE_MESSAGE(text, size) TracyMessage(text, size)
#define MOE_PROFILE_MESSAGE_LITERAL(text) TracyMessageL(text)
#define MOE_PROFILE_PLOT(name, value) TracyPlot(name, value)
#define MOE_PROFILE_ALLOC(ptr, size) TracyAlloc(ptr, size)
#define MOE_PROFILE_FREE(ptr) TracyFree(ptr)
#define MOE_PROFILE_LOCKABLE(type, var) TracyLockable(type, var)
#else
#define MOE_PROFILE_FRAME() ((void)0)
#define MOE_PROFILE_FRAME_NAMED(name) ((void)0)
#define MOE_PROFILE_ZONE() ((void)0)
#define MOE_PROFILE_ZONE_NAMED(name) ((void)0)
#define MOE_PROFILE_ZONE_DYNAMIC(name) ((void)0)
#define MOE_PROFILE_ZONE_COLOR(color) ((void)0)
#define MOE_PROFILE_ZONE_TEXT(text, size) ((void)0)
#define MOE_PROFILE_ZONE_VALUE(value) ((void)0)
#define MOE_PROFILE_THREAD(name) ((void)0)
#define MOE_PROFILE_MESSAGE(text, size) ((void)0)
#define MOE_PROFILE_MESSAGE_LITERAL(text) ((void)0)
#define MOE_PROFILE_PLOT(name, value) ((void)0)
#define MOE_PROFILE_ALLOC(ptr, size) ((void)0)
#define MOE_PROFILE_FREE(ptr) ((void)0)
#define MOE_PROFILE_LOCKABLE(type, var) type var
#endif
