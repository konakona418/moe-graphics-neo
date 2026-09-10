// ImHex Pattern for hako-ify metadata
import std.mem;

struct MetadataRegistry {
    u16 name_len;
    char name[name_len];
    u32 offset;
    u32 size;
};

std::mem::Section regsec = std::mem::create_section("Regs");

u32 offset = 0x00;

u16 num_registry @0x00;
MetadataRegistry registries[num_registry] @0x02;