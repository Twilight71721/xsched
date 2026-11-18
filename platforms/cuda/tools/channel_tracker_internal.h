#pragma once

// Linux RM ABI subset from NVIDIA open-gpu-kernel-modules, tag 595.71.05:
// nvos.h and ctrl/ctrl2080/ctrl2080fifo.h (MIT). Keep layout assertions below.

// Private implementation details; not installed as part of the HAL API.
#include <cstdint>
#include <cstddef>
#include <map>
#include <mutex>
#include <vector>
#include <sys/ioctl.h>
#include "xsched/cuda/hal/level3/channel_control.h"

namespace xsched::cuda::channel::tracker
{
struct Alloc21 { uint32_t client, parent, object, cls; uint64_t params; uint32_t size, status; };
struct Alloc64 { uint32_t client, parent, object, cls; uint64_t params, rights; uint32_t size, flags, status; };
struct Free { uint32_t client, parent, object, status; };
struct RmControl { uint32_t client, object, cmd, flags; uint64_t params; uint32_t size, status; };
struct Disable {
    uint8_t disable;
    uint32_t count;
    uint8_t only_scheduling, rewind;
    alignas(8) uint64_t event;
    uint32_t clients[64], channels[64];
};
static_assert(sizeof(Alloc21) == 32 && sizeof(Alloc64) == 48 && sizeof(Free) == 16);
static_assert(sizeof(RmControl) == 32 && sizeof(Disable) == 536);
static_assert(offsetof(Disable, event) == 16 && offsetof(Disable, clients) == 24);
using Key = std::pair<uint32_t, uint32_t>;
struct Object { uint32_t parent, cls; };
struct Owner {
    GroupState state = GroupState::Running;
    int fd = -1;
    unsigned subdevice = 0;
    std::vector<uint32_t> channels;
};
struct State {
    std::mutex mutex;
    std::map<Key, Object> objects;
    std::map<Key, Owner> owners;
};

State &GetState();
int RawIoctl(int fd, unsigned long request, void *argument);
int TrackIoctl(int fd, unsigned long request, void *argument);
ControlResult ControlGroup(int fd, unsigned client, unsigned subdevice,
                           unsigned *handle, int operation);
} // namespace xsched::cuda::channel::tracker
