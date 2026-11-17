#include <cstring>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <cuxtra/cuxtra.h>
#if defined(__linux__)
#include <dlfcn.h>
#endif

#include "xsched/cuda/hal/level3/channel_context.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

namespace
{
#if defined(__linux__) && defined(__x86_64__)
bool ValidatedDriver()
{
    FILE *file = std::fopen("/proc/driver/nvidia/version", "r");
    if (!file) return false;
    char line[512] = {};
    char version[64];
    std::snprintf(version, sizeof(version), " %s ", channel::ValidatedDriverVersion);
    bool supported = std::fgets(line, sizeof(line), file) &&
                     std::strstr(line, version);
    std::fclose(file);
    return supported;
}
#endif
}
std::unique_ptr<ChannelContext> ChannelContext::Create(CUcontext context)
{
#if defined(__linux__) && defined(__x86_64__)
    auto control = reinterpret_cast<Control>(dlsym(RTLD_DEFAULT, channel::ControlSymbol));
    if (!control || !ValidatedDriver()) {
        XWARN("CHANNEL requires the v2 tracker preloaded before CUDA initialization and driver %s",
              channel::ValidatedDriverVersion);
        return nullptr;
    }
    CUcontext current = nullptr;
    CUdevice device;
    CUDA_ASSERT(Driver::CtxGetCurrent(&current));
    XASSERT(current == context, "CHANNEL context does not match the current CUDA context");
    CUDA_ASSERT(Driver::CtxGetDevice(&device));
    auto guard = CudaContextGuard::AcquireExclusive(context, device);
    if (!guard) return nullptr;
    // Private ToolsRm ABI used by the bundled cuXtra; validated on driver 595.71.05.
    const unsigned char bytes[16] = {0x14,0x06,0x18,0x0d,0x72,0xd6,0xcb,0x47,
                                     0xab,0x6e,0xb9,0x63,0xe5,0x1b,0xf4,0xc6};
    CUuuid uuid;
    std::memcpy(&uuid, bytes, sizeof(uuid));
    const void *raw = nullptr;
    if (Driver::GetExportTable(&raw, &uuid) != CUDA_SUCCESS || !raw) {
        XWARN("CHANNEL ToolsRm export table unavailable");
        return nullptr;
    }
    auto table = static_cast<const uintptr_t *>(raw);
    if (table[0] < 3 * sizeof(uintptr_t) || !table[1] || !table[2]) {
        XWARN("CHANNEL ToolsRm ABI unavailable");
        return nullptr;
    }
    unsigned handles[6] = {24, 0, 0, 0, 0, 0};
    unsigned channel_client = 0;
    auto get_handles = reinterpret_cast<CUresult (*)(void *, CUcontext)>(table[1]);
    auto get_channel = reinterpret_cast<CUresult (*)(unsigned *, unsigned *, CUcontext)>(table[2]);
    unsigned group = 0;
    if (get_handles(handles, context) != CUDA_SUCCESS ||
        get_channel(&channel_client, &group, context) != CUDA_SUCCESS) {
        XWARN("CHANNEL cannot query ToolsRm context handles");
        return nullptr;
    }
    unsigned client = handles[3], subdevice = handles[5];
    int fd = cuXtraGetRmControlFd();
    if (!client || !subdevice || !group || client != channel_client || fd < 0) return nullptr;
    auto instance = std::unique_ptr<ChannelContext>(
        new ChannelContext(std::move(guard), control, fd, client, subdevice));
    int result = control(fd, client, subdevice, &group, channel::kAcquire);
    if (result != 0) {
        XWARN("CHANNEL registration failed (%d); no queue was registered", result);
        return nullptr;
    }
    instance->group_ = group;
    return instance;
#else
    UNUSED(context);
    XWARN("CHANNEL requires Linux x86_64");
    return nullptr;
#endif
}
ChannelContext::ChannelContext(std::unique_ptr<CudaContextGuard> guard, Control control, int fd,
                               unsigned client, unsigned subdevice)
    : context_guard_(std::move(guard)), control_(control), fd_(fd), client_(client), subdevice_(subdevice)
{
}
ChannelContext::~ChannelContext()
{
    if (!group_) return;
    Restore();
    int result = control_(fd_, client_, subdevice_, &group_, channel::kRelease);
    XASSERT(result == 0, "CHANNEL release failed: %d", result);
}
void ChannelContext::Interrupt()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (paused_) return;
    int result = 0;
    for (int attempt = 0; attempt < 3; ++attempt) {
        result = control_(fd_, client_, subdevice_, &group_, channel::kPause);
        if (!result) break;
        int recovery = control_(fd_, client_, subdevice_, &group_, channel::kResume);
        XASSERT(recovery == 0, "CHANNEL pause failed (%d), recovery failed (%d); hardware state unknown",
                result, recovery);
    }
    XASSERT(result == 0, "CHANNEL pause failed after bounded retries: %d; channels restored", result);
    paused_ = true;
}
void ChannelContext::Restore()
{
    std::lock_guard<std::mutex> lock(mutex_);
    if (!paused_) return;
    int result = control_(fd_, client_, subdevice_, &group_, channel::kResume);
    XASSERT(result == 0, "CHANNEL resume failed: %d", result);
    paused_ = false;
}

bool ChannelContext::IsPaused()
{
    std::lock_guard<std::mutex> lock(mutex_);
    return paused_;
}
