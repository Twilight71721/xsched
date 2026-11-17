#include "channel_tracker_internal.h"
#include <cstdarg>
#include <cstring>
#include <dlfcn.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace xsched::cuda::channel::tracker
{
int RawIoctl(int fd, unsigned long request, void *argument)
{
    return syscall(SYS_ioctl, fd, request, argument);
}
}

extern "C" int ioctl(int fd, unsigned long request, ...) noexcept
{
    va_list args;
    va_start(args, request);
    void *argument = va_arg(args, void *);
    va_end(args);
    return xsched::cuda::channel::tracker::TrackIoctl(fd, request, argument);
}

extern "C" int xsched_cuda_channel_control_v2(int fd, unsigned client, unsigned subdevice,
                                             unsigned *handle, int operation)
{
    return xsched::cuda::channel::tracker::ControlGroup(fd, client, subdevice, handle, operation);
}

// XSched normally deep-binds the real driver, which bypasses ioctl interposition.
// Only this explicitly preloaded experimental helper changes libcuda loading.
extern "C" void *dlopen(const char *filename, int flags) noexcept
{
    using Open = void *(*)(const char *, int);
    static auto real_open = reinterpret_cast<Open>(dlsym(RTLD_NEXT, "dlopen"));
    if (!real_open) return nullptr;
    if (filename) {
        const char *name = std::strrchr(filename, '/');
        name = name ? name + 1 : filename;
        if (std::strcmp(name, "libcuda.so") == 0 || std::strncmp(name, "libcuda.so.", 11) == 0)
            flags &= ~RTLD_DEEPBIND;
    }
    return real_open(filename, flags);
}
