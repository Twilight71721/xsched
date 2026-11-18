#include "channel_tracker_internal.h"
#include <cerrno>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace xsched::cuda::channel::tracker
{
int RawIoctl(int fd, unsigned long request, void *argument)
{
    return syscall(SYS_ioctl, fd, request, argument);
}
}

// Called by the x86-64 ABI trampoline. Keep the third register as an opaque
// word until both the request and device have been identified as NVIDIA RM.
extern "C" __attribute__((visibility("hidden")))
int xsched_cuda_tracker_ioctl(int fd, unsigned long request, uintptr_t argument) noexcept
{
    using namespace xsched::cuda::channel::tracker;
    if (request == _IOWR('F', 0x2b, Alloc21) || request == _IOWR('F', 0x2b, Alloc64) ||
        request == _IOWR('F', 0x29, Free)) {
        int saved_errno = errno;
        struct stat device{};
        // NVIDIA's control/GPU nodes use character-device major 195. Do not
        // parse an unrelated device's request even if its ioctl number matches.
        bool nvidia = fstat(fd, &device) == 0 && S_ISCHR(device.st_mode) && major(device.st_rdev) == 195;
        errno = saved_errno;
        if (nvidia) return TrackIoctl(fd, request, reinterpret_cast<void *>(argument));
    }
    return syscall(SYS_ioctl, fd, request, argument);
}

extern "C" xsched::cuda::channel::ControlResult
xsched_cuda_channel_control_v3(int fd, unsigned client, unsigned subdevice,
                               unsigned *handle, int operation)
{
    return xsched::cuda::channel::tracker::ControlGroup(fd, client, subdevice, handle, operation);
}
