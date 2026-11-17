#pragma once

#include <memory>
#include <mutex>

#include "xsched/cuda/hal/common/context_guard.h"
#include "xsched/cuda/hal/level3/channel_control.h"

namespace xsched::cuda
{

// Exclusive compute-group control; unlike TsgContext, instances are not shared across queues.
class ChannelContext
{
public:
    static std::unique_ptr<ChannelContext> Create(CUcontext context);
    ~ChannelContext();
    ChannelContext(const ChannelContext &) = delete;
    ChannelContext &operator=(const ChannelContext &) = delete;

    void Interrupt();
    void Restore();
    bool IsPaused();

private:
    using Control = channel::Control;
    ChannelContext(std::unique_ptr<CudaContextGuard> guard, Control control, int fd,
                   unsigned client, unsigned subdevice);

    // Released after the destructor has restored and unregistered the RM group.
    const std::unique_ptr<CudaContextGuard> context_guard_;
    const Control control_;
    const int fd_;
    const unsigned client_;
    const unsigned subdevice_;
    unsigned group_ = 0; // Nonzero only after successful RM acquisition.
    bool paused_ = false;
    std::mutex mutex_;
};

} // namespace xsched::cuda
