#include <utility>
#include "xsched/cuda/hal/common/levels.h"

#include "xsched/cuda/hal/level3/channel.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

std::shared_ptr<HwQueue> CudaQueueLv3Channel::Create(CUstream stream)
{
    XResult status;
    return Create(stream, status);
}

std::shared_ptr<HwQueue> CudaQueueLv3Channel::Create(CUstream stream, XResult &status)
{
    status = kXSchedErrorNotSupported;
    CUdevice device;
    CUDA_ASSERT(Driver::CtxGetDevice(&device));
    if (GetArch(device) != 120) {
        XWARN("experimental CHANNEL backend requires sm120");
        return nullptr;
    }
    CUcontext context = nullptr;
    CUDA_ASSERT(Driver::StreamGetCtx(stream, &context));
    auto channel_context = ChannelContext::Create(context, status);
    if (!channel_context) return nullptr;
    return std::shared_ptr<CudaQueueLv3Channel>(
        new CudaQueueLv3Channel(stream, std::move(channel_context)));
}

CudaQueueLv3Channel::CudaQueueLv3Channel(CUstream stream,
                                       std::unique_ptr<ChannelContext> channel_context)
    : CudaQueueLv1(stream), channel_context_(std::move(channel_context))
{
    XWARN("CHANNEL owns the context's compute group; independent copies may continue; DropCommands unsupported");
}

void CudaQueueLv3Channel::Deactivate()
{
    // At Level 2 this backend uses the stronger, state-preserving Level 3 operation.
    channel_context_->Interrupt();
}

void CudaQueueLv3Channel::Interrupt() { channel_context_->Interrupt(); }
void CudaQueueLv3Channel::Restore(const CommandLog &) { channel_context_->Restore(); }
void CudaQueueLv3Channel::Reactivate(const CommandLog &) { channel_context_->Restore(); }
void CudaQueueLv3Channel::SynchronizeOnSuspend()
{
    // Synchronous RM disable confirms suspension without completing retained work.
    if (level_ >= kPreemptLevelDeactivate) channel_context_->ConfirmPaused();
    else CudaQueueLv1::Synchronize();
}
void CudaQueueLv3Channel::OnPreemptLevelChange(XPreemptLevel level)
{
    XASSERT(level >= kPreemptLevelBlock && level <= kPreemptLevelInterrupt,
            "unsupported CHANNEL level: %d", level);
    level_ = level;
}
