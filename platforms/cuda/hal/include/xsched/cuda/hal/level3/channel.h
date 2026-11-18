#pragma once

#include "xsched/cuda/hal/level3/channel_context.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"

namespace xsched::cuda
{

// Experimental Linux RM backend. One managed queue owns the context's compute group.
class CudaQueueLv3Channel : public CudaQueueLv1
{
public:
    static std::shared_ptr<preempt::HwQueue> Create(CUstream stream);
    static std::shared_ptr<preempt::HwQueue> Create(CUstream stream, XResult &status);
    ~CudaQueueLv3Channel() override = default;
    void Deactivate() override;
    void Interrupt() override;
    void Restore(const preempt::CommandLog &) override;
    void Reactivate(const preempt::CommandLog &) override;
    void SynchronizeOnSuspend() override;
    void OnPreemptLevelChange(XPreemptLevel level) override;
    bool SupportNativeEventWait() const override { return true; }
    bool SupportDynamicLevel() override { return true; }
    XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }
    bool SupportResumeDropCommands() const override { return false; }

private:
    CudaQueueLv3Channel(CUstream stream, std::unique_ptr<ChannelContext> channel_context);
    std::unique_ptr<ChannelContext> channel_context_;
};

} // namespace xsched::cuda
