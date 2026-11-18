#include <memory>
#include <new>

#include "xsched/cuda/hal.h"
#include "xsched/cuda/hal/common/options.h"
#include "xsched/cuda/hal/level3/channel.h"
#include "xsched/xsched.h"

using namespace xsched::cuda;
using xsched::preempt::HwQueueManager;

struct CudaTaskDomain
{
    CUcontext ctx = nullptr;
    CUstream stream = nullptr;
    HwQueueHandle hwq = 0;
    XQueueHandle xq = 0;
    bool owns_cuda = false;
};

namespace
{
// A successful cuCtxCreate also pushes one entry on the calling thread's stack.
class PopContext
{
public:
    explicit PopContext(CUcontext ctx): ctx_(ctx) {}
    ~PopContext()
    {
        CUcontext popped;
        XASSERT(Driver::CtxPopCurrent_v2(&popped) == CUDA_SUCCESS && popped == ctx_,
                "task domain CUDA context stack is unbalanced");
    }
private:
    const CUcontext ctx_;
};

XResult InitializeBackend()
{
#if defined(__linux__) && defined(__x86_64__)
    if (GetCudaLv3Implementation() != kCudaLv3ImplementationChannel) return kXSchedErrorNotSupported;
    if (Driver::Init(0) != CUDA_SUCCESS) return kXSchedErrorHardware;
    int version;
    if (Driver::DriverGetVersion(&version) != CUDA_SUCCESS) return kXSchedErrorHardware;
    if (version >= 13000) return kXSchedSuccess;
#endif
    return kXSchedErrorNotSupported;
}

XResult BindQueue(CudaTaskDomain &task)
{
    XResult acquisition = kXSchedSuccess;
    const auto hwq = GetHwQueueHandle(task.stream);
    auto result = HwQueueManager::Add(hwq, [&] {
        return CudaQueueLv3Channel::Create(task.stream, acquisition);
    }, true);
    if (result != kXSchedSuccess) return acquisition != kXSchedSuccess ? acquisition : result;

    result = XQueueCreate(&task.xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone);
    if (result != kXSchedSuccess) {
        XASSERT(HwQueueDestroy(hwq) == kXSchedSuccess, "task domain bind rollback failed");
        return result;
    }
    task.hwq = hwq;
    return kXSchedSuccess;
}
}

EXPORT_C_FUNC XResult CudaTaskDomainCreate(CudaTaskDomain **out, int device_ordinal)
{
    if (!out) return kXSchedErrorInvalidValue;
    *out = nullptr;
    if (device_ordinal < 0) return kXSchedErrorInvalidValue;
    auto result = InitializeBackend();
    if (result != kXSchedSuccess) return result;
    CUdevice device;
    auto err = Driver::DeviceGet(&device, device_ordinal);
    if (err == CUDA_ERROR_INVALID_DEVICE) return kXSchedErrorInvalidValue;
    if (err != CUDA_SUCCESS) return kXSchedErrorHardware;

    std::unique_ptr<CudaTaskDomain> task(new (std::nothrow) CudaTaskDomain);
    if (!task || Driver::CtxCreate_v4(&task->ctx, nullptr, 0, device) != CUDA_SUCCESS)
        return kXSchedErrorHardware;
    task->owns_cuda = true;
    {
        PopContext pop(task->ctx);
        if (Driver::StreamCreate(&task->stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS) {
            result = BindQueue(*task);
            if (result != kXSchedSuccess)
                XASSERT(Driver::StreamDestroy_v2(task->stream) == CUDA_SUCCESS, "stream rollback failed");
        } else result = kXSchedErrorHardware;
    }
    if (result != kXSchedSuccess) {
        XASSERT(CudaContextGuard::DestroyContext(task->ctx, false) == CUDA_SUCCESS, "context rollback failed");
        return result;
    }
    *out = task.release();
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult CudaTaskDomainAttach(CudaTaskDomain **out, CUstream stream)
{
    if (!out) return kXSchedErrorInvalidValue;
    *out = nullptr;
    if (!stream || stream == CU_STREAM_LEGACY || stream == CU_STREAM_PER_THREAD)
        return kXSchedErrorInvalidValue;
    auto result = InitializeBackend();
    if (result != kXSchedSuccess) return result;
    CUcontext ctx = nullptr;
    CUgreenCtx green = nullptr;
    if (Driver::StreamGetCtx_v2(stream, &ctx, &green) != CUDA_SUCCESS || !ctx)
        return kXSchedErrorInvalidValue;
    if (green) return kXSchedErrorNotSupported;
    if (Driver::CtxPushCurrent_v2(ctx) != CUDA_SUCCESS) return kXSchedErrorHardware;
    PopContext pop(ctx);
    const auto state = Driver::StreamQuery(stream);
    if (state == CUDA_ERROR_NOT_READY) return kXSchedErrorNotAllowed;
    if (state != CUDA_SUCCESS) return kXSchedErrorHardware;

    std::unique_ptr<CudaTaskDomain> task(new (std::nothrow) CudaTaskDomain);
    if (!task) return kXSchedErrorHardware;
    task->ctx = ctx;
    task->stream = stream;
    result = BindQueue(*task);
    if (result == kXSchedSuccess) *out = task.release();
    return result;
}

EXPORT_C_FUNC XResult CudaTaskDomainGetQueue(const CudaTaskDomain *task, XQueueHandle *xq)
{
    if (!task || !xq) return kXSchedErrorInvalidValue;
    if (!task->xq) return kXSchedErrorNotAllowed;
    *xq = task->xq;
    return kXSchedSuccess;
}

EXPORT_C_FUNC XResult CudaTaskDomainWithStream(CudaTaskDomain *task, CudaTaskWork work, void *argument)
{
    if (!task || !work) return kXSchedErrorInvalidValue;
    if (!task->xq) return kXSchedErrorNotAllowed;
    if (Driver::CtxPushCurrent_v2(task->ctx) != CUDA_SUCCESS) return kXSchedErrorHardware;
    PopContext pop(task->ctx);
    return work(task->stream, argument);
}

EXPORT_C_FUNC XResult CudaTaskDomainDestroy(CudaTaskDomain **owner)
{
    if (!owner || !*owner) return kXSchedErrorInvalidValue;
    auto &task = **owner;
    if (Driver::CtxPushCurrent_v2(task.ctx) != CUDA_SUCCESS) return kXSchedErrorHardware;
    {
        PopContext pop(task.ctx);
        if (task.xq) {
            auto result = XQueueDestroy(task.xq); // Resume and drain retained commands.
            if (result != kXSchedSuccess) return result;
            task.xq = 0;
        }
        if (task.hwq) {
            if (Driver::StreamSynchronize(task.stream) != CUDA_SUCCESS) return kXSchedErrorHardware;
            auto result = HwQueueDestroy(task.hwq);
            if (result != kXSchedSuccess) return result;
            task.hwq = 0;
        }
        if (task.owns_cuda && task.stream) {
            if (Driver::StreamDestroy_v2(task.stream) != CUDA_SUCCESS) return kXSchedErrorHardware;
            task.stream = nullptr;
        }
    }
    if (task.owns_cuda && CudaContextGuard::DestroyContext(task.ctx, false) != CUDA_SUCCESS)
        return kXSchedErrorHardware;
    delete *owner;
    *owner = nullptr;
    return kXSchedSuccess;
}
