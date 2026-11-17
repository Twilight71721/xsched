#pragma once

#include <memory>

#include "xsched/cuda/hal/common/cuda.h"

namespace xsched::cuda
{

// Opt-in, exclusive context ownership for backends retaining driver resources.
// Keep the guard alive until those resources have been released.
class CudaContextGuard
{
public:
    static std::unique_ptr<CudaContextGuard> AcquireExclusive(CUcontext context, CUdevice device);
    static CUresult DestroyContext(CUcontext context, bool legacy);
    static CUresult ResetPrimaryContext(CUdevice device, bool legacy);
    ~CudaContextGuard();
    CudaContextGuard(const CudaContextGuard &) = delete;
    CudaContextGuard &operator=(const CudaContextGuard &) = delete;

private:
    explicit CudaContextGuard(CUcontext context) : context_(context) {}
    const CUcontext context_;
    bool registered_ = false;
};

} // namespace xsched::cuda
