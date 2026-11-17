#include <map>
#include <mutex>

#include "xsched/cuda/hal/common/context_guard.h"
#include "xsched/cuda/hal/common/cuda_assert.h"

using namespace xsched::cuda;

namespace
{
struct ContextOwners {
    std::mutex mutex;
    std::map<CUcontext, CUdevice> contexts;
};
ContextOwners &Owners() { static auto *owners = new ContextOwners; return *owners; }
}

std::unique_ptr<CudaContextGuard> CudaContextGuard::AcquireExclusive(CUcontext context, CUdevice device)
{
    if (!context) return nullptr;
    // The guard must outlive the lock on all return and exception paths.
    auto guard = std::unique_ptr<CudaContextGuard>(new CudaContextGuard(context));
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    if (!owners.contexts.emplace(context, device).second) {
        XWARN("CUDA context already has an exclusive owner");
        return nullptr;
    }
    guard->registered_ = true;
    return guard;
}

CudaContextGuard::~CudaContextGuard()
{
    if (!registered_) return;
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    owners.contexts.erase(context_);
}

CUresult CudaContextGuard::DestroyContext(CUcontext context, bool legacy)
{
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    if (owners.contexts.count(context)) {
        XWARN("release the CUDA context's backend resources before destroying it");
        return CUDA_ERROR_NOT_READY;
    }
    return legacy ? Driver::CtxDestroy(context) : Driver::CtxDestroy_v2(context);
}

CUresult CudaContextGuard::ResetPrimaryContext(CUdevice device, bool legacy)
{
    auto &owners = Owners();
    std::lock_guard<std::mutex> lock(owners.mutex);
    for (const auto &[context, owned_device] : owners.contexts) {
        UNUSED(context);
        if (owned_device == device) {
            XWARN("release backend context resources on this device before primary context reset");
            return CUDA_ERROR_NOT_READY;
        }
    }
    return legacy ? Driver::DevicePrimaryCtxReset(device) : Driver::DevicePrimaryCtxReset_v2(device);
}
