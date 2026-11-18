#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/common/options.h"

#if defined(__linux__)
void *xsched::cuda::Driver::GetSymbol(const char *symbol_name)
{
    static void *handle = [] {
        const auto path = FindLibrary(XSCHED_CUDA_LIB_ENV_NAME,
                                      {"libcuda.so", "libcuda.so.1"}, {});
        // CHANNEL observes RM allocations from CUDA initialization onward.
        // Other backends retain their existing deep-binding behavior.
        int flags = GetCudaLv3Implementation() == kCudaLv3ImplementationChannel
                  ? RTLD_NOW | RTLD_LOCAL : XSCHED_RTLD_FLAGS;
        auto library = dlopen(path.c_str(), flags);
        XASSERT(library != nullptr, "fail to dlopen %s", path.c_str());
        return library;
    }();
    auto symbol = RealDlSym(handle, symbol_name);
    XASSERT(symbol != nullptr, "fail to get symbol %s", symbol_name);
    return symbol;
}
#endif
