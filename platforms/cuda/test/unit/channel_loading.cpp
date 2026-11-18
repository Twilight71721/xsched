#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/hal/common/driver.h"

using namespace xsched::cuda;
#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

int main(int argc, char **argv)
{
    CHECK(argc == 2);
    bool first = std::strcmp(argv[1], "driver-first") == 0;
    bool deep = std::strcmp(argv[1], "deepbound-first") == 0;
    CHECK(first || deep || std::strcmp(argv[1], "hal-first") == 0);
    void *library = nullptr;
    if (first || deep) {
        const auto path = FindLibrary(XSCHED_CUDA_LIB_ENV_NAME, {"libcuda.so.1", "libcuda.so"}, {});
        library = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL | (deep ? RTLD_DEEPBIND : 0));
        CHECK(library != nullptr);
        auto init = reinterpret_cast<CUresult (*)(unsigned)>(dlsym(library, "cuInit"));
        CHECK(init && init(0) == CUDA_SUCCESS);
    }
    CHECK(Driver::Init(0) == CUDA_SUCCESS);
    CUcontext context;
    CHECK(Driver::CtxCreate_v2(&context, 0, 0) == CUDA_SUCCESS);
    CUstream stream;
    CHECK(Driver::StreamCreate(&stream, CU_STREAM_NON_BLOCKING) == CUDA_SUCCESS);
    HwQueueHandle hwq = 0;
    XResult result = CudaQueueCreate(&hwq, stream);
    if (deep) {
        // External DEEPBIND bypasses the tracker; reject missing history rather
        // than modifying another library's load flags or silently using LV1.
        CHECK(result != kXSchedSuccess && hwq == 0);
    } else {
        CHECK(result == kXSchedSuccess);
        XQueueHandle xq;
        CHECK(XQueueCreate(&xq, hwq, kPreemptLevelInterrupt, kQueueCreateFlagNone) == kXSchedSuccess);
        CHECK(XQueueSuspend(xq, kQueueSuspendFlagSyncHwQueue) == kXSchedSuccess);
        CHECK(XQueueResume(xq, kQueueResumeFlagNone) == kXSchedSuccess);
        CHECK(XQueueDestroy(xq) == kXSchedSuccess && HwQueueDestroy(hwq) == kXSchedSuccess);
    }
    CHECK(Driver::StreamDestroy_v2(stream) == CUDA_SUCCESS);
    CHECK(Driver::CtxDestroy_v2(context) == CUDA_SUCCESS);
    if (library) CHECK(dlclose(library) == 0);
    printf("PASS: CHANNEL loading mode %s\n", argv[1]);
}
