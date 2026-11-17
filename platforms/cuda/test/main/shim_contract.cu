#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <cuda_runtime.h>

#include "xsched/xsched.h"
#include "xsched/cuda/hal.h"
#include "xsched/cuda/hal/common/context_guard.h"
#include "xsched/cuda/hal/common/driver.h"
#include "xsched/cuda/hal/level1/cuda_queue.h"

using namespace xsched::cuda;
using namespace xsched::preempt;

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
#define CK(x) REQUIRE((x) == 0)

// A non-CHANNEL backend opting into native event dependencies.
class NativeEventQueue : public CudaQueueLv1
{
public:
    using CudaQueueLv1::CudaQueueLv1;
    bool SupportNativeEventWait() const override { return true; }
    bool SupportResumeDropCommands() const override { return false; }
};

__global__ void WaitForRelease(volatile unsigned *flags, unsigned *result,
                               unsigned long long max_clocks)
{
    flags[0] = 1;
    __threadfence_system();
    auto start = clock64();
    while (!flags[1] && clock64() - start < max_clocks) {}
    *result = flags[1] ? 42 : 0;
}

void CheckEventWait(bool native)
{
    cudaStream_t source, target;
    CK(cudaStreamCreateWithFlags(&source, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&target, cudaStreamNonBlocking));
    HwQueueHandle hwq = GetHwQueueHandle(source);
    CK(HwQueueManager::Add(hwq, [&]() -> std::shared_ptr<HwQueue> {
        if (native) return std::make_shared<NativeEventQueue>(source);
        return std::make_shared<CudaQueueLv1>(source);
    }));
    XQueueHandle xq;
    CK(XQueueCreate(&xq, hwq, kPreemptLevelBlock, kQueueCreateFlagNone));
    unsigned *host, *mapped, *result;
    CK(cudaHostAlloc(&host, 2 * sizeof(unsigned), cudaHostAllocMapped));
    CK(cudaHostGetDevicePointer(&mapped, host, 0));
    volatile unsigned *flags = host;
    flags[0] = 0;
    flags[1] = 0;
    CK(cudaMalloc(&result, sizeof(unsigned)));
    cudaEvent_t event, done;
    CK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    CK(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
    int clock_khz;
    CK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
    WaitForRelease<<<1, 1, 0, source>>>(mapped, result, (unsigned long long)clock_khz * 10000);
    CK(cudaGetLastError());
    CK(cudaEventRecord(event, source));
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!flags[0] && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    REQUIRE(flags[0]);
    std::atomic<bool> wait_returned{false};
    std::thread release([&] {
        auto until = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (!wait_returned.load() && std::chrono::steady_clock::now() < until)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        flags[1] = 1;
    });
    auto status = cudaStreamWaitEvent(target, event);
    bool returned_before_release = !flags[1];
    // On the native path the dependent work must still be waiting on the GPU.
    auto record_status = cudaEventRecord(done, target);
    auto query_status = cudaEventQuery(done);
    wait_returned.store(true);
    release.join();
    CK(status); CK(record_status);
    REQUIRE(returned_before_release == native);
    if (native) REQUIRE(query_status == cudaErrorNotReady);
    CK(cudaStreamSynchronize(target));
    CK(cudaStreamSynchronize(source));
    unsigned value = 0;
    CK(cudaMemcpy(&value, result, sizeof(value), cudaMemcpyDeviceToHost));
    REQUIRE(value == 42);
    CK(XQueueDestroy(xq));
    CK(HwQueueDestroy(hwq));
    CK(cudaEventDestroy(event)); CK(cudaEventDestroy(done));
    CK(cudaFree(result)); CK(cudaFreeHost(host));
    CK(cudaStreamDestroy(source)); CK(cudaStreamDestroy(target));
}

int main()
{
    CK(Driver::Init(0));
    CK(XHintSetScheduler(kSchedulerAppManaged, kPolicyUnknown));
    CUcontext context;
    CK(Driver::CtxCreate_v2(&context, 0, 0));
    // Exercise common lifecycle guards without CHANNEL or its preload helper.
    REQUIRE(!CudaContextGuard::AcquireExclusive(nullptr, 0));
    {
        auto guard = CudaContextGuard::AcquireExclusive(context, 0);
        REQUIRE(guard);
        REQUIRE(!CudaContextGuard::AcquireExclusive(context, 0));
        REQUIRE(cuCtxDestroy(context) == CUDA_ERROR_NOT_READY);
        REQUIRE(cuDevicePrimaryCtxReset(0) == CUDA_ERROR_NOT_READY);
        CUcontext other;
        CK(Driver::CtxCreate_v2(&other, 0, 0));
        CK(cuCtxDestroy(other));
        CK(Driver::CtxSetCurrent(context));
        REQUIRE(cuCtxDestroy(context) == CUDA_ERROR_NOT_READY);
    }
    {
        auto reacquired = CudaContextGuard::AcquireExclusive(context, 0);
        REQUIRE(reacquired);
    }
    CheckEventWait(false);
    CheckEventWait(true);
    CK(cuCtxDestroy(context));
    CK(cuDevicePrimaryCtxReset(0));
    puts("PASS: generic context guards, LV1 event fallback and non-CHANNEL native event capability");
}
