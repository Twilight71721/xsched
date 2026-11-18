#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <cuda.h>
#include <cuda_runtime.h>
#include "xsched/cuda/hal.h"
#include "xsched/xsched.h"

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)
#define OK(x) CHECK((x) == 0)
using namespace std::chrono_literals;

struct Counters
{
    volatile unsigned progress;
    volatile unsigned stop;
    unsigned long long sum;
};

__global__ void Compute(Counters *c)
{
    unsigned iterations = 0;
    unsigned long long sum = 0;
    while (!c->stop && iterations < 100000) {
        auto start = clock64();
        while (clock64() - start < 100000) {}
        sum += ++iterations;
        c->progress = iterations;
        __threadfence_system();
    }
    c->sum = sum;
}

struct Task
{
    CudaTaskDomain *domain = nullptr;
    XQueueHandle queue = 0;
    Counters *host = nullptr;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
};

XResult Launch(CUstream stream, void *arg)
{
    auto &task = *static_cast<Task *>(arg);
    task.stream = stream;
    OK(cuCtxGetCurrent(&task.context));
    HwQueueHandle hwq;
    XQueueHandle observed;
    OK(CudaQueueGet(&hwq, stream));
    OK(XQueueGet(&observed, hwq));
    CHECK(observed == task.queue);
    Counters *device;
    OK(cudaHostGetDevicePointer(&device, task.host, 0));
    Compute<<<1, 1, 0, stream>>>(device);
    OK(cudaGetLastError());
    return kXSchedSuccess;
}

void ExpectCurrent(CUcontext expected)
{
    CUcontext actual = nullptr;
    OK(cuCtxGetCurrent(&actual));
    CHECK(actual == expected);
}

void PauseOne(const Task &paused, const Task &running)
{
    OK(XQueueSuspend(paused.queue, kQueueSuspendFlagSyncHwQueue));
    unsigned frozen = paused.host->progress;
    for (int window = 0; window < 5; ++window) {
        unsigned before = running.host->progress;
        std::this_thread::sleep_for(10ms);
        CHECK(paused.host->progress == frozen);
        CHECK(running.host->progress > before);
    }
    OK(XQueueResume(paused.queue, kQueueResumeFlagNone));
}

int main(int argc, char **argv)
{
    OK(cuInit(0));
    OK(XHintSetScheduler(kSchedulerAppManaged, kPolicyUnknown));
    CUcontext primary;
    OK(cuDevicePrimaryCtxRetain(&primary, 0));
    OK(cuCtxPushCurrent(primary));
    Task a, b;
    if (argc == 2 && std::strcmp(argv[1], "unavailable") == 0) {
        CHECK(CudaTaskDomainCreate(&a.domain, 0) == kXSchedErrorNotSupported && !a.domain);
        ExpectCurrent(primary);
        CUcontext popped;
        OK(cuCtxPopCurrent(&popped));
        CHECK(popped == primary);
        OK(cuDevicePrimaryCtxRelease(0));
        puts("PASS unavailable backend rejected, caller context preserved");
        return 0;
    }
    CHECK(CudaTaskDomainCreate(&a.domain, -1) == kXSchedErrorInvalidValue && !a.domain);
    CHECK(CudaTaskDomainAttach(&a.domain, nullptr) == kXSchedErrorInvalidValue && !a.domain);
    bool both_new = argc == 2 && std::strcmp(argv[1], "both-new") == 0;
    OK(CudaTaskDomainCreate(&a.domain, 0));
    ExpectCurrent(primary);

    // Competing clients each supply an idle stream in the same dedicated context.
    CUstream streams[2];
    for (auto &stream : streams) OK(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    std::atomic<bool> release_host_work{false};
    auto wait_for_host = [](void *arg) {
        auto *release = static_cast<std::atomic<bool> *>(arg);
        while (!release->load()) std::this_thread::yield();
    };
    OK(cuLaunchHostFunc(streams[0], wait_for_host, &release_host_work));
    CudaTaskDomain *busy = nullptr;
    CHECK(CudaTaskDomainAttach(&busy, streams[0]) == kXSchedErrorNotAllowed && !busy);
    release_host_work.store(true);
    OK(cuStreamSynchronize(streams[0]));

    CudaTaskDomain *claims[2]{};
    XResult results[2]{};
    std::atomic<unsigned> ready{0};
    auto claim = [&](int i) {
        ready.fetch_add(1);
        while (ready.load() != 2) std::this_thread::yield();
        results[i] = CudaTaskDomainAttach(&claims[i], streams[i]);
        ExpectCurrent(nullptr);
    };
    std::thread t0(claim, 0), t1(claim, 1);
    t0.join(); t1.join();
    int winner = results[0] == kXSchedSuccess ? 0 : 1;
    CHECK(results[winner] == kXSchedSuccess && claims[winner]);
    CHECK(results[1 - winner] == kXSchedErrorNotAllowed && !claims[1 - winner]);
    b.domain = claims[winner];
    CudaTaskDomain *duplicate = nullptr;
    CHECK(CudaTaskDomainAttach(&duplicate, streams[winner]) == kXSchedErrorNotAllowed && !duplicate);

    if (both_new) {
        OK(CudaTaskDomainDestroy(&b.domain));
        OK(CudaTaskDomainCreate(&b.domain, 0));
    }
    auto reject_work = [](CUstream, void *) -> XResult { return kXSchedErrorInvalidValue; };
    CHECK(CudaTaskDomainWithStream(a.domain, reject_work, nullptr) == kXSchedErrorInvalidValue);
    ExpectCurrent(primary);
    for (auto task : {&a, &b}) {
        OK(CudaTaskDomainGetQueue(task->domain, &task->queue));
        OK(cudaHostAlloc(&task->host, sizeof(Counters), cudaHostAllocMapped | cudaHostAllocPortable));
        *task->host = {};
    }
    for (auto task : {&a, &b}) {
        int64_t before, after;
        OK(XQueueProfileHwCommandCount(task->queue, &before));
        OK(CudaTaskDomainWithStream(task->domain, Launch, task));
        OK(XQueueProfileHwCommandCount(task->queue, &after));
        CHECK(after > before);
        ExpectCurrent(primary);
    }
    CHECK(a.context != b.context && a.context != primary);
    if (both_new) CHECK(b.context != primary);
    else CHECK(b.context == primary && b.stream == streams[winner]);
    auto deadline = std::chrono::steady_clock::now() + 3s;
    while ((a.host->progress < 10 || b.host->progress < 10) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(1ms);
    CHECK(a.host->progress >= 10 && b.host->progress >= 10);
    for (int round = 0; round < 3; ++round) {
        PauseOne(a, b);
        PauseOne(b, a);
    }
    OK(XQueueSuspend(a.queue, kQueueSuspendFlagSyncHwQueue));
    a.host->stop = 1;
    b.host->stop = 1;
    for (auto task : {&a, &b}) {
        OK(CudaTaskDomainDestroy(&task->domain)); // Includes paused destruction.
        CHECK(!task->domain);
        ExpectCurrent(primary);
        unsigned long long n = task->host->progress;
        CHECK(n > 0 && n < 100000 && task->host->sum == n * (n + 1) / 2);
        OK(cudaFreeHost(task->host));
    }
    OK(CudaTaskDomainAttach(&b.domain, streams[1 - winner]));
    OK(CudaTaskDomainDestroy(&b.domain));
    for (auto stream : streams) OK(cuStreamDestroy(stream));
    CUcontext popped;
    OK(cuCtxPopCurrent(&popped));
    CHECK(popped == primary);
    OK(cuDevicePrimaryCtxRelease(0));
    puts("PASS 30 directional pause windows, arithmetic, context stack, exclusive admission, drain and reuse");
}
