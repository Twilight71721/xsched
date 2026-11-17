#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>

#include "xsched/preempt/xqueue/async_xqueue.h"

using namespace xsched::preempt;
using namespace std::chrono_literals;

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

// Gates force ordering at completion and hardware-confirmation boundaries without a GPU.
class Gate
{
public:
    void Reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = false;
        entered_ = false;
    }
    void Open()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        open_ = true;
        cv_.notify_all();
    }
    void Pass()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_ = true;
        cv_.notify_all();
        cv_.wait(lock, [this] { return open_; });
    }
    void WaitEntered()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        CHECK(cv_.wait_for(lock, 3s, [this] { return entered_; }));
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool open_ = true;
    bool entered_ = false;
};

void Finish(std::future<void> &task)
{
    CHECK(task.wait_for(3s) == std::future_status::ready);
    task.get();
}

class TestHwQueue : public HwQueue
{
public:
    Gate completion, confirmation, restored;
    std::atomic<int> interrupts{0}, restores{0}, confirmations{0};
    std::atomic<bool> paused{false};
    std::atomic<XPreemptLevel> level{kPreemptLevelUnknown};

    void Launch(std::shared_ptr<HwCommand>) override {}
    void Synchronize() override { completion.Pass(); }
    void SynchronizeOnSuspend() override
    {
        CHECK(paused.load());
        ++confirmations;
        confirmation.Pass();
        CHECK(paused.load());
    }
    void Interrupt() override { CHECK(!paused.exchange(true)); ++interrupts; }
    void Restore(const CommandLog &) override
    {
        CHECK(paused.exchange(false));
        ++restores;
        restored.Pass();
    }
    void OnPreemptLevelChange(XPreemptLevel value) override
    {
        CHECK(!paused.load());
        level.store(value);
    }
    XDevice GetDevice() override { return 0; }
    HwQueueHandle GetHandle() override { return reinterpret_cast<uintptr_t>(this); }
    bool SupportDynamicLevel() override { return true; }
    bool SupportResumeDropCommands() const override { return false; }
    XPreemptLevel GetMaxSupportedLevel() override { return kPreemptLevelInterrupt; }
};

class TestCommand : public HwCommand
{
public:
    void Synchronize() override { CHECK(false); }
    bool Synchronizable() override { return false; }
    bool EnableSynchronization() override { return false; }
};

class ObservedQueue : public AsyncXQueue
{
public:
    explicit ObservedQueue(std::shared_ptr<HwQueue> hwq)
        : AsyncXQueue(hwq, kPreemptLevelInterrupt) {}
    Gate before_wait, after_wait;
    void WaitAll() override
    {
        before_wait.Pass();
        AsyncXQueue::WaitAll();
        after_wait.Pass();
    }
};

void RepeatedConfirmation()
{
    auto hwq = std::make_shared<TestHwQueue>();
    auto queue = std::make_shared<ObservedQueue>(hwq);
    queue->Suspend(kQueueSuspendFlagNone);
    hwq->confirmation.Reset();
    auto confirm = std::async(std::launch::async, [&] {
        queue->Suspend(kQueueSuspendFlagSyncHwQueue | kQueueSuspendFlagWaitAll);
    });
    hwq->confirmation.WaitEntered();
    CHECK(confirm.wait_for(20ms) == std::future_status::timeout);
    auto resume = std::async(std::launch::async, [&] { queue->Resume(kQueueResumeFlagNone); });
    CHECK(resume.wait_for(20ms) == std::future_status::timeout);
    CHECK(hwq->interrupts == 1 && hwq->restores == 0);
    hwq->confirmation.Open();
    Finish(confirm);
    Finish(resume);
    CHECK(hwq->confirmations == 1 && hwq->restores == 1);
    queue->Suspend(kQueueSuspendFlagNone);
    queue->Suspend(kQueueSuspendFlagSyncHwQueue);
    CHECK(hwq->interrupts == 2 && hwq->confirmations == 2);
    queue->Resume(kQueueResumeFlagNone);
    puts("PASS repeated confirmation and serialized Resume");
}

void CompletionWaitAllowsControl(int64_t flags)
{
    auto hwq = std::make_shared<TestHwQueue>();
    auto queue = std::make_shared<ObservedQueue>(hwq);
    hwq->completion.Reset();
    queue->Submit(std::make_shared<TestCommand>());
    hwq->completion.WaitEntered();
    queue->before_wait.Reset();
    std::promise<void> started;
    auto waiting = std::async(std::launch::async, [&] {
        started.set_value();
        queue->Suspend(flags);
    });
    started.get_future().wait();
    if (flags == kQueueSuspendFlagWaitAll) queue->before_wait.WaitEntered();
    CHECK(waiting.wait_for(20ms) == std::future_status::timeout);
    auto control = std::async(std::launch::async, [&] {
        queue->Suspend(kQueueSuspendFlagNone);
        queue->Resume(kQueueResumeFlagNone);
    });
    Finish(control); // Must finish while the original completion wait is still blocked.
    queue->before_wait.Open();
    hwq->completion.Open();
    Finish(waiting);
    CHECK(hwq->paused && hwq->interrupts == 2 && hwq->restores == 1);
    queue->Resume(kQueueResumeFlagNone);
    queue->WaitAll();
    printf("PASS completion wait permits control (flags=%lld)\n", static_cast<long long>(flags));
}

void ConfigurationWaitsForResume(bool change_level)
{
    auto hwq = std::make_shared<TestHwQueue>();
    auto queue = std::make_shared<ObservedQueue>(hwq);
    // Force suspension between drain completion and acquiring the control mutex.
    queue->after_wait.Reset();
    auto config = std::async(std::launch::async, [&] {
        if (change_level) queue->SetPreemptLevel(kPreemptLevelBlock);
        else queue->SetLaunchConfig(8, 4);
    });
    queue->after_wait.WaitEntered();
    queue->Suspend(kQueueSuspendFlagNone);
    queue->after_wait.Open();
    CHECK(config.wait_for(20ms) == std::future_status::timeout);
    CHECK(hwq->level == kPreemptLevelInterrupt && hwq->restores == 0);
    auto resume = std::async(std::launch::async, [&] { queue->Resume(kQueueResumeFlagNone); });
    Finish(resume);
    Finish(config);
    CHECK(hwq->level == (change_level ? kPreemptLevelBlock : kPreemptLevelInterrupt));
    puts(change_level ? "PASS level change waits for Resume" : "PASS launch config waits for Resume");
}

void DestroyCancelsPendingControl()
{
    auto hwq = std::make_shared<TestHwQueue>();
    auto queue = std::make_shared<ObservedQueue>(hwq);
    queue->before_wait.Reset();
    auto suspend = std::async(std::launch::async, [&] { queue->Suspend(kQueueSuspendFlagWaitAll); });
    queue->before_wait.WaitEntered();
    queue->PrepareForDestroy();
    queue->before_wait.Open();
    Finish(suspend);
    queue->Suspend(kQueueSuspendFlagSyncHwQueue);
    CHECK(hwq->interrupts == 0 && hwq->confirmations == 0);

    // Also test the actual manager deletion path with an outstanding shared reference.
    auto managed_hwq = std::make_shared<TestHwQueue>();
    auto handle = managed_hwq->GetHandle();
    CHECK(HwQueueManager::Add(handle, [&] { return managed_hwq; }) == kXSchedSuccess);
    XQueueHandle xq;
    CHECK(XQueueCreate(&xq, handle, kPreemptLevelInterrupt, 0) == kXSchedSuccess);
    auto outstanding = XQueueManager::Get(xq);
    managed_hwq->completion.Reset();
    outstanding->Submit(std::make_shared<TestCommand>());
    managed_hwq->completion.WaitEntered();
    outstanding->Suspend(0);
    managed_hwq->restored.Reset();
    managed_hwq->restored.Open();
    auto destroy = std::async(std::launch::async, [&] { CHECK(XQueueDestroy(xq) == kXSchedSuccess); });
    managed_hwq->restored.WaitEntered();
    outstanding->Suspend(0); // A request that acquired the queue before deletion must not re-pause it.
    CHECK(!managed_hwq->paused && managed_hwq->interrupts == 1);
    CHECK(destroy.wait_for(20ms) == std::future_status::timeout);
    managed_hwq->completion.Open();
    Finish(destroy);
    CHECK(XQueueManager::Get(xq) == nullptr);
    outstanding.reset();
    CHECK(HwQueueDestroy(handle) == kXSchedSuccess);
    puts("PASS destruction rejects late suspension and drains work");
}

int main()
{
    RepeatedConfirmation();
    CompletionWaitAllowsControl(kQueueSuspendFlagWaitAll);
    CompletionWaitAllowsControl(kQueueSuspendFlagWaitIdle);
    ConfigurationWaitsForResume(true);
    ConfigurationWaitsForResume(false);
    DestroyCancelsPendingControl();
}
