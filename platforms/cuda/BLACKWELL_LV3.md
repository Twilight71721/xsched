# Experimental Blackwell Level 3: CHANNEL

This backend adds state-preserving compute preemption on the validated SM120
configuration below. Select it explicitly with `XSCHED_CUDA_LV3_IMPL=CHANNEL`
and preload `libcuda_channel_tracker.so` before CUDA initialization. The default
backend selection is unchanged. This PR does not include SM120 Guardian Level 2.

## Mechanism and scope

The backend queries private ToolsRm context handles, locates the owning compute
group in the process's tracked RM allocations, and disables all compute channels
in that group with `NV2080_CTRL_CMD_FIFO_DISABLE_CHANNELS`. The request uses
synchronous preemption, `bOnlyDisableScheduling=false`, `bRewindGpPut=false`, and
no event. Resume re-enables the original channel snapshot. XSched does not
instrument kernels or replay completed instructions on this path.

The control domain is a compute group, not an individual CUDA stream. Only one
managed queue may own a context/group. Other compute streams in that group also
stop. Use separate CUDA contexts for independently controlled tasks, and account
for any cross-context event/data dependencies when scheduling them.

The optional preload library observes this process's successful RM allocations
and frees on NVIDIA character devices (major 195). It does not enumerate or
control other processes' objects. Its Linux x86-64 SysV entry point forwards
the ioctl argument register without assuming the optional argument is a pointer;
only recognized RM allocation/free requests on NVIDIA devices are decoded.
Unrelated requests go directly to the system call with their argument unchanged.

Set `XSCHED_CUDA_LV3_IMPL=CHANNEL` before the first XSched CUDA call. XSched's
CUDA loader then opens the real driver without `RTLD_DEEPBIND`; other backends
retain their previous loading flags. The tracker no longer intercepts `dlopen`
or changes other libraries' loading choices. Normal external loading of the
driver before XSched is supported when the tracker was preloaded at process
startup. External deep-binding, a separate loader namespace, direct system calls
or loading the tracker after CUDA initialization can bypass tracking and are
unsupported. Missing allocation history makes queue acquisition fail; it must
not be repaired by guessing channel handles or silently falling back to LV1.

## Validated configuration

- Linux x86_64; NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU, SM120.
- NVIDIA driver **595.71.05**, CUDA Toolkit **13.0**, native `sm_120` kernels.
- Explicit application-managed queues; one managed owner per compute group.

Queue creation requires the v3 tracker symbol and the exact driver version from
`/proc/driver/nvidia/version`. An unavailable tracker, unsupported driver or GPU,
unrecognized handle, or duplicate owner returns an error without registering a
queue or silently falling back to Level 1. Other driver versions require ABI
review and hardware validation before changing the version check.

## Allocate a task domain

A task domain owns one scheduler-visible LV3 XQueue and one exclusive compute
group. Select CHANNEL and load the tracker and CUDA shim as described above.
Create a domain before the task initializes its CUDA libraries or device buffers.

- `CudaTaskDomainCreate(&domain, ordinal)` creates a dedicated ordinary CUDA
  context and a nonblocking stream. Each independently scheduled task gets its
  own domain. The calling thread's previous CUDA context is restored.
- `CudaTaskDomainAttach(&domain, stream)` borrows an idle, explicit stream in an
  ordinary context the application dedicates to this task. The caller must
  quiesce submitters during admission, keep the stream and context alive, and
  transfer submissions to the domain. A busy stream, registered stream, or
  managed group conflict returns `kXSchedErrorNotAllowed`. The allocator cannot
  discover unregistered application tasks that share that context.
- `CudaTaskDomainWithStream(domain, work, arg)` invokes a host callback with the
  domain context current. Initialize library handles in that context, bind them
  to the supplied stream, and enqueue work there. The callback returns an
  `XResult` and must leave the context stack balanced. Work on other streams
  can share hardware suspension yet escape this XQueue's submission accounting.
- `CudaTaskDomainGetQueue(domain, &queue)` supplies a borrowed handle for policy
  hints and XQueue control. The domain owns destruction of the queue. The
  allocator does not select a scheduler or change its policy.
- Stop submissions and join host users of the domain, including users issuing
  queue controls, before `CudaTaskDomainDestroy(&domain)`. Destruction resumes
  and drains the queue, releases its group, then frees owned CUDA resources.
  An attached stream and context remain owned by the application. Success
  clears the domain pointer. On a teardown error retain the handle for cleanup;
  a partially closed domain must not be used for new work. The underlying HAL
  retains its fail-stop behavior on persistent RM control failures.

The `sm120_task_domain` test uses a running kernel in each domain and checks
the command counters, bidirectional suspension, final arithmetic, concurrent
admission and resource reuse. The CHANNEL contract runner exercises both
new/new and new/attached task layouts.

## Queue contract

- `XQueueSuspend(..., kQueueSuspendFlagSyncHwQueue)` confirms compute suspension,
  not command completion or complete context quiescence. Independent copy-engine
  transfers may continue. Do not free buffers based on suspension alone.
- Ordinary `HwQueueSynchronize` retains completion semantics and may block until
  another host thread resumes the queue.
- An unmanaged stream waiting for an event recorded by a CHANNEL queue waits for
  record submission before inserting the native CUDA dependency. Record
  submission can itself wait if the source queue is paused.
- `kQueueResumeFlagDropCommands` returns `kXSchedErrorNotSupported` before changing
  queue state, including when a CHANNEL queue has been switched to Level 1.
- Destroy XQueue, then HwQueue, then stream/context. Destroying a paused CHANNEL
  XQueue resumes and drains its retained work. Ownership can then be reacquired.
- The shim rejects destruction of an owned context, and conservatively rejects
  primary-context reset while any CHANNEL queue owns a context on that device.
  The tracker rejects freeing owned groups/ancestors and allocation/free within
  paused or uncertain groups.
- Level 2 on this backend uses the same stronger pause operation. Level 1 uses
  ordinary blocking. Change levels only on a resumed queue with no concurrent
  queue operations. Suspend/resume controls on an AsyncXQueue are serialized.

A failed batch disable may have affected some channels. The tracker retains the
snapshot and attempts to enable the entire set. It owns all retries: at most
three disable attempts, with at most three enable attempts for each rollback.
A new disable is attempted only after successful restoration. Successful
rollback alone never means successful suspension; only a later successful
disable does. Normal resume also uses at most three enable attempts.

The tracker is the sole authority for hardware state:
`Running -> Pausing -> Paused -> Restoring -> Running`. An unsuccessful
restoration leaves `Unknown`, retaining the snapshot and rejecting pause,
confirmation, release and group lifetime changes until explicit restoration
succeeds. `ChannelContext` owns the binding and context lifetime guard but no
longer keeps a second pause flag, mutex or retry loop. Suspension confirmation
checks the tracker state; LV1 synchronization still waits for stream completion.
The private v3 result distinguishes validation, syscall and RM errors, includes
the final state and reports rollback failure separately from the original error.
HAL and tracker must be rebuilt together; v2 is deliberately not accepted.

Persistent runtime failures terminate with a diagnostic because the HAL control
methods return `void`. This is fail-stop handling, not a recoverable-error API.
The registry lock continues to serialize object lifetime changes with driver
control. It is intentionally held across ioctl calls; releasing it without
pinning the object graph would allow channels to escape the pause snapshot.

## Limitations

- This is not transparent independent scheduling of multiple streams sharing a
  context. Automatic per-stream queue creation conflicts with group ownership;
  the existing stream-create shim does not propagate `AutoCreate` failures to
  the CUDA caller. An unmanaged stream is not automatically included in the
  owner's queue-idle detection or service accounting. Use explicit APP-mode
  ownership and controlled task layouts until group-aware integration exists.
- Do not bypass the shim, release the last primary-context reference while a
  queue exists, or race context lifetime operations with queue creation/destruction.
  Green Context lifetime is not covered by these guards.
- CUDA graphs, cooperative/cluster/device launches, external semaphores, fork
  after CUDA initialization, other GPUs/drivers, and high-contention multi-context
  scaling have not been validated by this change.
- Synchronous RM calls carry driver/firmware/hardware costs. The backend does not
  promise a latency bound or a performance improvement for short tasks/time slices.
- Adding HAL virtual methods changes the internal C++ ABI. Rebuild preempt, HAL,
  shim and any out-of-tree HAL subclasses together.

## AsyncXQueue control ordering

- A repeated suspend with `kQueueSuspendFlagSyncHwQueue` confirms the existing
  suspension, including one initially requested without synchronization. It does
  not issue a second `Deactivate()` or `Interrupt()`. Confirmation and restoration
  remain serialized, so restoration cannot invalidate an in-progress confirmation.
- `WaitAll` and `WaitIdle` are preconditions for a new suspension. They are skipped
  when already suspended, and their completion waits do not hold the control mutex.
  Another thread can suspend/resume during those waits. A concurrent Resume does
  not cancel the waiting request: it may suspend the queue after its wait finishes.
- Level/launch-configuration changes wait for a paused queue to be resumed by
  another thread, then drain and apply under the control mutex. They recheck the
  state after draining, and never silently resume a paused queue themselves.
  Callers must still exclude concurrent submissions during configuration changes;
  this is not a general concurrent submission/configuration protocol.
- Before destruction drains the queue, `PrepareForDestroy()` marks control as
  terminated and resumes it. A pending or late Suspend holding an older shared
  reference cannot pause it again; pending configuration changes are abandoned.
  Callers must stop submissions and retain the underlying CUDA resources until
  destruction finishes. Concurrent destruction of CUDA resources is unsupported.
- HAL suspension confirmation must not require Resume on the same queue to
  complete. CHANNEL fulfills this with synchronous compute-channel disable;
  ordinary stream completion still requires the retained work to execute.

The GPU-free `async_xqueue_control_test` forces delayed confirmation, blocked
completion, configuration/suspension races and deletion with an outstanding
reference. Run it after building with `BUILD_TEST=ON`:

```bash
ctest --test-dir build-channel -R '^async_xqueue_control_test$' --output-on-failure
```

The preemption changes preserve the default suspension synchronization for older
backends; they do not validate TSG timeslice-zero semantics on other hardware.

## Build and test

Production builds use `BUILD_TEST=OFF` (the default). This omits test executables
and test syscall implementations. The tracker exports only `ioctl`
and the versioned internal `xsched_cuda_channel_control_v3` entry point; its
private implementation header is not installed. There is no public CHANNEL
inspection/test API. The scope and limitations above are the backend contract.

From the repository root, using an initialized recursive checkout:

```bash
cmake -S . -B build-channel -DPLATFORM_CUDA=ON -DSHIM_SOFTLINK=ON \
  -DBUILD_TEST=ON -DBUILD_SERVICE=OFF -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_INSTALL_PREFIX="$PWD/output-channel"
cmake --build build-channel -j8
cmake --install build-channel
ctest --test-dir build-channel -R '^(async_xqueue_control_test|cuda_channel_.*test)$' --output-on-failure
python3 platforms/cuda/test/scripts/run_lv3_contracts.py \
  --install output-channel --build build-channel \
  --output /tmp/xsched-channel-contracts-new --repeats 3
```

To validate common shim integration without CHANNEL or its tracker, run the
installed `bin/cuda/shim_contract` with `XSCHED_SCHEDULER=APP`,
`LD_LIBRARY_PATH` pointing to the install's `lib`, and no tracker `LD_PRELOAD`.

The output directory must be new. The suite runs GPU tests serially with timeouts
and saves metadata, binary hashes, stdout/stderr and a summary. It controls only
the test process's channels and requires no root privileges or driver changes.

The CPU test links the production tracker core with its own syscall implementation and covers
ownership, binding, lifecycle guards, partial disable, bounded retries, rollback,
unknown-state recovery, confirmation, error domains and overflow. It does not
inject actual driver/firmware failures. The ioctl tests exercise both native and
preloaded calls with no argument, scalar and pointer arguments, kernel errors,
and colliding RM request numbers on non-NVIDIA descriptors.
The GPU loading tests check XSched-first loading, ordinary external driver-first
loading, and rejection after external deep-bound initialization. Each mode runs
in a fresh process so loader state cannot leak between tests.

`sm120_lv3` checks real shim submission, concurrent streams, 3D grids/blocks,
repeated suspension of unfinished kernels, exact non-idempotent counts and
thread-local arithmetic, queued kernel/copy ordering, Level-1 switching and queue
recreation. Functions are preloaded to avoid lazy loading invalidating the
concurrent-work setup. `sm120_channel_contract` checks independent contexts,
copy/event dependencies, completion synchronization, unsupported-operation
rejection, paused destruction, reuse and concurrent host controls on an idle
queue. GPU tests exit normally and exercise cleanup; they do not use `_Exit`.

## Implementation map

- `hal/src/level3/channel.cpp`: XSched queue interface and delegation to an exclusive `ChannelContext`.
- `hal/src/level3/channel_context.cpp`: ownership, ToolsRm lookup and delegation to tracker transactions. Each queue owns its context controller; controllers are not shared like `TsgContext`. Context lifetime protection is held through a common RAII guard.
- `hal/include/xsched/cuda/hal/level3/channel_control.h`: internal tracker ABI.
- `tools/channel_tracker.cpp`: preload entry points and real system-call transport.
- `tools/channel_tracker_ioctl.S`: Linux x86-64 ioctl argument-register forwarding.
- `tools/channel_tracker_core.cpp`: RM object tracking, ownership and disable/enable transactions.
- `tools/channel_tracker_internal.h`: private RM layouts and implementation declarations.
- `tools/channel_tracker.map`: export allowlist for the two required preload symbols.
- `hal/src/common/driver.cpp`: CUDA-only loader policy; CHANNEL avoids deep-binding without a global `dlopen` wrapper.
- `preempt/src/xqueue/async_xqueue.cpp`: control serialization and suspension boundary.
- `hal/src/common/context_guard.cpp`: backend-independent, opt-in context ownership and guarded destroy/reset. An exclusive guard is held from backend acquisition through RM cleanup; it does not change ordinary LV1 queue ownership.
- `hal/include/xsched/cuda/hal/level1/cuda_queue.h`: `SupportNativeEventWait()` defaults to false. A backend opts in only if submitted native event dependencies remain valid across its preemption/restoration and cancellation behavior.
- `shim/src/shim.cpp`: common context guards and capability-based event waits; no CHANNEL type dependency.
- `test/main/shim_contract.cu`: common guards, legacy event waiting and a non-CHANNEL queue opting into native waits, without the tracker.
- `test/scripts/run_lv3_contracts.py`: repeatable correctness suite.

RM layouts are derived from NVIDIA's matching open driver definitions:
[control header](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.71.05/src/common/sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080fifo.h),
[allocation/control ABI](https://github.com/NVIDIA/open-gpu-kernel-modules/blob/595.71.05/src/common/sdk/nvidia/inc/nvos.h).
These definitions do not make the private ToolsRm interface a stable CUDA API.
