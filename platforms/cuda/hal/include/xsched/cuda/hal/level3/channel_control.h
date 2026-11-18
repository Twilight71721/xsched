#pragma once

#include <cstdint>

namespace xsched::cuda::channel
{

// Internal ABI between the HAL and the explicitly preloaded RM tracker.
inline constexpr char ControlSymbol[] = "xsched_cuda_channel_control_v3";
inline constexpr char ValidatedDriverVersion[] = "595.71.05";

enum Operation : int
{
    kAcquire = 0, // Input: compute channel. Output: its owning group.
    kPause = 1,
    kResume = 2,
    kRelease = 3,
    kConfirmPaused = 4, // Confirm suspension without waiting for command completion.
};

enum class GroupState : uint32_t { Unowned, Running, Pausing, Paused, Restoring, Unknown };
enum class ErrorDomain : uint32_t { None, Validation, System, Rm };

struct Error {
    ErrorDomain domain = ErrorDomain::None;
    uint32_t code = 0;
    bool Failed() const { return domain != ErrorDomain::None; }
};

struct ControlResult {
    Error error;
    Error recovery_error;
    GroupState state = GroupState::Unowned;
    bool Succeeded() const { return !error.Failed(); }
};
static_assert(sizeof(Error) == 8 && sizeof(ControlResult) == 20);

// The tracker owns state and all retries. A recovered failed pause is still an
// error unless a subsequent disable succeeds. Unknown requires restoration.
// All operations after acquire use the returned group and the original binding.
using Control = ControlResult (*)(int fd, unsigned client, unsigned subdevice,
                                  unsigned *handle, int operation);

} // namespace xsched::cuda::channel
