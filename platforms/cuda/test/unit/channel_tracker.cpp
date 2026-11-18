// Link the production core with a test-owned syscall implementation.
#include "../../tools/channel_tracker_internal.h"
#include <cerrno>
#include <cstdio>
using namespace xsched::cuda::channel;
using namespace xsched::cuda::channel::tracker;
#include <cstdlib>
#include <deque>
#include <set>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); std::exit(1); } } while (0)

namespace xsched::cuda::channel::tracker
{
std::deque<int> errors;
std::set<unsigned> disabled;
unsigned controls = 0;
unsigned allocated_object = 0;
int allocation_error = 0;
int RawIoctl(int, unsigned long req, void *arg)
{
    if (req == _IOWR('F', 0x2b, Alloc64)) {
        auto &p = *static_cast<Alloc64 *>(arg);
        if (allocation_error < 0) { errno = EIO; return -1; }
        p.status = allocation_error;
        if (!allocation_error && allocated_object) p.object = allocated_object;
        return 0;
    }
    if (req != _IOWR('F', 0x2a, RmControl)) return 0;
    ++controls;
    auto &c = *static_cast<RmControl *>(arg);
    auto &p = *reinterpret_cast<Disable *>(c.params);
    CHECK(c.cmd == 0x2080110b && !p.only_scheduling && !p.rewind && !p.event);
    const auto state = GetState().owners.at({c.client, 10}).state;
    CHECK(state == (p.disable ? GroupState::Pausing : GroupState::Restoring));
    int error = 0;
    if (!errors.empty()) { error = errors.front(); errors.pop_front(); }
    // A failing request can already have changed its first channel.
    unsigned count = error ? 1 : p.count;
    for (unsigned i = 0; i < count; ++i)
        if (p.disable) disabled.insert(p.channels[i]);
        else disabled.erase(p.channels[i]);
    if (error < 0) { errno = EIO; return -1; }
    c.status = error;
    return 0;
}

constexpr unsigned Client = 1, Group = 10, Subdevice = 3;
ControlResult Call(unsigned *handle, int op)
{
    return ControlGroup(7, Client, Subdevice, handle, op);
}
void Allocate(unsigned parent, unsigned object, unsigned cls)
{
    Alloc21 p{Client, parent, object, cls, 0, 0, 0};
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc21), &p) == 0);
}
void Setup()
{
    auto &s = GetState();
    s.objects.clear(); s.owners.clear();
    errors.clear(); disabled.clear(); controls = 0;
    allocated_object = 0; allocation_error = 0;
    Allocate(0, Client, 0);
    Allocate(Client, Subdevice, 0x2080);
    Allocate(Client, Group, 0xa06c);
    Allocate(Group, 11, 0xc96f);
    Allocate(Group, 12, 0xc96f);
}
}

void CheckError(const ControlResult &result, ErrorDomain domain, unsigned code,
                GroupState state, ErrorDomain recovery_domain = ErrorDomain::None,
                unsigned recovery_code = 0)
{
    CHECK(!result.Succeeded());
    CHECK(result.error.domain == domain && result.error.code == code);
    CHECK(result.state == state);
    CHECK(result.recovery_error.domain == recovery_domain && result.recovery_error.code == recovery_code);
}

int main()
{
    Setup();
    // Use returned handles and record only successful allocations, for both RM ABIs.
    Alloc64 allocation{Client, Group, 0, 0xc96f, 0, 0, 0, 0, 0};
    allocated_object = 13;
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == 0);
    CHECK(allocation.object == 13);
    CHECK(GetState().objects.at({Client, 13}).parent == Group);
    CHECK(GetState().objects.at({Client, 13}).cls == 0xc96f);
    unsigned handle = 13;
    auto acquired = Call(&handle, kAcquire);
    CHECK(acquired.Succeeded() && acquired.state == GroupState::Running && handle == Group);
    CHECK(Call(&handle, kPause).Succeeded() && disabled.size() == 3);
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == -1 && errno == EBUSY);
    CHECK(Call(&handle, kResume).Succeeded() && disabled.empty());
    CHECK(Call(&handle, kRelease).Succeeded());
    allocation.object = 14;
    allocated_object = 14;
    allocation_error = 0x40;
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == 0);
    CHECK(allocation.status == 0x40 && !GetState().objects.count({Client, 14}));
    allocation.status = 0;
    allocation_error = -1;
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc64), &allocation) == -1);
    CHECK(errno == EIO && !GetState().objects.count({Client, 14}));

    Setup();
    CheckError(Call(nullptr, kAcquire), ErrorDomain::Validation, EINVAL, GroupState::Unowned);
    handle = 11;
    CheckError(Call(&handle, 9), ErrorDomain::Validation, EINVAL, GroupState::Unowned);
    CHECK(Call(&handle, kAcquire).Succeeded() && handle == Group);
    unsigned duplicate = 12;
    CheckError(Call(&duplicate, kAcquire), ErrorDomain::Validation, EBUSY, GroupState::Unowned);
    CHECK(duplicate == 12);
    CheckError(ControlGroup(8, Client, Subdevice, &handle, kPause),
               ErrorDomain::Validation, EINVAL, GroupState::Unowned);
    CheckError(ControlGroup(7, Client, 99, &handle, kPause),
               ErrorDomain::Validation, EINVAL, GroupState::Unowned);
    CheckError(Call(&handle, kConfirmPaused), ErrorDomain::Validation, EBUSY, GroupState::Running);
    CHECK(controls == 0);
    Free parent{Client, 0, Client, 0};
    CHECK(TrackIoctl(7, _IOWR('F', 0x29, Free), &parent) == -1 && errno == EBUSY);
    auto paused = Call(&handle, kPause);
    CHECK(paused.Succeeded() && paused.state == GroupState::Paused && disabled.size() == 2);
    CHECK(Call(&handle, kPause).Succeeded() && controls == 1);
    CHECK(Call(&handle, kConfirmPaused).Succeeded() && controls == 1);
    CheckError(Call(&handle, kRelease), ErrorDomain::Validation, EBUSY, GroupState::Paused);
    Alloc21 escaped{Client, Group, 13, 0xc96f, 0, 0, 0};
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc21), &escaped) == -1 && errno == EBUSY);
    Free child{Client, Group, 11, 0};
    CHECK(TrackIoctl(7, _IOWR('F', 0x29, Free), &child) == -1 && errno == EBUSY);
    CHECK(Call(&handle, kResume).Succeeded() && disabled.empty());
    CHECK(Call(&handle, kResume).Succeeded() && controls == 2);
    CheckError(Call(&handle, kConfirmPaused), ErrorDomain::Validation, EBUSY, GroupState::Running);

    // The tracker alone retries a partial disable, only after complete rollback.
    unsigned before = controls;
    errors = {-1, 0, 0};
    CHECK(Call(&handle, kPause).Succeeded() && disabled.size() == 2);
    CHECK(controls == before + 3 && errors.empty());
    // Enable also retries, using the exact same channel snapshot.
    errors = {0x40, -1, 0};
    CHECK(Call(&handle, kResume).Succeeded() && disabled.empty() && errors.empty());

    // Exhausted disable attempts with successful rollbacks are NOT success.
    before = controls;
    errors = {-1, 0, -1, 0, -1, 0};
    CheckError(Call(&handle, kPause), ErrorDomain::System, EIO, GroupState::Running);
    CHECK(controls == before + 6 && disabled.empty() && errors.empty());
    CheckError(Call(&handle, kConfirmPaused), ErrorDomain::Validation, EBUSY, GroupState::Running);
    // The same numeric code from RM remains distinguishable from errno.
    errors = {EIO, 0, EIO, 0, EIO, 0};
    CheckError(Call(&handle, kPause), ErrorDomain::Rm, EIO, GroupState::Running);
    CHECK(disabled.empty() && errors.empty());

    // Failed rollback retains the snapshot and prohibits pause/lifetime changes.
    before = controls;
    errors = {0x40, 0x40, -1, 0x40};
    CheckError(Call(&handle, kPause), ErrorDomain::Rm, 0x40, GroupState::Unknown,
               ErrorDomain::Rm, 0x40);
    CHECK(controls == before + 4 && errors.empty());
    CHECK(GetState().owners.at({Client, Group}).channels.size() == 2);
    CheckError(Call(&handle, kRelease), ErrorDomain::Validation, EBUSY, GroupState::Unknown);
    before = controls;
    CheckError(Call(&handle, kPause), ErrorDomain::Validation, EIO, GroupState::Unknown);
    CheckError(Call(&handle, kConfirmPaused), ErrorDomain::Validation, EBUSY, GroupState::Unknown);
    CHECK(controls == before);
    CHECK(TrackIoctl(7, _IOWR('F', 0x2b, Alloc21), &escaped) == -1 && errno == EBUSY);
    CHECK(TrackIoctl(7, _IOWR('F', 0x29, Free), &child) == -1 && errno == EBUSY);
    CHECK(Call(&handle, kResume).Succeeded() && disabled.empty());
    CHECK(GetState().owners.at({Client, Group}).state == GroupState::Running);
    CHECK(Call(&handle, kPause).Succeeded());
    errors = {-1, -1, -1};
    CheckError(Call(&handle, kResume), ErrorDomain::System, EIO, GroupState::Unknown);
    CheckError(Call(&handle, kRelease), ErrorDomain::Validation, EBUSY, GroupState::Unknown);
    CHECK(Call(&handle, kResume).Succeeded() && disabled.empty());
    auto released = Call(&handle, kRelease);
    CHECK(released.Succeeded() && released.state == GroupState::Unowned);
    CHECK(TrackIoctl(7, _IOWR('F', 0x29, Free), &parent) == 0);
    CHECK(GetState().objects.empty());
    handle = 11;
    CheckError(Call(&handle, kAcquire), ErrorDomain::Validation, ENOENT, GroupState::Unowned);

    Setup();
    for (unsigned i = 13; i < 76; ++i) Allocate(Group, i, 0xc96f);
    handle = 11;
    CHECK(Call(&handle, kAcquire).Succeeded());
    CheckError(Call(&handle, kPause), ErrorDomain::Validation, EOVERFLOW, GroupState::Running);
    CHECK(controls == 0);
    CHECK(Call(&handle, kRelease).Succeeded());
    puts("PASS: allocation ABIs, ownership, binding, lifecycle, state transitions, error domains, bounded retries, rollback, recovery, overflow");
}
