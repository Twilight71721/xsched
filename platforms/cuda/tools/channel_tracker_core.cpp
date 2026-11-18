#include "channel_tracker_internal.h"
#include <cerrno>
#include <cstdio>

namespace xsched::cuda::channel::tracker
{
State &GetState() { static auto *state = new State; return *state; }

bool Descendant(State &s, uint32_t client, uint32_t object, uint32_t ancestor)
{
    for (size_t depth = 0; depth <= s.objects.size(); ++depth) {
        if (object == ancestor) return true;
        auto it = s.objects.find({client, object});
        if (it == s.objects.end() || it->second.parent == object) return false;
        object = it->second.parent;
    }
    return false;
}

// This preload library must be loaded before CUDA initialization. Never discovers
// objects belonging to other processes; only successful allocations are recorded.
int TrackIoctl(int fd, unsigned long req, void *arg)
{
    bool alloc21 = req == _IOWR('F', 0x2b, Alloc21);
    bool alloc64 = req == _IOWR('F', 0x2b, Alloc64);
    bool freeing = req == _IOWR('F', 0x29, Free);
    if (!arg || (!alloc21 && !alloc64 && !freeing)) return RawIoctl(fd, req, arg);
    auto &s = GetState();
    std::lock_guard<std::mutex> lock(s.mutex);
    uint32_t client, parent, object, cls = 0;
    if (freeing) {
        auto *p = static_cast<Free *>(arg);
        client = p->client; parent = p->parent; object = p->object;
    } else if (alloc64) {
        auto *p = static_cast<Alloc64 *>(arg);
        client = p->client; parent = p->parent; object = p->object; cls = p->cls;
    } else {
        auto *p = static_cast<Alloc21 *>(arg);
        client = p->client; parent = p->parent; object = p->object; cls = p->cls;
    }
    for (const auto &[key, owner] : s.owners) {
        if (key.first != client) continue;
        // Context/group lifetime must outlive its registered queue. New channels
        // while paused would escape the disabled set, so reject that allocation.
        if ((freeing && Descendant(s, client, key.second, object)) ||
            (owner.state != GroupState::Running &&
             Descendant(s, client, freeing ? object : parent, key.second))) {
            errno = EBUSY;
            return -1;
        }
    }
    int result = RawIoctl(fd, req, arg);
    int saved_errno = errno;
    uint32_t status = freeing ? static_cast<Free *>(arg)->status
        : alloc64 ? static_cast<Alloc64 *>(arg)->status : static_cast<Alloc21 *>(arg)->status;
    if (result == 0 && status == 0) {
        if (freeing) {
            std::vector<Key> removed;
            for (const auto &[key, value] : s.objects) {
                (void)value;
                if (key.first == client && Descendant(s, client, key.second, object)) removed.push_back(key);
            }
            for (auto key : removed) s.objects.erase(key);
        } else if (alloc64) {
            auto *p = static_cast<Alloc64 *>(arg);
            s.objects[{p->client, p->object}] = {parent, cls};
        } else {
            auto *p = static_cast<Alloc21 *>(arg);
            s.objects[{p->client, p->object}] = {parent, cls};
        }
    }
    errno = saved_errno;
    return result;
}

namespace
{
constexpr int MaxAttempts = 3;

Error Issue(int fd, unsigned client, unsigned subdevice, const Owner &owner, bool disable)
{
    Disable p{};
    p.disable = disable;
    p.count = owner.channels.size();
    for (size_t i = 0; i < owner.channels.size(); ++i) {
        p.clients[i] = client;
        p.channels[i] = owner.channels[i];
    }
    RmControl c{client, subdevice, 0x2080110b, 0, reinterpret_cast<uint64_t>(&p), sizeof(p), 0};
    if (RawIoctl(fd, _IOWR('F', 0x2a, RmControl), &c) != 0)
        return {ErrorDomain::System, uint32_t(errno ? errno : EIO)};
    if (c.status) return {ErrorDomain::Rm, c.status};
    return {};
}

Error Restore(int fd, unsigned client, unsigned subdevice, Owner &owner)
{
    owner.state = GroupState::Restoring;
    Error error;
    for (int attempt = 0; attempt < MaxAttempts; ++attempt) {
        error = Issue(fd, client, subdevice, owner, false);
        if (!error.Failed()) {
            owner.state = GroupState::Running;
            return {};
        }
    }
    owner.state = GroupState::Unknown;
    return error;
}

ControlResult Invalid(int error, GroupState state = GroupState::Unowned)
{
    return {{ErrorDomain::Validation, uint32_t(error)}, {}, state};
}
}

ControlResult ControlGroup(int fd, unsigned client, unsigned subdevice,
                           unsigned *handle, int op)
{
    if (!handle || fd < 0 || !client || !subdevice || op < kAcquire || op > kConfirmPaused)
        return Invalid(EINVAL);
    auto &s = GetState();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (op == kAcquire) {
        auto channel = s.objects.find({client, *handle});
        if (channel == s.objects.end() || channel->second.cls != 0xc96f) {
            fprintf(stderr, "CHANNEL tracker: missing compute channel %x (client %x, %zu tracked objects)\n", *handle, client, s.objects.size());
            return Invalid(ENOENT);
        }
        Key key{client, channel->second.parent};
        auto group = s.objects.find(key);
        if (group == s.objects.end() || group->second.cls != 0xa06c) return Invalid(ENOENT);
        Owner owner;
        owner.fd = fd;
        owner.subdevice = subdevice;
        if (!s.owners.emplace(key, std::move(owner)).second) return Invalid(EBUSY);
        *handle = key.second;
        return {{}, {}, GroupState::Running};
    }
    auto it = s.owners.find({client, *handle});
    if (it == s.owners.end()) return Invalid(ENOENT);
    auto &owner = it->second;
    if (owner.fd != fd || owner.subdevice != subdevice) return Invalid(EINVAL);
    if (op == kRelease) {
        if (owner.state != GroupState::Running) return Invalid(EBUSY, owner.state);
        s.owners.erase(it);
        return {};
    }
    if (op == kConfirmPaused) {
        if (owner.state != GroupState::Paused) return Invalid(EBUSY, owner.state);
        return {{}, {}, owner.state};
    }
    if ((op == kPause && owner.state == GroupState::Paused) ||
        (op == kResume && owner.state == GroupState::Running)) return {{}, {}, owner.state};
    if (op == kResume) {
        auto error = Restore(fd, client, subdevice, owner);
        if (!error.Failed()) owner.channels.clear();
        return {error, {}, owner.state};
    }
    if (owner.state != GroupState::Running) return Invalid(EIO, owner.state);
    owner.channels.clear();
    for (const auto &[key, obj] : s.objects) {
        if (key.first == client && obj.parent == *handle && obj.cls == 0xc96f)
            owner.channels.push_back(key.second);
    }
    if (owner.channels.empty() || owner.channels.size() > 64) {
        owner.channels.clear();
        return Invalid(EOVERFLOW, owner.state);
    }

    Error error;
    for (int attempt = 0; attempt < MaxAttempts; ++attempt) {
        owner.state = GroupState::Pausing;
        error = Issue(fd, client, subdevice, owner, true);
        if (!error.Failed()) {
            owner.state = GroupState::Paused;
            return {{}, {}, owner.state};
        }
        // A failed batch may have changed any subset. Never retry disable until
        // the entire original snapshot has been successfully re-enabled.
        auto recovery = Restore(fd, client, subdevice, owner);
        if (recovery.Failed()) return {error, recovery, owner.state};
    }
    owner.channels.clear();
    return {error, {}, owner.state};
}

} // namespace xsched::cuda::channel::tracker
