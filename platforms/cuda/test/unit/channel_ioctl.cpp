// Exercise the real preload entry point with non-CUDA descriptors.
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <initializer_list>
#include <sys/ioctl.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL line %d: %s (errno=%d)\n", __LINE__, #x, errno); std::exit(1); } } while (0)

int main()
{
    int pipe_fd[2];
    CHECK(pipe(pipe_fd) == 0);
    // No optional argument: the trampoline must not interpret register contents.
    CHECK(ioctl(pipe_fd[0], FIOCLEX) == 0);
    CHECK(fcntl(pipe_fd[0], F_GETFD) & FD_CLOEXEC);
    CHECK(ioctl(pipe_fd[0], FIONCLEX) == 0);
    CHECK(!(fcntl(pipe_fd[0], F_GETFD) & FD_CLOEXEC));

    CHECK(write(pipe_fd[1], "abc", 3) == 3);
    int count = 0;
    CHECK(ioctl(pipe_fd[0], FIONREAD, &count) == 0 && count == 3);
    int nonblocking = 1;
    CHECK(ioctl(pipe_fd[0], FIONBIO, &nonblocking) == 0);
    CHECK(fcntl(pipe_fd[0], F_GETFL) & O_NONBLOCK);
    CHECK(ioctl(pipe_fd[0], FIONREAD, reinterpret_cast<void *>(1)) == -1 && errno == EFAULT);
    CHECK(ioctl(-1, FIOCLEX) == -1 && errno == EBADF);

    // An integer-valued ioctl argument (not a pointer).
    int master = posix_openpt(O_RDWR | O_NOCTTY);
    CHECK(master >= 0 && grantpt(master) == 0 && unlockpt(master) == 0);
    int peer = ioctl(master, TIOCGPTPEER, O_RDWR | O_NOCTTY | O_CLOEXEC);
    CHECK(peer >= 0 && (fcntl(peer, F_GETFD) & FD_CLOEXEC));
    CHECK(close(peer) == 0 && close(master) == 0);

    // Colliding RM numbers on a non-NVIDIA device must not be dereferenced.
    int null_fd = open("/dev/null", O_RDWR);
    CHECK(null_fd >= 0);
    for (unsigned long request : {0xc020462bUL, 0xc030462bUL, 0xc0104629UL}) {
        CHECK(ioctl(null_fd, request, reinterpret_cast<void *>(1)) == -1 && errno == ENOTTY);
        CHECK(ioctl(-1, request, reinterpret_cast<void *>(1)) == -1 && errno == EBADF);
    }
    CHECK(close(null_fd) == 0 && close(pipe_fd[0]) == 0 && close(pipe_fd[1]) == 0);
    puts("PASS: no-argument, scalar, pointer, error and non-NVIDIA ioctl passthrough");
}
