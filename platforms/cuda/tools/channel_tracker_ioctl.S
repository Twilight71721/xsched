// Linux x86-64 SysV: fd, request and the optional argument occupy RDI, RSI, RDX.
// Forward the registers without fetching a nonexistent or differently typed
// C vararg. For no-argument requests the kernel ignores the third register.
.text
.globl ioctl
.type ioctl, @function
.hidden xsched_cuda_tracker_ioctl
ioctl:
    .cfi_startproc
    endbr64
    jmp xsched_cuda_tracker_ioctl
    .cfi_endproc
.size ioctl, .-ioctl
.section .note.GNU-stack,"",@progbits
