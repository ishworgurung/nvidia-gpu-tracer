// SPDX-License-Identifier: GPL-2.0
// eBPF GPU tracer: kprobe on nvidia_unlocked_ioctl.
// Decodes RM escape parameters from userspace via bpf_probe_read_user.

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

// ---- NV escape codes (ioctl "nr" field, bits 0-7 of ioctl cmd) ----
// These are the legacy RM escape numbers used by the proprietary driver.
// Also used as sub-commands inside NV_ESC_IOCTL_XFER_CMD (0xD3).
#define NV_ESC_RM_ALLOC_MEMORY 0x27
#define NV_ESC_RM_ALLOC 0x29
#define NV_ESC_RM_CONTROL 0x2A
#define NV_ESC_RM_FREE 0x2B
#define NV_ESC_RM_VID_HEAP_CONTROL 0x4A
// XFER_CMD = NV_IOCTL_BASE(200) + 11 = 211 = 0xD3
// Used to tunnel oversized RM args: arg = *nv_ioctl_xfer_t { cmd, size, ptr }
#define NV_ESC_IOCTL_XFER_CMD 0xD3

// Extract ioctl "nr" field (bits 0-7) from the full Linux ioctl cmd.
#define NV_CMD(cmd) ((cmd) & 0xFF)
// Extract ioctl "size" field (bits 16-29): sizeof the NVOS* struct passed in.
#define NV_SIZE(cmd) (((cmd) >> 16) & 0x3FFF)

// NV_ESC_RM_ALLOC accepts two struct versions, told apart by size.
#define NVOS21_SIZE 32
#define NVOS64_SIZE 48

// d3 value on exit when the RM status could not be read back.
#define RM_STATUS_UNKNOWN 0xFFFFFFFF

// ---- event types ----
#define EVT_MEM_ALLOC 1  // NV_ESC_RM_ALLOC_MEMORY: GPU memory allocation
#define EVT_OBJ_ALLOC 2  // NV_ESC_RM_ALLOC:        object allocation (channels, contexts, ...)
#define EVT_RM_CONTROL 3 // NV_ESC_RM_CONTROL:      resource manager control call
#define EVT_OBJ_FREE 4   // NV_ESC_RM_FREE:         object / memory free
#define EVT_VID_HEAP 5   // NV_ESC_RM_VID_HEAP_CONTROL

// is_exit flag values
#define IS_ENTRY 0
#define IS_EXIT 1

// ---- event structure sent to userspace ----
// Decoded fields vary by evt_type; see comments below.
// On exit events (is_exit=1): d2 = ioctl return value (0 or -errno),
//                              d3 = RM status from the NVOS* struct (0 = NV_OK,
//                                   RM_STATUS_UNKNOWN if unreadable),
//                              d4 = latency_ns.
// On entry events (is_exit=0): fields are decoded per evt_type.
struct gpu_event
{
    __u64 ts_ns;
    __u32 pid;
    __u32 tid;
    __u8 type;    // EVT_* above
    __u8 nv_cmd;  // raw escape code
    __u8 is_exit; // IS_ENTRY or IS_EXIT
    __u8 pad;
    __u64 arg_ptr; // userspace arg pointer (for correlation)
    char comm[16];
    // entry decoded payload (interpretation depends on type):
    __u32 d0; // EVT_OBJ_ALLOC: hClass  | EVT_RM_CONTROL: ctrl_cmd | EVT_OBJ_FREE: h_object | EVT_MEM_ALLOC: hClass
    __u32 d1; // EVT_OBJ_ALLOC: hParent | EVT_RM_CONTROL: h_object | EVT_MEM_ALLOC: flags
    __u32 d2; // EVT_OBJ_ALLOC: hNew    | EVT_RM_CONTROL: params_size | exit: ioctl ret
    __u32 d3; // exit: RM status (NV_OK = 0)
    __u64 d4; // EVT_MEM_ALLOC: allocation size in bytes (limit+1) | exit: latency_ns
};

// ---- maps ----
struct
{
    __uint(type, BPF_MAP_TYPE_PERF_EVENT_ARRAY);
    __uint(key_size, sizeof(__u32));
    __uint(value_size, sizeof(__u32));
} events SEC(".maps");

// Per-CPU scratch for building events (avoids large stack alloc).
struct
{
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct gpu_event);
} heap SEC(".maps");

// Stash across kprobe→kretprobe boundary.
struct kprobe_state
{
    __u64 entry_ts; // ktime_get_ns() at entry, for latency
    __u64 arg_ptr;
    __u8 nv_cmd;
    __u8 evt_type;
    __u16 arg_size; // sizeof the NVOS* struct, from the ioctl cmd or XFER header
};
struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 4096);
    __type(key, __u64); // pid_tgid
    __type(value, struct kprobe_state);
} active_ioctls SEC(".maps");

// ---- helpers ----
static __always_inline struct gpu_event *get_slot(void)
{
    __u32 zero = 0;
    return bpf_map_lookup_elem(&heap, &zero);
}

static __always_inline void fill_common(struct gpu_event *e, __u8 type,
                                        __u8 cmd, __u8 is_exit, __u64 arg)
{
    __u64 pt = bpf_get_current_pid_tgid();
    e->ts_ns = bpf_ktime_get_ns();
    e->pid = pt >> 32;
    e->tid = (__u32)pt;
    e->type = type;
    e->nv_cmd = cmd;
    e->is_exit = is_exit;
    e->pad = 0;
    e->arg_ptr = arg;
    e->d0 = e->d1 = e->d2 = e->d3 = 0;
    e->d4 = 0;
    bpf_get_current_comm(e->comm, sizeof(e->comm));
}

// Read a field from a userspace buffer using __builtin_memcpy to avoid
// potential unaligned-access rejections from the BPF verifier.
#define READ_FIELD(dst, buf, off) __builtin_memcpy(&(dst), (buf) + (off), sizeof(dst))

// ---- decode RM args from a 64-byte snapshot of a userspace NVOS* struct ----
static __always_inline void decode_rm_args(struct gpu_event *e, __u8 cmd,
                                           __u64 user_ptr)
{
    __u8 buf[64] = {};
    if (bpf_probe_read_user(buf, sizeof(buf), (void *)user_ptr) != 0)
        return;

    switch (cmd)
    {
    case NV_ESC_RM_ALLOC:
    {
        // NVOS21_PARAMETERS / NVOS64_PARAMETERS share the first 16 bytes:
        //   +0  hRoot       (u32)
        //   +4  hObjectParent (u32)
        //   +8  hObjectNew  (u32)
        //   +12 hClass      (u32)  ← object class (channel/context/memory type)
        //   +16 pAllocParms (u64, 8-byte aligned)
        // status: +28 in NVOS21, +40 in NVOS64 (see read_rm_status)
        __u32 h_class, h_parent, h_new;
        READ_FIELD(h_parent, buf, 4);
        READ_FIELD(h_new, buf, 8);
        READ_FIELD(h_class, buf, 12);
        e->d0 = h_class;
        e->d1 = h_parent;
        e->d2 = h_new;
        break;
    }
    case NV_ESC_RM_CONTROL:
    {
        // NVOS54_PARAMETERS layout:
        //   +0  hClient     (u32)
        //   +4  hObject     (u32)
        //   +8  cmd         (u32)  ← RM sub-command (the interesting part)
        //   +12 flags       (u32)
        //   +16 params      (u64, 8-byte aligned)
        //   +24 paramsSize  (u32)
        //   +28 status      (u32)
        __u32 ctrl_cmd, h_object, params_size;
        READ_FIELD(h_object, buf, 4);
        READ_FIELD(ctrl_cmd, buf, 8);
        READ_FIELD(params_size, buf, 24);
        e->d0 = ctrl_cmd;
        e->d1 = h_object;
        e->d2 = params_size;
        break;
    }
    case NV_ESC_RM_ALLOC_MEMORY:
    {
        // NVOS02_PARAMETERS layout:
        //   +0  hRoot       (u32)
        //   +4  hObjectParent (u32)
        //   +8  hObjectNew  (u32)
        //   +12 hClass      (u32)  ← memory class
        //   +16 flags       (u32)
        //   +24 pMemory     (u64, 8-byte aligned)
        //   +32 limit       (u64)  ← size - 1
        //   +40 status      (u32)
        __u32 h_class, flags;
        __u64 limit;
        READ_FIELD(h_class, buf, 12);
        READ_FIELD(flags, buf, 16);
        READ_FIELD(limit, buf, 32);
        e->d0 = h_class;
        e->d1 = flags;
        e->d4 = limit + 1; // size in bytes
        break;
    }
    case NV_ESC_RM_FREE:
    {
        // NVOS00_PARAMETERS layout:
        //   +0  hRoot          (u32)
        //   +4  hObjectParent  (u32)
        //   +8  hObjectOld     (u32) ← handle being freed
        //   +12 status         (u32)
        __u32 h_object, h_parent;
        READ_FIELD(h_parent, buf, 4);
        READ_FIELD(h_object, buf, 8);
        e->d0 = h_object;
        e->d1 = h_parent;
        break;
    }
    }
}

// ---- read the RM status the driver wrote back into the user's NVOS* struct ----
// The ioctl itself usually returns 0 even when the RM call fails; the real
// result is the struct's status field. Offsets per open-gpu-kernel-modules nvos.h.
static __always_inline __u32 read_rm_status(__u8 cmd, __u16 size, __u64 user_ptr)
{
    __u32 off;
    switch (cmd)
    {
    case NV_ESC_RM_ALLOC:
        if (size == NVOS21_SIZE)
            off = 28;
        else if (size == NVOS64_SIZE)
            off = 40;
        else
            return RM_STATUS_UNKNOWN;
        break;
    case NV_ESC_RM_CONTROL:
        off = 28; // NVOS54_PARAMETERS
        break;
    case NV_ESC_RM_ALLOC_MEMORY:
        off = 40; // NVOS02_PARAMETERS
        break;
    case NV_ESC_RM_FREE:
        off = 12; // NVOS00_PARAMETERS
        break;
    default:
        return RM_STATUS_UNKNOWN; // VID_HEAP layout not decoded
    }

    __u32 status;
    if (bpf_probe_read_user(&status, sizeof(status), (void *)(user_ptr + off)) != 0)
        return RM_STATUS_UNKNOWN;
    return status;
}

// ---- kprobe entry ----
SEC("kprobe/nvidia_unlocked_ioctl")
int BPF_KPROBE(trace_nvidia_ioctl_enter,
               struct file *filp, unsigned int cmd, unsigned long arg)
{
    __u8 nv_cmd = NV_CMD(cmd);
    __u64 real_arg = arg;
    __u8 real_cmd = nv_cmd;
    __u16 real_size = NV_SIZE(cmd);

    // Handle XFER_CMD tunnel: arg points to nv_ioctl_xfer_t { cmd(u32), size(u32), ptr(u64) }
    if (nv_cmd == NV_ESC_IOCTL_XFER_CMD)
    {
        __u8 xbuf[16] = {};
        if (bpf_probe_read_user(xbuf, sizeof(xbuf), (void *)arg) != 0)
            return 0;
        __u32 xcmd, xsize;
        __u64 xptr;
        READ_FIELD(xcmd, xbuf, 0);
        READ_FIELD(xsize, xbuf, 4);
        READ_FIELD(xptr, xbuf, 8);
        real_cmd = (__u8)xcmd;
        real_size = (__u16)xsize;
        real_arg = xptr;
    }

    // Filter to RM ops we care about
    if (real_cmd != NV_ESC_RM_ALLOC_MEMORY &&
        real_cmd != NV_ESC_RM_ALLOC &&
        real_cmd != NV_ESC_RM_CONTROL &&
        real_cmd != NV_ESC_RM_FREE &&
        real_cmd != NV_ESC_RM_VID_HEAP_CONTROL)
        return 0;

    __u8 evt_type;
    switch (real_cmd)
    {
    case NV_ESC_RM_ALLOC_MEMORY:
        evt_type = EVT_MEM_ALLOC;
        break;
    case NV_ESC_RM_ALLOC:
        evt_type = EVT_OBJ_ALLOC;
        break;
    case NV_ESC_RM_CONTROL:
        evt_type = EVT_RM_CONTROL;
        break;
    case NV_ESC_RM_FREE:
        evt_type = EVT_OBJ_FREE;
        break;
    default:
        evt_type = EVT_VID_HEAP;
        break;
    }

    // Stash for kretprobe
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u64 now = bpf_ktime_get_ns();
    struct kprobe_state st = {
        .entry_ts = now,
        .arg_ptr = real_arg,
        .nv_cmd = real_cmd,
        .evt_type = evt_type,
        .arg_size = real_size,
    };
    bpf_map_update_elem(&active_ioctls, &pid_tgid, &st, BPF_ANY);

    struct gpu_event *e = get_slot();
    if (!e)
        return 0;

    fill_common(e, evt_type, real_cmd, IS_ENTRY, real_arg);
    decode_rm_args(e, real_cmd, real_arg);

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, e, sizeof(*e));
    return 0;
}

// ---- kretprobe exit: emit latency event ----
SEC("kretprobe/nvidia_unlocked_ioctl")
int BPF_KRETPROBE(trace_nvidia_ioctl_exit, long ret)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    struct kprobe_state *st = bpf_map_lookup_elem(&active_ioctls, &pid_tgid);
    if (!st)
        return 0;

    // Exit event: d2 = ioctl ret, d3 = RM status, d4 = latency in ns.
    struct gpu_event *e = get_slot();
    if (!e)
        goto cleanup;

    fill_common(e, st->evt_type, st->nv_cmd, IS_EXIT, st->arg_ptr);
    e->d2 = (__u32)ret;
    e->d3 = read_rm_status(st->nv_cmd, st->arg_size, st->arg_ptr);
    e->d4 = bpf_ktime_get_ns() - st->entry_ts;

    bpf_perf_event_output(ctx, &events, BPF_F_CURRENT_CPU, e, sizeof(*e));

cleanup:
    bpf_map_delete_elem(&active_ioctls, &pid_tgid);
    return 0;
}

char _license[] SEC("license") = "GPL";
