# nv-gpu-tracer

eBPF + Rust tracer for the NVIDIA driver boundary. It kprobes `nvidia_unlocked_ioctl`,
decodes the Resource Manager (RM) escape calls a CUDA process makes, and prints each
call with its arguments, RM status and latency.

Use it to see what a GPU workload asks the driver for: memory allocations and their
sizes, channel and context creation, RM control calls, frees, and which of them fail.

## Quick start

```bash
# 1. Build deps (Ubuntu/Debian) + a Rust toolchain
sudo apt-get install -y clang libelf-dev zlib1g-dev pkg-config
# bpftool is only needed if src/bpf/vmlinux.h is missing:
#   Ubuntu: sudo apt-get install linux-tools-$(uname -r)   Debian: sudo apt-get install bpftool

# 2. Build
cargo build --release

# 3. Run (needs root)
sudo ./target/release/nv-gpu-tracer

# 4. In another terminal, generate GPU work
cd cuda-loadgen && uv run main.py
```

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--pid <PID>` | all | Only show events from this process ID |
| `--comm <STR>` | all | Only show processes whose name contains `STR` (e.g. `python`) |
| `--pages <N>` | `64` | Perf buffer pages per CPU. Raise it if you see `Lost N events` |

Filtering happens in userspace, so every NVIDIA ioctl on the host is still captured in the kernel.

```bash
sudo ./target/release/nv-gpu-tracer --comm python
sudo ./target/release/nv-gpu-tracer --pid "$(pgrep -f main.py)" --pages 256
```

## What gets traced

| Event | RM escape | Shown on entry |
|---|---|---|
| `mem_alloc` | `NV_ESC_RM_ALLOC_MEMORY` (0x27) | memory class, size, flags |
| `obj_alloc` | `NV_ESC_RM_ALLOC` (0x29) | object class (channel, compute/copy context, VA space…), parent and new handle |
| `rm_control` | `NV_ESC_RM_CONTROL` (0x2A) | control command (named when known), object handle, params size |
| `obj_free` | `NV_ESC_RM_FREE` (0x2B) | handle being freed, parent |
| `vid_heap` | `NV_ESC_RM_VID_HEAP_CONTROL` (0x4A) | raw arg pointer only |

Calls tunnelled through `NV_ESC_IOCTL_XFER_CMD` (0xD3) are unwrapped and decoded as the inner call.
Every traced call also gets an exit line with status and latency.

## Reading the output

```
[ts]            <event>   pid=<pid> comm=<name> <decoded args>
[ts]              └─ <event> done  pid=<pid> comm=<name> status=<status>  latency=<µs>
```

| `status=` | Meaning |
|---|---|
| `ok` | ioctl returned 0 and RM status is `NV_OK` |
| `err rm=0x…` | ioctl succeeded but RM rejected the call. Look the code up in `nvstatuscodes.h` |
| `err errno=N` | the ioctl itself failed (e.g. `-14` = `EFAULT`) |
| `unknown` | RM status couldn't be read (`vid_heap`, an unrecognised struct size, or an unreadable user page) |

Unknown classes and control commands print as hex. Add names in `class_name()` and
`ctrl_cmd_name()` in `src/main.rs`.

## How it works

```
CUDA app ──ioctl──▶ nvidia_unlocked_ioctl ──▶ RM
                │                         │
            kprobe                    kretprobe
     decode NVOS* args from        read NVOS*.status back,
     user memory, stash entry      compute latency, emit exit
     time keyed by pid_tgid        event
                │                         │
                └────── perf buffer ──────┘
                           ▼
                 nv-gpu-tracer (Rust, libbpf-rs)
                  filter, name, print
```

- BPF side: `src/bpf/gpu_tracer.bpf.c`. Struct offsets follow open-gpu-kernel-modules `nvos.h`.
- Userspace: `src/main.rs`. `GpuEvent` must match `struct gpu_event` field for field.
- `build.rs` compiles the BPF object and generates the skeleton with `libbpf-cargo`.

## Limits

- **Setup and teardown only, not kernel launches.** CUDA submits work by writing to GPFIFO
  rings and doorbells from userspace, so launches never hit an ioctl. You see the channels
  and memory being set up, not each kernel.
- **`/dev/nvidia-uvm` is not traced.** Unified memory and page-fault traffic go through a different driver entry point.
- **Driver-version sensitive.** The RM struct layouts and escape numbers can change between driver releases. If decoded fields look wrong, check them against your driver's `nvos.h`.
- **Needs root.** The binary checks for euid 0. `CAP_BPF` + `CAP_PERFMON` alone aren't enough yet.
- **x86_64 only** as built (`-D__TARGET_ARCH_x86` in `build.rs`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Failed to attach kprobes` | NVIDIA module not loaded (`lsmod \| grep nvidia`), or the symbol isn't there: `grep nvidia_unlocked_ioctl /proc/kallsyms` |
| `gelf.h` / `libelf.h: No such file` during build | `sudo apt-get install libelf-dev zlib1g-dev` |
| `bpftool not found` during build | Install bpftool. It's only needed when `src/bpf/vmlinux.h` is missing |
| BPF load fails on a different kernel | Regenerate BTF headers: `rm src/bpf/vmlinux.h && cargo build` |
| `Lost N events on CPU X` | Raise `--pages`, or filter with `--pid` / `--comm` |
| No output while CUDA runs | Check the app is really on the GPU (`nvidia-smi`) and that your filter matches its `comm` (15 chars max) |

## Test workload

`cuda-loadgen/main.py` runs about 30 s of 4096×4096 PyTorch matmuls, elementwise ops and
reductions on CUDA. Needs Python ≥ 3.14 and a CUDA build of PyTorch (`uv run main.py`).

## License

The BPF program is GPL-2.0 (required to use GPL-only BPF helpers).
