mod skel {
    include!(concat!(env!("OUT_DIR"), "/gpu_tracer.skel.rs"));
}

use anyhow::{Context, Result};
use clap::Parser;
use libbpf_rs::{
    PerfBufferBuilder,
    skel::{OpenSkel, Skel, SkelBuilder},
};
use std::{
    mem::MaybeUninit,
    sync::Arc,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use skel::GpuTracerSkelBuilder;

// Must match gpu_tracer.bpf.c exactly (repr(C), same field order).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuEvent {
    ts_ns: u64,
    pid: u32,
    tid: u32,
    evt_type: u8,
    nv_cmd: u8,
    is_exit: u8, // 0 = entry, 1 = exit
    pad: u8,
    arg_ptr: u64,
    comm: [u8; 16],
    d0: u32, // entry: hClass / ctrl_cmd / h_object
    d1: u32, // entry: hParent / h_object / flags
    d2: u32, // entry: hNew / params_size | exit: ioctl ret (0 or -errno)
    d3: u32, // exit: RM status (0 = NV_OK, RM_STATUS_UNKNOWN if unreadable)
    d4: u64, // entry: alloc size | exit: latency_ns
}

const RM_STATUS_UNKNOWN: u32 = 0xFFFF_FFFF;

const EVT_MEM_ALLOC: u8 = 1;
const EVT_OBJ_ALLOC: u8 = 2;
const EVT_RM_CONTROL: u8 = 3;
const EVT_OBJ_FREE: u8 = 4;
const EVT_VID_HEAP: u8 = 5;

// ---- known RM_CONTROL sub-commands ----
// Format: (class << 16) | method
fn ctrl_cmd_name(cmd: u32) -> Option<&'static str> {
    match cmd {
        // NV0000: root / client-level
        0x0000_0002 => Some("GPU_GET_ATTACHED_IDS"),
        0x0000_0106 => Some("GPU_GET_ID_INFO"),
        0x0000_0702 => Some("SYNC_BOOST_GROUP_INFO"),
        0x0000_0902 => Some("EVENT_SET_NOTIFICATION"),
        // NV2080: subdevice (GPU engine ops)
        0x2080_0101 => Some("GR_GET_CAPS"),
        0x2080_0110 => Some("GR_GET_INFO"),
        0x2080_0114 => Some("GR_CTXSW_PREEMPTION_BIND"),
        0x2080_0120 => Some("GR_GET_GLOBAL_SM_ORDER"),
        0x2080_0a06 => Some("CE_GET_CAPS_V2"),
        0x2080_1028 => Some("FB_FLUSH_GPU_CACHE"),
        0x2080_1301 => Some("FB_GET_INFO"),
        0x2080_1323 => Some("FB_GET_INFO_V2"),
        0x2080_1401 => Some("PERF_GET_CURRENT_PSTATE"),
        0x2080_1601 => Some("PERF_GPUMON_UTIL_SAMPLES"),
        0x2080_1702 => Some("PERF_GET_CLK_BOOST_TABLE"),
        0x2080_2802 => Some("TIMER_GET_TIME"),
        0x2080_0d01 => Some("BUS_GET_INFO"),
        0x2080_0d03 => Some("BUS_GET_PCI_INFO"),
        0x2080_2201 => Some("INTERNAL_GPU_START_FABRIC_PROBE"),
        // NVA06F: GPFIFO channel
        0xa06f_0101 => Some("GPFIFO_STOP_CHANNEL"),
        0xa06f_0103 => Some("GPFIFO_SCHEDULE"),
        // NV906F: older GPFIFO
        0x906f_0101 => Some("RESET_CHANNEL"),
        // NV503C: video memory
        0x503c_0102 => Some("REGISTER_VA_SPACE"),
        _ => None,
    }
}

// ---- known RM_ALLOC object class codes ----
fn class_name(class: u32) -> String {
    let known: Option<&str> = match class {
        0x0001 => Some("NV_ROOT_CLIENT"),
        0x0002 => Some("NV_DEVICE"),
        0x0005 => Some("NV_TIMER"),
        0x0041 => Some("MEM_LOCAL_USER"),
        0x0071 => Some("MEM_VIRTUAL"),
        0x0080 => Some("MEM_SYSTEM"),
        0x00C2 => Some("MEM_EXTENDED_USER"),
        0x00F1 => Some("MEM_FABRIC"),
        0x2080 => Some("SUBDEVICE"),
        0x503C => Some("DEFERRED_API"),
        // GPU FIFO channel types (compute channels → kernel launch capable)
        0xA06C => Some("CHAN_GPFIFO_KEPLER_C"),
        0xA06F => Some("CHAN_GPFIFO_KEPLER"),
        0xB06F => Some("CHAN_GPFIFO_MAXWELL"),
        0xC06F => Some("CHAN_GPFIFO"),
        0xC36F => Some("CHAN_GPFIFO_AMPERE"),
        0xC56F => Some("CHAN_GPFIFO_HOPPER"),
        // Compute engine contexts
        0xA297 => Some("COMPUTE_KEPLER_A"),
        0xB197 => Some("COMPUTE_MAXWELL_A"),
        0xC3C0 => Some("COMPUTE_AMPERE_A"),
        0xA0C0 => Some("COMPUTE_KEPLER_C"),
        // DMA copy engine contexts
        0xA0B5 => Some("DMA_COPY_KEPLER"),
        0xB0B5 => Some("DMA_COPY_MAXWELL"),
        0xC0B5 => Some("DMA_COPY_TURING"),
        0xC3B5 => Some("DMA_COPY_AMPERE"),
        0xC6B5 => Some("DMA_COPY_HOPPER"),
        // VA space
        0x90F1 => Some("VA_SPACE"),
        _ => None,
    };
    known
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{:#06x}", class))
}

fn evt_label(t: u8) -> &'static str {
    match t {
        EVT_MEM_ALLOC => "mem_alloc ",
        EVT_OBJ_ALLOC => "obj_alloc ",
        EVT_RM_CONTROL => "rm_control",
        EVT_OBJ_FREE => "obj_free  ",
        EVT_VID_HEAP => "vid_heap  ",
        _ => "unknown   ",
    }
}

fn comm_str(c: &[u8; 16]) -> &str {
    let n = c.iter().position(|&b| b == 0).unwrap_or(16);
    std::str::from_utf8(&c[..n]).unwrap_or("?")
}

fn human_bytes(n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    if n >= 1 << 30 {
        return format!("{:.1}GiB", n as f64 / (1u64 << 30) as f64);
    }
    if n >= 1 << 20 {
        return format!("{:.1}MiB", n as f64 / (1u64 << 20) as f64);
    }
    if n >= 1 << 10 {
        return format!("{:.1}KiB", n as f64 / (1u64 << 10) as f64);
    }
    format!("{}B", n)
}

fn handle_event(_cpu: i32, data: &[u8]) {
    if data.len() < std::mem::size_of::<GpuEvent>() {
        return;
    }
    let e: GpuEvent = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const GpuEvent) };

    let ts = e.ts_ns as f64 / 1_000_000_000.0;
    let comm = comm_str(&e.comm);

    // Exit events: show latency, RM status and the ioctl's own return value.
    // RM failures usually come back with ioctl ret = 0, so the RM status is
    // the field that says whether the call actually worked.
    if e.is_exit != 0 {
        let us = e.d4 as f64 / 1_000.0;
        let errno = e.d2 as i32;
        let ok = match (errno, e.d3) {
            (0, 0) => "ok".to_string(),
            (0, RM_STATUS_UNKNOWN) => "unknown".to_string(),
            (0, rm) => format!("err rm=0x{rm:x}"),
            (n, _) => format!("err errno={n}"),
        };
        println!(
            "[{ts:>16.6}]   └─ {lbl} done  pid={pid:<6} comm={comm:<16} status={ok}  latency={us:.1}µs",
            lbl = evt_label(e.evt_type),
            pid = e.pid,
        );
        return;
    }

    match e.evt_type {
        EVT_OBJ_ALLOC => println!(
            "[{ts:>16.6}] {lbl} pid={pid:<6} comm={comm:<16} class={cls:<24} hParent={par:#010x} hNew={new:#010x}",
            lbl = evt_label(e.evt_type),
            pid = e.pid,
            cls = class_name(e.d0),
            par = e.d1,
            new = e.d2,
        ),
        EVT_RM_CONTROL => {
            let cmd_s = ctrl_cmd_name(e.d0)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("{:#010x}", e.d0));
            println!(
                "[{ts:>16.6}] {lbl} pid={pid:<6} comm={comm:<16} cmd={cmd:<32} hObject={obj:#010x} paramsSize={psz}",
                lbl = evt_label(e.evt_type),
                pid = e.pid,
                cmd = cmd_s,
                obj = e.d1,
                psz = e.d2,
            );
        }
        EVT_MEM_ALLOC => println!(
            "[{ts:>16.6}] {lbl} pid={pid:<6} comm={comm:<16} class={cls:<20} size={sz:<10} flags={flags:#010x}",
            lbl = evt_label(e.evt_type),
            pid = e.pid,
            cls = class_name(e.d0),
            sz = human_bytes(e.d4),
            flags = e.d1,
        ),
        EVT_OBJ_FREE => println!(
            "[{ts:>16.6}] {lbl} pid={pid:<6} comm={comm:<16} hObject={obj:#010x} hParent={par:#010x}",
            lbl = evt_label(e.evt_type),
            pid = e.pid,
            obj = e.d0,
            par = e.d1,
        ),
        _ => println!(
            "[{ts:>16.6}] {lbl} pid={pid:<6} comm={comm:<16} arg={arg:#018x}",
            lbl = evt_label(e.evt_type),
            pid = e.pid,
            arg = e.arg_ptr,
        ),
    }
}

fn handle_lost(cpu: i32, count: u64) {
    eprintln!("Lost {count} events on CPU {cpu}");
}

#[derive(Parser)]
#[command(
    name = "nv-gpu-tracer",
    about = "eBPF-based NVIDIA GPU workload tracer"
)]
struct Cli {
    #[arg(long, default_value_t = 64)]
    pages: usize,

    /// Only show events from this PID (omit to trace all)
    #[arg(long)]
    pid: Option<u32>,

    /// Only show events from processes whose name contains this string
    #[arg(long)]
    comm: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    if unsafe { libc::geteuid() } != 0 {
        anyhow::bail!("nv-gpu-tracer requires root (or CAP_BPF + CAP_PERFMON)");
    }

    bump_memlock_rlimit();

    let skel_builder = GpuTracerSkelBuilder::default();
    let mut obj = MaybeUninit::uninit();
    let open_skel = skel_builder
        .open(&mut obj)
        .context("Failed to open BPF skeleton")?;
    let mut skel = open_skel.load().context("Failed to load BPF program")?;
    skel.attach().context("Failed to attach kprobes")?;

    let pid_filter = cli.pid;
    let comm_filter = cli.comm.clone();

    let perf = PerfBufferBuilder::new(&skel.maps.events)
        .pages(cli.pages)
        .sample_cb(move |cpu, data| {
            if data.len() < std::mem::size_of::<GpuEvent>() {
                return;
            }
            let e: GpuEvent = unsafe { std::ptr::read_unaligned(data.as_ptr() as *const GpuEvent) };
            if let Some(p) = pid_filter {
                if e.pid != p {
                    return;
                }
            }
            if let Some(ref c) = comm_filter {
                if !comm_str(&e.comm).contains(c.as_str()) {
                    return;
                }
            }
            handle_event(cpu, data);
        })
        .lost_cb(handle_lost)
        .build()
        .context("Failed to create perf buffer")?;

    println!("Tracing NVIDIA ioctl calls... (Ctrl-C to stop)\n");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || r.store(false, Ordering::SeqCst))?;

    while running.load(Ordering::SeqCst) {
        perf.poll(Duration::from_millis(100))?;
    }

    println!("\nDone.");
    Ok(())
}

fn bump_memlock_rlimit() {
    let rl = libc::rlimit {
        rlim_cur: libc::RLIM_INFINITY,
        rlim_max: libc::RLIM_INFINITY,
    };
    unsafe { libc::setrlimit(libc::RLIMIT_MEMLOCK, &rl) };
}
