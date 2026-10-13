use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(
        std::env::var("OUT_DIR").expect("OUT_DIR must be set in build script"),
    );
    let src = PathBuf::from("src/bpf");

    // Generate vmlinux.h if it doesn't exist yet (needs bpftool on PATH)
    let vmlinux = src.join("vmlinux.h");
    if !vmlinux.exists() {
        let status = std::process::Command::new("bpftool")
            .args(["btf", "dump", "file", "/sys/kernel/btf/vmlinux", "format", "c"])
            .stdout(std::fs::File::create(&vmlinux).expect("create vmlinux.h"))
            .status()
            .expect("bpftool not found — install bpftool to generate vmlinux.h");
        assert!(status.success(), "bpftool failed to generate vmlinux.h");
    }

    libbpf_cargo::SkeletonBuilder::new()
        .source(src.join("gpu_tracer.bpf.c"))
        .clang_args([
            "-I", src.to_str().unwrap(),
            "-D__TARGET_ARCH_x86",
        ])
        .build_and_generate(out.join("gpu_tracer.skel.rs"))
        .expect("BPF skeleton build failed");

    println!("cargo:rerun-if-changed=src/bpf/gpu_tracer.bpf.c");
}
