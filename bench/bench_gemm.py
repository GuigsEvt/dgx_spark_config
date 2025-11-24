import time
import torch


def bench_gemm_loop(
    M=4096,
    N=4096,
    K=4096,
    dtype=torch.float16,
    target_seconds=60.0,
    warmup=20,
):
    device = torch.device("cuda")
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)

    flops_per_matmul = 2.0 * M * N * K

    print(f"\n[Config]")
    print(f"  M,N,K        = {M}, {N}, {K}")
    print(f"  dtype        = {dtype}")
    print(f"  target time  = {target_seconds:.1f} s")
    print(f"  warmup iters = {warmup}")

    # Warmup
    print("\n[Warmup]")
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    print("\n[Burn-in]")
    start = time.perf_counter()
    last_report = start
    iters = 0
    total_flops = 0.0

    # Run until we've hit target_seconds
    while True:
        _ = a @ b
        iters += 1
        total_flops += flops_per_matmul

        # Every ~1s, print a status line
        now = time.perf_counter()
        elapsed = now - start
        if now - last_report >= 1.0:
            avg_time = elapsed / iters
            tflops = (flops_per_matmul / avg_time) / 1e12
            print(
                f"  t={elapsed:5.1f}s | iters={iters:5d} | "
                f"avg={avg_time*1e3:7.2f} ms | ~{tflops:7.2f} TFLOPs"
            )
            last_report = now

        if elapsed >= target_seconds:
            break

    torch.cuda.synchronize()
    end = time.perf_counter()
    elapsed = end - start
    avg_time = elapsed / iters
    tflops = (flops_per_matmul / avg_time) / 1e12

    print("\n[Summary]")
    print(f"  Total time   : {elapsed:.2f} s")
    print(f"  Total iters  : {iters}")
    print(f"  Avg per iter : {avg_time*1e3:.2f} ms")
    print(f"  Effective    : {tflops:.2f} TFLOPs (FP16)")
    return tflops


def main():
    print("=== DGX Spark FP16 GEMM Burn-in ===")

    print("\n[PyTorch info]")
    print("  torch.version        :", torch.__version__)
    print("  torch.cuda.version   :", torch.version.cuda)
    print("  cudnn.version        :", torch.backends.cudnn.version())

    if not torch.cuda.is_available():
        print("\n[ERROR] CUDA is not available. Are you in the right env / on the DGX?")
        return

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    print("\n[CUDA device]")
    print(f"  Name        : {props.name}")
    print(f"  Device ID   : {device}")
    print(f"  SM count    : {props.multi_processor_count}")
    print(f"  Total mem   : {props.total_memory / (1024**3):.2f} GB")
    print(f"  Compute cap : {props.major}.{props.minor}")

    free_mem, total_mem = torch.cuda.mem_get_info()
    print("\n[Memory]")
    print(f"  Free        : {free_mem / (1024**3):.2f} GB")
    print(f"  Total       : {total_mem / (1024**3):.2f} GB")

    # Tunable workload parameters
    M = N = K = 8192
    target_seconds = 60

    torch.cuda.empty_cache()
    bench_gemm_loop(
        M=M,
        N=N,
        K=K,
        dtype=torch.float16,
        target_seconds=target_seconds,
        warmup=20,
    )

    print("\nDone. Check your nvidia-smi timeline / DCGM metrics for this window.")


if __name__ == "__main__":
    main()
