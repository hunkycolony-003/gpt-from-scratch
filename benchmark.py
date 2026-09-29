import argparse
import time
import os
import csv
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import GPT, GPTconfig

def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

def measure_incremental_peak_memory(model, x, device):
    """
    Measures incremental forward-pass peak memory allocation in MB (above static model weights).
    """
    incremental_peak_mb = 0.0

    if device == "cuda":
        torch.cuda.synchronize()
        base_mem = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated(device)
        incremental_peak_mb = max(0.0, (peak_mem - base_mem) / (1024 * 1024))

    elif device == "mps":
        torch.mps.synchronize()
        before_mem = torch.mps.current_allocated_memory()
        with torch.no_grad():
            _ = model(x)
        torch.mps.synchronize()
        after_mem = torch.mps.current_allocated_memory()
        incremental_peak_mb = max(0.0, (after_mem - before_mem) / (1024 * 1024))

    return incremental_peak_mb

def measure_prefill_latency_and_throughput(model, x, device, n_warmup, n_runs):
    """
    Runs 'n_warmup' passes, then times 'n_runs' passes for prefill.
    Returns: mean_ms, median_ms, p10_ms, p90_ms, tokens_per_sec.
    """
    B, T = x.shape
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)

    run_latencies_ms = []
    if device == "cuda":
        for _ in range(n_runs):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            run_latencies_ms.append(start_event.elapsed_time(end_event))
    else:
        for _ in range(n_runs):
            sync(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            sync(device)
            t1 = time.perf_counter()
            run_latencies_ms.append((t1 - t0) * 1000)

    mean_ms = float(np.mean(run_latencies_ms))
    median_ms = float(np.median(run_latencies_ms))
    p10_ms = float(np.percentile(run_latencies_ms, 10))
    p90_ms = float(np.percentile(run_latencies_ms, 90))
    tokens_per_sec = (B * T * 1000.0) / max(median_ms, 1e-6)

    return mean_ms, median_ms, p10_ms, p90_ms, tokens_per_sec

def measure_cached_decode(model, prompt, max_decode_tokens, device):
    """
    Runs autoregressive decoding starting from 'prompt' using native cached decoding.
    Returns per-step latency (ms), decode tokens/sec, and cumulative latency.
    """
    model.eval()
    B, T_prompt = prompt.shape

    # 1. Prompt Prefill into dynamic cache
    with torch.no_grad():
        logits, _, kv_caches = model(prompt)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # 2. Autoregressive Decode Steps
    step_records = []
    total_time_ms = 0.0

    with torch.no_grad():
        for step in range(max_decode_tokens):
            sync(device)
            t0 = time.perf_counter()

            logits, _, kv_caches = model(next_token, kv_caches=kv_caches)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            sync(device)
            t1 = time.perf_counter()

            step_ms = (t1 - t0) * 1000.0
            total_time_ms += step_ms
            context_len = T_prompt + step + 1
            step_tokens_per_sec = (B * 1000.0) / max(step_ms, 1e-6)

            step_records.append({
                "step_idx": step + 1,
                "context_len": context_len,
                "step_latency_ms": step_ms,
                "decode_tokens_per_sec": step_tokens_per_sec,
                "cumulative_ms": total_time_ms
            })

    return step_records

def analytical_kv_cache_mb(config, seq_len, batch_size, bytes_per_param=2):
    """
    Analytical formula for KV cache memory footprint.
    """
    if config.attn_type == "mla":
        latent_dim = getattr(config, "kv_lora_rank", 128) + getattr(config, "qk_rope_dim", 32)
        total_bytes = config.n_layers * latent_dim * seq_len * batch_size * bytes_per_param
        return total_bytes / (1024 * 1024)

    kv_heads = config.n_heads if config.attn_type == "mha" else (
        1 if config.attn_type == "mqa" else config.n_heads // config.heads_per_group
    )
    head_dim = config.n_embed // config.n_heads
    total_bytes = 2 * config.n_layers * kv_heads * seq_len * head_dim * batch_size * bytes_per_param
    return total_bytes / (1024 * 1024)

def plot_and_save(output_dir):
    """
    Generates publication-quality Matplotlib plots from prefill and decode CSVs.
    """
    colors = {"mha": "#1f77b4", "mqa": "#ff7f0e", "gqa": "#2ca02c", "mla": "#d62728"}

    # 1. Plot Analytical & Measured Preallocated KV Cache Memory
    prefill_file = os.path.join(output_dir, "prefill.csv")
    if os.path.exists(prefill_file):
        rows = []
        with open(prefill_file, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        target_bsz = "32" if any(r["batch_size"] == "32" for r in rows) else rows[0]["batch_size"]
        plt.figure(figsize=(8, 5), dpi=300)
        for attn in ["mha", "gqa", "mqa", "mla"]:
            attn_rows = [r for r in rows if r["attention"] == attn and r["batch_size"] == target_bsz]
            if not attn_rows:
                continue
            attn_rows.sort(key=lambda x: int(x["seq_len"]))
            seqs = [int(r["seq_len"]) for r in attn_rows]
            kv_mbs = [float(r["analytical_cache_mb"]) for r in attn_rows]
            plt.plot(seqs, kv_mbs, marker="o", linewidth=2.2, label=f"{attn.upper()} (Preallocated)", color=colors.get(attn))

        plt.title(f"Preallocated KV Cache Footprint vs Sequence Length (Batch Size = {target_bsz})", fontsize=12, fontweight="bold")
        plt.xlabel("Sequence Length (Tokens)", fontsize=11)
        plt.ylabel("KV Cache Memory (MB)", fontsize=11)
        plt.legend(frameon=True, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "kv_cache_memory.png"))
        plt.close()

    # 2. Plot Prefill Throughput vs Batch Size
    if os.path.exists(prefill_file):
        rows = []
        with open(prefill_file, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        target_seq = "1024" if any(r["seq_len"] == "1024" for r in rows) else rows[0]["seq_len"]
        plt.figure(figsize=(8, 5), dpi=300)
        for attn in ["mha", "gqa", "mqa", "mla"]:
            attn_rows = [r for r in rows if r["attention"] == attn and r["seq_len"] == target_seq]
            if not attn_rows:
                continue
            attn_rows.sort(key=lambda x: int(x["batch_size"]))
            bszes = [int(r["batch_size"]) for r in attn_rows]
            tps = [float(r["tokens_per_sec"]) for r in attn_rows]
            plt.plot(bszes, tps, marker="s", linewidth=2.2, label=attn.upper(), color=colors.get(attn))

        plt.title(f"Prefill Throughput vs Batch Size (Seq Len = {target_seq})", fontsize=12, fontweight="bold")
        plt.xlabel("Batch Size", fontsize=11)
        plt.ylabel("Prefill Throughput (Tokens / Sec)", fontsize=11)
        plt.legend(frameon=True, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "throughput_scaling.png"))
        plt.savefig(os.path.join(output_dir, "prefill_throughput.png"))
        plt.close()

    # 3. Plot Decode Generation Latency & Throughput
    dec_file = os.path.join(output_dir, "decode.csv")
    if os.path.exists(dec_file):
        rows = []
        with open(dec_file, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append(r)

        # Plot 3A: Cumulative generation latency
        target_bsz = "1"
        plt.figure(figsize=(8, 5), dpi=300)
        for attn in ["mha", "gqa", "mqa", "mla"]:
            attn_rows = [r for r in rows if r["attention"] == attn and r["batch_size"] == target_bsz]
            if not attn_rows:
                continue
            attn_rows.sort(key=lambda x: int(x["step_idx"]))
            steps = [int(r["step_idx"]) for r in attn_rows]
            cum_ms = [float(r["cumulative_ms"]) for r in attn_rows]
            plt.plot(steps, cum_ms, marker="^", markersize=3, linewidth=2.0, label=attn.upper(), color=colors.get(attn))

        plt.title("Cumulative Generation Latency with Preallocated KV Cache", fontsize=12, fontweight="bold")
        plt.xlabel("Generated Token Step (T=1)", fontsize=11)
        plt.ylabel("Cumulative Latency (ms)", fontsize=11)
        plt.legend(frameon=True, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "generation_latency.png"))
        plt.close()

        # Plot 3B: Decode Tokens/Sec vs Batch Size
        plt.figure(figsize=(8, 5), dpi=300)
        for attn in ["mha", "gqa", "mqa", "mla"]:
            attn_bsz_groups = {}
            for r in rows:
                if r["attention"] == attn:
                    bsz = int(r["batch_size"])
                    attn_bsz_groups.setdefault(bsz, []).append(float(r["decode_tokens_per_sec"]))
            if not attn_bsz_groups:
                continue
            sorted_bsz = sorted(attn_bsz_groups.keys())
            mean_tps = [float(np.mean(attn_bsz_groups[b])) for b in sorted_bsz]
            plt.plot(sorted_bsz, mean_tps, marker="D", linewidth=2.2, label=attn.upper(), color=colors.get(attn))

        plt.title("Cached Autoregressive Decode Throughput vs Batch Size", fontsize=12, fontweight="bold")
        plt.xlabel("Batch Size", fontsize=11)
        plt.ylabel("Decode Throughput (Tokens / Sec)", fontsize=11)
        plt.legend(frameon=True, fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "decode_throughput.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Attention Mechanism Benchmark Suite")
    parser.add_argument("--attention", nargs="+", choices=["mha", "mqa", "gqa", "mla"], default=["mha", "mqa", "gqa", "mla"])
    parser.add_argument("--seq-len", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--batch-size", nargs="+", type=int, default=[1, 8, 16, 32])
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16", "float16"], default="bfloat16" if torch.cuda.is_available() else "float32")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--kernel", type=str, default="eager")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile")
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--decode-prompt-len", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    torch_dtype = getattr(torch, args.dtype)
    bytes_per_param = 2 if args.dtype in ["bfloat16", "float16"] else 4

    # Save reproducibility metadata
    metadata = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "device": args.device,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "seed": args.seed,
        "dtype": args.dtype,
        "kernel": f"{args.kernel}_compiled" if args.compile else args.kernel,
        "n_warmup": args.n_warmup,
        "n_runs": args.n_runs,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    }
    with open(os.path.join(args.output_dir, "benchmark_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    max_benchmark_seq = max(args.seq_len)

    # Base 124M GPT-2 scale config
    base_config = GPTconfig(
        n_layers=12,
        n_heads=12,
        n_embed=768,
        vocab_size=50257,
        heads_per_group=4,
        kv_lora_rank=128,
        qk_rope_dim=32,
        block_size=max_benchmark_seq + args.decode_tokens + 10,
    )

    prefill_results = []
    decode_results = []
    summary_mem_results = []

    print("\n=======================================================")
    print(f"🚀 RUNNING BENCHMARK ON {metadata['device_name']}")
    print(f"Kernel: {metadata['kernel'].upper()} | DType: {args.dtype} | Seed: {args.seed}")
    print("=======================================================\n")

    # =========================================================
    # WORKLOAD 1: PREFILL BENCHMARK
    # =========================================================
    print("--- WORKLOAD 1: PARALLEL PREFILL BENCHMARK ---")
    for attn in args.attention:
        config = base_config
        config.attn_type = attn
        model = GPT(config).to(device=args.device, dtype=torch_dtype).eval()
        if args.compile:
            model = torch.compile(model, dynamic=True)

        for seq_len in args.seq_len:
            for bsz in args.batch_size:
                print(f"[Prefill] {attn.upper():<4} | Seq: {seq_len:<4} | Batch: {bsz:<2} | Kernel: {metadata['kernel']}")
                x = torch.randint(0, config.vocab_size, (bsz, seq_len), device=args.device)

                # Incremental forward peak memory
                inc_peak_mb = measure_incremental_peak_memory(model, x, args.device)

                # Actual cache size for this config
                with torch.no_grad():
                    _, _, test_caches = model(x)
                actual_cache_bytes = 0
                if test_caches is not None:
                    for layer_cache in test_caches:
                        if layer_cache is not None:
                            for tensor in layer_cache:
                                if isinstance(tensor, torch.Tensor):
                                    actual_cache_bytes += tensor.nelement() * tensor.element_size()
                actual_cache_mb = actual_cache_bytes / (1024 * 1024)
                del test_caches

                # Analytical cache size
                anal_cache_mb = analytical_kv_cache_mb(config, seq_len, bsz, bytes_per_param=bytes_per_param)

                # Latency & Throughput
                mean_ms, med_ms, p10_ms, p90_ms, tps = measure_prefill_latency_and_throughput(
                    model, x, args.device, args.n_warmup, args.n_runs
                )

                prefill_results.append({
                    "attention": attn, "seq_len": seq_len, "batch_size": bsz, "kernel": metadata['kernel'],
                    "latency_mean_ms": mean_ms, "latency_median_ms": med_ms,
                    "latency_p10_ms": p10_ms, "latency_p90_ms": p90_ms,
                    "tokens_per_sec": tps, "incremental_peak_memory_mb": inc_peak_mb,
                    "actual_cache_mb": actual_cache_mb, "analytical_cache_mb": anal_cache_mb
                })

                summary_mem_results.append({
                    "attention": attn, "seq_len": seq_len, "batch_size": bsz, "kernel": metadata['kernel'],
                    "incremental_peak_mb": inc_peak_mb, "actual_cache_mb": actual_cache_mb,
                    "analytical_cache_mb": anal_cache_mb
                })

                del x

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

    # =========================================================
    # WORKLOAD 2: CACHED DECODE BENCHMARK
    # =========================================================
    print("\n--- WORKLOAD 2: CACHED AUTOREGRESSIVE DECODE BENCHMARK ---")
    decode_batch_sizes = [1, 8, 16, 32]

    for attn in args.attention:
        config = base_config
        config.attn_type = attn
        model = GPT(config).to(device=args.device, dtype=torch_dtype).eval()
        if args.compile:
            model = torch.compile(model, dynamic=True)

        for bsz in decode_batch_sizes:
            print(f"[Decode]  {attn.upper():<4} | Batch: {bsz:<2} | Prompt: {args.decode_prompt_len} | Decode: {args.decode_tokens} tokens")
            prompt = torch.randint(0, config.vocab_size, (bsz, args.decode_prompt_len), device=args.device)

            step_records = measure_cached_decode(model, prompt, args.decode_tokens, args.device)
            for rec in step_records:
                decode_results.append({
                    "attention": attn, "batch_size": bsz, "kernel": metadata['kernel'],
                    "step_idx": rec["step_idx"], "context_len": rec["context_len"],
                    "step_latency_ms": rec["step_latency_ms"],
                    "decode_tokens_per_sec": rec["decode_tokens_per_sec"],
                    "cumulative_ms": rec["cumulative_ms"]
                })

            del prompt

        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()
        elif args.device == "mps":
            torch.mps.empty_cache()

    # Save CSVs
    with open(os.path.join(args.output_dir, "prefill.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "attention", "seq_len", "batch_size", "kernel",
            "latency_mean_ms", "latency_median_ms", "latency_p10_ms", "latency_p90_ms",
            "tokens_per_sec", "incremental_peak_memory_mb", "actual_cache_mb", "analytical_cache_mb"
        ])
        writer.writeheader()
        writer.writerows(prefill_results)

    with open(os.path.join(args.output_dir, "decode.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "attention", "batch_size", "kernel",
            "step_idx", "context_len", "step_latency_ms", "decode_tokens_per_sec", "cumulative_ms"
        ])
        writer.writeheader()
        writer.writerows(decode_results)

    with open(os.path.join(args.output_dir, "memory.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "attention", "seq_len", "batch_size", "kernel",
            "incremental_peak_mb", "actual_cache_mb", "analytical_cache_mb"
        ])
        writer.writeheader()
        writer.writerows(summary_mem_results)

    print(f"\nSaved CSV results to {args.output_dir}/")
    print("Generating Matplotlib plots...")
    plot_and_save(args.output_dir)
    print(f"Saved plots to {args.output_dir}/*.png")

if __name__ == "__main__":
    main()
