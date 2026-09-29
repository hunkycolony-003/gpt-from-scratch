import os
import math
import csv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import matplotlib.pyplot as plt

from model import GPT, GPTconfig

class TextDataLoader:
    def __init__(self, tokens, B, T):
        self.tokens = tokens
        self.B = B
        self.T = T
        self.current_idx = 0

    def next_batch(self, device):
        B, T = self.B, self.T
        if self.current_idx + B * T + 1 > len(self.tokens):
            self.current_idx = 0
        buf = self.tokens[self.current_idx : self.current_idx + B * T + 1]
        x = buf[:-1].view(B, T).to(device)
        y = buf[1:].view(B, T).to(device)
        self.current_idx += B * T
        return x, y

def evaluate_perplexity(model, val_loader, num_val_batches, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(num_val_batches):
            x, y = val_loader.next_batch(device)
            logits, loss, _ = model(x, targets=y)
            total_loss += loss.item()
    avg_loss = total_loss / num_val_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

def train_and_eval(attn_type, train_tokens, val_tokens, steps=150, device="cpu"):
    torch.manual_seed(42)
    config = GPTconfig(
        attn_type=attn_type,
        n_layers=4,
        n_heads=4,
        n_embed=128,
        block_size=128,
        vocab_size=50257,
        heads_per_group=2,
        kv_lora_rank=32,
        qk_rope_dim=16
    )

    model = GPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    train_loader = TextDataLoader(train_tokens, B=8, T=64)
    val_loader = TextDataLoader(val_tokens, B=8, T=64)

    model.train()
    print(f"Training {attn_type.upper()} for {steps} steps on {device}...")
    t0 = time.time()
    for step in range(steps):
        x, y = train_loader.next_batch(device)
        optimizer.zero_grad()
        logits, loss, _ = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    t1 = time.time()

    val_loss, val_ppl = evaluate_perplexity(model, val_loader, num_val_batches=20, device=device)
    print(f"[{attn_type.upper()}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f} (Time: {t1-t0:.1f}s)")
    return val_loss, val_ppl

def main():
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Perplexity Degradation Evaluation on device: {device}")

    # Load and tokenize data
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()

    enc = tiktoken.get_encoding("gpt2")
    all_tokens = torch.tensor(enc.encode(text), dtype=torch.long)

    # Train / Val Split (90% / 10%)
    split_idx = int(0.9 * len(all_tokens))
    train_tokens = all_tokens[:split_idx]
    val_tokens = all_tokens[split_idx:]
    print(f"Tokens: {len(train_tokens):,} train, {len(val_tokens):,} val")

    variants = ["mha", "gqa", "mqa", "mla"]
    results = []

    for attn in variants:
        loss, ppl = train_and_eval(attn, train_tokens, val_tokens, steps=150, device=device)
        results.append({"attention": attn, "val_loss": loss, "perplexity": ppl})

    # Baseline is MHA
    mha_ppl = next(r["perplexity"] for r in results if r["attention"] == "mha")
    for r in results:
        deg = ((r["perplexity"] - mha_ppl) / mha_ppl) * 100
        r["degradation_pct"] = deg

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "perplexity.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attention", "val_loss", "perplexity", "degradation_pct"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved perplexity results to {csv_path}")

    # Plotting Perplexity Comparison
    plt.figure(figsize=(7, 4.5), dpi=300)
    names = [r["attention"].upper() for r in results]
    ppls = [r["perplexity"] for r in results]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    bars = plt.bar(names, ppls, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f"{height:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title("Validation Perplexity Comparison (nanoGPT Base)", fontsize=13, fontweight="bold")
    plt.ylabel("Validation Perplexity (Lower is better)", fontsize=11)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "perplexity_comparison.png")
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved perplexity chart to {chart_path}")

    print("\n" + "="*60)
    print(f"{'Attention':<12} {'Val Loss':<12} {'Perplexity':<15} {'Degradation vs MHA':<18}")
    print("="*60)
    for r in results:
        print(f"{r['attention'].upper():<12} {r['val_loss']:<12.4f} {r['perplexity']:<15.2f} {r['degradation_pct']:>+8.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
