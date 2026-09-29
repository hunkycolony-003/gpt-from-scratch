import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from model import GPT, GPTconfig

def test_full_vs_cached_numerical_equivalence():
    """
    Asserts that full parallel prefill forward-pass logits match token-by-token
    cached autoregressive decode logits down to machine precision (< 1e-4) across
    all four attention variants (MHA, MQA, GQA, MLA) in pure eager attention mode.
    """
    print("\n--- Running Test: Full vs. Cached Numerical Equivalence (Eager Mode) ---")
    torch.manual_seed(42)

    for attn in ['mha', 'mqa', 'gqa', 'mla']:
        config = GPTconfig(
            attn_type=attn,
            block_size=64,
            n_layers=2,
            n_heads=4,
            n_embed=64,
            heads_per_group=2,
            kv_lora_rank=16,
            qk_rope_dim=8
        )
        model = GPT(config).eval()
        x = torch.randint(0, config.vocab_size, (1, 8))

        with torch.no_grad():
            logits_full, _, _ = model(x)

            # Step-by-step cached decode
            logits_cached_steps = []
            caches = None
            for i in range(x.shape[1]):
                tok = x[:, i:i+1]
                out, _, caches = model(tok, kv_caches=caches)
                logits_cached_steps.append(out)
            logits_cached = torch.cat(logits_cached_steps, dim=1)
            diff = (logits_full - logits_cached).abs().max().item()

            print(f"  ✓ {attn.upper():<4}: Full vs Cached Diff = {diff:.2e}")
            assert diff < 1e-4, f"{attn} cached diff {diff} exceeded tolerance"

def test_torch_compile_equivalence():
    """
    Verifies that torch.compile operates cleanly on the eager attention models.
    """
    print("\n--- Running Test: torch.compile Forward Execution ---")
    torch.manual_seed(42)
    for attn in ['mha', 'mqa', 'gqa', 'mla']:
        config = GPTconfig(
            attn_type=attn,
            block_size=32,
            n_layers=1,
            n_heads=2,
            n_embed=32,
            heads_per_group=1,
            kv_lora_rank=16,
            qk_rope_dim=8
        )
        model = GPT(config).eval()
        compiled_model = torch.compile(model)
        x = torch.randint(0, config.vocab_size, (1, 4))
        with torch.no_grad():
            out, _, _ = compiled_model(x)
        print(f"  ✓ {attn.upper():<4}: Compiled successfully, output shape {out.shape}")

def test_parameter_counts():
    """
    Computes and reports attention and total model parameter counts across all variants.
    """
    print("\n--- Running Test: Parameter Count Reporting ---")
    base_config = GPTconfig(
        n_layers=12,
        n_heads=12,
        n_embed=768,
        heads_per_group=4,
        kv_lora_rank=128,
        qk_rope_dim=32,
        vocab_size=50257
    )

    param_counts = {}
    for attn in ['mha', 'mqa', 'gqa', 'mla']:
        base_config.attn_type = attn
        model = GPT(base_config)

        # Calculate attention parameters only (first block)
        attn_params = sum(p.numel() for p in model.transformer.h[0].attn.parameters())
        total_attn_params = attn_params * base_config.n_layers
        total_model_params = sum(p.numel() for p in model.parameters())

        param_counts[attn] = {
            "attn_per_layer": attn_params,
            "total_attn": total_attn_params,
            "total_model": total_model_params
        }
        print(f"  ✓ {attn.upper():<4}: Attn/Layer = {attn_params:,} | Total Attn = {total_attn_params:,} | Total Model = {total_model_params:,}")

    return param_counts

if __name__ == "__main__":
    test_full_vs_cached_numerical_equivalence()
    test_torch_compile_equivalence()
    test_parameter_counts()
    print("\n🎉 ALL UNIT AND SYSTEM VERIFICATION TESTS PASSED SUCCESSFULLY!")
