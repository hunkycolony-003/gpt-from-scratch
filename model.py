from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = ATTENTION_REGSITRY[config.attn_type](config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = FeedForward(config)

    def forward(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln_1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.config = config

        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.head_size = config.n_embed // config.n_heads
        self.rope = RotaryPositionalEmbedding(self.config)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, kv_cache=None):
        """
        x: (B, T, C)
        out: (B, T, C)
        """
        B, T, C = x.shape
        T_total = T

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.config.n_embed, 2) # each are (B, T, n_embed)

        # convert to (B, nh, T, n_embed)
        q = q.view(B, T, -1, self.head_size).transpose(1, 2)
        k = k.view(B, T, -1, self.head_size).transpose(1, 2)
        v = v.view(B, T, -1, self.head_size).transpose(1, 2)

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache[0].shape[-2]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        q = self.rope(q, positions)
        k = self.rope(k, positions)


        if kv_cache is not None:
            k_prev, v_prev = kv_cache

            k = torch.concat([k_prev, k], dim=-2) # k = (B, nh, T_total, head_size)
            v = torch.concat([v_prev, v], dim=-2) # k = (B, nh, T_total, head_size)

            T_total = k.shape[-2]

        new_cache = [k, v] # (B, nh, T_total, head_size)

        att = q @ k.transpose(-2, -1) * self.head_size**-0.5 # (B, H, T, T_total)
        if kv_cache is None:
            att = att.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B, nh, T, head_size)
        out = out.transpose(1, 2) # (B, T, nh, head_size)
        out = out.contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out, new_cache


class MultiQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.config = config
        self.head_size = config.n_embed // config.n_heads

        self.kv_proj = nn.Linear(config.n_embed, 2 * self.head_size)
        self.q_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)

        self.rope = RotaryPositionalEmbedding(self.config)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, kv_cache=None):

        B, T, C = x.shape

        kv = self.kv_proj(x) # (B, T, head_size * 2)
        k, v = kv.split(self.head_size, dim=-1) # (B, T, head_size)
        k = k.unsqueeze(-3) # (B, 1, T, head_size)
        v = v.unsqueeze(-3) # (B, 1, T, head_size)

        q = self.q_proj(x) # (B, T, n_embed)
        q = q.view(B, T, -1, self.head_size).transpose(1, 2) # (B, nh, T, head_size)

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache[0].shape[-2]

        positions = torch.arange(past_len, past_len+T, device=x.device)
        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.concat([k_prev, k], dim=-2) # k = (B, nh, T_total, head_size)
            v = torch.concat([v_prev, v], dim=-2) # k = (B, nh, T_total, head_size)
            T_total = k.shape[-2]

        new_cache = [k, v]

        att = q @ k.transpose(-2, -1) *self.head_size**-0.5 # (B, nH, T, T)
        att = att.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v # (B, nh, T, head_size)
        out = out.transpose(1, 2) # (B, T, nh, head_size)
        out = out.contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out, new_cache


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0

        self.config = config
        self.head_size = config.n_embed // config.n_heads
        self.n_groups = config.n_heads // config.heads_per_group
        self.heads_per_group = config.heads_per_group

        self.kv_proj = nn.Linear(config.n_embed, 2 * self.n_groups * self.head_size, bias=False)
        self.q_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)

        self.rope = RotaryPositionalEmbedding(self.config)

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, kv_cache=None, flash_attention=False):
        B, T, C = x.shape

        kv = self.kv_proj(x)
        k, v = kv.split(self.n_groups * self.head_size , dim=-1) # (B, T, n_groups * head_size)
        k = k.view(B, T, self.n_groups, self.head_size).transpose(-2, -3) # (B, n_groups, T, head_size)
        v = v.view(B, T, self.n_groups, self.head_size).transpose(-2, -3)

        q = self.q_proj(x)
        q = q.view(B, T, -1, self.head_size).transpose(-2, -3) # (B, nh, T, head_size)

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache[0].shape[-2]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        q = self.rope(q, positions)
        k = self.rope(k, positions)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.concat([k_prev, k], dim=-2)
            v = torch.concat([v_prev, v], dim=-2)
            T_total = k.shape[-2]

        new_cache = [k, v]


        if flash_attention:
            # next two lines is equivalent to enable_gqa=True
            # k = k.repeat_interleave(self.heads_per_group, -3) # (B, n_groups, 1, T, head_size)
            # v = v.repeat_interleave(self.heads_per_group, -3) # (B, n_groups, 1, T, head_size)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            out = out.transpose(-2, -3)
        else:
            k = k.unsqueeze(-3) # (B, n_groups, 1, T, head_size)
            v = v.unsqueeze(-3) # (B, n_groups, 1, T, head_size)
            q = q.view(B, self.n_groups, -1, T, self.head_size) # (B, n_groups, heads_per_group, T, head_size)
            att = q @ k.transpose(-2, -1) *self.head_size**-0.5 # (B, n_groups, heads_per_group, T, T)
            att = att.masked_fill(self.bias[:T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = att @ v # (B, n_groups, heads_per_group , T, head_size)
            out = out.permute(0, 3, 1, 2, 4) # (B, T, n_groups, heads_per_group , head_size)

        out = out.contiguous().view(B, T, C)
        out = self.c_proj(out)

        return out, new_cache


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.config = config
        self.n_heads = config.n_heads
        self.head_size = config.n_embed // config.n_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_rope_dim = config.qk_rope_dim

        # kv down projection
        self.w_dkv = nn.Linear(config.n_embed, config.kv_lora_rank, bias=False)

        # kv up projection
        self.w_uk = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_size, bias=False)
        self.w_uv = nn.Linear(config.kv_lora_rank, config.n_heads * self.head_size, bias=False)

        # decoupled positional key projection
        self.w_kr = nn.Linear(config.n_embed, self.qk_rope_dim, bias=False)

        # query projections
        self.w_dq = nn.Linear(config.n_embed, config.n_heads * self.head_size, bias=False)
        self.w_q_r = nn.Linear(config.n_embed, config.n_heads * self.qk_rope_dim, bias=False)

        self.rope = RotaryPositionalEmbedding(config, dim=self.qk_rope_dim)

        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=False)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape

        past_len = 0
        if kv_cache is not None:
            past_len = kv_cache[0].shape[-2]
        positions = torch.arange(past_len, past_len + T, device=x.device)

        q_c = self.w_dq(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, nh, T, d_h)
        q_r = self.w_q_r(x).view(B, T, self.n_heads, self.qk_rope_dim).transpose(1, 2)  # (B, nh, T, d_R)
        q_r = self.rope(q_r, positions)

        c_kv = self.w_dkv(x)  # (B, T, kv_lora_rank)
        k_r = self.w_kr(x).unsqueeze(1)  # (B, 1, T, d_R) 1 head shared across all query heads
        k_r = self.rope(k_r, positions)

        # 1. Update and persist KV Cache
        if kv_cache is not None:
            c_kv_prev, k_r_prev = kv_cache
            c_kv = torch.concat([c_kv_prev, c_kv], dim=-2)
            k_r = torch.concat([k_r_prev, k_r], dim=-2)

        new_cache = [c_kv, k_r]
        T_total = c_kv.shape[-2]
        scale = (self.head_size + self.qk_rope_dim) ** -0.5

        # 2. Decoding Phase with Matrix Absorption (T == 1)
        if kv_cache is not None and T == 1:
            # Absorb W_uk into Query
            w_uk = self.w_uk.weight.view(self.n_heads, self.head_size, self.kv_lora_rank)
            q_c_absorbed = torch.matmul(q_c, w_uk)  # (B, nh, 1, d_c)

            # Content & Positional scores directly against cached representations
            score_c = q_c_absorbed @ c_kv.unsqueeze(1).transpose(-2, -1)  # (B, nh, 1, T_total)
            score_r = q_r @ k_r.transpose(-2, -1)                         # (B, nh, 1, T_total)
            att = F.softmax((score_c + score_r) * scale, dim=-1)         # (B, nh, 1, T_total)

            # Attend over latent values and absorb W_uv into Output
            out_latent = att @ c_kv.unsqueeze(1)                          # (B, nh, 1, d_c)
            w_uv = self.w_uv.weight.view(self.n_heads, self.head_size, self.kv_lora_rank)
            out = torch.matmul(out_latent, w_uv.transpose(-2, -1))        # (B, nh, 1, d_h)

            out = out.transpose(1, 2).contiguous().view(B, 1, -1)
            return self.c_proj(out), new_cache

        # 3. Parallel Prefill / Training Phase (T > 1)
        k_c = self.w_uk(c_kv).view(B, -1, self.n_heads, self.head_size).transpose(1, 2)  # (B, nh, T_total, d_h)
        v = self.w_uv(c_kv).view(B, -1, self.n_heads, self.head_size).transpose(1, 2)    # (B, nh, T_total, d_h)

        score_c = q_c @ k_c.transpose(-2, -1)  # (B, nh, T, T_total)
        score_r = q_r @ k_r.transpose(-2, -1)  # (B, nh, T, T_total)
        att = (score_c + score_r) * scale

        if kv_cache is None or past_len == 0:
            att = att.masked_fill(self.bias[:T, :T_total] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        out = att @ v  # (B, nh, T, head_size)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.c_proj(out)

        return out, new_cache


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)

    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, config, dim=None):
        super().__init__()
        self.n_embed = config.n_embed
        self.block_size = config.block_size
        self.base = config.base
        self.head_size = dim if dim is not None else config.n_embed // config.n_heads

        positions = torch.arange(self.block_size).float()

        pair_idx = torch.arange(0, self.head_size, 2)
        freq = self.base ** (-pair_idx/self.head_size)

        angles = torch.outer(positions, freq) # (block_size, n_embed / 2)

        # cache
        cos_cache = torch.cos(angles) # (block_size, n_embed / 2)
        sin_cache = torch.sin(angles)

        self.register_buffer("cos_cached", cos_cache, persistent=False)
        self.register_buffer("sin_cached", sin_cache, persistent=False)

    def forward(self, x, token_positions):
        """
        x: (..., T, n_embed)
        token_positions: (..., T)
        """
        cos = self.cos_cached[token_positions] # (..., T, n_embed / 2)
        sin = self.sin_cached[token_positions] # (..., T, n_embed / 2)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        out = torch.empty_like(x)

        out[..., ::2] = x_even*cos - x_odd*sin
        out[..., 1::2] = x_odd*cos + x_even*sin

        return out


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict( dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # weight initialisation
        self.apply(self._weight_init)

    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.1)

    def forward(self, idx, targets=None, kv_caches=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequnce length greater than {T}"

        x = self.transformer.wte(idx)

        new_caches = [None] * self.config.n_layers
        for i, block in enumerate(self.transformer.h):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_caches[i] = block(x, layer_cache)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(-1))

        return logits, loss, new_caches


    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt:", model_type)

        config_args = {
            'gpt2':  dict(n_layers=12, n_heads=12, n_embed=768), # 124M params
            'gpt2-medium':  dict(n_layers=24, n_heads=16, n_embed=1024), # 124M params
            'gpt2-large':  dict(n_layers=36, n_heads=20, n_embed=1280), # 124M params
            'gpt2-xl':  dict(n_layers=48, n_heads=25, n_embed=1600), # 124M params
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        config = GPTconfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_attn.bias")]
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


ATTENTION_REGSITRY = {
    "mha": CausalSelfAttention,
    "mqa": MultiQueryAttention,
    "gqa": GroupQueryAttention,
    "mla": MultiHeadLatentAttention
}

@dataclass
class GPTconfig:
    base: int = 10000
    block_size: int = 1024
    vocab_size: int = 50257
    n_layers: int = 12
    n_heads: int = 12
    heads_per_group: int = 4
    n_embed: int = 768
    attn_type: str = "mha"
    kv_lora_rank: int = 128
    qk_rope_dim: int = 32
