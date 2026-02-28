import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config (matches GPTConfig shape & usage)
# -----------------------------------------------------------------------------

@dataclass
class LlamaConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float = 0.0
    bias: bool = False
    graph_emb_dim: int = 500


# -----------------------------------------------------------------------------
# RMSNorm (LLaMA-style LayerNorm replacement)
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


# -----------------------------------------------------------------------------
# RoPE helpers
# -----------------------------------------------------------------------------

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)

def apply_rope(x, freqs):
    return (x * freqs.cos()) + (rotate_half(x) * freqs.sin())


# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------

class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, freqs, padding_mask=None):
        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        if padding_mask is not None:
            padding_mask = padding_mask[:, None, None, :]

        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=padding_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


# -----------------------------------------------------------------------------
# SwiGLU MLP
# -----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w1 = nn.Linear(dim, 2 * dim, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        x1, x2 = self.w1(x).chunk(2, dim=-1)
        return self.w2(F.silu(x1) * x2)


# -----------------------------------------------------------------------------
# Transformer Block (same role as GPT Block)
# -----------------------------------------------------------------------------

class LlamaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.n_embd)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.attn = LlamaAttention(config)
        self.mlp = SwiGLU(config.n_embd)

    def forward(self, x, freqs, padding_mask=None):
        x = x + self.attn(self.attn_norm(x), freqs, padding_mask)
        x = x + self.mlp(self.ffn_norm(x))
        return x


# -----------------------------------------------------------------------------
# LLaMA Model (GPT-compatible)
# -----------------------------------------------------------------------------

class Llama(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.graph_emb_proj = nn.Linear(config.graph_emb_dim, config.n_embd)

        self.blocks = nn.ModuleList([
            LlamaBlock(config) for _ in range(config.n_layer)
        ])

        self.norm = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        # RoPE cache
        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
        pos = torch.arange(config.block_size)
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        self.register_buffer("freqs", torch.cat([freqs, freqs], dim=-1))

    # -------------------------------------------------------------------------
    # Forward (IDENTICAL signature to GPT)
    # -------------------------------------------------------------------------
    def forward(
        self,
        idx,
        graph_emb,
        targets=None,
        padding_mask=None,
        preserve_time_dim=False
    ):
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.tok_emb(idx)
        x = x + self.graph_emb_proj(graph_emb).unsqueeze(1)

        freqs = self.freqs[:T][None, None, :, :]

        for block in self.blocks:
            x = block(x, freqs, padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0
            )
        else:
            loss = None
            if not preserve_time_dim:
                logits = logits[:, [-1], :]

        return logits, loss

    # -------------------------------------------------------------------------
    # Optimizer (same as GPT)
    # -------------------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay = [p for p in param_dict.values() if p.dim() < 2]

        optim_groups = [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': nodecay, 'weight_decay': 0.0},
        ]

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'

        return torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=use_fused
        )

    # -------------------------------------------------------------------------
    # Generation (same behavior as GPT)
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, graph_emb, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(
                idx_cond,
                graph_emb,
                preserve_time_dim=False
            )
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx