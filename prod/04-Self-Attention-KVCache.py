#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产篇 04 · 自注意力与 KV Cache：GQA 32→8 逐层对齐 HF，手写 FlashAttention / PagedAttention 对账

对应文章：同目录 04-Self-Attention-KVCache.md（建议先读 基础篇/04-Self-Attention-KVCache.py 与 .md）
运行：    python3 04-Self-Attention-KVCache.py
依赖：    numpy + torch（CPU）+ transformers（离线用 LlamaConfig 随机初始化一个 tiny GQA Llama，
          不下载任何预训练权重、不联网）。flash-attn / vLLM 等重依赖不安装，
          归为"方向性说明 + 调用骨架"，运行结束在报表里逐条标注真跑 / 未验证。
产物：    ./output/生产篇04-Self-Attention-KVCache/ 下的报表、GQA sweep 表、KV/Flash/Paged 账、配图

基础篇 04 用纯 Python 数清了"重算 vs 缓存"的乘加次数；生产篇换一副眼镜——把 KV Cache 当成
**推理服务的显存预算问题**来重做。全文闭环七件事：

    A. 手写 GQA（KV 头数 32→8）并与 HF LlamaForCausalLM 逐层数值对齐；repeat vs repeat_interleave 静默 bug
    B. KV Cache 显存账：2×层数×KV头数×头维度×字节数，tiny 真张量实测 + 三个真实模型口径外推
    C. GQA sweep 32→{32,16,8,4,1}：参数 / KV 字节 / decode 带宽 / 不训练的质量代理
    D. 手写 FlashAttention 等价实现：分块 + online softmax，数学精确零近似；HBM 落地量账
    E. 手写 PagedAttention 分页 KV 模拟器：block table 记账、碎片账、并发账、数学不变性断言
    F. RoPE × KV Cache：缓存的 K 已旋转，前缀复用要重旋转；FP32 算角再 cast 的精度账
    G. 线上问题清单（显存超限 / 并发上不去 / 长序列 OOM / 静默 bug 两连）+ 工业库骨架

哪些是本机真跑验证的、哪些只是"工业库/内核调用示例未验证"，运行结束会在报表里逐条标明。
"""

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '生产篇04-Self-Attention-KVCache')

VERIFY_LOG = []
W = 74


def log_verify(item, status, note=''):
    VERIFY_LOG.append((item, status, note))
    print('  [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================================================
#  复用组件（与生产篇 03 同式：RMSNorm / RoPE / SwiGLU）
# ==================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight.float() * xf).to(x.dtype)


def build_rope_cache(seq_len, head_dim, theta=10000.0, device='cpu'):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ==================================================================
#  A. 手写 GQA + 与 HF LlamaForCausalLM(32→8) 逐层对齐
# ==================================================================
def expand_kv(t, n_rep, mode='repeat_interleave'):
    """把 [B, H_kv, S, Dh] 的 K/V 扩展成 [B, H_q, S, Dh]，让第 h 个 Q 头用上第 h//n_rep 个 KV 头。

    repeat_interleave 与 group（reshape）是两种正确写法，bitwise 相同；
    repeat 是经典静默 bug：它把 KV 头"整块平铺"，第 h 个 Q 头会错配到第 h%H_kv 个 KV 头。
    """
    if mode == 'repeat_interleave':
        return t.repeat_interleave(n_rep, dim=1)
    if mode == 'group':
        B, Hkv, S, Dh = t.shape
        return (t[:, :, None, :, :].expand(B, Hkv, n_rep, S, Dh)
                .reshape(B, Hkv * n_rep, S, Dh))
    if mode == 'repeat':
        return t.repeat(1, n_rep, 1, 1)
    raise ValueError('未知 expand 模式：%s' % mode)


class GroupedQueryAttention(nn.Module):
    """GQA：Q 头数 n_heads，KV 头数 n_kv_heads（n_heads 必须能被 n_kv_heads 整除）。
       K/V 投影的输出维度是 n_kv_heads·head_dim，比 MHA 瘦 n_heads/n_kv_heads 倍。"""

    def __init__(self, dim, n_heads, n_kv_heads, theta=10000.0, expand='repeat_interleave'):
        super().__init__()
        assert dim % n_heads == 0, '维度不匹配：hidden 必须能被 Q 头数整除'
        assert n_heads % n_kv_heads == 0, '维度不匹配：Q 头数必须能被 KV 头数整除'
        self.n_heads, self.n_kv_heads = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = dim // n_heads
        self.theta, self.expand = theta, expand
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = expand_kv(k, self.n_rep, self.expand)
        v = expand_kv(v, self.n_rep, self.expand)
        scores = (q @ k.transpose(-1, -2)) * self.scaling
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(ctx)


class CausalSelfAttentionMHA(nn.Module):
    """MHA 参照实现（与生产篇 03 的 CausalSelfAttention 同构）：用于 g==h 退化回归。"""

    def __init__(self, dim, n_heads, theta=10000.0):
        super().__init__()
        assert dim % n_heads == 0, '维度不匹配：hidden 必须能被头数整除'
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.theta = theta
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * self.scaling
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(ctx)


class LlamaBlockGQA(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ffn_hidden, eps=1e-5, theta=10000.0,
                 expand='repeat_interleave'):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.self_attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, theta, expand)
        self.mlp = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlamaGQA(nn.Module):
    def __init__(self, vocab, dim, n_layers, n_heads, n_kv_heads, ffn_hidden,
                 eps=1e-5, theta=10000.0, expand='repeat_interleave'):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [LlamaBlockGQA(dim, n_heads, n_kv_heads, ffn_hidden, eps, theta, expand)
             for _ in range(n_layers)])
        self.norm = RMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.dim, self.n_heads, self.theta = dim, n_heads, theta

    def forward(self, ids, collect_hidden=False):
        B, S = ids.shape
        cos, sin = build_rope_cache(S, self.dim // self.n_heads, self.theta, ids.device)
        h = self.embed_tokens(ids)
        hidden = [h] if collect_hidden else None
        for blk in self.layers:
            h = blk(h, cos, sin)
            if collect_hidden:
                hidden.append(h)
        h = self.norm(h)
        logits = self.lm_head(h)
        return (logits, hidden) if collect_hidden else logits


def build_hf_tiny_llama_gqa(vocab, dim, n_layers, n_heads, n_kv_heads, ffn_hidden,
                            eps, theta, seed):
    """离线构造随机初始化的 tiny GQA LlamaForCausalLM（只给 config，不下载任何权重）。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    set_all_seeds(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=dim, intermediate_size=ffn_hidden,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,             # GQA：KV 头数 < Q 头数（本篇主戏）
        max_position_embeddings=max(64, 4 * dim), rms_norm_eps=eps,
        rope_theta=theta, attention_bias=False, mlp_bias=False,
        tie_word_embeddings=False)
    return LlamaForCausalLM(cfg).eval(), cfg


def copy_hf_into_hand_gqa(hf, hand):
    hs = hf.state_dict()
    hand.embed_tokens.weight.data.copy_(hs['model.embed_tokens.weight'])
    for i, blk in enumerate(hand.layers):
        p = 'model.layers.%d.' % i
        blk.self_attn.q_proj.weight.data.copy_(hs[p + 'self_attn.q_proj.weight'])
        blk.self_attn.k_proj.weight.data.copy_(hs[p + 'self_attn.k_proj.weight'])
        blk.self_attn.v_proj.weight.data.copy_(hs[p + 'self_attn.v_proj.weight'])
        blk.self_attn.o_proj.weight.data.copy_(hs[p + 'self_attn.o_proj.weight'])
        blk.mlp.gate_proj.weight.data.copy_(hs[p + 'mlp.gate_proj.weight'])
        blk.mlp.up_proj.weight.data.copy_(hs[p + 'mlp.up_proj.weight'])
        blk.mlp.down_proj.weight.data.copy_(hs[p + 'mlp.down_proj.weight'])
        blk.input_layernorm.weight.data.copy_(hs[p + 'input_layernorm.weight'])
        blk.post_attention_layernorm.weight.data.copy_(hs[p + 'post_attention_layernorm.weight'])
    hand.norm.weight.data.copy_(hs['model.norm.weight'])
    hand.lm_head.weight.data.copy_(hs['lm_head.weight'])


def layerwise_align_gqa(vocab, dim, n_layers, n_heads, n_kv_heads, ffn_hidden,
                        eps, theta, seq, batch):
    """同权重下比对手写 GQA 与 HF 的每层 hidden_states / logits；并给出 repeat 错配的反面数字。"""
    hf, cfg = build_hf_tiny_llama_gqa(vocab, dim, n_layers, n_heads, n_kv_heads,
                                      ffn_hidden, eps, theta, SEED)
    set_all_seeds(SEED + 7)
    hand = TinyLlamaGQA(vocab, dim, n_layers, n_heads, n_kv_heads, ffn_hidden, eps, theta).eval()
    hand_bad = TinyLlamaGQA(vocab, dim, n_layers, n_heads, n_kv_heads, ffn_hidden, eps, theta,
                            expand='repeat').eval()
    copy_hf_into_hand_gqa(hf, hand)
    copy_hf_into_hand_gqa(hf, hand_bad)

    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    with torch.no_grad():
        hf_out = hf(ids, output_hidden_states=True)
        hand_logits, hand_hidden = hand(ids, collect_hidden=True)
        bad_logits = hand_bad(ids)
    diffs = []
    # HF 约定：hidden_states[-1] 是过了末端 RMSNorm 的最终隐状态（生产篇 03 踩过这个坑）
    for i in range(n_layers):
        d = (hf_out.hidden_states[i] - hand_hidden[i]).abs().max().item()
        diffs.append(('embed' if i == 0 else 'layer%d' % (i - 1), d))
    with torch.no_grad():
        hand_final = hand.norm(hand_hidden[n_layers])
    diffs.append(('layer%d+norm' % (n_layers - 1),
                  (hf_out.hidden_states[n_layers] - hand_final).abs().max().item()))
    logit_diff = (hf_out.logits - hand_logits).abs().max().item()
    expand_wrong_diff = (hand_logits - bad_logits).abs().max().item()
    return diffs, logit_diff, ids, expand_wrong_diff, cfg


def hf_cache_equivalence(model, ids, prefill):
    """HF DynamicCache 增量解码 vs 一次全量前向。

    位置 t 的 logits 是"看完 0..t 后预测 t+1"，所以增量第 t 步的输出与全量位置 t 的 logits
    **同下标对齐**；若错开一位比对，会得到一个 ~1e-1 的假失配。
    """
    from transformers import DynamicCache
    B, S = ids.shape
    with torch.no_grad():
        full = model(ids).logits
        cache = DynamicCache()
        out = model(ids[:, :prefill], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        inc = [out.logits]
        for t in range(prefill, S):
            out = model(ids[:, t:t + 1], past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            inc.append(out.logits)
        inc = torch.cat(inc, dim=1)                       # [B, S, V]，与 full 同下标
        aligned = (full - inc).abs().max().item()
        misaligned = (full[:, 1:, :] - inc[:, :-1, :]).abs().max().item()
        seq_len = cache.get_seq_length()
        per_layer = [(layer.keys.shape, layer.keys.numel() * layer.keys.element_size()
                      + layer.values.numel() * layer.values.element_size())
                     for layer in cache.layers]
    return aligned, misaligned, seq_len, per_layer


# ==================================================================
#  B. KV Cache 显存账
# ==================================================================
def kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes):
    """每 token 的 KV Cache 字节数：2(K+V) × 层数 × KV头数 × 头维度 × 每元素字节数。"""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


# 引用口径（未验证）：三个真实模型的 KV 形状，用于把 tiny 实测外推到生产量级
REF_MODELS = [
    ('Llama-2-7B  MHA  32层/32KV头/128维 fp16', 32, 32, 128, 2),
    ('Llama-3-8B  GQA  32层/8KV头/128维  fp16', 32, 8, 128, 2),
    ('Llama-2-70B GQA  80层/8KV头/128维  fp16', 80, 8, 128, 2),
]


def concurrency_budget(total_gb, weights_gb, bytes_per_token, ctx):
    """给定显存总预算与权重占用，算 4K 上下文下能并发多少路。整数除法，不真分配显存。"""
    budget = int((total_gb - weights_gb) * 1024 ** 3)
    per_req = bytes_per_token * ctx
    return budget // per_req


# MLA（多潜向量注意力）：不砍 KV 头数，改砍"每个 token 存多少标量"。
# 512 / 64 / 61 层是 DeepSeek-V3 与 Kimi K2 公开 config 里的已核项（见文章 3.5 与番外说明）。
# 两个 MHA 基线是本文自选的对照档（Llama 档 32 头、DeepSeek 档 128 头），不是论文自己的对比对象。
MLA_CFG = {'kv_lora_rank': 512, 'qk_rope_head_dim': 64, 'n_layers': 61,
           'mha_heads': 32, 'mha_head_dim': 128, 'ds_heads': 128, 'dtype_bytes': 2,
           'ctx': 131072, 'batch': 8}


def mla_scalars_per_token(kv_lora_rank, qk_rope_head_dim):
    """MLA 每 token 每层缓存的标量数 = 一个压缩低秩潜向量 + 一条解耦的 RoPE 键。
       账本里没有铁律①那个 2：K 和 V 都从同一个潜向量上行投影出来，缓存的就是潜向量本身。"""
    return kv_lora_rank + qk_rope_head_dim


def mha_scalars_per_token(n_heads, head_dim):
    """同位置基线 MHA：K 与 V 各存一份，所以是 2 × 头数 × 头维。"""
    return 2 * n_heads * head_dim


def gqa_equiv_groups(scalars_per_token, head_dim):
    """把"每 token 标量数"折回 GQA 口径：等于几个 KV 头。可以是小数——
       GQA 只能在整数个头上调，MLA 调的是表示维度，这就是两条路线正交的原因。"""
    return scalars_per_token / (2 * head_dim)


# ==================================================================
#  C. GQA sweep
# ==================================================================
def attention_params_formula(dim, n_heads, n_kv_heads):
    """每层注意力参数量闭式：q(d²) + k/v(各 n_kv·dh·d) + o(d²) = d²·(1 + 2g/h + 1)。"""
    dh = dim // n_heads
    return dim * dim + 2 * (n_kv_heads * dh) * dim + dim * dim


def decode_bandwidth_bytes(n_layers, n_kv_heads, head_dim, dtype_bytes, cached_tokens):
    """decode 一步要把全部 KV Cache 从显存读一遍：2·层·KV头·头维·字节·已缓存 token 数。"""
    return kv_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes) * cached_tokens


def gqa_degrade_proxy(K, V, Q, g):
    """不训练的质量代理：把每 n_rep=H/g 个头的 K/V 取组均值共享，度量信息损失。

    确定性、毫秒级。MSE / KL 随 g 减小单调不减；max|Δ| 不保证单调（g=1 时可能回落），只打印。
    """
    H, S, D = K.shape
    n_rep = H // g
    Km = K.view(g, n_rep, S, D).mean(1, keepdim=True).expand(g, n_rep, S, D).reshape(H, S, D)
    sc = torch.einsum('hsd,hsd->hs', Q, K) / math.sqrt(D)
    sd = torch.einsum('hsd,hsd->hs', Q, Km) / math.sqrt(D)
    p = torch.softmax(sc, dim=-1)
    pd = torch.softmax(sd, dim=-1)
    mse = ((Km - K) ** 2).mean().item()
    kl = (p * torch.log((p + 1e-12) / (pd + 1e-12))).sum(-1).mean().item()
    maxd = (Km - K).abs().max().item()
    return mse, kl, maxd


# ==================================================================
#  D. 手写 FlashAttention 等价实现（分块 + online softmax）
# ==================================================================
def naive_attention(q, k, v):
    """朴素因果注意力：显式造 [S,S] 分数矩阵。分块实现要与它逐元素对账。"""
    S = q.shape[-2]
    sc = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    causal = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
    sc = sc.masked_fill(causal, float('-inf'))
    w = torch.softmax(sc, dim=-1)          # 按输入 dtype 算：fp32 走 fp32、fp64 走 fp64
    return w @ v


def online_softmax_step(m, l, o, s_block, v_block):
    """FlashAttention 的递推核心：把一个新块并进运行统计量 (m, l, o)。

    m_new = max(m, rowmax(s))；alpha = exp(m - m_new)；l = l·alpha + rowsum(p)；o = o·alpha + p·V。
    首块 m=-inf 时 alpha=exp(-inf)=0，用 where 兜底防 nan。数学上精确，零近似。
    """
    m_new = torch.maximum(m, s_block.max(dim=-1).values)
    alpha = torch.exp(m - m_new)
    alpha = torch.where(torch.isinf(alpha), torch.zeros_like(alpha), alpha)
    p = torch.exp(s_block - m_new.unsqueeze(-1))
    l = l * alpha + p.sum(dim=-1)
    o = o * alpha.unsqueeze(-1) + (p @ v_block)
    return m_new, l, o


def flash_attention_tiled(q, k, v, br=64, bc=64):
    """分块因果注意力：[S,S] 分数矩阵从不整体落地，只有 [br,bc] 的小块存在。"""
    B, H, S, D = q.shape
    o = torch.zeros(B, H, S, D, dtype=q.dtype)
    for i0 in range(0, S, br):
        qi = q[:, :, i0:i0 + br, :]
        mi = torch.full((B, H, qi.shape[2]), float('-inf'), dtype=q.dtype)
        li = torch.zeros(B, H, qi.shape[2], dtype=q.dtype)
        oi = torch.zeros(B, H, qi.shape[2], D, dtype=q.dtype)
        rows = torch.arange(i0, i0 + qi.shape[2]).view(-1, 1)
        for j0 in range(0, S, bc):
            if j0 > i0 + br - 1:                       # 因果：只看左下块
                break
            kj = k[:, :, j0:j0 + bc, :]
            vj = v[:, :, j0:j0 + bc, :]
            s = (qi @ kj.transpose(-1, -2)) / math.sqrt(D)
            cols = torch.arange(j0, j0 + kj.shape[2]).view(1, -1)
            s = s.masked_fill(cols > rows, float('-inf'))
            mi, li, oi = online_softmax_step(mi, li, oi, s, vj)
        o[:, :, i0:i0 + br, :] = oi / li.unsqueeze(-1)
    return o


def hbm_accounting(S, d, br):
    """HBM 落地量账（元素个数，整数精确）。

    朴素：scores 写+读、probs 写+读 → 4·S² 个元素落地，[S,S] 级张量落地 2 次。
    分块：只有 O(S·d) 的输出与 O(S) 的运行统计量落地，[S,S] 级张量落地 0 次。
    注意"总读写字节"并不是 O(S)：分块要把 K/V 重读 S/br 遍，是 Θ(S²·d/br)。两个数都报。
    """
    naive_landed = 4 * S * S
    naive_squares = 2
    tiled_landed = S * d + 2 * S
    tiled_squares = 0
    naive_total = (3 * S * d + 4 * S * S + S * d)          # 读QKV + 落地4S² + 写O
    tiled_total = (2 * S * d + 2 * S * S * d // br)        # 读Q写O + K/V 重读 S/br 遍
    return dict(naive_landed=naive_landed, naive_squares=naive_squares,
                tiled_landed=tiled_landed, tiled_squares=tiled_squares,
                naive_total=naive_total, tiled_total=tiled_total,
                ratio_landed=naive_landed / tiled_landed)


# ==================================================================
#  E. 手写 PagedAttention 分页 KV 模拟器
# ==================================================================
BLOCK = 16


class PagedKVPool:
    """分页 KV 池：KV 按 BLOCK 个 token 一块存放，请求用 block_table 记账。
       只复现"记账与碎片"，不含 vLLM 的调度 / 抢占 / 前缀共享 / CUDA kernel。"""

    def __init__(self, n_blocks, n_kv_heads, head_dim):
        self.n_blocks = n_blocks
        self.n_kv_heads, self.head_dim = n_kv_heads, head_dim
        self.free = list(range(n_blocks))
        self.store = {}            # block_id -> [K[BLOCK,H,Dh], V[BLOCK,H,Dh]]
        self.block_table = {}      # req -> [block_id, ...]
        self.filled = {}           # req -> 已写 token 数

    def _new_block(self):
        if not self.free:
            return None
        bid = self.free.pop()
        H, D = self.n_kv_heads, self.head_dim
        self.store[bid] = [torch.zeros(BLOCK, H, D), torch.zeros(BLOCK, H, D)]
        return bid

    def write_kv(self, req, k_rows, v_rows):
        """把 [L,H,D] 的 K/V 按块写进池子，返回实际占用的块数。"""
        L = k_rows.shape[0]
        table = []
        for s in range(0, L, BLOCK):
            bid = self._new_block()
            if bid is None:
                raise RuntimeError('分页池块耗尽')
            table.append(bid)
            ke, ve = k_rows[s:s + BLOCK], v_rows[s:s + BLOCK]
            self.store[bid][0][:ke.shape[0]] = ke
            self.store[bid][1][:ve.shape[0]] = ve
        self.block_table[req] = table
        self.filled[req] = L
        return len(table)

    def gather(self, req):
        """按 block table 把 K/V 拼回 [L,H,D]——只改记账，不改数据。"""
        L = self.filled[req]
        ks, vs = [], []
        for bid in self.block_table[req]:
            ks.append(self.store[bid][0])
            vs.append(self.store[bid][1])
        K = torch.cat(ks, dim=0)[:L]
        V = torch.cat(vs, dim=0)[:L]
        return K, V

    def blocks_used(self, req):
        return len(self.block_table[req])


def contiguous_blocks_needed(max_len):
    return math.ceil(max_len / BLOCK)


def admit_greedy(lengths, budget, blocks_of):
    """请求按 trace 顺序（循环到达）持续申请，每条占用 blocks_of(L) 块，直到预算放不下为止。
       返回 (served, used_blocks, served_lengths)。连续预留传常数块、分页传 ceil(L/BLOCK)。"""
    served, used, served_lens = 0, 0, []
    n = len(lengths)
    i = 0
    while True:
        L = lengths[i % n]
        need = blocks_of(L)
        if used + need > budget:
            break
        used += need
        served += 1
        served_lens.append(L)
        i += 1
    return served, used, served_lens


def make_request_trace(seed, n_req=64, lo=128, hi=512):
    rng = random.Random(seed)
    return [rng.randint(lo, hi) for _ in range(n_req)]


def paged_attention_ctx(K, V, q_row):
    """用 gather 出来的 K/V 算一个查询行的因果注意力上下文（最后一行，全可见）。"""
    L, H, D = K.shape
    sc = (q_row[None, None, :, :] * K[None, ...]).sum(-1) / math.sqrt(D)   # [1,H,L]
    w = torch.softmax(sc, dim=-1)
    return (w[..., None] * V[None, ...]).sum(-2)                            # [1,H,D]


# ==================================================================
#  F. RoPE × KV Cache
# ==================================================================
def rope_spectrum(head_dim=128, theta=10000.0):
    """RoPE 每个维度的波长：λ_i = 2π·theta^(2i/head_dim)。低频维管长程、高频维管局部。"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    return 2 * math.pi / inv


def rotate_by_delta(k_rot, delta, head_dim, theta=10000.0):
    """缓存里的 K 已带位置 m 的旋转；要把它搬到 m+Δ，利用 R(m+Δ)=R(Δ)·R(m) 再旋一次 R(Δ)。"""
    cos, sin = build_rope_cache(delta + 1, head_dim, theta)
    c, s = cos[delta], sin[delta]
    return k_rot * c + rotate_half(k_rot) * s


def demo_prefix_reuse(prefix, q_vec, delta, head_dim, theta=10000.0):
    """前缀复用三对照：① 从头重算（真值）② 直接复用旧缓存 K（位置错位）③ 重旋转 Δ。
       场景：一段前缀当初缓存时位于位置 [0,P)，如今复用时它整体挪到了 [Δ,Δ+P)。"""
    P = prefix.shape[0]
    cos_t, sin_t = build_rope_cache(P + delta + 1, head_dim, theta)
    k_true = prefix * cos_t[delta:delta + P] + rotate_half(prefix) * sin_t[delta:delta + P]
    q_new = q_vec * cos_t[P + delta] + rotate_half(q_vec) * sin_t[P + delta]

    cos_old, sin_old = build_rope_cache(P, head_dim, theta)
    k_cached = prefix * cos_old + rotate_half(prefix) * sin_old      # 缓存里存的样子（位置 0..P-1）
    k_fixed = rotate_by_delta(k_cached, delta, head_dim, theta)      # R(Δ)·R(m)=R(m+Δ) 搬到 Δ..Δ+P-1

    def ctx_of(k):
        sc = (q_new[None, :] * k).sum(-1) / math.sqrt(head_dim)
        w = torch.softmax(sc, dim=-1)
        return (w[:, None] * k).sum(0)

    return ctx_of(k_true), ctx_of(k_cached), ctx_of(k_fixed)


def demo_rope_precision(pos, head_dim, theta=10000.0):
    """FP32 算角再 cast vs 角度先 cast 成 fp16：长位置下后者误差爆炸。参照用 fp64。"""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2).double() / head_dim))
    ang64 = pos * inv
    ref = torch.cat([ang64.cos(), ang64.cos()], dim=-1)

    inv32 = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang32 = pos * inv32
    good = torch.cat([ang32.cos(), ang32.cos()], dim=-1).half().float()

    inv16 = inv32.half().float()
    ang16 = (pos * inv16).half().float()
    bad = torch.cat([ang16.cos(), ang16.cos()], dim=-1)

    err_good = (good - ref.float()).abs().max().item()
    err_bad = (bad - ref.float()).abs().max().item()
    return err_good, err_bad


# ==================================================================
#  G. 工业库骨架（未验证）
# ==================================================================
FLASH_SKELETON = '''# flash-attn（GPU-only，本机未验证）：调用方式二选一
from flash_attn import flash_attn_func          # out = flash_attn_func(q, k, v, causal=True)
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # PyTorch 2.x SDPA 自动选后端
# with sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...             # 显式指定后端'''

VLLM_SKELETON = '''# vLLM（需下载模型，本机未验证）：PagedAttention 由引擎内部管理
from vllm import LLM, SamplingParams
llm = LLM(model=..., max_model_len=4096, gpu_memory_utilization=0.9)
out = llm.generate(prompts, SamplingParams(temperature=0.7, max_tokens=256))'''


def try_industrial_libs():
    for name, mod in (('flash_attn', 'flash_attn'), ('vllm', 'vllm')):
        try:
            __import__(mod)
            log_verify('%s 本体' % name, '真跑验证', '本机已安装')
        except ImportError:
            log_verify('%s 本体' % name, '未验证', '本机不安装（GPU-only 或需下载），仅调用骨架')


# ---------------------------------------------------------------- 配图
def draw(sweep, hbm_rows, paged_stats, ref_rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[提示] 未安装 matplotlib，跳过画图')
        return None
    plt.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei',
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))

    gs = [r['g'] for r in sweep]
    axes[0].bar([str(g) for g in gs], [r['kv_bytes'] for r in sweep], color='#4a9d6a',
                label='KV 字节/token')
    ax0 = axes[0].twinx()
    ax0.plot([str(g) for g in gs], [r['params'] for r in sweep], 'o-', color='#d08c4a',
             label='每层注意力参数')
    ax0.set_ylabel('每层注意力参数', color='#d08c4a')
    ax0.tick_params(axis='y', labelcolor='#d08c4a')
    axes[0].set_title('GQA sweep：KV 头数 32→1\n省显存/参数/带宽，FLOPs 不变')
    axes[0].set_xlabel('KV 头数 g')
    axes[0].set_ylabel('KV 字节/token', color='#4a9d6a')
    axes[0].tick_params(axis='y', labelcolor='#4a9d6a')
    axes[0].text(0.02, 0.95, '注意力 FLOPs 不变', color='#cc5a5a', fontsize=11,
                 transform=axes[0].transAxes, fontweight='bold')

    Ss = [r['S'] for r in hbm_rows]
    axes[1].plot(Ss, [r['naive_landed'] for r in hbm_rows], 'o--', color='#cc5a5a',
                 label='朴素落地 4S²')
    axes[1].plot(Ss, [r['tiled_landed'] for r in hbm_rows], 's--', color='#4a9d6a',
                 label='分块落地 S·d+2S')
    axes[1].set_xscale('log'); axes[1].set_yscale('log')
    axes[1].set_title('HBM 落地量（log-log）\n斜率 2 vs 1：[S,S] 从不落地')
    axes[1].set_xlabel('序列长度 S')
    axes[1].set_ylabel('落地元素数 (log)')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which='both')

    cats = ['连续预留', '分页']
    xw = np.arange(len(cats))
    axes[2].bar(xw - 0.2, [paged_stats['waste_contig'] * 100, paged_stats['waste_paged'] * 100],
                width=0.38, color='#cc5a5a', label='显存浪费率 %（左轴）')
    ax2 = axes[2].twinx()
    ax2.bar(xw + 0.2, [paged_stats['served_contig'], paged_stats['served_paged']],
            width=0.38, color='#4a9d6a', label='同预算并发路数（右轴）')
    axes[2].set_xticks(xw)
    axes[2].set_xticklabels(cats)
    axes[2].set_ylabel('显存浪费率 %', color='#cc5a5a')
    axes[2].tick_params(axis='y', labelcolor='#cc5a5a')
    ax2.set_ylabel('同预算并发路数', color='#4a9d6a')
    ax2.tick_params(axis='y', labelcolor='#4a9d6a')
    axes[2].set_title('碎片与并发：连续预留 vs 分页')

    names = [r[0].split()[0] + '+' + r[0].split()[1] for r in ref_rows]
    ctxs = [4096, 32768, 131072]
    xs = np.arange(len(names))
    wbar = 0.26
    for ci, ctx in enumerate(ctxs):
        vals = [r[1] * ctx for r in ref_rows]
        axes[3].bar(xs + (ci - 1) * wbar, vals, width=wbar,
                    label='ctx=%dK' % (ctx // 1024),
                    color=['#4a9d6a', '#d08c4a', '#8888cc'][ci])
    axes[3].set_xticks(xs)
    axes[3].set_xticklabels(names, fontsize=8)
    axes[3].set_yscale('log')
    axes[3].axhline(80 * 1024 ** 3, color='#cc5a5a', ls='--', lw=1, label='80GB 预算')
    axes[3].set_title('单请求 KV 显存（三口径 × 4K/32K/128K）')
    axes[3].set_ylabel('字节 (log)')
    axes[3].legend(fontsize=7)

    path = os.path.join(OUT, 'fig-04-gqa-kvcache.png')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    set_all_seeds(SEED)
    t_main = time.perf_counter()

    # ---------------- A. GQA 与 HF 逐层对齐
    ALIGN = dict(vocab=64, dim=256, n_layers=2, n_heads=32, n_kv_heads=8,
                 ffn_hidden=512, eps=1e-5, theta=10000.0, seq=12, batch=2)
    diffs, logit_diff, ids, expand_wrong_diff, cfg = layerwise_align_gqa(**ALIGN)
    max_diff = max(d for _, d in diffs + [('logits', logit_diff)])
    head_dim = ALIGN['dim'] // ALIGN['n_heads']
    log_verify('手写 GQA（KV 头数 32→8）+ 与 HF LlamaForCausalLM 逐层数值对齐', '真跑验证',
               '离线随机 config（不下载权重），同权重比对每层 hidden_states + logits')

    # repeat_interleave vs group：两种正确写法 bitwise 相同
    g = torch.Generator().manual_seed(SEED)
    tk = torch.randn(2, 8, 6, 8, generator=g)
    eq_correct = torch.equal(expand_kv(tk, 4, 'repeat_interleave'), expand_kv(tk, 4, 'group'))
    wrong_vs_right = (expand_kv(tk, 4, 'repeat') - expand_kv(tk, 4, 'repeat_interleave')) \
        .abs().max().item()

    # HF DynamicCache 增量解码 vs 全量前向
    hf_model, _ = build_hf_tiny_llama_gqa(ALIGN['vocab'], ALIGN['dim'], ALIGN['n_layers'],
                                          ALIGN['n_heads'], ALIGN['n_kv_heads'],
                                          ALIGN['ffn_hidden'], ALIGN['eps'], ALIGN['theta'], SEED)
    cache_aligned, cache_misaligned, cache_seq, cache_per_layer = \
        hf_cache_equivalence(hf_model, ids, prefill=4)
    measured_bytes = sum(b for _, b in cache_per_layer)
    measured_per_token = measured_bytes // (ids.shape[0] * cache_seq)
    formula_bytes = kv_bytes_per_token(ALIGN['n_layers'], ALIGN['n_kv_heads'], head_dim, 4)
    log_verify('HF DynamicCache 增量解码 == 全量前向', '真跑验证',
               '同下标对齐比对 logits；错开一位会得到假失配')

    # ---------------- B. KV 显存账
    ref_rows = [(name, kv_bytes_per_token(L, g_, dh, db)) for name, L, g_, dh, db in REF_MODELS]
    mha_conc = concurrency_budget(80, 14, ref_rows[0][1], 4096)
    gqa_conc = concurrency_budget(80, 14, ref_rows[1][1], 4096)
    mla_s = mla_scalars_per_token(MLA_CFG['kv_lora_rank'], MLA_CFG['qk_rope_head_dim'])
    mha_s = mha_scalars_per_token(MLA_CFG['mha_heads'], MLA_CFG['mha_head_dim'])
    ds_s = mha_scalars_per_token(MLA_CFG['ds_heads'], MLA_CFG['mha_head_dim'])
    mla_save = 1.0 - mla_s / mha_s
    ds_save = 1.0 - mla_s / ds_s
    mla_groups = gqa_equiv_groups(mla_s, MLA_CFG['mha_head_dim'])
    mla_layer_bytes = mla_s * MLA_CFG['dtype_bytes']
    mla_req_bytes = mla_layer_bytes * MLA_CFG['n_layers'] * MLA_CFG['ctx']
    mla_batch_bytes = mla_req_bytes * MLA_CFG['batch']
    mla_gib = mla_batch_bytes / 1024.0 ** 3
    mha_gib = mla_batch_bytes * mha_s / mla_s / 1024.0 ** 3
    ds_gib = mla_batch_bytes * ds_s / mla_s / 1024.0 ** 3
    log_verify('MLA 显存账（低秩潜向量 512+64=576）', '未验证',
               'MLA 本体未实现；512/64/61 层核到的是 DeepSeek-V3 与 Kimi K2 的公开 config，'
               '论文正文有 (d_c+d_h^R)×层数 的式子，但两个 MHA 基线与 61 层×128K×batch8 的 GiB 数'
               '都是本文自选对照档的算术推导，不是论文口径')

    # ---------------- C. GQA sweep
    set_all_seeds(SEED + 11)
    H_SW, S_SW, D_SW = ALIGN['n_heads'], 64, head_dim
    Kg = torch.randn(H_SW, S_SW, D_SW, generator=torch.Generator().manual_seed(SEED))
    Vg = torch.randn(H_SW, S_SW, D_SW, generator=torch.Generator().manual_seed(SEED + 1))
    Qg = torch.randn(H_SW, S_SW, D_SW, generator=torch.Generator().manual_seed(SEED + 2))
    sweep = []
    for g_ in (32, 16, 8, 4, 1):
        set_all_seeds(SEED + 3)
        attn = GroupedQueryAttention(ALIGN['dim'], ALIGN['n_heads'], g_)
        meas = sum(p.numel() for p in attn.parameters())
        mse, kl, maxd = gqa_degrade_proxy(Kg, Vg, Qg, g_)
        sweep.append(dict(g=g_, params=meas,
                          formula=attention_params_formula(ALIGN['dim'], ALIGN['n_heads'], g_),
                          d2=meas / ALIGN['dim'] ** 2,
                          kv_bytes=kv_bytes_per_token(ALIGN['n_layers'], g_, head_dim, 4),
                          rel=kv_bytes_per_token(ALIGN['n_layers'], 32, head_dim, 4)
                          / kv_bytes_per_token(ALIGN['n_layers'], g_, head_dim, 4),
                          bw=decode_bandwidth_bytes(ALIGN['n_layers'], g_, head_dim, 4, 4096),
                          mse=mse, kl=kl, maxd=maxd))

    # ---------------- D. FlashAttention 等价 + HBM 账
    flash_rows = []
    for S in (256, 512, 1024):
        qq = torch.randn(1, 4, S, 64, generator=torch.Generator().manual_seed(SEED))
        kk = torch.randn(1, 4, S, 64, generator=torch.Generator().manual_seed(SEED + 1))
        vv = torch.randn(1, 4, S, 64, generator=torch.Generator().manual_seed(SEED + 2))
        ref = naive_attention(qq, kk, vv)
        d32 = {}
        for br in (16, 64, 128):
            d32[br] = (flash_attention_tiled(qq, kk, vv, br, br) - ref).abs().max().item()
        q64, k64, v64 = qq.double(), kk.double(), vv.double()
        d64 = (flash_attention_tiled(q64, k64, v64, 64, 64)
               - naive_attention(q64, k64, v64)).abs().max().item()
        flash_rows.append(dict(S=S, d32=d32, d64=d64, **hbm_accounting(S, 64, 64)))
    log_verify('手写 FlashAttention 等价实现（分块 + online softmax）', '真跑验证',
               'fp32 阈值 1e-5、fp64 阈值 1e-12，与朴素逐元素对账；[S,S] 从不整体落地')

    # ---------------- E. PagedAttention 分页模拟器
    lengths = make_request_trace(SEED, n_req=64, lo=128, hi=512)
    MAX_LEN = 1024
    BUDGET = 4096
    H_P, D_P = 4, 8
    # 连续预留：每请求固定占 ceil(MAX_LEN/BLOCK) 块（哪怕实际只有 200 token 也预留 1024）
    served_contig, used_contig, contig_lens = admit_greedy(
        lengths, BUDGET, lambda L: contiguous_blocks_needed(MAX_LEN))
    # 分页：每请求只占 ceil(L/BLOCK) 块（按需分配，用多少占多少）
    served_paged, used_paged, paged_lens = admit_greedy(
        lengths, BUDGET, lambda L: math.ceil(L / BLOCK))
    sum_Lc, sum_Lp = sum(contig_lens), sum(paged_lens)
    waste_contig = 1.0 - sum_Lc / (served_contig * MAX_LEN)
    waste_paged = 1.0 - sum_Lp / (used_paged * BLOCK)
    tail_frags = [BLOCK * math.ceil(L / BLOCK) - L for L in paged_lens]
    # 真跑：把分页服务到的请求写进真实分页池，验证 used_paged 与池实际占块一致
    pool = PagedKVPool(BUDGET, H_P, D_P)
    pool_used = 0
    for i, L in enumerate(paged_lens):
        K = torch.randn(L, H_P, D_P, generator=torch.Generator().manual_seed(SEED + 100 + (i % 64)))
        V = torch.randn(L, H_P, D_P, generator=torch.Generator().manual_seed(SEED + 200 + (i % 64)))
        pool_used += pool.write_kv(i, K, V)
    assert pool_used == used_paged, '分页池实际占块 %d != 记账 %d' % (pool_used, used_paged)
    # 数学不变性：同一段 K/V，连续张量 vs 分页池 gather，注意力上下文 bitwise 相同
    K0 = torch.randn(96, H_P, D_P, generator=torch.Generator().manual_seed(SEED + 300))
    V0 = torch.randn(96, H_P, D_P, generator=torch.Generator().manual_seed(SEED + 301))
    pool2 = PagedKVPool(64, H_P, D_P)
    pool2.write_kv('probe', K0, V0)
    Kp, Vp = pool2.gather('probe')
    qr = torch.randn(H_P, D_P, generator=torch.Generator().manual_seed(SEED + 302))
    ctx_contig = paged_attention_ctx(K0, V0, qr)
    ctx_paged = paged_attention_ctx(Kp, Vp, qr)
    paged_math_diff = (ctx_contig - ctx_paged).abs().max().item()
    gather_diff = (K0 - Kp).abs().max().item() + (V0 - Vp).abs().max().item()
    paged_stats = dict(waste_contig=waste_contig, waste_paged=waste_paged,
                       served_contig=served_contig, served_paged=served_paged,
                       max_tail=max(tail_frags), budget_blocks=BUDGET)
    log_verify('手写 PagedAttention 分页 KV 模拟器', '真跑验证',
               'block table 记账 + 碎片账 + 并发账；gather 后注意力数学 bitwise 不变')

    # ---------------- F. RoPE × KV Cache
    lam = rope_spectrum(128, 10000.0)
    prefix = torch.randn(32, 64, generator=torch.Generator().manual_seed(SEED))
    qv = torch.randn(64, generator=torch.Generator().manual_seed(SEED + 1))
    ctx_true, ctx_stale, ctx_fixed = demo_prefix_reuse(prefix, qv, delta=17, head_dim=64)
    reuse_fixed_diff = (ctx_true - ctx_fixed).abs().max().item()
    reuse_stale_diff = (ctx_true - ctx_stale).abs().max().item()
    err_good, err_bad = demo_rope_precision(4096, 64)
    log_verify('RoPE × KV Cache：前缀复用需重旋转 + FP32 算角再 cast', '真跑验证',
               '重旋转 == 从头重算；直接复用旧缓存位置错位；fp16 先算角误差放大百倍以上')

    # ---------------- G. 工业库
    try_industrial_libs()

    secs = time.perf_counter() - t_main

    # ---------------- 报表
    lines = []
    lines.append('=' * W)
    lines.append('生产篇 04 · 自注意力与 KV Cache 报表（seed=%d）' % SEED)
    lines.append('=' * W)

    lines.append('\n【A】手写 GQA 与 HF LlamaForCausalLM 逐层数值对齐（KV 头数 32→8）')
    lines.append('    对齐配置：vocab=%d dim=%d layers=%d Q头=%d KV头=%d head_dim=%d ffn=%d seq=%d batch=%d'
                 % (ALIGN['vocab'], ALIGN['dim'], ALIGN['n_layers'], ALIGN['n_heads'],
                    ALIGN['n_kv_heads'], head_dim, ALIGN['ffn_hidden'], ALIGN['seq'], ALIGN['batch']))
    lines.append('    k_proj.weight 形状 %s（MHA 应为 [%d,%d]）→ KV 投影瘦 %d 倍'
                 % (tuple(hf_model.model.layers[0].self_attn.k_proj.weight.shape),
                    ALIGN['dim'], ALIGN['dim'], ALIGN['n_heads'] // ALIGN['n_kv_heads']))
    lines.append('    同权重前向，逐层 hidden_states / 最终 logits 的最大绝对误差：')
    for name, d in diffs:
        lines.append('      %-10s max|Δ| = %.3e' % (name, d))
    lines.append('      %-10s max|Δ| = %.3e' % ('logits', logit_diff))
    lines.append('    → 全层最大误差 %.3e，阈值 1e-4 内即判等价（实测：%s）'
                 % (max_diff, '通过' if max_diff <= 1e-4 else '超阈值'))
    lines.append('    静默 bug 两连（都不报错、都"能跑"）：')
    lines.append('      ① expand=repeat 头序错配：logits max|Δ| = %.3e（正确写法的 %.0f 倍）'
                 % (expand_wrong_diff, expand_wrong_diff / max(max_diff, 1e-30)))
    lines.append('      ② 增量解码错开一位比对：max|Δ| = %.3e（同下标对齐仅 %.3e）'
                 % (cache_misaligned, cache_aligned))
    lines.append('    两种正确写法：repeat_interleave vs group reshape，torch.equal = %s' % eq_correct)
    lines.append('    repeat vs repeat_interleave 的 K 张量 max|Δ| = %.3e' % wrong_vs_right)

    lines.append('\n【B】KV Cache 显存账：2 × 层数 × KV头数 × 头维度 × 字节数')
    lines.append('    tiny 实测（DynamicCache 真张量）：%d 字节 / (token·batch) == 闭式 %d  %s'
                 % (measured_per_token, formula_bytes,
                    '一致' if measured_per_token == formula_bytes else '不一致'))
    lines.append('    缓存序列长度 = %d；逐层 (K 形状, K+V 字节) = %s'
                 % (cache_seq, [(tuple(s), b) for s, b in cache_per_layer]))
    lines.append('    真实模型口径外推（引用口径，本机无权重，标未验证）：')
    for name, b in ref_rows:
        lines.append('      %-40s %7d B/token = %6.1f KB' % (name, b, b / 1024.0))
    lines.append('    并发账（80GB 总预算 − 14GB 权重，4K 上下文，整数除法，不真分配显存）：')
    lines.append('      MHA 512KB/token → %d 路；GQA-8 128KB/token → %d 路；比值 %.1f×'
                 % (mha_conc, gqa_conc, gqa_conc / mha_conc))
    lines.append('    第四条路线 MLA（低秩潜向量；512/64/61 层核自 DeepSeek-V3 与 Kimi K2 的公开 config，账本身是整数算术）：')
    lines.append('      每 token 每层 = kv_lora_rank %d + qk_rope_head_dim %d = %d 个标量'
                 '（账本里没有那个 2：K/V 共享同一份潜向量）'
                 % (MLA_CFG['kv_lora_rank'], MLA_CFG['qk_rope_head_dim'], mla_s))
    lines.append('      对照档 A（Llama 式 MHA %d头×%d维）= %d 标量 → MLA 省 %.2f%%（1 − %d/%d）'
                 % (MLA_CFG['mha_heads'], MLA_CFG['mha_head_dim'], mha_s,
                    100.0 * mla_save, mla_s, mha_s))
    lines.append('      对照档 B（DeepSeek 式 MHA %d头×%d维）= %d 标量 → MLA 省 %.2f%%（1 − %d/%d）'
                 % (MLA_CFG['ds_heads'], MLA_CFG['mha_head_dim'], ds_s,
                    100.0 * ds_save, mla_s, ds_s))
    lines.append('      %d / (2×%d) = %.2f → 论文原话"KV cache 等于只有 2.25 组的 GQA"；'
                 'GQA 只能整数个头上调，MLA 调的是表示维度'
                 % (mla_s, MLA_CFG['mha_head_dim'], mla_groups))
    lines.append('      同一 head_dim=%d 的 GQA 档对照：g=8 → %d 标量（MLA 再低 %.2f×）、'
                 'g=1(MQA) → %d 标量（MLA 高 %.2f×）'
                 % (MLA_CFG['mha_head_dim'], 2 * 8 * MLA_CFG['mha_head_dim'],
                    2 * 8 * MLA_CFG['mha_head_dim'] / mla_s, 2 * MLA_CFG['mha_head_dim'],
                    mla_s / (2 * MLA_CFG['mha_head_dim'])))
    lines.append('      %d 层 fp16 · %dK 上下文 · batch %d：MLA = %d B = %.2f GiB；'
                 '档 A MHA = %.1f GiB；档 B MHA = %.1f GiB（后两个单卡根本放不下）'
                 % (MLA_CFG['n_layers'], MLA_CFG['ctx'] // 1024, MLA_CFG['batch'],
                    mla_batch_bytes, mla_gib, mha_gib, ds_gib))
    lines.append('      口径提醒：DeepSeek-V2 摘要自述"KV cache 减少 93.3%"，比的是 DeepSeek 67B，'
                 '与上面两个百分数是不同基线，不可互换（引用口径，未复现）')
    lines.append('      同摘要自述最大生成吞吐提升到 5.76×、训练成本省 42.5%（对 DeepSeek 67B，引用口径）；'
                 '出处：MLA 由 DeepSeek-V2 提出，V3 摘要自述 adopts MLA，Kimi K2 公开 config 同为 512+64')

    lines.append('\n【C】GQA sweep：KV 头数 32→{32,16,8,4,1}（MHA→MQA 全谱）')
    lines.append('    %-5s %12s %12s %12s %8s %12s %10s %10s'
                 % ('KV头', '每层注意力参数', '闭式(d²倍数)', 'KV字节/token', '相对MHA',
                    'decode读带宽', '降级MSE', '降级KL'))
    for r in sweep:
        lines.append('    %-5d %12d %9.3f·d² %12d %7.2f× %12d %10.3e %10.3e'
                     % (r['g'], r['params'], r['d2'], r['kv_bytes'], r['rel'], r['bw'],
                        r['mse'], r['kl']))
    lines.append('    → g=8 时每层注意力参数 = %.3f·d²（MHA 的 4·d² 省 %.1f%%）；KV 字节 32→8 精确 %.1f×、32→1 精确 %.0f×'
                 % (sweep[2]['d2'], 100.0 * (1 - sweep[2]['d2'] / 4.0),
                    sweep[0]['kv_bytes'] / sweep[2]['kv_bytes'],
                    sweep[0]['kv_bytes'] / sweep[4]['kv_bytes']))
    lines.append('    → 降级代理 max|Δ|（只打印不断言，g=1 可能回落）：%s'
                 % ', '.join('g=%d:%.3e' % (r['g'], r['maxd']) for r in sweep))
    lines.append('    → 注意：decode 读带宽随 g 线性降，但注意力 FLOPs 与 g 无关——省显存不等于省计算')

    lines.append('\n【D】手写 FlashAttention 等价实现：分块 + online softmax（数学精确，零近似）')
    lines.append('    %-6s %22s %12s %14s %14s %10s'
                 % ('S', 'fp32 max|Δ| (br=16/64/128)', 'fp64 max|Δ|', '朴素落地元素',
                    '分块落地元素', '落地比'))
    for r in flash_rows:
        lines.append('    %-6d %22s %12.3e %14d %14d %9.1f×'
                     % (r['S'], '/'.join('%.2e' % r['d32'][b] for b in (16, 64, 128)),
                        r['d64'], r['naive_landed'], r['tiled_landed'], r['ratio_landed']))
    lines.append('    [S,S] 级张量落地次数：朴素 %d 次 → 分块 %d 次'
                 % (flash_rows[0]['naive_squares'], flash_rows[0]['tiled_squares']))
    lines.append('    总读写字节（不是 O(S)！分块要重读 K/V S/br 遍）：S=1024 朴素 %d vs 分块 %d'
                 % (flash_rows[-1]['naive_total'], flash_rows[-1]['tiled_total']))
    lines.append('    诚实边界：CPU 上没有 SRAM/HBM 之分，本机分块实现不会更快（大概率更慢）；')
    lines.append('              量的是"账"不是加速比。GPU 侧只能引原文口径（本机未复现）：')
    lines.append('              注意力计算 7.6×（A100、GPT-2、head_dim 64、seq 1024、前向+反向、vs PyTorch 实现）；')
    lines.append('              端到端 15%（BERT-large seq 512，对比 MLPerf 1.1 记录）、'
                 '3×（GPT-2 seq 1K）、2.4×（Long Range Arena）；')
    lines.append('              HBM 访问量的原文界是 Θ(Nd+N²) → Θ(N²d²/M)，最多约 9× 更少')

    lines.append('\n【E】手写 PagedAttention 分页 KV 模拟器（BLOCK=%d token/块）' % BLOCK)
    lines.append('    请求 trace：%d 条，长度 %d~%d（seed 确定性）' % (len(lengths), min(lengths), max(lengths)))
    lines.append('    连续预留浪费率 = %.2f%%（满负载解析上界 = 1 − E[L]/max_len，'
                 '本 trace E[L]≈%.1f → %.2f%%；vLLM 正文常引的 60-80%% 是生产分布口径，摘要只写到 near-zero waste）'
                 % (100.0 * waste_contig, float(np.mean(lengths)),
                    100.0 * (1 - float(np.mean(lengths)) / MAX_LEN)))
    lines.append('    分页浪费率     = %.2f%%（< 4%%；只有尾块内部碎片）' % (100.0 * waste_paged))
    lines.append('    尾块内部碎片最大值 = %d token/请求（上界 BLOCK-1 = %d）' % (max(tail_frags), BLOCK - 1))
    lines.append('    同预算 4096 块：连续 %d 路 vs 分页 %d 路（%.2f×）'
                 % (served_contig, served_paged, served_paged / served_contig))
    lines.append('    数学不变性：gather 后 K/V max|Δ| = %.1e，注意力上下文 max|Δ| = %.1e（bitwise）'
                 % (gather_diff, paged_math_diff))
    lines.append('    诚实标注：只复现记账与碎片；调度/抢占/前缀共享/CUDA kernel 属 vLLM，服务化留生产篇 13')
    lines.append('              上面的 %.2f× 依赖这条 trace 的长度分布（请求全打满 max_len 时退化为 1.0×）；'
                 % (served_paged / served_contig))
    lines.append('              引用口径（已核摘要原句）：vLLM 把流行 LLM 的吞吐提升 2-4×，'
                 '对照组是 FasterTransformer / Orca，不是朴素 HF 循环')
    lines.append('              前缀共享（本模拟器未实现）：分页把"逻辑序列"与"物理位置"解绑，'
                 '多请求的 block_table 前缀可指向同一批物理块，写时才拆分（copy-on-write）')

    lines.append('\n【F】RoPE × KV Cache')
    lines.append('    波长谱（head_dim=128, theta=1e4）：λ_0 = %.3f token，λ_63 = %.0f token'
                 % (lam[0].item(), lam[-1].item()))
    lines.append('    前缀复用（Δ=17）：重旋转 vs 从头重算 max|Δ| = %.3e；直接复用旧缓存 max|Δ| = %.3e'
                 % (reuse_fixed_diff, reuse_stale_diff))
    lines.append('    精度账（pos=4096）：fp32 算角再 cast = %.3e；角度先 cast fp16 = %.3e（%.0f 倍）'
                 % (err_good, err_bad, err_bad / max(err_good, 1e-30)))
    lines.append('    位置外推四件套（只改"位置→角"这张表，KV 字节账不变；原文口径，本机未复现）：')
    lines.append('      PI：位置索引线性下缩放，原文报 32768 窗口只需微调 1000 步；')
    lines.append('      NTK-aware：改 base theta、按维非均匀缩放（社区帖出身，无正式论文）；')
    lines.append('      YaRN：NTK-by-parts 分段 + 注意力 logit 温度，原文报少 10× token、2.5× 训练步；')
    lines.append('      LongRoPE：按维非均匀插值 + 渐进两段扩窗，原文报最长 2048k token；')
    lines.append('      已核事实：DeepSeek-V3 与 Kimi K2 的 config 里 rope_scaling.type 就是 yarn')

    lines.append('\n【G】线上问题清单（现象→定位→解决，全部真跑）')
    lines.append('    显存超限：tiny 实测 %d B/token → 外推 512KB/token → 4K 上下文临界并发 %d→%d 路'
                 % (measured_per_token, mha_conc, gqa_conc))
    lines.append('    并发上不去：连续预留浪费 %.1f%% → 同预算 %d 路；分页浪费 %.1f%% → %d 路'
                 % (100.0 * waste_contig, served_contig, 100.0 * waste_paged, served_paged))
    lines.append('    长序列 OOM：朴素 [S,S] 在 S=32768 单头单层 fp16 = %.2f GB（×32 头 = %.0f GB）'
                 % (32768 * 32768 * 2 / 1024 ** 3, 32768 * 32768 * 2 * 32 / 1024 ** 3))
    lines.append('               分块只落地 S·d+2S = %d 元素（S=1024,d=64），[S,S] 落地 0 次'
                 % flash_rows[-1]['tiled_landed'])
    lines.append('    静默 bug 两连：repeat 头序错配（%.3e）+ 旧缓存位置错位（%.3e）'
                 % (expand_wrong_diff, reuse_stale_diff))

    lines.append('\n    工业库调用骨架（完整讲解放文章）：')
    for s in FLASH_SKELETON.splitlines():
        lines.append('      ' + s)
    for s in VLLM_SKELETON.splitlines():
        lines.append('      ' + s)

    lines.append('')
    lines.append('=' * W)
    lines.append('【验证状态清单】哪些是本机真跑、哪些是未验证示例')
    lines.append('=' * W)
    for item, status, note in VERIFY_LOG:
        lines.append('    [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))

    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(os.path.join(OUT, 'gqa_sweep.tsv'), 'w', encoding='utf-8') as f:
        f.write('kv_heads\tattn_params_measured\tattn_params_formula\td2_multiple\t'
                'kv_bytes_per_token\trel_to_mha\tdecode_bw_bytes\tdegrade_mse\tdegrade_kl\t'
                'degrade_maxdiff\n')
        for r in sweep:
            f.write('%d\t%d\t%d\t%.6f\t%d\t%.6f\t%d\t%.6e\t%.6e\t%.6e\n'
                    % (r['g'], r['params'], r['formula'], r['d2'], r['kv_bytes'], r['rel'],
                       r['bw'], r['mse'], r['kl'], r['maxd']))
    with open(os.path.join(OUT, 'kv_flash_paged.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED,
                   'align': {'config': ALIGN, 'diffs': diffs, 'logit_diff': logit_diff,
                             'expand_wrong_diff': expand_wrong_diff,
                             'cache_aligned': cache_aligned, 'cache_misaligned': cache_misaligned,
                             'measured_bytes_per_token': measured_per_token,
                             'formula_bytes_per_token': formula_bytes},
                   'ref_models': [{'name': n, 'bytes_per_token': b} for n, b in ref_rows],
                   'concurrency': {'mha': mha_conc, 'gqa8': gqa_conc},
                   'mla': dict(MLA_CFG, scalars_per_token=mla_s,
                               mha_baseline_scalars=mha_s, ds_baseline_scalars=ds_s,
                               save_ratio=mla_save, ds_save_ratio=ds_save,
                               gqa_equiv_groups=mla_groups,
                               batch_bytes=mla_batch_bytes,
                               mla_gib=mla_gib, mha_gib=mha_gib, ds_gib=ds_gib),
                   'flash': [{k: v for k, v in r.items()} for r in flash_rows],
                   'paged': paged_stats,
                   'rope': {'lambda_0': lam[0].item(), 'lambda_63': lam[-1].item(),
                            'reuse_fixed_diff': reuse_fixed_diff,
                            'reuse_stale_diff': reuse_stale_diff,
                            'fp32_then_cast': err_good, 'cast_first': err_bad}},
                  f, ensure_ascii=False, indent=2)

    fig = draw(sweep, flash_rows, paged_stats, ref_rows)
    if fig:
        print('\n[配图] %s' % fig)
    print('\n    （本机 CPU 全程 %.1f 秒；秒表只打印不落盘，保证产物逐字节可复现）' % secs)

    # ---------------- 自校验
    print('\n' + '=' * W)
    print('【自校验】')
    print('=' * W)
    assert max_diff <= 1e-4, 'GQA 逐层对齐超阈值 max|Δ|=%.3e' % max_diff
    print('  1. 手写 GQA vs HF LlamaForCausalLM 逐层 max|Δ| = %.3e ≤ 1e-4  通过' % max_diff)
    assert logit_diff <= 1e-4, 'logits 对齐超阈值 %.3e' % logit_diff
    print('  2. 最终 logits max|Δ| = %.3e ≤ 1e-4  通过' % logit_diff)
    assert cache_aligned <= 1e-4, 'DynamicCache 增量 != 全量：%.3e' % cache_aligned
    assert cache_misaligned > 100 * cache_aligned, '错开一位的假失配应远大于真失配'
    print('  3. DynamicCache 增量 == 全量（同下标）max|Δ| = %.3e；错开一位 = %.3e（假失配）  通过'
          % (cache_aligned, cache_misaligned))
    assert expand_wrong_diff > 100 * max_diff, 'repeat 错配应远大于正确对齐'
    print('  4. expand=repeat 头序错配 max|Δ| = %.3e > 100× 正确对齐(%.3e)  通过'
          % (expand_wrong_diff, max_diff))
    assert eq_correct, 'repeat_interleave 与 group 应 bitwise 相同'
    print('  5. repeat_interleave vs group reshape：torch.equal = True  通过')
    set_all_seeds(SEED + 5)
    mha_ref = CausalSelfAttentionMHA(64, 8).eval()
    gqa_deg = GroupedQueryAttention(64, 8, 8).eval()
    gqa_deg.load_state_dict(mha_ref.state_dict())
    xdeg = torch.randn(2, 6, 64, generator=torch.Generator().manual_seed(SEED))
    cosd, sind = build_rope_cache(6, 8)
    with torch.no_grad():
        deg_diff = (mha_ref(xdeg, cosd, sind) - gqa_deg(xdeg, cosd, sind)).abs().max().item()
    assert deg_diff < 1e-12, 'g==h 应精确退化为 MHA：%.3e' % deg_diff
    print('  6. g==h 退化：GQA 与 MHA 同构实现 max|Δ| = %.1e（跨篇回归生产篇 03）  通过' % deg_diff)
    for r in sweep:
        assert r['params'] == r['formula'], 'g=%d 参数实测 %d != 闭式 %d' % (r['g'], r['params'], r['formula'])
    assert abs(sweep[2]['d2'] - 2.5) < 1e-9, 'g=8 应为 2.5·d²'
    print('  7. 每层注意力参数实测 == 闭式 d²(1+2g/h+1)（5 档整数）；g=8 = %.2f·d² 省 %.1f%%  通过'
          % (sweep[2]['d2'], 100.0 * (1 - 2.5 / 4.0)))
    assert measured_per_token == formula_bytes, 'KV 字节实测 != 闭式'
    assert sweep[0]['kv_bytes'] / sweep[2]['kv_bytes'] == 4.0, '32→8 应精确 4 倍'
    assert sweep[0]['kv_bytes'] / sweep[4]['kv_bytes'] == 32.0, '32→1 应精确 32 倍'
    assert mla_s == 576 and mha_s == 8192, 'MLA/MHA 每 token 标量数闭合：%d vs %d' % (mla_s, mha_s)
    assert mla_save == 0.9296875, 'MLA 相对 32 头 MHA 的节省比应整数闭合：%.10f' % mla_save
    assert ds_s == 32768 and ds_save == 0.982421875, '128 头档应省 98.24%%：%.10f' % ds_save
    assert mla_groups == 2.25, '576 应折合 2.25 个 KV 头：%.6f' % mla_groups
    assert mha_gib / mla_gib - mha_s / mla_s < 1e-9, '同一批标量比在字节账上应一致'
    assert mla_req_bytes * MLA_CFG['batch'] == mla_batch_bytes, '单请求 × batch 应整数闭合'
    print('  8. KV 字节实测 == 闭式；32→8 精确 %.1f×、32→1 精确 %.0f×；'
          'MLA 账本 %d 标量 vs 档A %d / 档B %d → 省 %.2f%% / %.2f%%、折合 %.2f 组  通过'
          % (sweep[0]['kv_bytes'] / sweep[2]['kv_bytes'],
             sweep[0]['kv_bytes'] / sweep[4]['kv_bytes'],
             mla_s, mha_s, ds_s, 100.0 * mla_save, 100.0 * ds_save, mla_groups))
    assert sweep[0]['mse'] == 0.0, 'g=32 降级 MSE 应为 0'
    ms = [r['mse'] for r in sweep]
    ks = [r['kl'] for r in sweep]
    assert all(ms[i] <= ms[i + 1] + 1e-12 for i in range(len(ms) - 1)), 'MSE 应单调不减'
    assert all(ks[i] <= ks[i + 1] + 1e-12 for i in range(len(ks) - 1)), 'KL 应单调不减'
    print('  9. 降级代理 MSE(g=32)=0 且 MSE/KL 随 g 减小单调不减（max|Δ| 只打印）  通过')
    for r in flash_rows:
        assert max(r['d32'].values()) <= 1e-5, 'fp32 分块超阈值'
        assert r['d64'] <= 1e-12, 'fp64 分块超阈值'
    d32s = [max(r['d32'].values()) for r in flash_rows]
    assert max(d32s) - min(d32s) < 1e-4, 'fp32 误差应与 block size 无关'
    print('  10. 分块 == 朴素：fp32 ≤ %.2e（与 br 无关）、fp64 ≤ %.2e（零近似硬证据）  通过'
          % (max(d32s), max(r['d64'] for r in flash_rows)))
    for r in flash_rows:
        assert r['naive_landed'] == 4 * r['S'] ** 2 and r['tiled_landed'] == r['S'] * 64 + 2 * r['S']
        assert r['naive_squares'] == 2 and r['tiled_squares'] == 0
    print('  11. HBM 落地元素整数精确：4S² vs S·d+2S；[S,S] 落地 2→0 次  通过')
    assert gather_diff == 0.0 and paged_math_diff == 0.0, '分页 gather 应 bitwise 不变'
    print('  12. 分页 gather：K/V 与注意力上下文 max|Δ| = %.1e（只改记账不改数学）  通过' % paged_math_diff)
    assert waste_contig > 0.60, '连续预留浪费应落在 60-80%% 区间：%.3f' % waste_contig
    assert waste_paged < 0.04, '分页浪费应 < 4%%：%.3f' % waste_paged
    assert max(tail_frags) <= BLOCK - 1, '尾块碎片上界 BLOCK-1'
    assert served_paged >= 2.5 * served_contig, '分页并发应 ≥ 2.5× 连续'
    print('  13. 碎片与并发：连续浪费 %.1f%% / 分页 %.1f%%；尾块 ≤ %d；并发 %d vs %d（%.2f×）  通过'
          % (100 * waste_contig, 100 * waste_paged, max(tail_frags),
             served_contig, served_paged, served_paged / served_contig))
    assert reuse_fixed_diff < 1e-6, '重旋转应等于从头重算'
    assert reuse_stale_diff > 1e-2, '直接复用旧缓存应明显错位'
    assert abs(lam[0].item() - 2 * math.pi) < 1e-4, 'λ_0 应为 2π'
    assert err_bad > 100 * err_good, 'fp16 先算角误差应放大百倍以上'
    print('  14. RoPE×Cache：重旋转 == 重算(%.1e)、旧缓存错位(%.1e)、λ_0=2π、fp16 算角 %.0f×  通过'
          % (reuse_fixed_diff, reuse_stale_diff, err_bad / err_good))
    print('\n  全部断言通过。')
    print('\n[产物] 报表 / gqa_sweep.tsv / kv_flash_paged.json / 配图 → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
