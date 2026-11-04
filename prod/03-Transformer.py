#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产篇 03 · Transformer：手写现代 Llama block，与 HF LlamaForCausalLM 逐层数值对齐

对应文章：同目录 03-Transformer.md（建议先读 基础篇/03-Transformer.py 与 .md）
运行：    python3 03-Transformer.py
依赖：    numpy + torch（CPU）+ transformers（离线用 LlamaConfig 随机初始化一个 tiny Llama，
          不下载任何预训练权重、不联网）。flash-attn / vLLM / Megatron 等重依赖不安装，
          归为"方向性说明 + 调用骨架"，运行结束在报表里逐条标注真跑 / 未验证。
产物：    ./output/生产篇03-Transformer/ 下的报表、逐层对齐 diff 表、loss 曲线配图

基础篇把单头 Post-Norm 的机理讲透、维度流跑通；生产篇换一副眼镜——把整块现代解码器
（RMSNorm + RoPE + SwiGLU + 因果多头注意力 + Pre-Norm）用 torch 手写出来，然后：

    A. 手写现代 Llama block —— 逐组件写清 RMSNorm / RoPE / SwiGLU / 因果 MHA / Pre-Norm 残差
    B. 与 HF LlamaForCausalLM 逐层前向数值对齐 —— 同权重下比对每层 hidden_states 与最终 logits
    C. Pre-Norm vs Post-Norm 训练稳定性实验 —— 同语料同超参，看谁 spike / 谁平稳收敛
    D. 参数量与显存分解 —— 每个子层占多少参数，词表/维度怎么放大成本
    E. 线上问题清单 —— 维度不匹配 / 梯度爆炸 / 收敛慢（现象→定位→解决，含可跑 demo）

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
OUT = os.path.join(HERE, 'output', '生产篇03-Transformer')

VERIFY_LOG = []


def log_verify(item, status, note=''):
    VERIFY_LOG.append((item, status, note))
    print('  [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================================================
#  A. 手写现代 Llama block（torch，float32，逐组件）
# ==================================================================
class RMSNorm(nn.Module):
    """Llama 用 RMSNorm 取代 LayerNorm：只除以均方根、不减均值，也没有 bias。
       前向 = x / sqrt(mean(x^2) + eps) * weight。比 LayerNorm 少一次均值统计，更快。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # 与 HF LlamaRMSNorm 一致：在 float32 上算方差，再乘回 weight
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        xf = xf * torch.rsqrt(var + self.eps)
        return (self.weight.float() * xf).to(x.dtype)


def build_rope_cache(seq_len, head_dim, theta=10000.0, device='cpu'):
    """RoPE 旋转位置编码的 cos/sin 缓存。
       inv_freq[i] = theta^(-2i/head_dim)；freqs = outer(pos, inv_freq)；emb = [freqs, freqs]。
       返回 cos, sin，形状 [seq_len, head_dim]。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                 # [seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)            # [seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x):
    """把向量后半段取负、与前半段交换：[-x2, x1]。RoPE 的旋转核心。"""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """对 q、k 施加旋转位置编码（与 HF apply_rotary_pos_emb 逐式一致）。
       q/k: [B, H, S, Dh]；cos/sin: [S, Dh] → 广播到 [1,1,S,Dh]。"""
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q_e = q * cos + rotate_half(q) * sin
    k_e = k * cos + rotate_half(k) * sin
    return q_e, k_e


def rope_dot(q, k, m, n, head_dim, theta=10000.0):
    """把向量 q 放在位置 m、k 放在位置 n，各施加 RoPE 后做点积。
       RoPE 的关键性质：这个点积只依赖相对位置 (m-n)，整体平移不改变结果。"""
    cos, sin = build_rope_cache(max(m, n) + 1, head_dim, theta)
    qv, kv = q.view(1, 1, 1, head_dim), k.view(1, 1, 1, head_dim)
    qe = qv * cos[m] + rotate_half(qv) * sin[m]
    ke = kv * cos[n] + rotate_half(kv) * sin[n]
    return torch.sum(qe * ke).item()


class CausalSelfAttention(nn.Module):
    """因果多头自注意力（MHA：KV 头数 == Q 头数）。GQA 是生产篇 04 的主题，这里先不做。
       q/k/v/o 均为无 bias 线性层，与 Llama 一致。"""

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

    def forward(self, x, cos, sin, return_weights=False):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        scores = (q @ k.transpose(-1, -2)) * self.scaling          # [B,H,S,S]
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))          # 只能看左边（含自己）
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(ctx)
        return (out, w) if return_weights else out


class SwiGLU(nn.Module):
    """Llama 的前馈层：down( silu(gate(x)) * up(x) )。三个无 bias 线性层，
       比原始 Transformer 的"两层 ReLU FFN"多一个门控分支，参数多约 1.5 倍但更稳更好训。"""

    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    """Pre-Norm 解码器块：先归一化再进子层，残差直接加在"未归一化"的主干上。
       h = x + Attn(RMSNorm(x))；out = h + MLP(RMSNorm(h))。与 LlamaDecoderLayer 完全同构。"""

    def __init__(self, dim, n_heads, ffn_hidden, eps=1e-5, theta=10000.0, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.self_attn = CausalSelfAttention(dim, n_heads, theta)
        self.mlp = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        if self.pre_norm:
            x = x + self.self_attn(self.input_layernorm(x), cos, sin)
            x = x + self.mlp(self.post_attention_layernorm(x))
        else:
            # Post-Norm（原始 Transformer 写法）：先残差相加再归一化，留作稳定性对照
            x = self.input_layernorm(x + self.self_attn(x, cos, sin))
            x = self.post_attention_layernorm(x + self.mlp(x))
        return x


class TinyLlama(nn.Module):
    """手写的迷你 Llama：embedding → N×LlamaBlock → 末端 RMSNorm → lm_head。
       刻意与 HF LlamaForCausalLM 的子层命名/形状对齐，方便逐层灌权重与比对。"""

    def __init__(self, vocab, dim, n_layers, n_heads, ffn_hidden,
                 eps=1e-5, theta=10000.0, pre_norm=True):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [LlamaBlock(dim, n_heads, ffn_hidden, eps, theta, pre_norm) for _ in range(n_layers)])
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


# ==================================================================
#  B. 与 HF LlamaForCausalLM 逐层前向数值对齐
# ==================================================================
def build_hf_tiny_llama(vocab, dim, n_layers, n_heads, ffn_hidden, eps, theta, seed):
    """离线构造一个随机初始化的 tiny LlamaForCausalLM（只给 config，不下载任何权重）。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    set_all_seeds(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=dim, intermediate_size=ffn_hidden,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_heads,                # MHA：KV 头数 == Q 头数（GQA 见 04）
        max_position_embeddings=max(64, 4 * dim), rms_norm_eps=eps,
        rope_theta=theta, attention_bias=False, mlp_bias=False,
        tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


def copy_hf_into_hand(hf, hand):
    """把 HF 的权重逐张量灌进手写模型（同命名），保证两边前向用的是同一套参数。"""
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


def layerwise_align(vocab, dim, n_layers, n_heads, ffn_hidden, eps, theta, seq, batch):
    """同权重下，比对手写模型与 HF 的每层 hidden_states 与最终 logits 的最大绝对误差。"""
    hf, _ = build_hf_tiny_llama(vocab, dim, n_layers, n_heads, ffn_hidden, eps, theta, SEED)
    set_all_seeds(SEED + 7)
    hand = TinyLlama(vocab, dim, n_layers, n_heads, ffn_hidden, eps, theta, pre_norm=True).eval()
    copy_hf_into_hand(hf, hand)

    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, vocab, (batch, seq), generator=g)
    with torch.no_grad():
        hf_out = hf(ids, output_hidden_states=True)
        hand_logits, hand_hidden = hand(ids, collect_hidden=True)
    diffs = []
    # HF 的 hidden_states[0] 是 embedding 输出，[1..n-1] 是各层"未过末端 norm"的输出，
    # 而 hidden_states[-1] 是过了末端 RMSNorm 的最终隐状态——比对时要对齐这个细节。
    for i in range(n_layers):
        d = (hf_out.hidden_states[i] - hand_hidden[i]).abs().max().item()
        diffs.append(('embed' if i == 0 else 'layer%d' % (i - 1), d))
    with torch.no_grad():
        hand_final_normed = hand.norm(hand_hidden[n_layers])
    d_last = (hf_out.hidden_states[n_layers] - hand_final_normed).abs().max().item()
    diffs.append(('layer%d+norm' % (n_layers - 1), d_last))
    # 反面教材（可复现正文 3.2 的踩坑）：若漏掉末端 norm、直接拿"未过 norm"的最后一层
    # 去比 HF 的 hidden_states[-1]，误差会大好几个数量级。读者可对照这一行理解那个 2.8 从哪来。
    d_last_wrong = (hf_out.hidden_states[n_layers] - hand_hidden[n_layers]).abs().max().item()
    logit_diff = (hf_out.logits - hand_logits).abs().max().item()
    return diffs, logit_diff, ids, d_last_wrong


# ==================================================================
#  C. Pre-Norm vs Post-Norm 训练稳定性实验
# ==================================================================
def make_char_corpus(rng, n=600, seq_len=48, alpha=24):
    """合成一份"有可学结构"的字符序列：用几条带依赖的模式拼接，让 LM 有信号可学。
       返回 (data tensor [N, seq_len], vocab)。纯合成、确定性、不联网。"""
    # 模式：形如 A B A B ... / 递增计数 / 固定短语循环，制造可预测的下一步
    vocab = alpha + 2                      # 0..alpha-1 为符号，alpha=pad，alpha+1=bos
    seqs = []
    for _ in range(n):
        base = [rng.randrange(alpha) for _ in range(rng.randint(2, 5))]
        s = []
        while len(s) < seq_len:
            s.extend(base)
            if rng.random() < 0.3:
                base = [rng.randrange(alpha) for _ in range(rng.randint(2, 5))]
        seqs.append(s[:seq_len])
    data = torch.tensor(seqs, dtype=torch.long)
    return data, vocab


def train_stability(pre_norm, data, vocab, dim=64, n_layers=8, n_heads=4,
                    lr=1e-2, steps=120, warmup=0, seed=SEED):
    """同一份数据、同一套超参、同一随机种子，分别训 Pre-Norm / Post-Norm 的字符 LM。
       无 warmup + 相对高的 lr + 较深的层数，是 Post-Norm 训练失稳的经典配方。
       warmup>0 时，前 warmup 步把学习率从 ~0 线性升到 lr（Post-Norm 的救命稻草）。
       返回每步 loss、每步梯度范数、是否出现 NaN/Inf。"""
    set_all_seeds(seed)
    ffn = int(dim * 8 / 3 / n_heads) * n_heads      # Llama 风格：约 8/3·dim，取整到头数倍数
    model = TinyLlama(vocab, dim, n_layers, n_heads, ffn, pre_norm=pre_norm)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = data.shape[0]
    losses, gnorms = [], []
    nan_hit = False
    for step in range(steps):
        if warmup > 0:                              # 线性 warmup：lr 从 0 爬升
            cur_lr = lr * min(1.0, (step + 1) / warmup)
            for g in opt.param_groups:
                g['lr'] = cur_lr
        idx = torch.randint(0, n, (32,), generator=torch.Generator().manual_seed(seed + step))
        xb, yb = data[idx], torch.roll(data[idx], -1, dims=1)
        yb[:, -1] = vocab - 1                        # 末位目标置为 pad，避免跨样本
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, vocab), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()  # 不裁剪，只测量
        opt.step()
        lv = loss.item()
        if not math.isfinite(lv) or not math.isfinite(gn):
            nan_hit = True
        losses.append(lv)
        gnorms.append(gn)
    return dict(losses=losses, gnorms=gnorms, nan=nan_hit,
                first=losses[0], last=losses[-1],
                peak_gnorm=max((g for g in gnorms if math.isfinite(g)), default=float('nan')))


# ==================================================================
#  D. 参数量与显存分解
# ==================================================================
def param_breakdown(model):
    """按子层统计参数量，看清"钱花在哪儿"：注意力 vs MLP vs embedding/head vs norm。"""
    groups = {'embed_tokens': 0, 'lm_head': 0, 'attention': 0, 'mlp': 0, 'norm': 0}
    for name, p in model.named_parameters():
        n = p.numel()
        if 'embed_tokens' in name:
            groups['embed_tokens'] += n
        elif 'lm_head' in name:
            groups['lm_head'] += n
        elif 'self_attn' in name:
            groups['attention'] += n
        elif 'mlp' in name:
            groups['mlp'] += n
        elif 'norm' in name or 'layernorm' in name:
            groups['norm'] += n
        else:
            raise ValueError('未归类的参数：%s（请补充 param_breakdown 的分组规则）' % name)
    total = sum(groups.values())
    return groups, total


# ==================================================================
#  E. 线上问题 demo：维度不匹配
# ==================================================================
def demo_dim_mismatch(cases):
    """现象：hidden 不能被头数整除，构造即报错；定位：head_dim = dim/n_heads 非整数；
       解决：保证 hidden % n_heads == 0。cases 为 [(dim, n_heads), ...]，返回各是否被拦下。"""
    caught = []
    for dim, n_heads in cases:
        try:
            CausalSelfAttention(dim, n_heads)      # dim 不能被 n_heads 整除
            caught.append((dim, n_heads, False))
        except AssertionError:
            caught.append((dim, n_heads, True))
    return caught


def demo_causal_mask(vocab, dim, n_layers, n_heads, ffn, seq, t, seed):
    """因果掩码的行为验证：只改位置 t 的 token，重跑前向。
       因果掩码下位置 i 只能看 j<=i，所以位置 <t 的 logits 完全不变、位置 >=t 的 logits 改变。
       与基础篇"无掩码、扰动一处全序列都变"正好对照。返回 (before_maxdiff, after_maxdiff)。"""
    set_all_seeds(seed)
    model = TinyLlama(vocab, dim, n_layers, n_heads, ffn, pre_norm=True).eval()
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (1, seq), generator=g)
    ids2 = ids.clone()
    ids2[0, t] = (ids2[0, t] + 1) % vocab          # 只动位置 t 的 token
    with torch.no_grad():
        l1, l2 = model(ids), model(ids2)
    before = (l1[0, :t] - l2[0, :t]).abs().max().item() if t > 0 else 0.0
    after = (l1[0, t:] - l2[0, t:]).abs().max().item()
    return before, after


def demo_rope_relative(head_dim=16, theta=10000.0, seed=SEED):
    """RoPE 相对位置不变性：固定 q、k，把两者的位置同时平移 Δ，点积应保持不变。
       返回 (原始点积, 平移后点积)，二者应在浮点误差内相等。"""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(head_dim, generator=g)
    k = torch.randn(head_dim, generator=g)
    m, n, shift = 3, 7, 5
    base = rope_dot(q, k, m, n, head_dim, theta)
    moved = rope_dot(q, k, m + shift, n + shift, head_dim, theta)
    return base, moved


# ---------------------------------------------------------------- 配图
def draw(pre, post, post_warm, diffs, logit_diff, groups):
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

    steps = range(1, len(pre['losses']) + 1)
    axes[0].plot(steps, pre['losses'], '-', color='#4a9d6a', label='Pre-Norm')
    axes[0].plot(steps, post['losses'], '-', color='#d08c4a', label='Post-Norm')
    axes[0].plot(steps, post_warm['losses'], '--', color='#cc5a5a', label='Post-Norm+warmup')
    axes[0].set_yscale('log')
    axes[0].set_title('训练 loss（无 warmup + 高 lr + 8 层）')
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('loss (log)')
    axes[0].legend(fontsize=9)

    axes[1].plot(steps, pre['gnorms'], '-', color='#4a9d6a', label='Pre-Norm')
    axes[1].plot(steps, post['gnorms'], '-', color='#d08c4a', label='Post-Norm')
    axes[1].plot(steps, post_warm['gnorms'], '--', color='#cc5a5a', label='Post-Norm+warmup')
    axes[1].set_yscale('log')
    axes[1].set_title('梯度范数（Post-Norm 易爆）')
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('grad norm (log)')
    axes[1].legend(fontsize=9)

    labels = [d[0] for d in diffs] + ['logits']
    vals = [d[1] for d in diffs] + [logit_diff]
    axes[2].bar(range(len(vals)), vals, color='#8888cc')
    axes[2].set_yscale('log')
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    axes[2].axhline(1e-4, color='#cc5a5a', ls='--', lw=1, label='阈值 1e-4')
    axes[2].set_title('手写 vs HF 逐层 max|Δ|')
    axes[2].set_ylabel('最大绝对误差 (log)')
    axes[2].legend(fontsize=8)

    keys = list(groups.keys())
    axes[3].barh(keys, [groups[k] for k in keys], color='#4a9d6a')
    axes[3].set_title('参数量分解（tiny Llama）')
    axes[3].set_xlabel('参数量')
    axes[3].invert_yaxis()

    path = os.path.join(OUT, 'fig-03-transformer.png')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    set_all_seeds(SEED)

    # ---------------- A/B. 手写 block 与 HF 逐层对齐
    ALIGN = dict(vocab=64, dim=32, n_layers=3, n_heads=4, ffn_hidden=64,
                 eps=1e-5, theta=10000.0, seq=12, batch=2)
    diffs, logit_diff, ids, wrong_last_diff = layerwise_align(**ALIGN)
    max_diff = max(d for _, d in diffs + [('logits', logit_diff)])
    log_verify('手写现代 Llama block（RMSNorm+RoPE+SwiGLU+因果MHA+Pre-Norm）', '真跑验证',
               'torch 前向，逐组件手写')
    log_verify('与 HF LlamaForCausalLM 逐层数值对齐', '真跑验证',
               '离线随机 config（不下载权重），同权重比对每层 hidden_states + logits')

    # ---------------- RoPE 相对位置不变性 + 因果掩码行为
    rope_base, rope_moved = demo_rope_relative()
    log_verify('RoPE 相对位置不变性', '真跑验证',
               'q/k 位置同时平移，点积不变（只依赖相对位置）')
    cm_before, cm_after = demo_causal_mask(vocab=64, dim=32, n_layers=2, n_heads=4,
                                           ffn=64, seq=12, t=5, seed=SEED)
    log_verify('因果掩码行为（不偷看未来）', '真跑验证',
               '改位置 t 的 token，logits[:t] 不变、logits[t:] 改变')

    # ---------------- C. Pre-Norm vs Post-Norm 训练稳定性
    rng = random.Random(SEED)
    data, vocab = make_char_corpus(rng)
    t0 = time.perf_counter()
    pre = train_stability(True, data, vocab, lr=1e-2, steps=120)
    post = train_stability(False, data, vocab, lr=1e-2, steps=120)
    post_warm = train_stability(False, data, vocab, lr=1e-2, steps=120, warmup=30)
    train_secs = time.perf_counter() - t0
    log_verify('Pre-Norm vs Post-Norm 训练稳定性实验', '真跑验证',
               '同语料同超参，CPU 真训 %d 步 ×3 臂（含 Post-Norm+warmup），记录 loss / 梯度范数' % 120)

    # ---------------- D. 参数量分解
    set_all_seeds(SEED)
    demo_model = TinyLlama(vocab, 64, 8, 4, int(64 * 8 / 3 / 4) * 4, pre_norm=True)
    groups, total = param_breakdown(demo_model)

    # ---------------- E. 维度不匹配 demo（两种非整除场景）
    dim_cases = demo_dim_mismatch([(65, 4), (64, 3)])   # 65%4!=0；64 是 2 的幂但 64%3!=0
    dim_all_caught = all(c for _, _, c in dim_cases)
    log_verify('线上问题 demo（维度不匹配拦截）', '真跑验证',
               'hidden 不能被头数整除时构造即断言失败（65%4、64%3 两例）')

    # ---------------- 报表
    W = 74
    lines = []
    lines.append('=' * W)
    lines.append('生产篇 03 · Transformer 报表（seed=%d）' % SEED)
    lines.append('=' * W)

    lines.append('\n【A/B】手写现代 Llama block 与 HF LlamaForCausalLM 逐层数值对齐')
    lines.append('    对齐配置：vocab=%d dim=%d layers=%d heads=%d ffn=%d seq=%d batch=%d'
                 % (ALIGN['vocab'], ALIGN['dim'], ALIGN['n_layers'], ALIGN['n_heads'],
                    ALIGN['ffn_hidden'], ALIGN['seq'], ALIGN['batch']))
    lines.append('    同权重前向，逐层 hidden_states / 最终 logits 的最大绝对误差：')
    for name, d in diffs:
        lines.append('      %-8s max|Δ| = %.3e' % (name, d))
    lines.append('      %-8s max|Δ| = %.3e' % ('logits', logit_diff))
    lines.append('    → 全层最大误差 %.3e，阈值 1e-4 内即判等价（实测：%s）'
                 % (max_diff, '通过' if max_diff <= 1e-4 else '超阈值'))
    lines.append('    反面教材：若漏掉末端 norm、直接拿未过 norm 的最后一层比 hidden_states[-1]，')
    lines.append('               max|Δ| = %.3e（大好几个数量级）——这正是正文 3.2 那个"2.8 级"踩坑的来源。'
                 % wrong_last_diff)

    lines.append('\n【C】Pre-Norm vs Post-Norm 训练稳定性（8 层 / dim64 / lr=1e-2 / 120 步）')
    lines.append('    数据：合成字符序列 %d 条 × %d（vocab=%d，带可学依赖，确定性生成）'
                 % (data.shape[0], data.shape[1], vocab))
    lines.append('    %-18s %-12s %-12s %-14s %-8s'
                 % ('方案', '初始loss', '末尾loss', '峰值梯度范数', 'NaN/Inf'))
    for name, r in (('Pre-Norm', pre), ('Post-Norm', post), ('Post-Norm+warmup', post_warm)):
        last = r['last'] if math.isfinite(r['last']) else float('nan')
        peak = r['peak_gnorm'] if math.isfinite(r['peak_gnorm']) else float('nan')
        lines.append('    %-18s %-12.4f %-12.4f %-14.3e %-8s'
                     % (name, r['first'], last, peak, '是' if r['nan'] else '否'))
    lines.append('    → 结论：Pre-Norm 平稳收敛；Post-Norm 在无 warmup + 高 lr 下梯度范数飙升 / loss 更高；')
    lines.append('             给 Post-Norm 加 warmup（前 30 步线性升 lr）后明显回稳——说明 Post-Norm 不是不能训，')
    lines.append('             而是需要配套设计（warmup / 门控残差，见文章第四节"延伸"）。')
    lines.append('    （本机 CPU 训练 %.1f 秒；loss/梯度为实测，量级随 lr/深度而变，趋势稳定）' % train_secs)

    lines.append('\n【D】参数量分解（tiny Llama：vocab=%d dim=64 layers=8 heads=4）' % vocab)
    for k in ('embed_tokens', 'lm_head', 'attention', 'mlp', 'norm'):
        lines.append('    %-14s %8d  (%.1f%%)' % (k, groups[k], 100.0 * groups[k] / total))
    lines.append('    %-14s %8d' % ('合计', total))
    lines.append('    → MLP(SwiGLU) 通常吃掉最大头；embedding+lm_head 随 vocab 线性膨胀（见生产篇 02）')
    lines.append('    → 本表是 MHA（KV 头数=Q 头数）；若换 GQA（如 KV 头数=Q 的 1/4），KV 投影瘦身到 1/4，')
    lines.append('       注意力总参数从 4·d² 降到约 2.5·d²（GQA 细节见生产篇 04）')

    lines.append('\n【E】核心性质 + 线上问题清单 demo')
    lines.append('    RoPE 相对位置不变性：q/k 位置 (3,7) 点积 = %.6f，同时平移 +5 到 (8,12) 点积 = %.6f'
                 % (rope_base, rope_moved))
    lines.append('      → 二者之差 %.2e（应 ≈ 0）：点积只依赖相对位置 m-n，整体平移不变'
                 % abs(rope_base - rope_moved))
    lines.append('    因果掩码行为：只改位置 t=5 的 token → logits[:5] max|Δ|=%.2e（应=0，看不到未来）'
                 % cm_before)
    lines.append('                   logits[5:] max|Δ|=%.3e（应>0，位置>=t 都受影响）' % cm_after)
    lines.append('    维度不匹配：%s'
                 % '；'.join('hidden=%d heads=%d 拦下=%s' % (d, h, c) for d, h, c in dim_cases))
    lines.append('    梯度爆炸：见【C】Post-Norm 峰值梯度范数（同超参下 Pre-Norm 低若干数量级）')
    lines.append('    收敛慢：Post-Norm 需 warmup 才稳（见【C】第三臂）；Pre-Norm 无 warmup 也能起')

    lines.append('')
    lines.append('=' * W)
    lines.append('【验证状态清单】哪些是本机真跑、哪些是未验证示例')
    lines.append('=' * W)
    for item, status, note in VERIFY_LOG:
        lines.append('    [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))
    lines.append('    [未验证] flash-attn / vLLM / Megatron 等重依赖 —— 本机不安装（GPU-only 或需下载），')
    lines.append('               方向性说明 + 调用骨架见文章第四节，未在本机跑通')

    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    with open(os.path.join(OUT, 'align_diff.tsv'), 'w', encoding='utf-8') as f:
        f.write('layer\tmax_abs_diff\n')
        for name, d in diffs + [('logits', logit_diff)]:
            f.write('%s\t%.6e\n' % (name, d))
    with open(os.path.join(OUT, 'train_curves.json'), 'w', encoding='utf-8') as f:
        json.dump({'pre_norm': pre['losses'], 'post_norm': post['losses'],
                   'post_norm_warmup': post_warm['losses'],
                   'pre_gnorm': pre['gnorms'], 'post_gnorm': post['gnorms'],
                   'post_warmup_gnorm': post_warm['gnorms']}, f)

    fig = draw(pre, post, post_warm, diffs, logit_diff, groups)
    if fig:
        print('\n[配图] %s' % fig)

    # ---------------- 自校验
    print('\n' + '=' * W)
    print('【自校验】')
    print('=' * W)
    assert max_diff <= 1e-4, '逐层数值对齐超阈值 max|Δ|=%.3e' % max_diff
    print('  1. 手写 vs HF LlamaForCausalLM 逐层 max|Δ| = %.3e ≤ 1e-4  通过' % max_diff)
    assert logit_diff <= 1e-4, '最终 logits 对齐超阈值 %.3e' % logit_diff
    print('  2. 最终 logits max|Δ| = %.3e ≤ 1e-4  通过' % logit_diff)
    assert wrong_last_diff > 100 * max(d for _, d in diffs), \
        '漏掉末端 norm 的误差应远大于正确对齐（否则踩坑演示无意义）'
    print('  3. 漏掉末端 norm 的错误对齐 max|Δ| = %.3e（远大于正确的 %.3e）→ 复现踩坑  通过'
          % (wrong_last_diff, max(d for _, d in diffs)))
    assert abs(rope_base - rope_moved) < 1e-4, \
        'RoPE 相对位置不变性被破坏：%.6f vs %.6f' % (rope_base, rope_moved)
    print('  4. RoPE 相对位置不变性：位置 (3,7) 与平移后 (8,12) 点积差 %.2e < 1e-4  通过'
          % abs(rope_base - rope_moved))
    assert cm_before < 1e-6, '因果掩码失效：位置 <t 的 logits 不应改变（%.2e）' % cm_before
    assert cm_after > 1e-3, '因果掩码失效：位置 >=t 的 logits 应改变（%.3e）' % cm_after
    print('  5. 因果掩码：改 token[t] 后 logits[:t] max|Δ|=%.2e(=0)、logits[t:] max|Δ|=%.3e(>0)  通过'
          % (cm_before, cm_after))
    assert math.isfinite(pre['last']) and pre['last'] < pre['first'], \
        'Pre-Norm 应收敛（末尾 loss 有限且低于初始）'
    print('  6. Pre-Norm 收敛：loss %.4f → %.4f（有限且下降）  通过' % (pre['first'], pre['last']))
    post_bad = post['nan'] or (not math.isfinite(post['last'])) or (post['last'] > pre['last'])
    assert post_bad, 'Post-Norm 应在无 warmup+高 lr 下比 Pre-Norm 更差（NaN/发散/更高 loss）'
    print('  7. Post-Norm 失稳：NaN=%s，末尾 loss=%s，劣于 Pre-Norm  通过'
          % (post['nan'], ('%.4f' % post['last']) if math.isfinite(post['last']) else 'NaN/Inf'))
    assert post['peak_gnorm'] > pre['peak_gnorm'], 'Post-Norm 峰值梯度范数应高于 Pre-Norm'
    print('  8. 峰值梯度范数 Post-Norm(%.2e) > Pre-Norm(%.2e)  通过'
          % (post['peak_gnorm'], pre['peak_gnorm']))
    assert math.isfinite(post_warm['last']) and post_warm['last'] < post['last'], \
        'Post-Norm+warmup 应比无 warmup 的 Post-Norm 收敛更好'
    print('  9. Post-Norm+warmup 回稳：末尾 loss %.4f < 无 warmup 的 %.4f  通过'
          % (post_warm['last'], post['last']))
    assert dim_all_caught, '维度不匹配应全部被断言拦下'
    print('  10. 维度不匹配（65%4、64%3）均被断言拦下  通过')
    assert total == sum(p.numel() for p in demo_model.parameters()), '参数分解求和应等于总参数量'
    print('  11. 参数量分解求和 == 模型总参数量（%d）  通过' % total)
    print('\n  全部断言通过。')
    print('\n[产物] 报表 / 对齐 diff 表 / 训练曲线 / 配图 → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
