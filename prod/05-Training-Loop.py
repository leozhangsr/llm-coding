#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产篇 05 · 训练循环：手写生产级 AdamW + warmup + cosine，与 HF Trainer 真跑逐 step 对账

对应文章：同目录 05-Training-Loop.md（建议先读 基础篇/05-Training-Loop.py 与 .md）
运行：    python3 05-Training-Loop.py
依赖：    numpy + torch（CPU）+ transformers + accelerate（离线用 LlamaConfig 随机初始化一个
          tiny Llama，不下载任何预训练权重、不联网；accelerate 用于真跑 HF Trainer 做对照，
          缺失则自动降级为"调用骨架 + 标未验证"）。DeepSpeed / FSDP / Megatron 等分布式重依赖
          不安装，归为"方向性说明 + 调用骨架"，运行结束在报表里逐条标注真跑 / 未验证。
产物：    ./output/生产篇05-Training-Loop/ 下的报表、spike/裁剪对照表、断点续训账、配图

基础篇 05 把"一步优化器更新到底在算什么"和 bf16/fp16 的位分配讲透、demo 跑通；生产篇换一副
眼镜——把训练循环当成**一条会被脏数据、显存上限和机器重启反复打断的流水线**来重做。
全文闭环七件事：

    A. 生产训练循环骨架 + loss spike 五臂（裁剪 × 注入坏 batch × 有限性跳步）
    B. 梯度累积：K=4 累积 vs 一次 4×batch 大步，逐参数对账 + 显存峰值账
    C. 梯度检查点：手写 ManualCheckpoint vs torch.utils.checkpoint，前向次数整数账 + saved 张量元素账
    D. 混合精度：bf16 autocast 真跑，fp16 上溢/下溢反面教材（为什么生产用 bf16 而不是 fp16）
    E. 断点续训：逐位复现 + 三个反面教材（不存 optimizer state / 不存 step / 不存数据流位置）
    F. 训练 OOM 显存账本：tiny 实测 + meta device 真数 7B 结构参数量做外推
    G. 三大线上问题清单 + HF Trainer 真跑逐 step 对账（decay 掩码 / lr 调度 / grad_norm 三项硬断言）

哪些是本机真跑验证的、哪些只是"工业库/框架调用示例未验证"，运行结束会在报表里逐条标明。
"""

import copy
import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '生产篇05-Training-Loop')

VERIFY_LOG = []
W = 74

# ---------------- 本章锁定配置（全部确定性，改这里就能复现实验）----------------
DIM = 64
N_LAYERS = 4
N_HEADS = 4
FFN = 168                    # Llama 风格约 8/3·dim，取整到头数倍数（与生产篇 03 同式）
SEQ = 48
BATCH = 32
STEPS = 80
WARMUP = 10
LR = 3e-3
MIN_LR_RATE = 0.1            # eta_min = LR × 0.1
WD = 0.1
CLIP = 1.0
ACCUM_K = 4
SPIKE_STEP = 40              # 0-based；报表里按 1-based 展示为第 41 步
SPIKE_GAIN = 50.0
RESUME_AT = 30
ALPHA = 24                   # 合成语料的符号数 → vocab = ALPHA + 2 = 26
PROBE_PARAM = 'layers.0.self_attn.q_proj.weight'   # 尺度不变性逐元素拆解用的探针参数
LR_SCALE_K = ACCUM_K         # lr 缩放账里大臂的 effective batch 倍数
LR_SCALE_STEPS = 20          # 小臂走的优化器步数；大臂 = LR_SCALE_STEPS / LR_SCALE_K 步，消费同一批数据

# 7B 结构口径的 config（Llama-2-7B 的结构参数），用于 meta device 真数参数量
BIG_CFG = dict(vocab_size=32000, hidden_size=4096, intermediate_size=11008,
               num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=32)


def log_verify(item, status, note=''):
    VERIFY_LOG.append((item, status, note))
    print('  [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ==================================================================
#  复用组件（与生产篇 03/04 同式：RMSNorm / RoPE / 因果 MHA / SwiGLU / Pre-Norm block）
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


class CausalSelfAttention(nn.Module):
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
        causal = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal, float('-inf'))
        w = torch.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        ctx = (w @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(ctx)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    """Pre-Norm 解码器块（03 篇已用三臂实验证明它比 Post-Norm 稳）。本章只保留这一支。"""

    def __init__(self, dim, n_heads, ffn_hidden, eps=1e-5, theta=10000.0):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps)
        self.post_attention_layernorm = RMSNorm(dim, eps)
        self.self_attn = CausalSelfAttention(dim, n_heads, theta)
        self.mlp = SwiGLU(dim, ffn_hidden)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TinyLlama(nn.Module):
    def __init__(self, vocab, dim, n_layers, n_heads, ffn_hidden, eps=1e-5, theta=10000.0):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            [LlamaBlock(dim, n_heads, ffn_hidden, eps, theta) for _ in range(n_layers)])
        self.norm = RMSNorm(dim, eps)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.dim, self.n_heads, self.theta = dim, n_heads, theta

    def forward(self, ids):
        B, S = ids.shape
        cos, sin = build_rope_cache(S, self.dim // self.n_heads, self.theta, ids.device)
        h = self.embed_tokens(ids)
        for blk in self.layers:
            h = blk(h, cos, sin)
        return self.lm_head(self.norm(h))


def build_hf_tiny_llama(vocab, dim, n_layers, n_heads, ffn_hidden, eps, theta, seed):
    """离线构造一个随机初始化的 tiny LlamaForCausalLM（只给 config，不下载任何权重）。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    set_all_seeds(seed)
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=dim, intermediate_size=ffn_hidden,
        num_hidden_layers=n_layers, num_attention_heads=n_heads,
        num_key_value_heads=n_heads, max_position_embeddings=max(64, 4 * SEQ),
        rms_norm_eps=eps, rope_theta=theta, attention_bias=False, mlp_bias=False,
        tie_word_embeddings=False)
    return LlamaForCausalLM(cfg)


def copy_hf_into_hand(hf, hand):
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


def make_char_corpus(rng, n, seq_len=SEQ, alpha=ALPHA):
    """合成一份"有可学结构"的字符序列（与生产篇 03 同式）：纯合成、确定性、不联网。"""
    vocab = alpha + 2
    seqs = []
    for _ in range(n):
        base = [rng.randrange(alpha) for _ in range(rng.randint(2, 5))]
        s = []
        while len(s) < seq_len:
            s.extend(base)
            if rng.random() < 0.3:
                base = [rng.randrange(alpha) for _ in range(rng.randint(2, 5))]
        seqs.append(s[:seq_len])
    return torch.tensor(seqs, dtype=torch.long), vocab


# ==================================================================
#  A. 生产训练循环骨架：调度 / decay 掩码 / 裁剪 / spike / 有限性跳步
# ==================================================================
def lr_at(t, warmup, total, peak, eta_min):
    """线性 warmup + cosine 退火，口径与 HF get_cosine_with_min_lr_schedule_with_warmup 一致：
       t 是 0-based 的优化器步序号；warmup 段 lr = peak·t/warmup（t=0 时 lr=0，第 0 步是空步）；
       退火段 lr = eta_min + (peak-eta_min)·0.5·(1+cos(π·(t-warmup)/(total-warmup)))。"""
    if warmup > 0 and t < warmup:
        return peak * t / warmup
    prog = (t - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return eta_min + 0.5 * (peak - eta_min) * (1.0 + math.cos(math.pi * prog))


def wsd_lr_at(t, warmup, stable_end, decay_end, peak, eta_min):
    """WSD（Warmup–Stable–Decay）：线性 warmup 到 peak，稳定段恒定 peak，退火段 cosine 降到 eta_min。
       与 HF transformers 的 get_wsd_schedule(decay_type='cosine') 同口径（脚本里逐 step 对账）。
       与 cosine 的关键区别：稳定段的那一段 lr 与"总共要训多少步"无关。"""
    if warmup > 0 and t < warmup:
        return peak * t / warmup
    if t < stable_end:
        return peak
    if t >= decay_end:
        return eta_min
    prog = (t - stable_end) / max(1, decay_end - stable_end)
    return eta_min + 0.5 * (peak - eta_min) * (1.0 + math.cos(math.pi * prog))


def wsd_resume_ledger(warmup, peak, eta_min):
    """续训语义账：在同一条 lr 曲线上取"已走过的历史步"和"未来步"，然后只改一个变量——
       总步数（cosine 的 total / WSD 的 decay_end）——看哪些步的 lr 真的变了。
       分三种续训点：WSD 的稳定段内、WSD 的退火段内、cosine 的任意点。"""
    T1, T2 = 100, 160            # 原计划总步数 vs 续训时延长后的总步数
    stable_end, decay_end = 80, T1
    out = {}
    for kind in ('cosine', 'wsd_stable', 'wsd_decay'):
        if kind == 'cosine':
            resume = 50
            f1 = lambda t: lr_at(t, warmup, T1, peak, eta_min)
            f2 = lambda t: lr_at(t, warmup, T2, peak, eta_min)
        elif kind == 'wsd_stable':
            resume = 50
            f1 = lambda t: wsd_lr_at(t, warmup, stable_end, decay_end, peak, eta_min)
            f2 = lambda t: wsd_lr_at(t, warmup, T2 - 20, T2, peak, eta_min)
        else:
            resume = 90
            f1 = lambda t: wsd_lr_at(t, warmup, stable_end, decay_end, peak, eta_min)
            f2 = lambda t: wsd_lr_at(t, warmup, stable_end, T2, peak, eta_min)
        past = [abs(f2(t) - f1(t)) for t in range(0, resume)]
        fut = [(f2(t) - f1(t), t) for t in range(resume, T1)]
        changed = [d for d, _ in fut if abs(d) > 0.0]
        out[kind] = dict(
            resume=resume,
            past_steps=len(past),
            past_changed=sum(1 for d in past if abs(d) > 0.0),
            past_max_diff=max(past) if past else 0.0,
            fut_steps=len(fut),
            fut_changed=len(changed),
            fut_max_diff=max((abs(d) for d, _ in fut), default=0.0),
            fut_rel_max=max((abs(d) / max(f1(t), 1e-30) for d, t in fut), default=0.0),
            lr_at_resume_1=f1(resume), lr_at_resume_2=f2(resume))
    return out


def wsd_vs_hf(n_steps, warmup, peak, min_lr_rate):
    """手写 WSD 与 HF transformers 的 get_wsd_schedule(decay_type='cosine') 逐 step 对账。

       读出来的口径差（本机实测）：HF 的 lambda 是 `factor·(1-min_lr_ratio)+min_lr_ratio`，
       连 warmup 段也被抬了同一个下限，所以它的第 0 步 lr = min_lr_ratio·peak 而不是 0；
       稳定段与退火段两边逐位一致。这不是 WSD 的定义差异，是 HF 这一个实现的细节——
       同库的 cosine_with_min_lr 的 warmup 段就没有这个下限（本文与 Trainer 的对账里
       lr(0)=0 是精确成立的）。所以这里同时报"全步 max|Δ|"和"warmup 之后 max|Δ|"。"""
    from transformers.optimization import get_wsd_schedule
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=peak)
    n_stable = (n_steps - warmup) - 20            # 退火段固定 20 步
    sch = get_wsd_schedule(opt, num_warmup_steps=warmup, num_decay_steps=20,
                           num_stable_steps=n_stable, min_lr_ratio=min_lr_rate)
    eta_min = peak * min_lr_rate
    diff_all = diff_after = 0.0
    seq = []
    for t in range(n_steps):
        hf_lr = opt.param_groups[0]['lr']
        seq.append(hf_lr)
        d = abs(hf_lr - wsd_lr_at(t, warmup, warmup + n_stable, n_steps, peak, eta_min))
        diff_all = max(diff_all, d)
        if t >= warmup:
            diff_after = max(diff_after, d)
        opt.step()                                # 与 Trainer 一致的调用顺序：先 opt 后 sched
        sch.step()
    return dict(max_diff=diff_all, max_diff_from_warmup=diff_after,
                n_steps=n_steps, n_stable=n_stable, warmup_first_lr=seq[0],
                plateau=seq[warmup:warmup + n_stable], peak=peak, last=seq[-1])


DECAY_FORBIDDEN = (r'bias', r'layernorm', r'rmsnorm', r'(?:^|\.)norm(?:$|\.)', r'_norm(?:$|\.)')


def decay_param_names(model):
    """复刻 HF Trainer.get_decay_parameter_names 的掩码规则：LayerNorm 类模块的参数，
       以及名字命中 bias / layernorm / rmsnorm / norm / _norm 的参数，不施加 weight decay。
       规则抄自 transformers/trainer.py 的 forbidden_name_patterns（真跑里会与 Trainer 对账）。"""
    names = []
    for mname, mod in model.named_modules():
        if isinstance(mod, nn.LayerNorm):
            continue
        for pname, _ in mod.named_parameters(recurse=False):
            full = pname if mname == '' else mname + '.' + pname
            if any(re.search(p, full) for p in DECAY_FORBIDDEN):
                continue
            names.append(full)
    return names


def build_optimizer(model, lr, wd):
    """两个 param group：decay 组按 wd，no-decay 组按 0。与 Trainer.create_optimizer 同构。"""
    decay = set(decay_param_names(model))
    decay_ps, nodecay_ps = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay_ps if n in decay else nodecay_ps).append(p)
    return torch.optim.AdamW(
        [dict(params=decay_ps, weight_decay=wd),
         dict(params=nodecay_ps, weight_decay=0.0)],
        lr=lr, betas=(0.9, 0.999), eps=1e-8)


def causal_loss(logits, xb):
    """标准因果 LM 损失：丢掉最后一个位置的 logit，与 HF ForCausalLMLoss 的内部 shift 同口径。"""
    V = logits.shape[-1]
    return F.cross_entropy(logits[:, :-1, :].reshape(-1, V).float(), xb[:, 1:].reshape(-1))


def grad_total_norm(model):
    """只测量不裁剪的梯度总范数（与 clip_grad_norm_ 同式：各参数 L2 范数再取 L2）。"""
    norms = [p.grad.detach().norm(2.0) for p in model.parameters() if p.grad is not None]
    return torch.stack(norms).norm(2.0).item() if norms else 0.0


def _params_with_grad(values):
    """造一批只带梯度、不参与训练的叶子参数，用来单测两种裁剪函数对非有限值的行为。"""
    ps = []
    for v in values:
        p = torch.zeros(len(v), requires_grad=True)
        p.grad = torch.tensor(v, dtype=torch.float32)
        ps.append(p)
    return ps


def clip_strategy_probe(max_norm=1.0):
    """按范数裁剪 vs 按值裁剪：同一份带 inf 的梯度，两种裁剪各自把梯度变成什么。
       按范数裁剪的系数是 max_norm/(total_norm+1e-6)，total_norm=inf 时系数为 0，
       于是 0×inf=NaN —— 裁剪环节自己造出 NaN。按值裁剪是逐元素 clamp 到 ±max_norm，
       inf 被 clamp 成有限值，不乘系数，所以不会造 NaN；代价是梯度方向被改变。"""
    out = {}
    for kind in ('norm', 'value'):
        # 三种坏法：inf、NaN、以及全有限的大梯度（用来看方向是否保持）
        rows = {}
        for case, vecs in (
                ('inf', [[1.0, 2.0, float('inf'), -float('inf')], [0.5, 1.0, 2.0, 4.0]]),
                ('nan', [[1.0, float('nan'), 3.0, 4.0], [0.5, 1.0, 2.0, 4.0]]),
                ('finite', [[3.0, -4.0, 0.2, -0.1], [50.0, -25.0, 1.0, -2.0]])):
            ps = _params_with_grad(vecs)
            flat_before = torch.cat([p.grad for p in ps])
            ret = None
            if kind == 'norm':
                ret = torch.nn.utils.clip_grad_norm_(ps, max_norm).item()
            else:
                torch.nn.utils.clip_grad_value_(ps, max_norm)
            flat_after = torch.cat([p.grad for p in ps])
            finite_now = bool(torch.isfinite(flat_after).all())
            # 方向保持度：裁剪前后两向量的余弦（含 inf/NaN 时不定义，记 None）
            cos = None
            if torch.isfinite(flat_before).all() and finite_now:
                a = flat_before / flat_before.norm()
                b = flat_after / flat_after.norm()
                cos = float((a * b).sum())
            rows[case] = dict(
                returned=ret,
                n_nan_before=int(torch.isnan(flat_before).sum()),
                n_inf_before=int(torch.isinf(flat_before).sum()),
                n_nan=int(torch.isnan(flat_after).sum()),
                n_inf=int(torch.isinf(flat_after).sum()),
                all_finite=finite_now,
                max_abs=float(flat_after.abs().max()) if finite_now else None,
                direction_cos=cos)
        out[kind] = rows
    return out


def make_stream(data, n_batches, batch, seed):
    """用一条持久 generator 生成乱序 batch 流，并记下每步之前的 generator 状态。
       这样"断点续训必须保存数据流位置"才有可复现的对照物（真实工程里它就是 dataloader 进度）。"""
    g = torch.Generator().manual_seed(seed)
    n = data.shape[0]
    batches, states = [], []
    for _ in range(n_batches):
        states.append(g.get_state().clone())
        idx = torch.randperm(n, generator=g)[:batch]
        batches.append(data[idx])
    return batches, states


def train_loop(model, batches, *, total=STEPS, start_step=0, end_step=None, warmup=WARMUP,
               peak_lr=LR, min_lr_rate=MIN_LR_RATE, wd=WD, clip=CLIP, decay_mask=True,
               forward_fn=None, batch_offset=0, spike_at=None, spike_kind=None,
               spike_gain=SPIKE_GAIN, skip_nonfinite=False, autocast_bf16=False,
               opt_state=None, model_state=None, probe_at=None, probe_param=None,
               schedule='cosine', stable_end=None, accum_k=1):
    """生产训练循环骨架：warmup+cosine 调度、decay 掩码、梯度裁剪、坏 batch 注入、
       有限性跳步、bf16 autocast、从 state 续训。返回逐步 loss / 梯度范数 / lr。

       total / start_step / end_step：调度器按 total 算（总步数），实际只跑 [start_step, end_step)。
                                      断点续训必须沿用原来的 total，否则 cosine 的分母就变了。
       decay_mask=False  把 weight decay 施加到所有参数上（反面教材，用来复现"语义不一致"）
       spike_kind=None     不注入（干净基线）
       spike_kind='gain'   该步 loss × spike_gain —— 模拟脏数据/标签错位造成的异常大梯度
       spike_kind='nan'    该步 loss × inf —— 模拟 fp16 上溢 / 除零造成的非有限梯度
                          （注意不能用 loss + inf：加常数不改梯度，注入不进反向）
       skip_nonfinite=True 梯度总范数非有限时跳过这一步（不 step、清零梯度），生产正解
       probe_at/probe_param 在该步的 opt.step() 前后取一次 AdamW 状态快照（m、v、梯度、参数差），
                          用来把"全局 L2 范数的比值"拆成逐元素分布（2.3 节的尺度不变性）
       schedule='cosine'   线性 warmup + cosine 退火（默认，与全文所有臂同口径）
       schedule='wsd'      线性 warmup + 恒定 + cosine 退火（WSD），需给 stable_end；
                          退火段固定在 [stable_end, total)，所以 stable_end 之前的那段 lr
                          与"总共还要训多少步"无关——这正是 WSD 能随时延长 total 的原因
       schedule='constant' 恒定 lr（= peak_lr，不做 warmup 也不退火），
                          只用于"同样多数据、不同 effective batch"的 lr 缩放对照
       accum_k>1           一个优化器步累积 k 个连续 micro-batch（effective batch = k·B），
                          步数 n 对应消费 batches[0 : n·k]；与 schedule='constant' 搭配做 lr 缩放账
    """
    assert accum_k == 1 or spike_at is None, '累积模式只用于 lr 缩放对照，不支持坏 batch 注入'
    assert accum_k == 1 or forward_fn is None, '累积模式不做 forward_fn 覆写'
    fwd = forward_fn or (lambda m, x: m(x))
    eta_min = peak_lr * min_lr_rate
    end = total if end_step is None else end_step
    if schedule == 'wsd':
        se = stable_end if stable_end is not None else warmup
        def cur_lr_of(t):
            return wsd_lr_at(t, warmup, se, total, peak_lr, eta_min)
    elif schedule == 'constant':
        def cur_lr_of(t):
            return peak_lr
    else:
        def cur_lr_of(t):
            return lr_at(t, warmup, total, peak_lr, eta_min)
    opt = build_optimizer(model, peak_lr, wd) if decay_mask else \
        torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=wd,
                          betas=(0.9, 0.999), eps=1e-8)
    if model_state is not None:
        model.load_state_dict(model_state)
    if opt_state is not None:
        opt.load_state_dict(opt_state)
    rec = dict(losses=[], raw_losses=[], gnorms=[], applied=[], upd=[], lrs=[],
               nan_steps=0, skipped=0, probe=None)
    for t in range(start_step, end):
        cur_lr = cur_lr_of(t)
        for grp in opt.param_groups:
            grp['lr'] = cur_lr
        opt.zero_grad(set_to_none=True)
        spike = spike_at is not None and t == spike_at
        if accum_k == 1:
            xb = batches[t + batch_offset]
            with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast_bf16):
                raw = causal_loss(fwd(model, xb), xb)
            loss = (raw * spike_gain if (spike and spike_kind == 'gain')
                    else raw * float('inf') if spike else raw)
            loss.backward()
            lv = loss.item()
            raw_v = raw.item()
        else:
            # 累积模式：一个优化器步吃 accum_k 个连续 micro-batch（effective batch = accum_k·B）。
            # 各 micro-batch 同大小，所以"逐 micro-batch 求 loss 再平均"= 大 batch 的 loss。
            raw_v = 0.0
            for j in range(accum_k):
                xb = batches[t * accum_k + batch_offset + j]
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast_bf16):
                    l = causal_loss(fwd(model, xb), xb)
                (l / accum_k).backward()
                raw_v += l.detach().item() / accum_k
            lv = raw_v
        if clip is None:
            gn = grad_total_norm(model)
            applied = gn                       # 不裁剪：优化器吃进去的就是它
        else:
            # clip_grad_norm_ 返回的是「裁剪前」的总范数，裁剪后真正被 step 吃进去的范数
            # 要再量一次才准（torch 的系数是 clip/(norm+1e-6)，所以实测会略小于阈值）。
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip).item()
            applied = grad_total_norm(model)
        rec['losses'].append(lv)
        rec['raw_losses'].append(raw_v)
        rec['gnorms'].append(gn)
        rec['applied'].append(applied)
        rec['lrs'].append(cur_lr)
        if not math.isfinite(lv) or not math.isfinite(gn):
            rec['nan_steps'] += 1
        if skip_nonfinite and not math.isfinite(gn):
            rec['skipped'] += 1
            rec['upd'].append(0.0)
            opt.zero_grad(set_to_none=True)
            continue
        before = [p.detach().clone() for p in model.parameters()]
        named = dict(model.named_parameters())
        snap = None
        if probe_at is not None and t == probe_at:
            pp = named[probe_param]
            st = opt.state.get(pp, {})
            assert 'exp_avg' in st, 'probe_at 必须晚于首个 step，否则 AdamW 还没有 m/v 状态'
            snap = dict(grad=pp.grad.detach().float().clone(),
                        m=st['exp_avg'].detach().float().clone(),
                        v=st['exp_avg_sq'].detach().float().clone(),
                        p_before=pp.detach().float().clone())
        opt.step()
        rec['upd'].append(math.sqrt(sum(float(((p.detach() - b) ** 2).sum())
                                        for p, b in zip(model.parameters(), before))))
        if snap is not None:
            snap['p_after'] = named[probe_param].detach().float().clone()
            snap['upd_elem'] = (snap['p_after'] - snap['p_before']).abs()
            snap['upd_by_param'] = {n: (p.detach() - b).norm().item()
                                    for (n, p), b in zip(model.named_parameters(), before)}
            rec['probe'] = snap
    rec['last'] = rec['raw_losses'][-1] if rec['raw_losses'] else float('nan')
    rec['peak_gnorm'] = max((g for g in rec['gnorms'] if math.isfinite(g)), default=float('nan'))
    rec['peak_applied'] = max((g for g in rec['applied'] if math.isfinite(g)), default=float('nan'))
    rec['peak_upd'] = max((u for u in rec['upd'] if math.isfinite(u)), default=float('nan'))
    rec['n_clipped'] = sum(1 for g in rec['gnorms'] if math.isfinite(g) and g > clip) \
        if clip is not None else 0
    rec['_opt'] = opt
    rec['_model'] = model
    return rec


def recovery_steps(rec, spike_at, consec=1):
    """注入之后，raw loss 需要多少步才回到「注入前一步（同一条臂）」的水平；回不去返回 -1。
       从 spike_at+1 开始数：注入步本身的 raw loss 是未被放大的真值，拿它算恢复没有意义。
       consec>1 是更稳健的生产判据：要求连续 consec 步都在参考值以下才算恢复。"""
    ref = rec['raw_losses'][spike_at - 1]
    run = 0
    for i in range(spike_at + 1, len(rec['raw_losses'])):
        v = rec['raw_losses'][i]
        run = run + 1 if (math.isfinite(v) and v <= ref) else 0
        if run == consec:
            return i - consec + 1 - spike_at
    return -1


def bounce_back_steps(rec, spike_at):
    """首次恢复之后又弹回参考值以上的步数（假恢复的度量）。参考值取注入前一步的 raw loss。"""
    ref = rec['raw_losses'][spike_at - 1]
    first = None
    for i in range(spike_at + 1, len(rec['raw_losses'])):
        v = rec['raw_losses'][i]
        if first is None:
            if math.isfinite(v) and v <= ref:
                first = i
        elif math.isfinite(v) and v > ref:
            return i - first
    return 0


def adam_scale_ratios(clean_probe, gain_probe, gain=SPIKE_GAIN, betas=(0.9, 0.999)):
    """把"梯度 ×50、参数更新只放大 1.59×"这一句拆开：逐元素比较两条臂在注入步的
       m、√v 与 |Δp|。两条臂在注入步之前的 m/v 必须完全相同（下面用 m_state_diff 断言），
       这样才是同一起点的干净对照。"""
    b1, b2 = betas
    cold = (1.0 - b1) * gain / math.sqrt((1.0 - b2) * gain * gain)  # m=v=0 的元素的更新放大

    def quant(x):
        return [float(torch.quantile(x, p)) for p in (0.1, 0.5, 0.9)]

    g_c, g_g = clean_probe['grad'], gain_probe['grad']
    live_g = g_c.abs() > g_c.abs().max() * 1e-6          # 排除本来就没梯度的元素（近零除法会主导分布）
    grad_ratio = (g_g.abs() / g_c.abs().clamp_min(1e-30))[live_g]
    m_c = (b1 * clean_probe['m'] + (1.0 - b1) * g_c).abs()
    m_g = (b1 * gain_probe['m'] + (1.0 - b1) * g_g).abs()
    sv_c = (b2 * clean_probe['v'] + (1.0 - b2) * g_c * g_c).clamp_min(0.0).sqrt()
    sv_g = (b2 * gain_probe['v'] + (1.0 - b2) * g_g * g_g).clamp_min(0.0).sqrt()
    u_c, u_g = clean_probe['upd_elem'], gain_probe['upd_elem']
    live_u = u_c > u_c.max() * 1e-6
    r_upd = u_g / u_c.clamp_min(1e-30)
    ru_act, ug_act, uc_act = r_upd[live_u], u_g[live_u], u_c[live_u]
    hi = ru_act > cold                      # 比值超过"冷元素闭式值"的那批元素
    l2_c = sum(v * v for v in clean_probe['upd_by_param'].values())
    l2_g = sum(v * v for v in gain_probe['upd_by_param'].values())
    en_c = float((uc_act ** 2).sum())       # 探针参数内，干净臂的更新平方和（L2 的加权口径）
    en_g = float((ug_act ** 2).sum())
    return dict(
        m_state_diff=max((clean_probe['m'] - gain_probe['m']).abs().max().item(),
                         (clean_probe['v'] - gain_probe['v']).abs().max().item()),
        grad_ratio_min=float(grad_ratio.min()), grad_ratio_max=float(grad_ratio.max()),
        ratio_m=quant(m_g / m_c.clamp_min(1e-30)),
        ratio_sv=quant(sv_g / sv_c.clamp_min(1e-30)),
        ratio_upd_all=quant(r_upd),
        ratio_upd_active=quant(r_upd[live_u]),
        n_active=int(live_u.sum()), n_elem=int(u_c.numel()),
        n_hi_cold=int(hi.sum()),
        share_hi_clean=float((uc_act[hi] ** 2).sum() / en_c),
        share_hi_gain=float((ug_act[hi] ** 2).sum() / en_g),
        l2_ratio_probe=math.sqrt(en_g / en_c),
        cold_ratio=cold,
        l2_ratio=math.sqrt(l2_g / l2_c),
        upd_c=math.sqrt(l2_c), upd_g=math.sqrt(l2_g))


# ==================================================================
#  B. 梯度累积：以时间换显存峰值
# ==================================================================
def accumulation_equivalence(model_factory, micro_batches, big_batch, lr, wd):
    """路径一：K 个 micro-batch 各自 backward（loss/K）后累积，再一次 step。
       路径二：把 K 个 micro-batch 拼成一个大 batch，一次 forward/backward 后 step。
       两条路径起点权重相同，比较梯度与 step 之后的参数差。

       注意口径：逐元素 max|Δ| 会被"梯度接近 0 的那几个元素"主导（绝对误差极小、相对误差极大），
       所以这里同时给绝对量与相对量——相对量才是判断"两条路径是否等价"的正确尺子。"""
    out = {}
    for tag, accum in (('accum', True), ('big', False)):
        model = model_factory()
        opt = build_optimizer(model, lr, wd)
        opt.zero_grad(set_to_none=True)
        if accum:
            for xb in micro_batches:
                (causal_loss(model(xb), xb) / len(micro_batches)).backward()
        else:
            causal_loss(model(big_batch), big_batch).backward()
        grads = {n: p.grad.detach().clone() for n, p in model.named_parameters()}
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        gn = grad_total_norm(model)
        opt.step()
        out[tag] = dict(params={n: p.detach().clone() for n, p in model.named_parameters()},
                        grads=grads, before=before, gnorm=gn)
    a, b = out['accum'], out['big']
    grad_diff = max((a['grads'][n] - b['grads'][n]).abs().max().item() for n in a['grads'])
    total_gnorm = a['gnorm']
    param_diff = max((a['params'][n] - b['params'][n]).abs().max().item() for n in a['params'])
    update_abs = max((a['params'][n] - a['before'][n]).abs().max().item() for n in a['params'])
    param_scale = max(a['params'][n].abs().max().item() for n in a['params'])
    return dict(grad_diff=grad_diff, grad_rel=grad_diff / total_gnorm,
                param_diff=param_diff, param_rel=param_diff / update_abs,
                param_scale=param_scale, update_abs=update_abs,
                gnorm=total_gnorm, gnorm_diff=abs(a['gnorm'] - b['gnorm']))


def count_saved(fn):
    """用 saved_tensors_hooks 数一次前向"为反向保存了多少张量、多少元素"。
       这是激活显存的直接代理量（实测，不是估算）。"""
    stats = {'pack': 0, 'elems': 0}

    def pack(t):
        stats['pack'] += 1
        stats['elems'] += t.numel()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return stats


# ==================================================================
#  C. 梯度检查点：手写 vs 官方
# ==================================================================
class ManualCheckpoint(torch.autograd.Function):
    """手写梯度检查点（reentrant 风格）：前向只保存输入、不保存中间激活；反向时把这一块
       重新算一遍，再在重建的图上 backward，把参数梯度累加进去。
       "以算换存"的本质就体现在 backward 里那次 recompute。"""

    @staticmethod
    def forward(ctx, x, cos, sin, blk, counters):
        ctx.blk = blk
        ctx.cos = cos
        ctx.sin = sin
        ctx.counters = counters
        ctx.save_for_backward(x)
        counters['fwd'] += 1
        with torch.no_grad():
            return blk(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        ctx.counters['recompute'] += 1
        with torch.enable_grad():
            x2 = x.detach().requires_grad_(True)
            out = ctx.blk(x2, ctx.cos, ctx.sin)
        torch.autograd.backward(out, grad_out)      # 参数梯度累加进 blk 的 .grad
        return x2.grad, None, None, None, None


def run_blocks(model, ids, mode, counters, phase):
    """三种跑法走同一条主干：none（全保存）/ manual（手写检查点）/ official（官方检查点）。
       phase['now'] 用来给官方 checkpoint 的内部计数分辨"首次前向"还是"反向重算"。"""
    B, S = ids.shape
    cos, sin = build_rope_cache(S, model.dim // model.n_heads, model.theta, ids.device)
    h = model.embed_tokens(ids)
    for blk in model.layers:
        if mode == 'none':
            counters['fwd'] += 1
            h = blk(h, cos, sin)
        elif mode == 'manual':
            h = ManualCheckpoint.apply(h, cos, sin, blk, counters)
        else:
            def counted(x, c, s, b, ph=phase):
                counters[ph['now']] += 1
                return b(x, c, s)
            h = checkpoint(counted, h, cos, sin, blk, use_reentrant=False)
    return model.lm_head(model.norm(h))


def checkpoint_compare(model_factory, ids):
    """三种模式下：前向/重算次数、为反向保存的元素数、以及参数梯度与 none 模式的差。"""
    rows = {}
    for mode in ('none', 'manual', 'official'):
        model = model_factory()
        counters = {'fwd': 0, 'recompute': 0}
        phase = {'now': 'fwd'}
        saved = {'pack': 0, 'elems': 0}

        def pack(t, s=saved):
            s['pack'] += 1
            s['elems'] += t.numel()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            logits = run_blocks(model, ids, mode, counters, phase)
            loss = causal_loss(logits, ids)
        phase['now'] = 'recompute'
        model.zero_grad(set_to_none=True)
        loss.backward()
        rows[mode] = dict(counters=dict(counters), saved=dict(saved), loss=loss.item(),
                          grads={n: p.grad.detach().clone()
                                 for n, p in model.named_parameters() if p.grad is not None})
    base = rows['none']['grads']
    for mode in ('manual', 'official'):
        rows[mode]['grad_diff'] = max(
            (base[n] - rows[mode]['grads'][n]).abs().max().item() for n in base)
        rows[mode]['loss_diff'] = abs(rows[mode]['loss'] - rows['none']['loss'])
    rows['none']['grad_diff'] = 0.0
    rows['none']['loss_diff'] = 0.0
    return rows


# ==================================================================
#  D. 混合精度：bf16 vs fp16
# ==================================================================
def cast_table(vals):
    """把 fp32 值分别 cast 到 fp16 / bf16，记录上溢、下溢与相对误差。
       对应基础篇 05 的位级结论：fp16 尾数 10 位、指数 5 位、上限 65504、最小正规数约 6.1e-5；
       bf16 尾数 7 位、指数与 fp32 同宽 8 位、上限约 3.39e38。"""
    t = torch.tensor(vals, dtype=torch.float32)
    rows = []
    for v, hv, bv in zip(vals, t.half().tolist(), t.bfloat16().tolist()):
        rows.append(dict(value=v, fp16=hv, bf16=bv,
                         fp16_overflow=(not math.isfinite(hv)),
                         fp16_underflow=(hv == 0.0 and v != 0.0),
                         bf16_overflow=(not math.isfinite(bv)),
                         bf16_relerr=(abs(bv - v) / abs(v) if v else 0.0),
                         fp16_relerr=(abs(hv - v) / abs(v) if (v and math.isfinite(hv)) else float('inf'))))
    return rows


def fp16_subnormal_probe():
    """fp16 下溢侧的精确边界，以及它和 Adam 的 eps 的关系。
       fp16 的最小正规数是 2^-14≈6.104e-05，但真正决定"多小的数会变成 0"的是最小亚正规数
       2^-24≈5.9605e-08：小于它的数直接下溢为 0。AdamW 默认 eps=1e-8 恰好落在这条线以下，
       所以 fp16 里 eps 自己也变成 0——梯度下溢和 eps 消失是同一枚硬币的两面。"""
    finfo = np.finfo(np.float16)
    smallest_sub = float(finfo.smallest_subnormal)
    smallest_normal = float(finfo.tiny)
    eps_default = float(inspect.signature(torch.optim.AdamW.__init__).parameters['eps'].default)

    def h(x):
        return float(np.float16(x))

    # 分母塌缩链：fp16 下 √v̂=0 且 eps 也下溢为 0，分母为 0，m̂/0 在 IEEE 里给 inf
    m16 = np.float16(1e-4)
    den16 = float(np.float16(0.0) + np.float16(eps_default))
    div16 = float(m16 / den16) if den16 else float('inf')
    return dict(
        smallest_subnormal=smallest_sub, smallest_normal=smallest_normal,
        pow2_24=2.0 ** -24, pow2_14=2.0 ** -14,
        fp16_of_eps=h(eps_default), fp16_of_1e8=h(1e-8),
        fp16_of_6e8=h(6e-8), eps_default=eps_default,
        eps_below_subnormal=(eps_default < smallest_sub),
        denominator_collapse=den16, division_result=div16,
        bf16_tiny=float(torch.finfo(torch.bfloat16).tiny),
        bf16_min_sub=2.0 ** -133)


def rng_state_ledger(n_workers=4):
    """续训到底要存几套 RNG。本机是 CPU 单进程，能真数的只有：主进程那一份 torch 状态
       （5056 字节的 Philox 状态，实测一次 randn 之后只有"消耗计数"那段变、头 8 字节不动，
       所以只存 seed 再 manual_seed 是接不上流的，必须整份存）、numpy 的 MT19937 状态、
       Python 的 Mersenne Twister 状态。
       GPU 与多 worker 那两类本机量不到：
         - CUDA 每个设备各有一份 RNG（torch.cuda.get_rng_state_all() 返回逐设备列表），
           本机 torch.cuda.is_available() = False，取不到列表，只能标未验证；
         - DataLoader 起 worker 时，torch 在 _worker_init_fn 里做
           `seed = base_seed + worker_id; torch.manual_seed(seed)`，而 base_seed 是从
           loader.generator 抽出来的一个 int64。本机沙箱里 worker 进程会立刻退出（试过，
           RuntimeError: DataLoader worker exited unexpectedly），所以这里**按 torch 自己的
           算式重算**一遍：真起进程量不到，但算式是官方源码里的，结果可复核。"""
    g = torch.Generator().manual_seed(SEED)
    base = torch.empty((), dtype=torch.int64).random_(generator=g).item()
    worker_seeds = [base + i for i in range(n_workers)]
    try:
        from torch.utils.data._utils.worker import _generate_state
        np_seeds = [_generate_state(base, i) for i in range(n_workers)]
    except Exception as exc:                                # 私有 API 变动就退化为只报 torch 侧
        np_seeds = None
        np_seed_err = repr(exc)
    else:
        np_seed_err = None
    epoch_bases = []
    gg = torch.Generator().manual_seed(SEED)
    for _ in range(3):                                      # 每个 epoch 重新抽 base_seed
        epoch_bases.append(torch.empty((), dtype=torch.int64).random_(generator=gg).item())
    st_before = torch.get_rng_state().numpy().tobytes()
    torch.randn(1)
    st_after = torch.get_rng_state().numpy().tobytes()
    head_unchanged = st_before[:8] == st_after[:8]
    changed_bytes = [i for i in range(len(st_before)) if st_before[i] != st_after[i]]
    torch.set_rng_state(torch.frombuffer(bytearray(st_before), dtype=torch.uint8))
    assert torch.equal(torch.get_rng_state(),
                       torch.frombuffer(bytearray(st_before), dtype=torch.uint8))
    return dict(
        torch_rng_states_cpu=1,
        torch_state_bytes=len(torch.get_rng_state().numpy().tobytes()),
        cuda_available=torch.cuda.is_available(),
        cuda_rng_states=(len(torch.cuda.get_rng_state_all())
                         if torch.cuda.is_available() else None),
        numpy_state_words=len(np.random.get_state()[1]),
        py_rng_version=random.getstate()[0],
        n_workers=n_workers,
        dataloader_base_seed=base,
        worker_seeds=worker_seeds,
        worker_seeds_distinct=len(set(worker_seeds)) == n_workers,
        worker_np_seeds=np_seeds,
        worker_np_seed_err=np_seed_err,
        epoch_bases=epoch_bases,
        epoch_bases_distinct=len(set(epoch_bases)) == len(epoch_bases),
        torch_state_head_unchanged=head_unchanged,
        torch_state_changed_bytes=len(changed_bytes),
        torch_state_first_changed=changed_bytes[0] if changed_bytes else None)


# ==================================================================
#  E. 断点续训：逐位复现与三个反面教材
# ==================================================================
def save_ckpt(path, model, opt, step, gen_state):
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                    gen_state=gen_state, torch_rng=torch.get_rng_state(),
                    np_rng=np.random.get_state(), py_rng=random.getstate()), path)


def resume_variants(model_factory, batches, total, resume_at, good_state):
    """从同一段"已跑 30 步"的现场出发，四种续法各跑完剩下的 50 步：
         good     model + optimizer + step + 数据流位置 全都恢复（生产正解）
         no_opt   只恢复权重，optimizer 从零开始（m/v 丢了）
         no_step  权重和 optimizer 都恢复，但调度器从 t=0 重来（step 丢了）
         no_data  权重和 optimizer 都恢复，但数据流从头重放（dataloader 进度丢了）
       四种续法消费的都是 batches[resume_at:total]，唯一变量是"少存了什么"。"""
    out = {}
    specs = {
        'good':    dict(start_step=resume_at, end_step=total, batch_offset=0, use_opt=True),
        'no_opt':  dict(start_step=resume_at, end_step=total, batch_offset=0, use_opt=False),
        'no_step': dict(start_step=0, end_step=total - resume_at, batch_offset=resume_at,
                        use_opt=True),
        'no_data': dict(start_step=resume_at, end_step=total, batch_offset=-resume_at,
                        use_opt=True),
    }
    for tag, sp in specs.items():
        model = model_factory()
        kw = dict(model_state=good_state['model'], total=total, start_step=sp['start_step'],
                  end_step=sp['end_step'], batch_offset=sp['batch_offset'], clip=CLIP)
        if sp['use_opt']:
            kw['opt_state'] = good_state['opt']
        out[tag] = train_loop(model, batches, **kw)['raw_losses']
    return out


# ==================================================================
#  F. 训练 OOM：显存账本
# ==================================================================
def big_param_count():
    """在 meta device 上构造 7B 结构的 LlamaForCausalLM 真数参数量——不分配任何实张量，
       所以能在 CPU 上跑；这是"本机真数"，不是抄来的引用数字。"""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=BIG_CFG['vocab_size'], hidden_size=BIG_CFG['hidden_size'],
                      intermediate_size=BIG_CFG['intermediate_size'],
                      num_hidden_layers=BIG_CFG['num_hidden_layers'],
                      num_attention_heads=BIG_CFG['num_attention_heads'],
                      num_key_value_heads=BIG_CFG['num_key_value_heads'],
                      max_position_embeddings=4096, rms_norm_eps=1e-5,
                      attention_bias=False, mlp_bias=False, tie_word_embeddings=False)
    with torch.device('meta'):
        model = LlamaForCausalLM(cfg)
    return sum(p.numel() for p in model.parameters())


def big_param_formula(vocab, d, L, ffn):
    """闭式：embedding + lm_head（未共享）+ L×(4d² 注意力 + 3·d·ffn SwiGLU + 2d 两个 RMSNorm) + d 末端 norm。"""
    return 2 * vocab * d + L * (4 * d * d + 3 * d * ffn + 2 * d) + d


def memory_ledger(n_params, act_elems_plain, act_elems_ckpt, act_scale):
    """四项账：权重 / 梯度 / 优化器状态 / 激活。
       前三项按 ZeRO 论文的混合精度口径（bf16 权重 2P + bf16 梯度 2P + fp32 主权重 4P +
       m 4P + v 4P = 16P 字节，引用口径）；激活项用本机实测的 saved 元素数换算（真跑口径），
       再按 act_scale（层数比 × hidden 比）做线性外推，外推值明确标为估算口径。"""
    gb = 1024 ** 3
    return dict(
        params=n_params,
        bf16_weights_bytes=2 * n_params,
        bf16_grads_bytes=2 * n_params,
        fp32_master_bytes=4 * n_params,
        adam_m_v_bytes=8 * n_params,
        total_16P_bytes=16 * n_params,
        total_16P_gb=16 * n_params / gb,
        tiny_act_plain_bytes=act_elems_plain * 4,
        tiny_act_ckpt_bytes=act_elems_ckpt * 4,
        act_saved_ratio=act_elems_plain / act_elems_ckpt if act_elems_ckpt else float('inf'),
        act_scale_estimate=act_scale,
        big_act_plain_gb_est=act_elems_plain * act_scale * 4 / gb,
        big_act_ckpt_gb_est=act_elems_ckpt * act_scale * 4 / gb)


# ==================================================================
#  G. HF Trainer 真跑对照
# ==================================================================
TRAINER_SKELETON = '''\
# ---- HF Trainer 调用骨架（本机缺 accelerate 时只看这段，标"未验证"）----
from transformers import (LlamaConfig, LlamaForCausalLM, Trainer,
                          TrainingArguments, TrainerCallback)

class LossTap(TrainerCallback):          # 逐 step 把 loss / grad_norm / lr 抓出来
    def __init__(self): self.rows = []
    def on_log(self, args, state, control, logs=None, **kw):
        if logs and 'loss' in logs:
            self.rows.append(dict(step=state.global_step, **logs))

args = TrainingArguments(
    output_dir=tmpdir,
    per_device_train_batch_size=BATCH,
    gradient_accumulation_steps=1,
    max_steps=STEPS,
    learning_rate=LR,
    weight_decay=WD,                    # 只作用于非 norm / 非 bias 参数（Trainer 自动掩码）
    adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
    lr_scheduler_type='cosine_with_min_lr',
    lr_scheduler_kwargs={'min_lr_rate': MIN_LR_RATE},
    warmup_steps=WARMUP,
    max_grad_norm=CLIP,                 # Trainer 内部就是 clip_grad_norm_
    optim='adamw_torch',                # 默认 adamw_torch_fused；对照实验要显式关掉 fused
    train_sampling_strategy='sequential',     # 关掉 shuffle，才能和手写循环同序喂数据
    dataloader_drop_last=True,
    logging_steps=1, save_strategy='no', report_to=[],
    seed=SEED, use_cpu=True, disable_tqdm=True)

tap = LossTap()
trainer = Trainer(model=hf_model, args=args, train_dataset=ds, callbacks=[tap])
trainer.train()
# 对账三件套：trainer.get_decay_parameter_names(model) / tap.rows 里的 learning_rate / loss
'''

DEEPSPEED_SKELETON = '''\
# ---- 分布式训练骨架（DeepSpeed ZeRO / FSDP / Megatron，本机未安装，标"未验证"，属番外篇）----
# 显存账本里那 16P 字节单卡装不下时的三条出路，本质都是"把状态切开"：
#   ZeRO-1：只切优化器状态（12P → 12P/N），改动最小，通信量基本不变
#   ZeRO-2：再切梯度（2P → 2P/N）
#   ZeRO-3 / FSDP：连参数也切（4P → 4P/N），前向要 all-gather 参数，通信量上升
# TrainingArguments(deepspeed='ds_zero2.json') / TrainingArguments(fsdp='full_shard auto_wrap')
# Megatron-LM：张量并行 + 流水并行，切的是"算"不是"状态"，要改模型定义，超出本章
'''


def run_trainer_align(cfg_dict, hf_weights, stream_rows, steps):
    """真跑 HF Trainer，并把逐 step 的 loss / grad_norm / lr 抓出来供手写循环对账。
       返回 (None, 错误说明) 表示本机跑不了（缺 accelerate），调用方走降级骨架。"""
    try:
        import accelerate                       # noqa: F401  Trainer.__init__ 要构造 Accelerator
        from transformers import (LlamaConfig, LlamaForCausalLM, Trainer,
                                  TrainingArguments, TrainerCallback)
    except ImportError as e:
        return None, '缺依赖：%s' % e

    def new_hf():
        set_all_seeds(SEED + 3)
        m = LlamaForCausalLM(LlamaConfig(**cfg_dict))
        m.load_state_dict(hf_weights)
        return m

    class DS(torch.utils.data.Dataset):
        """行序 = 手写循环的 batch 顺序展平，配 sequential 采样器 → 两条循环同序同数据。"""

        def __len__(self):
            return stream_rows.shape[0]

        def __getitem__(self, i):
            row = stream_rows[i]
            return dict(input_ids=row.clone(),
                        attention_mask=torch.ones(row.shape[0], dtype=torch.long),
                        labels=row.clone())

    class LossTap(TrainerCallback):
        def __init__(self):
            self.rows = []

        def on_log(self, args, state, control, logs=None, **kw):
            if logs is not None and 'loss' in logs:
                rec = dict(logs)
                rec['step'] = state.global_step
                self.rows.append(rec)

    def one_run(gradient_checkpointing):
        tap = LossTap()
        with tempfile.TemporaryDirectory() as tmp:
            args = TrainingArguments(
                output_dir=tmp, per_device_train_batch_size=BATCH,
                gradient_accumulation_steps=1, max_steps=steps,
                learning_rate=LR, weight_decay=WD,
                adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
                lr_scheduler_type='cosine_with_min_lr',
                lr_scheduler_kwargs=dict(min_lr_rate=MIN_LR_RATE),
                warmup_steps=WARMUP, max_grad_norm=CLIP, optim='adamw_torch',
                train_sampling_strategy='sequential', dataloader_drop_last=True,
                logging_steps=1, save_strategy='no', report_to=[], seed=SEED,
                use_cpu=True, disable_tqdm=True,
                gradient_checkpointing=gradient_checkpointing,
                gradient_checkpointing_kwargs=dict(use_reentrant=False))
            model = new_hf()
            trainer = Trainer(model=model, args=args, train_dataset=DS(), callbacks=[tap])
            trainer.train()
            return dict(logs=list(tap.rows),
                        decay_names=sorted(trainer.get_decay_parameter_names(model)))

    return dict(base=one_run(False), ckpt=one_run(True)), None


# ---------------------------------------------------------------- 配图
def draw(arms, ckpt_rows, resume):
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
    colors = {'clean': '#4a9d6a', 'clip': '#8888cc', 'gain': '#d08c4a',
              'nan': '#cc5a5a', 'skip': '#3a7ca5',
              'good': '#4a9d6a', 'no_opt': '#d08c4a', 'no_step': '#cc5a5a', 'no_data': '#8888cc'}

    for tag, rec in arms.items():
        xs = range(1, len(rec['losses']) + 1)
        ys = [v if (math.isfinite(v) and v < 1e4) else float('nan') for v in rec['losses']]
        axes[0].plot(xs, ys, '-', color=colors[tag], lw=1.4, label=tag)
    axes[0].axvline(SPIKE_STEP + 1, color='#999999', ls=':', lw=1)
    axes[0].set_yscale('log')
    axes[0].set_title('① loss 五臂（第 %d 步注入坏 batch）' % (SPIKE_STEP + 1))
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('loss (log)')
    axes[0].legend(fontsize=8)

    for tag, rec in arms.items():
        xs = range(1, len(rec['gnorms']) + 1)
        ys = [v if math.isfinite(v) else float('nan') for v in rec['gnorms']]
        axes[1].plot(xs, ys, '-', color=colors[tag], lw=1.4, label=tag)
    axes[1].axhline(CLIP, color='#cc5a5a', ls='--', lw=1, label='clip=%.1f' % CLIP)
    axes[1].set_yscale('log')
    axes[1].set_title('② 梯度总范数（裁剪把峰值精确压到阈值）')
    axes[1].set_xlabel('step')
    axes[1].set_ylabel('grad norm (log)')
    axes[1].legend(fontsize=8)

    modes = ['none', 'manual', 'official']
    xs = np.arange(len(modes))
    wbar = 0.36
    fwd = [ckpt_rows[m]['counters']['fwd'] for m in modes]
    rec_ = [ckpt_rows[m]['counters']['recompute'] for m in modes]
    elems = [ckpt_rows[m]['saved']['elems'] for m in modes]
    axes[2].bar(xs - wbar / 2, fwd, width=wbar, color='#4a9d6a', label='首次前向次数')
    axes[2].bar(xs - wbar / 2, rec_, width=wbar, bottom=fwd, color='#d08c4a', label='反向重算次数')
    ax2b = axes[2].twinx()
    ax2b.bar(xs + wbar / 2, elems, width=wbar, color='#8888cc', label='saved 元素数')
    ax2b.set_ylabel('saved 元素数', color='#8888cc')
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(['不检查点', '手写检查点', '官方 checkpoint'], fontsize=9)
    axes[2].set_ylabel('块执行次数')
    axes[2].set_title('③ 以算换存：L → 2L 次前向，saved 元素 %d → %d' % (elems[0], elems[1]))
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')

    for tag, ls in resume.items():
        xs = range(RESUME_AT + 1, RESUME_AT + 1 + len(ls))
        ys = [v if math.isfinite(v) else float('nan') for v in ls]
        axes[3].plot(xs, ys, '-', color=colors[tag], lw=1.4, label=tag)
    axes[3].axvline(RESUME_AT + 0.5, color='#999999', ls=':', lw=1)
    axes[3].set_title('④ 断点续训：第 %d 步中断后的四种续法' % RESUME_AT)
    axes[3].set_xlabel('step')
    axes[3].set_ylabel('loss')
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig-05-training-loop.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    set_all_seeds(SEED)
    t_main = time.perf_counter()

    # ---------------- 数据与模型（全章共用同一套权重与 batch 流）
    rng = random.Random(SEED)
    data, VOCAB = make_char_corpus(rng, n=STEPS * BATCH * 2, seq_len=SEQ, alpha=ALPHA)
    batches, gen_states = make_stream(data, STEPS, BATCH, SEED)
    stream_rows = torch.cat(batches, dim=0)          # Trainer 用同一批数据的同一顺序

    hf_src = build_hf_tiny_llama(VOCAB, DIM, N_LAYERS, N_HEADS, FFN, 1e-5, 10000.0, SEED)
    hf_weights = {k: v.clone() for k, v in hf_src.state_dict().items()}
    hf_cfg_dict = {k: v for k, v in copy.deepcopy(hf_src.config).to_dict().items()
                   if k not in ('architectures', 'torch_dtype', 'transformers_version')}

    def make_hand():
        set_all_seeds(SEED + 7)
        m = TinyLlama(VOCAB, DIM, N_LAYERS, N_HEADS, FFN)
        copy_hf_into_hand(hf_src, m)
        return m

    def make_hf():
        from transformers import LlamaConfig, LlamaForCausalLM
        set_all_seeds(SEED + 3)
        m = LlamaForCausalLM(LlamaConfig(**hf_cfg_dict))
        m.load_state_dict(hf_weights)
        return m

    hand0 = make_hand()
    N_PARAMS = sum(p.numel() for p in hand0.parameters())
    HAND_NAMES = [n for n, _ in hand0.named_parameters()]
    HAND_DECAY = set(decay_param_names(hand0))
    HAND_NODECAY = [n for n in HAND_NAMES if n not in HAND_DECAY]

    lines = []
    lines.append('=' * W)
    lines.append('生产篇 05 · 训练循环：AdamW + warmup + cosine + 裁剪 + 累积 + 两类检查点 + 断点续训')
    lines.append('=' * W)
    lines.append('配置：dim=%d 层=%d 头=%d ffn=%d seq=%d batch=%d steps=%d warmup=%d lr=%.1e'
                 % (DIM, N_LAYERS, N_HEADS, FFN, SEQ, BATCH, STEPS, WARMUP, LR))
    lines.append('      eta_min=lr×%.2f=%.1e  weight_decay=%.2f  clip=%.1f  累积K=%d  注入步=%d'
                 % (MIN_LR_RATE, LR * MIN_LR_RATE, WD, CLIP, ACCUM_K, SPIKE_STEP + 1))
    lines.append('      合成语料 vocab=%d、%d 条 × %d token；手写 tiny Llama 参数量 = %d'
                 % (VOCAB, data.shape[0], SEQ, N_PARAMS))
    lines.append('      权重来源：离线 LlamaConfig 随机初始化（不下载任何权重、不联网），'
                 '手写模型与 HF 模型同权重')
    lines.append('      weight decay 掩码：%d 个参数施加 decay、%d 个不施加（每块 2 个 RMSNorm + 末端 1 个）'
                 % (len(HAND_DECAY), len(HAND_NODECAY)))

    # ---------------- A. 训练循环 + loss spike 五臂
    lines.append('\n' + '=' * W)
    lines.append('【A】生产训练循环骨架 + loss spike 五臂（第 %d 步注入坏 batch，gain=%.0f×）'
                 % (SPIKE_STEP + 1, SPIKE_GAIN))
    lines.append('=' * W)
    arms = {}
    arms['clean'] = train_loop(make_hand(), batches, clip=None,
                               probe_at=SPIKE_STEP, probe_param=PROBE_PARAM)
    arms['clip'] = train_loop(make_hand(), batches, clip=CLIP)
    arms['gain'] = train_loop(make_hand(), batches, clip=None,
                              spike_at=SPIKE_STEP, spike_kind='gain',
                              probe_at=SPIKE_STEP, probe_param=PROBE_PARAM)
    arms['nan'] = train_loop(make_hand(), batches, clip=CLIP,
                             spike_at=SPIKE_STEP, spike_kind='nan')
    arms['skip'] = train_loop(make_hand(), batches, clip=CLIP, skip_nonfinite=True,
                              spike_at=SPIKE_STEP, spike_kind='nan')
    clip_desc = {'clean': 'off', 'clip': 'clip=%.1f' % CLIP, 'gain': 'off',
                 'nan': 'clip=%.1f' % CLIP, 'skip': 'clip=%.1f' % CLIP}
    lines.append('    %-6s %-10s %-14s %-14s %-10s %-8s %-10s' %
                 ('臂', '裁剪', '峰值原始范数', '注入步loss', 'NaN步数', '跳步数', '末raw loss'))
    for tag, r in arms.items():
        lines.append('    %-6s %-10s %-14.6g %-14.6g %-10d %-8d %-10.6f' %
                     (tag, clip_desc[tag], r['peak_gnorm'], r['losses'][SPIKE_STEP],
                      r['nan_steps'], r['skipped'], r['last']))
    lines.append('    两种梯度范数口径互证：第 1 步「只测量不裁剪」= %.10f，'
                 '「clip_grad_norm_ 返回值」= %.10f（同权重同数据，差 %.1e）'
                 % (arms['clean']['gnorms'][0], arms['clip']['gnorms'][0],
                    abs(arms['clean']['gnorms'][0] - arms['clip']['gnorms'][0])))
    lines.append('    注入是否生效：不裁剪臂注入步梯度范数 = %.6g，干净臂同一步 = %.6g（比值 %.2f×，'
                 '注入 gain=%.0f×）'
                 % (arms['gain']['gnorms'][SPIKE_STEP], arms['clean']['gnorms'][SPIKE_STEP],
                    arms['gain']['gnorms'][SPIKE_STEP] / arms['clean']['gnorms'][SPIKE_STEP],
                    SPIKE_GAIN))
    lines.append('    裁剪是否生效：clip 臂的原始范数峰值 = %.6f（>阈值 %.1f，说明真的有步被裁），'
                 '而裁剪后优化器实际吃进去的范数峰值 = %.6f'
                 % (arms['clip']['peak_gnorm'], CLIP, arms['clip']['peak_applied']))
    lines.append('      被裁的步数 = %d/%d；口径提醒：clip_grad_norm_ 的返回值是「裁剪前」的总范数，'
                 '日志里看到它 >1.0 不代表裁剪失效' % (arms['clip']['n_clipped'], STEPS))
    upd_ratio = arms['gain']['upd'][SPIKE_STEP] / arms['clean']['upd'][SPIKE_STEP]
    lines.append('    Adam 的尺度不变性：梯度整体 ×%.0f（%.6g → %.6g），但那一步的参数更新 L2 范数'
                 '只从 %.6g 变到 %.6g（%.2f×），远不是 ×%.0f'
                 % (SPIKE_GAIN, arms['clean']['gnorms'][SPIKE_STEP],
                    arms['gain']['gnorms'][SPIKE_STEP], arms['clean']['upd'][SPIKE_STEP],
                    arms['gain']['upd'][SPIKE_STEP], upd_ratio, SPIKE_GAIN))
    lines.append('      —— 因为 AdamW 的更新量是 lr·m̂/(√v̂+eps)，梯度整体放大时 m̂ 与 √v̂ 同向放大，'
                 '大部分被约掉（m 的 β1=%.2f、v 的 β2=%.3f 响应速度不同，所以剩下这 %.2f×）'
                 % (0.9, 0.999, upd_ratio))
    sc = adam_scale_ratios(arms['clean']['probe'], arms['gain']['probe'])
    q3 = lambda v: '[%.3f, %.3f, %.3f]' % tuple(v)
    lines.append('      把这 %.2f× 拆成逐元素账（探针参数 %s，在该步 step() 前后各取一次快照）：'
                 % (upd_ratio, PROBE_PARAM))
    lines.append('        同起点核对：两臂进入注入步时的 AdamW 状态 m/v 逐元素最大差 = %.1e'
                 '（必须是 0，否则就不是干净对照）' % sc['m_state_diff'])
    lines.append('        梯度本身：非零元素的 |Δg| 比值 min = %.4f、max = %.4f —— '
                 'loss ×%.0f 把**每一个**梯度元素都放大到 ×%.0f 的 ±0.1%% 浮点带内，'
                 % (sc['grad_ratio_min'], sc['grad_ratio_max'], SPIKE_GAIN, SPIKE_GAIN))
    lines.append('          所以"放大只落在少数参数上、多数参数没拿到大梯度"这种解释是不成立的；'
                 '缩小发生在分母 √v̂ 上（它有 β2=%.3f 的记忆，一步顶不动）' % 0.999)
    lines.append('        逐元素比值分位（p10/p50/p90）：m̂ %s、√v̂ %s、更新量 |Δp|（有更新的 %d/%d 个元素）%s'
                 % (q3(sc['ratio_m']), q3(sc['ratio_sv']), sc['n_active'], sc['n_elem'],
                    q3(sc['ratio_upd_active'])))
    lines.append('        冷元素（m=v=0，即这一步才第一次拿到梯度的坐标）的闭式放大 = '
                 '(1−β1)·k / √((1−β2)·k²) = %.3f×' % sc['cold_ratio'])
    lines.append('          它是"无历史包袱"那一类元素的比值，**不是分布上界**：分位里 p90 = %.3f× 比它大。'
                 '比值 > %.3f× 的元素共 %d 个（占 %d 个里的 %.1f%%），'
                 % (sc['ratio_upd_active'][2], sc['cold_ratio'], sc['n_hi_cold'],
                    sc['n_active'], 100.0 * sc['n_hi_cold'] / sc['n_active']))
    lines.append('          它们在**干净臂**的更新平方和里只占 %.2f%%，在**放大臂**里占 %.2f%% —— '
                 '高比值来自干净臂自身更新几乎相消（分母小），所以这些坐标是"比值大、位移小"。'
                 '而 L2 比值 = √(Σu_放大²/Σu_干净²) 恰好是按**干净臂幅度**加权后再开方，'
                 '这批元素在权重里只占 %.2f%%，所以 p90 高达 %.3f× 也压不动 L2。'
                 % (100.0 * sc['share_hi_clean'], 100.0 * sc['share_hi_gain'],
                    100.0 * sc['share_hi_clean'], sc['ratio_upd_active'][2]))
    lines.append('          探针参数自己按同一口径开出的 L2 = %.3f×（比它自己的逐元素中位数 %.3f× 还低，'
                 '这就是加权的后果）。注意口径：逐元素分布量的是这一个参数（%d 个元素），'
                 '下面那个"全局 L2"是 39 个参数合起来的账，两者不同口径，不要当成同一个数读。'
                 % (sc['l2_ratio_probe'], sc['ratio_upd_active'][1], sc['n_active']))
    lines.append('        全局 L2 比值 = %.3f×（与上面 %.2f× 同一个量，两条算法互证），'
                 '它不等于任何单个元素的比值：' % (sc['l2_ratio'], upd_ratio))
    lines.append('          被 √v̂ 的历史压住的那些元素（v 已经有量级、k·g 顶不动它）比值低，冷元素闭式 %.3f×。'
                 '同口径的三明治是 探针 p10 %.3f× ≤ 探针 L2 %.3f× ≤ 冷元素 %.3f×（断言口径）；'
                 '探针 L2 %.3f× 甚至低于它自己的逐元素中位数 %.3f×，因为加权用的是干净臂幅度，'
                 '而低比值的那批元素恰恰是干净臂里位移最大的那些'
                 % (sc['cold_ratio'], sc['ratio_upd_active'][0], sc['l2_ratio_probe'],
                    sc['cold_ratio'], sc['l2_ratio_probe'], sc['ratio_upd_active'][1]))
    lines.append('      优化器替代口径（本文不实现）：本节全部账都以 AdamW 为默认。'
                 'Muon（MOMentedum Orthogonalized by Newton–Schulz，Keller Jordan 提出、用于 nanoGPT speedrun；'
                 '规模化见 Moonlight arXiv:2502.16982）这一类替代思路是把动量缓冲过的梯度矩阵用 Newton–Schulz 迭代正交化'
                 '（约 5 次迭代、只作用于 2D 隐藏矩阵），再按隐藏维与输出维的长宽比缩放步长；'
                 'embedding、lm_head、norm 与一维张量通常仍交给 AdamW。'
                 '工业使用：Kimi K2 用 MuonClip，GLM-4.5 除 word embedding / bias / RMSNorm 外全部用 Muon'
                 '（N=5 次迭代、momentum 0.95、update RMS 0.2）。归属为外部事实、已核原文；'
                 '它与本节这套账的关系是"把尺度不变性换成谱范数意义上的等步长"，'
                 '裁剪/warmup/断点续训这些工程约束一条都不省。本机未实现该优化器，标未验证。')
    lines.append('      代价不在注入的那一步，而在后面：末 raw loss 从 %.6f 恶化到 %.6f（+%.1f%%），'
                 '一次异常梯度污染了 m/v 的滑动平均'
                 % (arms['clean']['last'], arms['gain']['last'],
                    100 * (arms['gain']['last'] / arms['clean']['last'] - 1)))
    lines.append('      裁剪的真实作用就是这个：把污染 m/v 的幅度限制住，'
                 '而不是"把步子按倍数缩小"')
    lines.append('    NaN 注入：clip=%.1f 也救不了 —— NaN 步数 %d，末 raw loss = %s'
                 % (CLIP, arms['nan']['nan_steps'],
                    'nan' if not math.isfinite(arms['nan']['last']) else '%.6f' % arms['nan']['last']))
    lines.append('    有限性跳步：NaN 步数 %d、跳步 %d 次，末 raw loss = %.6f（同为 clip=%.1f 的 '
                 'clip 臂 %.6f，差 %.3e）'
                 % (arms['skip']['nan_steps'], arms['skip']['skipped'], arms['skip']['last'],
                    CLIP, arms['clip']['last'], abs(arms['skip']['last'] - arms['clip']['last'])))
    for tag in ('gain', 'nan', 'skip'):
        r = arms[tag]
        rc = recovery_steps(r, SPIKE_STEP)
        rc3 = recovery_steps(r, SPIKE_STEP, consec=3)
        bounce = bounce_back_steps(r, SPIKE_STEP)
        lines.append('    恢复步数（raw loss 回到本臂注入前 %.6f 以下）：%-5s 首次=%s，'
                     '连续 3 步站稳=%s，站稳后又弹回=%s'
                     % (r['raw_losses'][SPIKE_STEP - 1], tag,
                        ('%d 步' % rc) if rc >= 0 else '未恢复（全程 NaN）',
                        ('%d 步' % rc3) if rc3 >= 0 else '未恢复',
                        ('%d 步后弹回' % bounce) if bounce > 0 else '没弹回'))
    lines.append('      口径：首次恢复是单点判据，容易被一次侥幸的低 loss 骗过（弹回=%d 步就是证据）；'
                 '生产上应要求连续 N 步都低于参考值，本机 N=3 与 N=1 的结论差异见上一行'
                 % max(bounce_back_steps(arms[t], SPIKE_STEP) for t in ('gain', 'nan', 'skip')))
    log_verify('生产训练循环骨架（AdamW + warmup/cosine + decay 掩码 + 裁剪 + 有限性跳步）',
               '真跑验证', '五臂 × %d 步真训，坏 batch 为合成注入' % STEPS)
    log_verify('Adam 尺度不变性的逐元素拆解（m̂ / √v̂ / |Δp| 分布 + 冷元素闭式）', '真跑验证',
               '梯度逐元素比值全部落在 ×%.0f 的 ±0.1%% 带内；同口径三明治 探针 p10 %.3f× ≤ 探针 L2 %.3f× '
               '≤ 冷元素闭式 %.3f×，高比值元素在干净臂更新平方和只占 %.2f%%'
               % (SPIKE_GAIN, sc['ratio_upd_active'][0], sc['l2_ratio_probe'], sc['cold_ratio'],
                  100.0 * sc['share_hi_clean']))
    log_verify('Muon（Newton–Schulz 正交化动量）作为 AdamW 的替代', '外部引用（非本机可验证）',
               '本机未实现，只给机制说明与"哪些工程约束不省"的判断；归属已核原文：'
               'Muon 出自 Keller Jordan 的 MOMentedum Orthogonalized by Newton-Schulz（nanoGPT speedrun），'
               '规模化见 Moonlight arXiv:2502.16982，工业使用见 Kimi K2（MuonClip）与 GLM-4.5 技术报告')

    # ---------------- 裁剪策略对照：按范数 clip vs 按值 clamp
    cs = clip_strategy_probe(CLIP)
    lines.append('\n' + '=' * W)
    lines.append('【裁剪策略】clip_grad_norm_（按范数缩放）vs clip_grad_value_（逐元素 clamp 到 ±%.1f）'
                 % CLIP)
    lines.append('=' * W)
    lines.append('    %-18s %-12s %-8s %-8s %-9s %-8s %-11s %-11s' %
                 ('裁剪函数 × 坏法', '裁前NaN/inf', '返回值', 'NaN 个数', 'inf 个数', '是否有限',
                  'max|g|', '方向余弦'))
    for kind in ('norm', 'value'):
        for case in ('inf', 'nan', 'finite'):
            d = cs[kind][case]
            lines.append('    %-18s %-12s %-8s %-8d %-9d %-8s %-11s %-11s'
                         % ('%s / %s' % ('grad_norm_' if kind == 'norm' else 'grad_value_', case),
                            '%d/%d' % (d['n_nan_before'], d['n_inf_before']),
                            ('%.6g' % d['returned']) if d['returned'] is not None else 'None',
                            d['n_nan'], d['n_inf'], '是' if d['all_finite'] else '否',
                            ('%.6g' % d['max_abs']) if d['max_abs'] is not None else '—',
                            ('%.6f' % d['direction_cos']) if d['direction_cos'] is not None else '—'))
    lines.append('    读表（三行是关键）：')
    lines.append('      norm / inf：裁前实测 NaN=%d、inf=%d；返回值 = %.6g，裁剪后 NaN 个数 = %d —— '
                 '梯度里本来只有 inf、没有 NaN，是**裁剪自己造出的 NaN**（系数 = %.1f/(inf+1e-6) = 0，'
                 '0×inf = NaN）。'
                 % (cs['norm']['inf']['n_nan_before'], cs['norm']['inf']['n_inf_before'],
                    cs['norm']['inf']['returned'], cs['norm']['inf']['n_nan'], CLIP))
    lines.append('      value / inf：全部有限（max|g| = %.6g、NaN 个数 %d）—— 逐元素 clamp 不乘系数，'
                 'inf 直接变成一个有限值，天然免疫这个坑。'
                 % (cs['value']['inf']['max_abs'], cs['value']['inf']['n_nan']))
    lines.append('      代价在 finite 那两行：同一个大梯度（范数 %.4g > 阈值），按范数裁剪后方向余弦 = %.6f'
                 '（严格保方向，只缩长度），' % (cs['norm']['finite']['returned'],
                                                 cs['norm']['finite']['direction_cos']))
    lines.append('        按值裁剪后方向余弦 = %.6f（每个元素被独立 clamp 到 ±%.1f，方向被掰弯）。'
                 '所以"按值更稳"不是免费的：它把梯度方向改了。'
                 % (cs['value']['finite']['direction_cos'], CLIP))
    lines.append('      value / nan：NaN 个数 = %d —— clamp 不扩散 NaN（只那一个元素坏），'
                 '但也修不掉它；norm / nan 则把 %d 个元素全部变成 NaN（系数 NaN 乘遍每个元素）。'
                 % (cs['value']['nan']['n_nan'], cs['norm']['nan']['n_nan']))
    lines.append('    两条口径提醒：① clip_grad_norm_ 的返回值永远是**裁剪前**的总范数（上表 finite 行 %.4g），'
                 '日志里它 >%.1f 不代表裁剪失效（与【A】的 peak_applied=%.6f 对照）；'
                 % (cs['norm']['finite']['returned'], CLIP, arms['clip']['peak_applied']))
    lines.append('      ② 生产默认仍是按范数裁剪 + 非有限就跳步（跳步是挡 NaN/inf 的那一道，'
                 '不是裁剪）。两个相关口径：torch 的 clip_grad_norm_ 本身带 error_if_nonfinite 参数'
                 '（本机 inspect 到签名：%s），打开它非有限就直接抛错而不是静默扩散；'
                 '另有 AGC（adaptive gradient clipping）这类变体——阈值不取全局常数，'
                 '而是按每个参数自身的范数按比例定，属方向说明、本机未实现（未验证）。'
                 % ', '.join(inspect.signature(torch.nn.utils.clip_grad_norm_).parameters))
    log_verify('按范数裁剪造 NaN / 按值裁剪免疫 inf 的对照', '真跑验证',
               '同一份带 inf 的梯度：norm 后 NaN=%d，value 后全部有限（max|g|=%.1f）'
               % (cs['norm']['inf']['n_nan'], cs['value']['inf']['max_abs']))
    log_verify('AGC（adaptive gradient clipping）', '未验证',
               '只给方向说明，未实现算法；阈值口径来自论文，本轮未逐字核到原文')

    # ---------------- lr 调度变体：WSD（与 cosine 平级的另一条曲线）
    from transformers import __version__ as HF_V
    wsd_hf = wsd_vs_hf(STEPS, WARMUP, LR, MIN_LR_RATE)
    wsd_ledger = wsd_resume_ledger(WARMUP, LR, LR * MIN_LR_RATE)
    wsd_arm = train_loop(make_hand(), batches, clip=CLIP,
                         schedule='wsd', stable_end=STEPS - 20)
    hot_cos = sum(1 for v in arms['clip']['lrs'] if v >= 0.9 * LR)
    hot_wsd = sum(1 for v in wsd_arm['lrs'] if v >= 0.9 * LR)
    sum_lr_cos = sum(arms['clip']['lrs'])
    sum_lr_wsd = sum(wsd_arm['lrs'])
    lines.append('\n' + '=' * W)
    lines.append('【lr 调度】WSD（warmup–stable–decay）：与 HF 逐 step 对账 + 与 cosine 真跑对照 + 续训语义账')
    lines.append('=' * W)
    lines.append('    出处（外部事实，已逐字核到原文，非本机可验证）：WSD 的命名与三阶段设计出自 MiniCPM'
                 '（arXiv:2404.06395 §4.2，退火取约 10% token，原文写"复用退火前的模型、以原高 lr 继续训"）；'
                 'Kimi K2 §2.5 用 WSD 并配 MuonClip，MiniCPM4 是 7T 稳定段 + 1.3T 退火；'
                 '理论解释见 Wen et al. 的 river valley 损失地形（arXiv:2410.05192）。'
                 '反例同样要记：GLM-4.5 §2.4 明确选 cosine 而非 WSD，DeepSeek-V3 用的是 cosine decay。'
                 '库支持：HF transformers 的 SchedulerType.warmup_stable_decay（get_wsd_schedule，自 v4.41.0 起），'
                 'Megatron-Bridge 有 WSD 的 lr_decay_style，DeepSpeed 没有。')
    lines.append('    手写 wsd_lr_at vs HF get_wsd_schedule(decay_type="cosine"、num_cycles=0.5)：'
                 '逐 step 对账 %d 步（transformers %s）' % (STEPS, HF_V))
    lines.append('      全步 max|Δ| = %.3e，但 warmup 之后（t ≥ %d）max|Δ| = %.3e —— '
                 '稳定段与退火段两边逐位一致，唯一的差别在 warmup 段'
                 % (wsd_hf['max_diff'], WARMUP, wsd_hf['max_diff_from_warmup']))
    lines.append('      差别从哪来：HF 的 lambda 是 factor·(1−min_lr_ratio)+min_lr_ratio，'
                 '连 warmup 也被抬了同一个下限，所以它第 0 步 lr = %.3e = %.1f×peak 而不是 0；'
                 '同库的 cosine_with_min_lr 的 warmup 段没有这个下限（【A】与 Trainer 的对账里 '
                 'lr(0)=0 精确成立）。这是 HF 这一个实现的细节，不是 WSD 的定义。'
                 % (wsd_hf['warmup_first_lr'], wsd_hf['warmup_first_lr'] / wsd_hf['peak'], ))
    lines.append('      稳定段实测去重后的 lr 值 = %d 个（%d 步恒定），等于 peak = %.3e'
                 '（恒定段是真的恒定，不是"缓慢下降看着像恒定"）'
                 % (len(set(wsd_hf['plateau'])), len(wsd_hf['plateau']), wsd_hf['peak']))
    lines.append('      末步 lr = %.3e ≈ eta_min = %.3e（min_lr_ratio=%.2f×peak）'
                 % (wsd_hf['last'], LR * MIN_LR_RATE, MIN_LR_RATE))
    lines.append('    同一份权重、同一批数据、同一 clip，两条曲线各真跑 %d 步：' % STEPS)
    lines.append('      cosine：末 raw loss = %.6f，lr ≥ 0.9×peak 的步数 %d/%d，Σlr = %.4e'
                 % (arms['clip']['last'], hot_cos, STEPS, sum_lr_cos))
    lines.append('      WSD   ：末 raw loss = %.6f，lr ≥ 0.9×peak 的步数 %d/%d，Σlr = %.4e'
                 % (wsd_arm['last'], hot_wsd, STEPS, sum_lr_wsd))
    lines.append('      末 loss 差 = %.3e —— 本机这个 toy 任务上两条曲线跑得一样好；'
                 'WSD 的真正卖点不是"终点更低"，而是下面这本续训账'
                 % abs(wsd_arm['last'] - arms['clip']['last']))
    lines.append('    续训语义账：原计划 %d 步，中途改成 %d 步（延长 %d 步），'
                 '看历史步和未走步的 lr 各变了多少' % (100, 160, 60))
    lines.append('      %-12s %-8s %-16s %-16s %-10s %-16s' %
                 ('续训点', '已走步', '已走步里变了几个', '未走步数', '未走变了几个', '未走max|Δlr|'))
    for kind in ('cosine', 'wsd_stable', 'wsd_decay'):
        d = wsd_ledger[kind]
        lines.append('      %-12s %-8d %-16d %-16d %-10d %-16.3e'
                     % (kind, d['resume'], d['past_changed'], d['fut_steps'],
                        d['fut_changed'], d['fut_max_diff']))
    lines.append('      读法一（cosine 的代价）：在第 %d 步把 total 从 100 改成 160，'
                 '%d 个未走步的 lr **全**变了（最大相对变化 %.0f%%）；'
                 '已走的 %d 步里也有 %d 步被改写（max|Δlr| = %.3e）——'
                 % (wsd_ledger['cosine']['resume'], wsd_ledger['cosine']['fut_steps'],
                    100 * wsd_ledger['cosine']['fut_rel_max'], wsd_ledger['cosine']['past_steps'],
                    wsd_ledger['cosine']['past_changed'], wsd_ledger['cosine']['past_max_diff']))
    lines.append('        改写的是退火段整体形状（分母从 total−warmup 变成新的 total−warmup），'
                 'warmup 那 %d 步不受影响，所以"没变"的只有 warmup 段。'
                 % WARMUP)
    lines.append('        也就是说 cosine 下想加训，只能改 total，一改就把"这份 checkpoint 是怎么来的"'
                 '这条 lr 曲线本身改写了——历史 loss 曲线从此无法用任何一条 lr 序列复现出来。')
    lines.append('      读法二（WSD 稳定段的红利）：在第 %d 步（稳定段内）延长 total，'
                 '%d 个已走步一个都没变（max|Δlr| = %.1e），未走的 %d 步里只有 %d 步变、'
                 '而且变的全是退火段（被整体推后），稳定段那 %d 步的 lr 一个都没动。'
                 % (wsd_ledger['wsd_stable']['resume'], wsd_ledger['wsd_stable']['past_steps'],
                    wsd_ledger['wsd_stable']['past_max_diff'], wsd_ledger['wsd_stable']['fut_steps'],
                    wsd_ledger['wsd_stable']['fut_changed'],
                    wsd_ledger['wsd_stable']['fut_steps'] - wsd_ledger['wsd_stable']['fut_changed']))
    lines.append('        稳定段的 lr 只等于 peak，与 total 无关 → "先一直训着，看到要停了再退火"是合法操作，'
                 '这就是 WSD 支持随时中断/延长/发布一个 checkpoint 的机理')
    lines.append('      读法三（WSD 也不是万能）：在第 %d 步（退火段内）延长 decay_end，'
                 '未走的 %d 个退火步全变（max|Δlr| = %.3e）——而且连**已经走过**的 %d 步退火也被改写'
                 '（max|Δlr| = %.3e）。'
                 % (wsd_ledger['wsd_decay']['resume'], wsd_ledger['wsd_decay']['fut_steps'],
                    wsd_ledger['wsd_decay']['fut_max_diff'], wsd_ledger['wsd_decay']['past_changed'],
                    wsd_ledger['wsd_decay']['past_max_diff']))
    lines.append('        所以工程纪律是：**稳定段可以随便延长，退火段一旦开始就必须锁定 total**；'
                 '退火段中途改 total 会把 WSD 唯一那个红利（不动历史）赔掉——这时它比 cosine 更值得警惕，'
                 '因为使用者正是冲着"不动历史"才选它的。')
    log_verify('手写 WSD 与 HF get_wsd_schedule 逐 step 对账', '真跑验证',
               'transformers %s；稳定段+退火段 max|Δ|=%.1e，warmup 段差在 HF 的下限口径 %.1e'
               % (HF_V, wsd_hf['max_diff_from_warmup'], wsd_hf['max_diff']))
    log_verify('WSD 与 cosine 同数据各真跑 %d 步' % STEPS, '真跑验证',
               '末 loss %.6f vs %.6f；lr≥0.9×peak 步数 %d vs %d'
               % (wsd_arm['last'], arms['clip']['last'], hot_wsd, hot_cos))
    log_verify('WSD 的续训语义账（延长 total 后逐步比 lr）', '真跑验证',
               '纯调度器算式对照，不训练；cosine 未来步全变、WSD 稳定段一个不变')
    log_verify('WSD 的出处与"哪些工业模型在用它"这类归属', '外部引用（非本机可验证）',
               '本机只验证算式与 HF get_wsd_schedule 一致；归属已逐字核到原文：WSD 命名与三阶段设计出自 '
               'MiniCPM arXiv:2404.06395 §4.2（退火约 10% token），Kimi K2 §2.5 用 WSD 并配 MuonClip；'
               '两个反例也要记：GLM-4.5 §2.4 明确说选 cosine 而非 WSD，DeepSeek-V3 用的是 cosine decay'
               '（"WSD 由 DeepSeek-V3 采用"这一流传说法不成立）')

    # ---------------- B. 梯度累积
    micro = batches[:ACCUM_K]
    big = torch.cat(micro, dim=0)
    accum = accumulation_equivalence(make_hand, micro, big, LR, WD)
    probe_model = make_hand()
    saved_small = count_saved(lambda: causal_loss(probe_model(micro[0]), micro[0]))
    saved_big = count_saved(lambda: causal_loss(probe_model(big), big))
    lines.append('\n' + '=' * W)
    lines.append('【B】梯度累积：K=%d 累积 vs 一次 %d×batch=%d 的大步' % (ACCUM_K, ACCUM_K, len(big)))
    lines.append('=' * W)
    lines.append('    梯度：逐元素 max|Δ| = %.3e，相对梯度总范数（%.6f）= %.3e'
                 % (accum['grad_diff'], accum['gnorm'], accum['grad_rel']))
    lines.append('      %.2e 正是 float32 的求和顺序噪声量级：两条路的差别只在"K 次小 mean 再相加"'
                 '还是"一次大 mean"，' % accum['grad_rel'])
    lines.append('      数学上完全同一个式子，浮点上不可能逐位相同')
    lines.append('    一次 step 之后：参数逐元素 max|Δ| = %.3e，相对单步更新幅度（max|Δp| = %.3e）'
                 '= %.3e' % (accum['param_diff'], accum['update_abs'], accum['param_rel']))
    lines.append('      相对差从 %.1e（梯度）放大到 %.1e（更新），是 AdamW 干的：更新量是 '
                 'm̂/(√v̂+eps)，' % (accum['grad_rel'], accum['param_rel']))
    lines.append('      第 1 步 v̂ 极小，分母把梯度接近 0 的那些坐标的末位噪声放大成了可见的相对差；'
                 '绝对幅度仍只有 %.1e' % accum['param_diff'])
    lines.append('      口径提醒：判断两条路径是否等价要看相对量，不要看绝对 max|Δ|——'
                 '后者会被"梯度接近 0 的那几个元素"主导')
    lines.append('    为反向保存的元素数：batch=%d → %d；batch=%d → %d（比值 %.3f×，应 ≈ K=%d）'
                 % (BATCH, saved_small['elems'], len(big), saved_big['elems'],
                    saved_big['elems'] / saved_small['elems'], ACCUM_K))
    lines.append('    结论：累积省的是"激活显存峰值"（同一时刻只驻留 1/K 的激活），'
                 '不省算力（总 token 数不变，前向次数反而从 1 次变成 K 次）')
    log_verify('梯度累积 == 一次大 batch（逐参数对账）', '真跑验证',
               '梯度相对差 %.2e；saved 元素 %.2f× ≈ K=%d'
               % (accum['grad_rel'], saved_big['elems'] / saved_small['elems'], ACCUM_K))

    # ---------------- lr 缩放规则：累积把 effective batch 放大时，lr 该不该跟着放大
    ls = {}
    ls['small'] = train_loop(make_hand(), batches, total=LR_SCALE_STEPS, clip=CLIP,
                             schedule='constant', peak_lr=LR)
    ls['big_lr'] = train_loop(make_hand(), batches, total=LR_SCALE_STEPS // LR_SCALE_K,
                             clip=CLIP, schedule='constant', peak_lr=LR, accum_k=LR_SCALE_K)
    ls['big_lrK'] = train_loop(make_hand(), batches, total=LR_SCALE_STEPS // LR_SCALE_K,
                              clip=CLIP, schedule='constant', peak_lr=LR * LR_SCALE_K,
                              accum_k=LR_SCALE_K)
    lines.append('\n' + '=' * W)
    lines.append('【lr 缩放】累积把 effective batch 放大 K=%d 倍时，lr 要不要跟着放大（三臂真跑）'
                 % LR_SCALE_K)
    lines.append('=' * W)
    lines.append('    三臂消费**同一批数据**（batches[0:%d]，前向次数也相同），只差"每步吃多大 batch、'
                 'lr 多少"；恒定 lr、不 warmup 不退火，避免把调度器形状混进来' % LR_SCALE_STEPS)
    lines.append('    %-12s %-14s %-12s %-12s %-12s %-12s' %
                 ('臂', 'effective batch', '优化器步数', 'Σlr', '累计更新L2', '末 raw loss'))
    for tag, desc in (('small', 'B'), ('big_lr', 'K·B'), ('big_lrK', 'K·B')):
        r = ls[tag]
        eff = BATCH * (LR_SCALE_K if tag != 'small' else 1)
        lines.append('    %-12s %-14s %-12d %-12.4e %-12.4e %-12.6f'
                     % (tag, '%d（%s）' % (eff, desc), len(r['upd']), sum(r['lrs']),
                        sum(r['upd']), r['last']))
    u_small, u_big, u_bigK = (sum(ls[t]['upd']) for t in ('small', 'big_lr', 'big_lrK'))
    r_plain, r_scaled = u_big / u_small, u_bigK / u_small
    lines.append('    先给结论：AdamW 一个优化器步的实际位移 = lr × 饱和比（|m̂|/√v̂ 的逐元素均值），'
                 '前一项由调度器给，后一项**随 effective batch 上升**——')
    lines.append('      所以"累计更新量只由 Σlr 决定"不成立，本机把这两个因子分开量了出来。')
    lines.append('      小臂 Σlr = %.4e、累计更新 %.4e；大臂不缩放：Σlr = %.4e（除以 %d）、'
                 '累计更新 %.4e（比值 %.3f）'
                 % (sum(ls['small']['lrs']), u_small, sum(ls['big_lr']['lrs']), LR_SCALE_K, u_big,
                    r_plain))
    lines.append('      大臂按线性规则缩放（lr×%d）：Σlr = %.4e 与小臂**精确相等**，'
                 '累计更新 %.4e（比值 %.3f）' % (LR_SCALE_K, sum(ls['big_lrK']['lrs']), u_bigK, r_scaled))
    lines.append('      两行合起来是一个乘性结构：位移比 = Σlr 比 × s，s 只依赖 effective batch。'
                 's = %.3f（大臂 vs 小臂，同样 Σlr）；用它反推不缩放臂 = s/%d = %.4f，'
                 '实测 %.4f（差 %.1f%%）→ 结构本机成立'
                 % (r_scaled, LR_SCALE_K, r_scaled / LR_SCALE_K, r_plain,
                    100 * abs(r_plain - r_scaled / LR_SCALE_K) / (r_scaled / LR_SCALE_K)))
    lines.append('      于是缩放规则的落点很清楚：lr 不动 → 学习量只有小臂的 %.2f×（欠 %.1f 倍）；'
                 'lr×%d → 反而走到 %.2f×（超调）；本机标定的等效倍数 = K/s = %.2f，'
                 '介于 √K=%.1f 与 K=%d 之间'
                 % (r_plain, 1 / r_plain, LR_SCALE_K, r_scaled, LR_SCALE_K / r_scaled,
                    math.sqrt(LR_SCALE_K), LR_SCALE_K))
    lines.append('      末 loss：小臂 %.6f / 大臂不缩放 %.6f / 大臂×%d %.6f —— 三臂终点差别只有 %.1e '
                 '量级，本机 toy 任务分不出"哪条 lr 更快更好"；'
                 % (ls['small']['last'], ls['big_lr']['last'], LR_SCALE_K, ls['big_lrK']['last'],
                    max(ls['small']['last'], ls['big_lr']['last'], ls['big_lrK']['last'])
                    - min(ls['small']['last'], ls['big_lr']['last'], ls['big_lrK']['last'])))
    lines.append('        能断言的只有这本"学习量"账（口径清楚、可直接量），'
                 '等效缩放倍数 %.2f 是这个 toy 上的标定值，不外推到真实大模型' % (LR_SCALE_K / r_scaled))
    lines.append('    本文的立场（明说）：【B】的等价性是"同一个 effective batch 内部拆成 K 次前向"，'
                 'effective batch 与优化器步数都没变，所以 lr 一个都不用动；')
    lines.append('      一旦累积是为了**把 effective batch 放大 K 倍**（本文其它臂没做这件事），'
                 '就进入缩放规则的适用域：lr 要跟着放大（工业实践从线性 ×K 起步，配更长 warmup；'
                 '本机这一跑标定在 √K 与 K 之间），warmup 必须保留——大 lr 直接开步最容易炸，'
                 '见【A】的注入实验。')
    lines.append('    口径提醒（贯穿【B】【C】【F】）：saved_tensors_hooks 数出来的是'
                 '**autograd 为反向保存的中间激活**元素数')
    lines.append('      它不含参数、梯度、优化器状态，也不含推理侧的 KV Cache（那是 04 篇的账）；'
                 '用它做激活显存的代理量只在"同模型同 batch 同 seq 的相对比较"意义下成立')
    log_verify('累积 × effective batch × lr 缩放规则的三臂账', '真跑验证',
               '同样 Σlr 下大 batch 位移比 %.3f×；不缩放只有 %.3f×；本机等效缩放倍数 %.2f（K=%d）'
               % (r_scaled, r_plain, LR_SCALE_K / r_scaled, LR_SCALE_K))

    # ---------------- C. 梯度检查点
    ids_c = batches[0]
    ckpt_rows = checkpoint_compare(make_hand, ids_c)
    L = N_LAYERS
    saved_none = ckpt_rows['none']['saved']['elems']
    saved_man = ckpt_rows['manual']['saved']['elems']
    saved_off = ckpt_rows['official']['saved']['elems']
    lines.append('\n' + '=' * W)
    lines.append('【C】梯度检查点（activation checkpointing）：手写 ManualCheckpoint vs 官方 checkpoint')
    lines.append('=' * W)
    lines.append('    %-9s %-7s %-7s %-7s %-11s %-11s %-11s %-10s' %
                 ('模式', '前向', '重算', '合计', 'saved张量', 'saved元素', 'loss差', '梯度max|Δ|'))
    for m in ('none', 'manual', 'official'):
        r, c = ckpt_rows[m], ckpt_rows[m]['counters']
        lines.append('    %-9s %-7d %-7d %-7d %-11d %-11d %-11.3e %-10.3e' %
                     (m, c['fwd'], c['recompute'], c['fwd'] + c['recompute'],
                      r['saved']['pack'], r['saved']['elems'], r['loss_diff'], r['grad_diff']))
    lines.append('    整数账：不检查点 = L = %d 次；两种检查点都 = 2L = %d 次（前向 %d + 反向重算 %d）'
                 % (L, 2 * L, L, L))
    lines.append('    saved 元素数：%d → 手写 %d（降 %.1f%%）/ 官方 %d（降 %.1f%%）'
                 % (saved_none, saved_man, 100 * (1 - saved_man / saved_none),
                    saved_off, 100 * (1 - saved_off / saved_none)))
    lines.append('    理论上界：每块只存输入时 saved 元素 ≈ 1/L，L=%d → 降 %.1f%%；'
                 '实测官方降 %.1f%%（embedding/lm_head/末端 norm 的激活不在检查点范围内）'
                 % (L, 100 * (1 - 1.0 / L), 100 * (1 - saved_off / saved_none)))
    lines.append('    只改工程不改数学：手写与官方两条路的 loss 差 = %.3e / %.3e，'
                 '梯度 max|Δ| = %.3e / %.3e'
                 % (ckpt_rows['manual']['loss_diff'], ckpt_rows['official']['loss_diff'],
                    ckpt_rows['manual']['grad_diff'], ckpt_rows['official']['grad_diff']))
    log_verify('手写 ManualCheckpoint（torch.autograd.Function）', '真跑验证',
               '前向 L 次 / 反向重算 L 次，梯度与不检查点 max|Δ|=%.1e' % ckpt_rows['manual']['grad_diff'])
    log_verify('torch.utils.checkpoint(use_reentrant=False)', '真跑验证',
               '内部前向调用计数 = 2L = %d，梯度 max|Δ|=%.1e' % (2 * L, ckpt_rows['official']['grad_diff']))

    # ---------------- D. 混合精度
    mp_rows = cast_table([70000.0, 65504.0, 65520.0, 1.0 + 2 ** -8, 1.0 + 2 ** -9, 1e-8])
    fp32_run = arms['clip']                    # 与【A】的 clip 臂同配置，直接复用不重跑
    bf16_run = train_loop(make_hand(), batches, clip=CLIP, autocast_bf16=True)
    mp_diff = abs(fp32_run['last'] - bf16_run['last'])
    lines.append('\n' + '=' * W)
    lines.append('【D】混合精度：为什么生产用 bf16 而不是 fp16')
    lines.append('=' * W)
    lines.append('    %-13s %-13s %-13s %-8s %-8s %-11s' %
                 ('fp32 值', 'cast fp16', 'cast bf16', 'fp16上溢', 'fp16下溢', 'bf16相对误差'))
    for r in mp_rows:
        lines.append('    %-13.9g %-13.9g %-13.9g %-8s %-8s %-11.3e' %
                     (r['value'], r['fp16'], r['bf16'],
                      '是' if r['fp16_overflow'] else '否',
                      '是' if r['fp16_underflow'] else '否', r['bf16_relerr']))
    lines.append('    fp16：尾数 10 位、指数 5 位，上限 = %.0f、最小正规数 = %.3e（小于它就开始下溢）'
                 % (torch.finfo(torch.float16).max, torch.finfo(torch.float16).tiny))
    lines.append('    bf16：尾数 7 位、指数与 fp32 同宽 8 位，上限 = %.4e（比 fp16 高 %d 个数量级）'
                 % (torch.finfo(torch.bfloat16).max,
                    int(round(math.log10(torch.finfo(torch.bfloat16).max
                                         / torch.finfo(torch.float16).max)))))
    lines.append('    真跑 %d 步：fp32 臂末 loss = %.6f，bf16 autocast 臂末 loss = %.6f，差 = %.3e'
                 % (STEPS, fp32_run['last'], bf16_run['last'], mp_diff))
    lines.append('    原始范数峰值：fp32 臂 %.6f，bf16 臂 %.6f —— 同量级，bf16 不需要 GradScaler；'
                 'fp16 因为指数范围窄必须配 loss scaling'
                 % (fp32_run['peak_gnorm'], bf16_run['peak_gnorm']))
    lines.append('    取舍：bf16 用"尾数少 3 位"换"指数范围与 fp32 一致"。训练里的爆炸风险来自'
                 '范围（上溢/下溢）而不是精度，所以生产默认 bf16（承接生产篇 03 的伏笔）')
    sub = fp16_subnormal_probe()
    lines.append('    下溢侧的精确边界（本机用 numpy 的 fp16 cast 真数，不是引用公式）：')
    lines.append('      最小正规数 = %.6e = 2^−14；最小**亚**正规数 = %.6e = 2^−24 —— '
                 '真正决定"多小的数变成 0"的是后者'
                 % (sub['smallest_normal'], sub['smallest_subnormal']))
    lines.append('      fp16(1e-8) = %.1f（低于亚正规数下限 → 直接 0）、fp16(6e-8) = %.6e（正好落在下限）'
                 % (sub['fp16_of_1e8'], sub['fp16_of_6e8']))
    lines.append('      torch.optim.AdamW 的默认 eps = %.1e，而 fp16(eps) = %.1f —— '
                 '**eps 自己在 fp16 里也是 0**（eps < 亚正规下限：%s）'
                 % (sub['eps_default'], sub['fp16_of_eps'], sub['eps_below_subnormal']))
    lines.append('      于是 fp16 下的分母塌缩链是：√v̂ 下溢 0 + eps 下溢 0 → 分母 = %.1f，'
                 'm̂/0 在 IEEE 里给 %s —— 不是"梯度变小了"，是这一步直接变成 inf/NaN。'
                 % (sub['denominator_collapse'], sub['division_result']))
    lines.append('      所以"1e-8 这种小梯度被 fp16 清零"和"Adam 的 eps 在 fp16 里失效"是**同一枚硬币的'
                 '两面**（同一个 2^−24 下限），不是两个恰好同量级的独立问题；'
                 'bf16 的指数与 fp32 同宽，最小正规数 %.4e、最小亚正规数 %.4e，'
                 '1e-8 与 eps 都远在其上，两个问题一起消失'
                 % (sub['bf16_tiny'], sub['bf16_min_sub']))
    log_verify('bf16 autocast（CPU）真跑 %d 步 + fp16 上溢/下溢反面教材' % STEPS, '真跑验证',
               '末 loss 差 %.2e；70000→fp16 inf / bf16 %.0f；fp16 亚正规下限 %.3e，'
               'AdamW 默认 eps 在 fp16 里也 cast 成 0'
               % (mp_diff, mp_rows[0]['bf16'], sub['smallest_subnormal']))

    # ---------------- E. 断点续训
    pre_model = make_hand()
    head = train_loop(pre_model, batches, end_step=RESUME_AT, clip=CLIP)
    good_state = dict(model={k: v.clone() for k, v in pre_model.state_dict().items()},
                      opt=copy.deepcopy(head['_opt'].state_dict()))
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, 'ckpt_step%d.pt' % RESUME_AT)
        save_ckpt(ckpt_path, pre_model, head['_opt'], RESUME_AT, gen_states[RESUME_AT])
        loaded = torch.load(ckpt_path, weights_only=False)
        ckpt_keys = sorted(loaded.keys())
        ckpt_bytes = os.path.getsize(ckpt_path)
        assert torch.equal(loaded['gen_state'], gen_states[RESUME_AT]), '数据流状态回读不一致'
    base_full = arms['clip']                        # 与 good 臂同配置的不中断基线
    resume = resume_variants(make_hand, batches, STEPS, RESUME_AT, good_state)
    base_win = base_full['raw_losses'][RESUME_AT:STEPS]

    def win_diff(ls):
        vals = [abs(a - b) for a, b in zip(ls, base_win)
                if math.isfinite(a) and math.isfinite(b)]
        return max(vals) if vals else float('inf')

    lines.append('\n' + '=' * W)
    lines.append('【E】断点续训：第 %d 步中断，四种续法各跑完剩下的 %d 步' % (RESUME_AT, STEPS - RESUME_AT))
    lines.append('=' * W)
    lines.append('    落盘 checkpoint 的键：%s' % ', '.join(ckpt_keys))
    lines.append('    checkpoint 文件大小 = %.1f KB（tiny 模型；生产上它随参数量线性膨胀，见【F】）'
                 % (ckpt_bytes / 1024))
    lines.append('    %-9s %-16s %-14s %-14s' % ('续法', '恢复窗口max|Δ|', '窗口末loss', '与基线末loss差'))
    for tag in ('good', 'no_opt', 'no_step', 'no_data'):
        last = resume[tag][-1]
        lines.append('    %-9s %-16.3e %-14.6f %-14.3e' %
                     (tag, win_diff(resume[tag]), last, abs(last - base_full['last'])))
    lines.append('    正解（good）与不中断基线在恢复窗口内 max|Δ| = %.1e（%d 个 step 全比，逐位复现）'
                 % (win_diff(resume['good']), STEPS - RESUME_AT))
    lines.append('    反面教材一（no_opt）：optimizer 的 m/v 从零重建，等效于又一次冷启动，'
                 '窗口 max|Δ| = %.3e' % win_diff(resume['no_opt']))
    lines.append('    反面教材二（no_step）：调度器从 warmup 重来，lr 掉回 %.2e（本该 %.2e），'
                 '窗口 max|Δ| = %.3e'
                 % (lr_at(0, WARMUP, STEPS, LR, LR * MIN_LR_RATE),
                    lr_at(RESUME_AT, WARMUP, STEPS, LR, LR * MIN_LR_RATE),
                    win_diff(resume['no_step'])))
    lines.append('    反面教材三（no_data）：数据流从头重放 —— 恢复窗口喂的是 batch[0:%d] 而不是 '
                 'batch[%d:%d]，' % (STEPS - RESUME_AT, RESUME_AT, STEPS))
    lines.append('      其中 batch[0:%d] 被第二次使用，batch[%d:%d] 一次都没见过。'
                 '窗口 max|Δ| = %.3e（四个臂里最大），'
                 % (RESUME_AT, STEPS - RESUME_AT, STEPS, win_diff(resume['no_data'])))
    lines.append('      末 loss %.6f vs 基线 %.6f（%+.2f%%）：曲线看着还在降，'
                 '但它降的不是同一条曲线 —— 这是"进度丢失"最隐蔽的样子'
                 % (resume['no_data'][-1], base_full['last'],
                    100 * (resume['no_data'][-1] / base_full['last'] - 1)))
    rngl = rng_state_ledger()
    lines.append('    RNG 完备性账（本机 CPU 单进程能真数的部分）：')
    lines.append('      本机 torch CPU RNG 状态条数 = %d，单条 %d 字节；'
                 '实测一次 randn 之后头 8 字节不变（%s）、只有后面 %d 个字节变 → '
                 '只存 seed 再 manual_seed 接不上流，必须整份存'
                 % (rngl['torch_rng_states_cpu'], rngl['torch_state_bytes'],
                    rngl['torch_state_head_unchanged'], rngl['torch_state_changed_bytes']))
    lines.append('      numpy 状态 = %d 个 uint32、python random 版本 = %d（两者也都是整份存取）'
                 % (rngl['numpy_state_words'], rngl['py_rng_version']))
    lines.append('      CUDA：本机 torch.cuda.is_available() = %s → 每设备一份 RNG 状态'
                 '（torch.cuda.get_rng_state_all() 返回逐设备列表）这一条本机量不到，标未验证'
                 % rngl['cuda_available'])
    lines.append('      DataLoader 多 worker：按 torch 官方算式重算（_worker_init_fn 里 '
                 'base_seed 从 loader.generator 抽一个 int64，然后 seed = base_seed + worker_id）：'
                 'base_seed = %d → %d 个 worker 的 seed = %s（互不相同：%s）'
                 % (rngl['dataloader_base_seed'], rngl['n_workers'], rngl['worker_seeds'],
                    rngl['worker_seeds_distinct']))
    if rngl['worker_np_seeds'] is not None:
        lines.append('        numpy 侧还要再走一道 _generate_state(base_seed, worker_id)：'
                     '%s' % rngl['worker_np_seeds'])
    lines.append('        每个 epoch 还会重抽 base_seed（本机连抽 3 次 = %s，互不相同：%s）→ '
                 '"哪个 batch 落到哪个 worker"是随 epoch 变化的'
                 % (rngl['epoch_bases'], rngl['epoch_bases_distinct']))
    lines.append('        结论：本机这七件套（%s）对**CPU 单进程**是完备的；'
                 '一旦上 GPU 或多 worker，还要各自加"逐设备 RNG 状态"和'
                 '"loader.generator 状态 + 各 worker 种子"两类，缺一不可逐位复现。'
                 % ', '.join(ckpt_keys))
    lines.append('    边界说明：本文只管 checkpoint **内容**的完备性（存什么才能逐位复现），'
                 '不涉及保存频率 / step 轮换 / 异步写盘 / 分布式一致性这些保存**策略**问题')
    log_verify('checkpoint 内容的 RNG 完备性（CPU 单进程真数 + 官方 worker 播种算式重算）',
               '真跑验证', '主进程 torch 状态 %d 字节；worker seed = base_seed+id，%d 个互不相同'
               % (rngl['torch_state_bytes'], rngl['n_workers']))
    log_verify('CUDA 逐设备 RNG 状态、DataLoader worker 进程内实测', '未验证',
               '本机 CPU（cuda.is_available=False）；沙箱里真起 worker 会立刻退出'
               '（RuntimeError: worker exited unexpectedly），故只按 torch 官方算式重算')
    log_verify('断点续训逐位复现 + 三个反面教材', '真跑验证',
               'model/optimizer/step/数据流位置全存时窗口 max|Δ|=%.1e' % win_diff(resume['good']))

    # ---------------- F. 显存账本
    big_params = big_param_count()
    formula_params = big_param_formula(BIG_CFG['vocab_size'], BIG_CFG['hidden_size'],
                                       BIG_CFG['num_hidden_layers'], BIG_CFG['intermediate_size'])
    act_scale = (BIG_CFG['num_hidden_layers'] / N_LAYERS) * (BIG_CFG['hidden_size'] / DIM)
    ledger = memory_ledger(big_params, saved_none, saved_off, act_scale)
    lines.append('\n' + '=' * W)
    lines.append('【F】训练 OOM：显存账本（tiny 实测 + meta device 真数 7B 结构参数量）')
    lines.append('=' * W)
    lines.append('    7B 结构 config（vocab=%d d=%d L=%d ffn=%d heads=%d）在 meta device 上真数：'
                 % (BIG_CFG['vocab_size'], BIG_CFG['hidden_size'], BIG_CFG['num_hidden_layers'],
                    BIG_CFG['intermediate_size'], BIG_CFG['num_attention_heads']))
    lines.append('      实测参数量 = %d，闭式 = %d，差 = %d（整数精确）'
                 % (big_params, formula_params, big_params - formula_params))
    lines.append('    状态项（ZeRO 论文混合精度口径 16 字节/参数，引用口径）：')
    lines.append('      bf16 权重 2P = %.1f GB + bf16 梯度 2P = %.1f GB'
                 % (ledger['bf16_weights_bytes'] / 1024 ** 3, ledger['bf16_grads_bytes'] / 1024 ** 3))
    lines.append('      + fp32 主权重 4P = %.1f GB + m/v 8P = %.1f GB → 合计 16P = %.1f GB'
                 % (ledger['fp32_master_bytes'] / 1024 ** 3, ledger['adam_m_v_bytes'] / 1024 ** 3,
                    ledger['total_16P_gb']))
    lines.append('      P = %.3fB（本机 meta device 真数，不是引用数字）' % (big_params / 1e9))
    lines.append('    激活项（本机实测口径）：')
    lines.append('      tiny（batch=%d seq=%d L=%d d=%d）：不检查点 saved %d 元素 = %.2f MB，'
                 '检查点 %d 元素 = %.2f MB（降 %.1f%%）'
                 % (BATCH, SEQ, N_LAYERS, DIM, saved_none, saved_none * 4 / 1024 ** 2,
                    saved_off, saved_off * 4 / 1024 ** 2, 100 * (1 - saved_off / saved_none)))
    lines.append('      外推到 7B 结构（同 batch/seq，按 层数比 %.0f× × hidden 比 %.0f× = %.0f× '
                 '线性放大，估算口径）：不检查点 %.1f GB → 检查点 %.1f GB'
                 % (BIG_CFG['num_hidden_layers'] / N_LAYERS, BIG_CFG['hidden_size'] / DIM,
                    act_scale, ledger['big_act_plain_gb_est'], ledger['big_act_ckpt_gb_est']))
    lines.append('      注意这条外推是线性的，真实激活还含注意力的 S² 项（04 篇的 FlashAttention '
                 '正是为它而生），所以这里是下界口径')
    lines.append('      口径（与【B】【C】同一条）：这里的激活项只算 autograd 为反向保存的中间量，'
                 '参数/梯度/优化器状态走上面对应的 2P/2P/8P 项，不重复计；'
                 '推理侧的 KV Cache 不属于训练显存账（那是 04 篇的账本）')
    lines.append('    OOM 的定位顺序（生产口径）：先算 16P 状态项是否超过单卡显存 → 再算激活项'
                 '（batch/seq 是不是被拉长）→ 最后才怀疑碎片')
    lines.append('    诚实标注：本机是 CPU，没有真实显存分配，这一节量的是"账"；'
                 'ZeRO / FSDP / Megatron 的切分只给骨架，标未验证')
    log_verify('7B 结构参数量（meta device 真数）+ 闭式对账', '真跑验证',
               '%d == %d，整数精确' % (big_params, formula_params))
    log_verify('16 字节/参数的混合精度显存口径', '未验证',
               'ZeRO 论文引用口径；本机 CPU 无真实显存分配可测，只做算式换算')
    log_verify('激活显存外推到 7B 结构', '未验证',
               '线性估算口径，未计注意力 S² 项；tiny 的 saved 元素数是本机实测')
    log_verify('DeepSpeed ZeRO / FSDP / Megatron 切分', '未验证',
               '本机未安装，属番外篇；只给方向说明与骨架')

    # ---------------- G. 三大线上问题 + Trainer 真跑对照
    lines.append('\n' + '=' * W)
    lines.append('【G】三大线上问题清单（现象→定位→解决，全部有真跑证据）+ HF Trainer 真跑对照')
    lines.append('=' * W)
    lines.append('    1) loss spike')
    lines.append('       现象：loss 曲线单点尖峰，之后要么自己回落、要么一路 NaN')
    lines.append('       定位：看同一 step 的梯度总范数（Trainer 日志里的 grad_norm 就是它）——'
                 '本机实测注入步 %.6g vs 基线 %.6g'
                 % (arms['gain']['gnorms'][SPIKE_STEP], arms['clean']['gnorms'][SPIKE_STEP]))
    lines.append('       解决：clip_grad_norm_ 压幅度 + 非有限就跳步挡 NaN + 回滚到上一个 checkpoint '
                 '重放该段数据')
    lines.append('             本机实测：跳步臂末 loss %.6f vs clip 臂 %.6f；不跳步臂全程 NaN（%d 步）'
                 % (arms['skip']['last'], arms['clip']['last'], arms['nan']['nan_steps']))
    lines.append('    2) 训练 OOM')
    lines.append('       现象：CUDA out of memory，且往往在"换了更长序列 / 更大 batch"之后才出现')
    lines.append('       定位：先算 16P 状态项（%.1f GB），再算激活项；本机实测检查点把 saved 元素降 %.1f%%'
                 % (ledger['total_16P_gb'], 100 * (1 - saved_off / saved_none)))
    lines.append('       解决：梯度累积（省激活峰值，本机 K=%d 相对差 %.1e）→ 梯度检查点 → '
                 'ZeRO/FSDP 切状态 → 缩 batch/seq' % (ACCUM_K, accum['grad_rel']))
    lines.append('    3) 断点续训丢进度')
    lines.append('       现象：重启后 loss 曲线"接不上"——要么跳回冷启动水平，要么假性变低')
    lines.append('       定位：逐个核 checkpoint 里的键（本机落盘键：%s）' % ', '.join(ckpt_keys))
    lines.append('       解决：model + optimizer + step + 数据流位置 + RNG 全存；'
                 '本机实测全存时恢复窗口 max|Δ| = %.1e，少存任何一项都差 %.1e 以上'
                 % (win_diff(resume['good']),
                    min(win_diff(resume[t]) for t in ('no_opt', 'no_step', 'no_data'))))

    align, align_err = run_trainer_align(hf_cfg_dict, hf_weights, stream_rows, STEPS)
    trainer_ok = align is not None
    lines.append('')
    if trainer_ok:
        logs = align['base']['logs']
        tr_loss = [r['loss'] for r in logs]
        tr_gnorm = [r.get('grad_norm', float('nan')) for r in logs]
        tr_lr = [r.get('learning_rate', float('nan')) for r in logs]
        hand_hf = train_loop(make_hf(), batches, clip=CLIP,
                             forward_fn=lambda m, x: m(input_ids=x).logits)
        hand_tiny = train_loop(make_hand(), batches, clip=CLIP)
        # 反面教材：把 weight decay 施加到所有参数（含 RMSNorm 权重），复现"语义不一致"的量级差
        hand_wrong_decay = train_loop(make_hand(), batches, clip=CLIP, decay_mask=False)

        def dmax(a, b):
            n = min(len(a), len(b))
            vals = [abs(a[i] - b[i]) for i in range(n)
                    if math.isfinite(a[i]) and math.isfinite(b[i])]
            return max(vals) if vals else float('inf')

        loss_diff_hf = dmax(hand_hf['raw_losses'], tr_loss)
        loss_diff_tiny = dmax(hand_tiny['raw_losses'], tr_loss)
        gn_diff = dmax(hand_hf['gnorms'], tr_gnorm)
        lr_diff = dmax([lr_at(t, WARMUP, STEPS, LR, LR * MIN_LR_RATE) for t in range(STEPS)], tr_lr)
        early = max(abs(hand_tiny['raw_losses'][i] - tr_loss[i]) for i in range(10))
        late = max(abs(hand_tiny['raw_losses'][i] - tr_loss[i]) for i in range(STEPS - 10, STEPS))
        wrong_diff = dmax(hand_wrong_decay['raw_losses'], tr_loss)
        wrong_last = abs(hand_wrong_decay['raw_losses'][-1] - tr_loss[-1])
        tr_ckpt_diff = dmax(tr_loss, [r['loss'] for r in align['ckpt']['logs']])
        n_decay = len(align['base']['decay_names'])
        hf_names = [n for n, _ in make_hf().named_parameters()]
        n_all = len(hf_names)
        n_ours = len(decay_param_names(make_hf()))

        lines.append('    HF Trainer 真跑（accelerate 已安装；离线 config 随机初始化，不下载权重）：')
        lines.append('      TrainingArguments：optim=adamw_torch（显式关掉默认的 adamw_torch_fused）、'
                     'lr_scheduler_type=cosine_with_min_lr(min_lr_rate=%.2f)、warmup_steps=%d、'
                     'max_grad_norm=%.1f、weight_decay=%.2f' % (MIN_LR_RATE, WARMUP, CLIP, WD))
        lines.append('      train_sampling_strategy=sequential + dataloader_drop_last=True：'
                     '把同一批数据按同一顺序喂给两条循环，才能逐 step 对账')
        lines.append('      decay 掩码：Trainer 判定 %d/%d 个参数施加 weight decay，'
                     '手写掩码判定 %d 个，集合完全一致' % (n_decay, n_all, n_ours))
        lines.append('        （被排除的是 norm 与 bias；embed_tokens 与 lm_head 是被包含的——'
                     '这一点常被误解）')
        lines.append('      lr 调度：手写 lr_at(t) vs Trainer 逐步记录的 learning_rate，max|Δ| = %.3e'
                     % lr_diff)
        lines.append('        Trainer 口径的 warmup 是 lr = peak·t/warmup（t 为 0-based），'
                     '所以第 0 步 lr=0 是一个空步；手写循环若用 (t+1)/warmup 就会整体差一步')
        lines.append('      loss 逐 step 对账：')
        lines.append('        手写循环（HF 模型，同权重同数据同序） vs Trainer：max|Δ| = %.3e'
                     % loss_diff_hf)
        lines.append('        手写循环（手写 TinyLlama，同权重） vs Trainer：max|Δ| = %.3e'
                     % loss_diff_tiny)
        lines.append('        这个差没有随 step 放大：前 10 步 max|Δ| = %.3e → 后 10 步 %.3e。'
                     '原因是 cosine 把 lr 退火到 eta_min，' % (early, late))
        lines.append('        单步更新越来越小，前向实现差异（RoPE 缓存的求和顺序）没机会在优化器里复利；'
                     '换成恒定 lr 才会看到缓慢放大')
        lines.append('      反面教材：手写循环把 weight decay 施加到全部 %d 个参数（漏掉 decay 掩码），'
                     '其余完全相同' % n_all)
        lines.append('        vs Trainer：max|Δ| = %.3e，末 loss 差 = %.3e —— 比数值差大 %.0f 倍'
                     % (wrong_diff, wrong_last, wrong_diff / max(loss_diff_tiny, 1e-300)))
        lines.append('        这就是判据的分界线：语义错（掩码 / 调度 / 数据序）是量级差，'
                     '数值差（浮点求和顺序）只在小数点后第 6-7 位')
        lines.append('      grad_norm 逐 step 对账（手写 HF 臂 vs Trainer）：max|Δ| = %.3e' % gn_diff)
        lines.append('      Trainer 开 gradient_checkpointing=True 再跑一遍：loss 与不开时 '
                     'max|Δ| = %.3e —— 检查点只改显存不改数学（跨框架复证【C】）' % tr_ckpt_diff)
        lines.append('      Trainer 末 loss = %.6f，手写 HF 臂末 loss = %.6f，'
                     '手写 TinyLlama 臂末 loss = %.6f'
                     % (tr_loss[-1], hand_hf['raw_losses'][-1], hand_tiny['raw_losses'][-1]))
        lines.append('      判据总结：对不上时先看"掩码/调度/数据序"这三项语义，'
                     '再看"浮点求和顺序"这项数值；前者是量级差，后者是末位差')
        log_verify('HF Trainer 真跑逐 step 对账（loss / grad_norm / lr / decay 掩码）', '真跑验证',
                   'accelerate 已装；手写 HF 臂 loss max|Δ|=%.2e，手写 TinyLlama 臂 %.2e，'
                   '漏掉 decay 掩码则 %.2e' % (loss_diff_hf, loss_diff_tiny, wrong_diff))
        log_verify('Trainer gradient_checkpointing=True 与关闭时 loss 一致', '真跑验证',
                   'max|Δ|=%.2e' % tr_ckpt_diff)
    else:
        loss_diff_hf = loss_diff_tiny = gn_diff = lr_diff = tr_ckpt_diff = float('inf')
        early = late = wrong_diff = wrong_last = float('inf')
        tr_loss, hand_hf, hand_tiny = [], None, None
        hf_names = [n for n, _ in make_hf().named_parameters()]
        n_decay = n_ours = len(decay_param_names(make_hf()))
        n_all = len(hf_names)
        lines.append('    HF Trainer 真跑对照：本机跑不了（%s）→ 降级为调用骨架，标未验证' % align_err)
        for s in TRAINER_SKELETON.splitlines():
            lines.append('      ' + s)
        log_verify('HF Trainer 真跑逐 step 对账', '未验证', align_err)
        log_verify('Trainer gradient_checkpointing=True 与关闭时 loss 一致', '未验证', align_err)
    lines.append('')
    lines.append('    分布式（DeepSpeed ZeRO / FSDP / Megatron）调用骨架（本机未安装，标未验证）：')
    for s in DEEPSPEED_SKELETON.splitlines():
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

    with open(os.path.join(OUT, 'spike_clip.tsv'), 'w', encoding='utf-8') as f:
        f.write('arm\tclip\tskip_nonfinite\tspike_step_1based\tspike_kind\tpeak_gnorm_pre\t'
                'peak_gnorm_applied\tn_clipped\tloss_at_spike\traw_loss_at_spike\t'
                'upd_norm_at_spike\tlast_raw_loss\tnan_steps\tskipped\trecovery_steps\n')
        for tag, r in arms.items():
            kind = {'clean': 'none', 'clip': 'none',
                    'gain': 'gain%.0f' % SPIKE_GAIN, 'nan': 'inf', 'skip': 'inf'}[tag]
            f.write('%s\t%s\t%s\t%d\t%s\t%.6e\t%.6e\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%d\t%d\t%d\n'
                    % (tag, 'off' if clip_desc[tag] == 'off' else '%.1f' % CLIP,
                       str(tag == 'skip').lower(), SPIKE_STEP + 1, kind, r['peak_gnorm'],
                       r['peak_applied'], r['n_clipped'],
                       r['losses'][SPIKE_STEP], r['raw_losses'][SPIKE_STEP],
                       r['upd'][SPIKE_STEP], r['last'],
                       r['nan_steps'], r['skipped'], recovery_steps(r, SPIKE_STEP)))

    def jsonable(o):
        """把 NaN / ±inf 换成字符串，产物才是严格合法的 JSON（NaN 臂的末 loss 就是 nan）。"""
        if isinstance(o, dict):
            return {k: jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [jsonable(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return repr(o)
        return o

    with open(os.path.join(OUT, 'ckpt_resume.json'), 'w', encoding='utf-8') as f:
        json.dump(jsonable({'seed': SEED,
                   'config': dict(dim=DIM, n_layers=N_LAYERS, n_heads=N_HEADS, ffn=FFN,
                                  seq=SEQ, batch=BATCH, steps=STEPS, warmup=WARMUP, lr=LR,
                                  min_lr_rate=MIN_LR_RATE, wd=WD, clip=CLIP, accum_k=ACCUM_K,
                                  spike_step_1based=SPIKE_STEP + 1, spike_gain=SPIKE_GAIN,
                                  resume_at=RESUME_AT, vocab=VOCAB, n_params=N_PARAMS,
                                  decay_params=len(HAND_DECAY), no_decay_params=len(HAND_NODECAY)),
                   'arms': {t: dict(peak_gnorm_pre=r['peak_gnorm'],
                                    peak_gnorm_applied=r['peak_applied'],
                                    n_clipped=r['n_clipped'],
                                    peak_upd_norm=r['peak_upd'],
                                    upd_norm_at_spike=r['upd'][SPIKE_STEP],
                                    last_raw_loss=r['last'],
                                    nan_steps=r['nan_steps'], skipped=r['skipped'],
                                    gnorm_at_spike=r['gnorms'][SPIKE_STEP],
                                    loss_at_spike=r['losses'][SPIKE_STEP],
                                    recovery_steps=recovery_steps(r, SPIKE_STEP))
                            for t, r in arms.items()},
                   'lr_schedule_wsd': dict(
                       hf_align=dict(max_diff=wsd_hf['max_diff'],
                                     max_diff_from_warmup=wsd_hf['max_diff_from_warmup'],
                                     warmup_first_lr=wsd_hf['warmup_first_lr'],
                                     last=wsd_hf['last'], n_stable=wsd_hf['n_stable'],
                                     n_steps=wsd_hf['n_steps']),
                       resume_ledger=wsd_ledger,
                       arms=dict(cosine=dict(last=arms['clip']['last'],
                                             sum_lr=sum_lr_cos, hot_steps=hot_cos),
                                 wsd=dict(last=wsd_arm['last'], sum_lr=sum_lr_wsd,
                                          hot_steps=hot_wsd, stable_end=STEPS - 20),
                                 loss_diff=abs(wsd_arm['last'] - arms['clip']['last']))),
                   'adam_scale_breakdown': sc,
                   'clip_strategy': cs,
                   'lr_scaling': {t: dict(effective_batch=BATCH * (LR_SCALE_K if t != 'small' else 1),
                                          opt_steps=len(ls[t]['upd']), sum_lr=sum(ls[t]['lrs']),
                                          upd_total=sum(ls[t]['upd']), last=ls[t]['last'])
                                  for t in ('small', 'big_lr', 'big_lrK')},
                   'recovery': {t: dict(first=recovery_steps(arms[t], SPIKE_STEP),
                                        consec3=recovery_steps(arms[t], SPIKE_STEP, consec=3),
                                        bounce_back=bounce_back_steps(arms[t], SPIKE_STEP))
                                for t in ('gain', 'nan', 'skip')},
                   'accumulation': dict(param_diff=accum['param_diff'],
                                        grad_diff=accum['grad_diff'],
                                        grad_rel=accum['grad_rel'],
                                        param_rel=accum['param_rel'],
                                        update_abs=accum['update_abs'],
                                        gnorm=accum['gnorm'], gnorm_diff=accum['gnorm_diff'],
                                        saved_elems_small=saved_small['elems'],
                                        saved_elems_big=saved_big['elems']),
                   'gradient_checkpoint': {
                       m: dict(counters=ckpt_rows[m]['counters'], saved=ckpt_rows[m]['saved'],
                               loss=ckpt_rows[m]['loss'], grad_diff=ckpt_rows[m]['grad_diff'],
                               loss_diff=ckpt_rows[m]['loss_diff'])
                       for m in ('none', 'manual', 'official')},
                   'mixed_precision': dict(cast_table=mp_rows,
                                           fp16_max=float(torch.finfo(torch.float16).max),
                                           fp16_tiny=float(torch.finfo(torch.float16).tiny),
                                           bf16_max=float(torch.finfo(torch.bfloat16).max),
                                           fp32_last=fp32_run['last'], bf16_last=bf16_run['last'],
                                           fp32_peak_gnorm=fp32_run['peak_gnorm'],
                                           bf16_peak_gnorm=bf16_run['peak_gnorm'],
                                           fp16_subnormal=sub),
                   'resume': dict(ckpt_keys=ckpt_keys, ckpt_bytes=ckpt_bytes,
                                  rng_ledger=rngl,
                                  baseline_last=base_full['last'],
                                  window_maxdiff={t: win_diff(resume[t])
                                                  for t in ('good', 'no_opt', 'no_step', 'no_data')},
                                  window_last={t: resume[t][-1]
                                               for t in ('good', 'no_opt', 'no_step', 'no_data')}),
                   'memory': dict(big_cfg=BIG_CFG, big_params=big_params,
                                  formula_params=formula_params, ledger=ledger),
                   'trainer': (dict(ok=True, n_decay=n_decay, n_all=n_all, n_ours=n_ours,
                                    loss_diff_hand_hf=loss_diff_hf,
                                    loss_diff_hand_tiny=loss_diff_tiny,
                                    loss_diff_early=early, loss_diff_late=late,
                                    wrong_decay_diff=wrong_diff,
                                    wrong_decay_last_diff=wrong_last,
                                    wrong_decay_last=hand_wrong_decay['raw_losses'][-1],
                                    gnorm_diff=gn_diff, lr_diff=lr_diff,
                                    ckpt_on_off_diff=tr_ckpt_diff, trainer_last=tr_loss[-1],
                                    hand_hf_last=hand_hf['raw_losses'][-1],
                                    hand_tiny_last=hand_tiny['raw_losses'][-1])
                               if trainer_ok else dict(ok=False, err=align_err))}),
                  f, ensure_ascii=False, indent=2)

    secs = time.perf_counter() - t_main
    fig = draw(arms, ckpt_rows, resume)
    if fig:
        print('\n[配图] %s' % fig)
    print('\n    （本机 CPU 全程 %.1f 秒；秒表只打印不落盘，保证产物逐字节可复现）' % secs)

    # ---------------- 自校验
    print('\n' + '=' * W)
    print('【自校验】')
    print('=' * W)
    assert abs(lr_at(0, WARMUP, STEPS, LR, LR * MIN_LR_RATE)) < 1e-15, 'warmup 起点 lr 应为 0'
    assert abs(lr_at(WARMUP, WARMUP, STEPS, LR, LR * MIN_LR_RATE) - LR) < 1e-15, 'warmup 末端应为 peak'
    tail = lr_at(STEPS - 1, WARMUP, STEPS, LR, LR * MIN_LR_RATE)
    assert abs(tail - LR * MIN_LR_RATE) < 1e-5, '退火末端应趋近 eta_min：%.3e' % tail
    print('  1. lr 调度边界：lr(0)=0、lr(warmup)=peak=%.1e、lr(%d)=%.4e≈eta_min  通过'
          % (LR, STEPS - 1, tail))
    assert len(HAND_NODECAY) == 2 * N_LAYERS + 1, 'no-decay 应为 2L+1 个 norm：%s' % HAND_NODECAY
    assert all('norm' in n for n in HAND_NODECAY), 'no-decay 组里混进了非 norm 参数'
    assert len(HAND_DECAY) + len(HAND_NODECAY) == len(HAND_NAMES), '掩码划分应无遗漏无重复'
    print('  2. decay 掩码：手写模型 %d 个参数 = decay %d + no-decay %d（=2L+1 个 RMSNorm），'
          '无遗漏无重复  通过' % (len(HAND_NAMES), len(HAND_DECAY), len(HAND_NODECAY)))
    assert abs(arms['clean']['gnorms'][0] - arms['clip']['gnorms'][0]) < 1e-12, \
        '两种梯度范数口径第 1 步应完全一致'
    print('  3. 梯度范数两种口径互证：第 1 步「只测量」与「clip_grad_norm_ 返回值」差 = %.1e  通过'
          % abs(arms['clean']['gnorms'][0] - arms['clip']['gnorms'][0]))
    if trainer_ok:
        assert set(align['base']['decay_names']) == set(decay_param_names(make_hf())), \
            'decay 掩码与 Trainer.get_decay_parameter_names 不一致'
        print('  4. decay 掩码集合 == Trainer.get_decay_parameter_names（%d/%d 个施加 decay）  通过'
              % (n_decay, n_all))
        assert lr_diff <= 1e-12, 'lr 调度与 Trainer 不一致：%.3e' % lr_diff
        print('  5. lr 序列 vs Trainer 记录的 learning_rate：max|Δ| = %.1e（同口径同数值）  通过'
              % lr_diff)
        assert loss_diff_hf <= 1e-4, '手写循环（HF 模型）与 Trainer loss 差过大：%.3e' % loss_diff_hf
        assert gn_diff <= 1e-3, 'grad_norm 与 Trainer 差过大：%.3e' % gn_diff
        print('  6. 手写循环（HF 模型，同权重同数据同序）vs Trainer：loss max|Δ| = %.3e、'
              'grad_norm max|Δ| = %.3e  通过' % (loss_diff_hf, gn_diff))
        assert loss_diff_tiny <= 1e-5, '手写 TinyLlama 与 Trainer 的差应在浮点末位量级：%.3e' % loss_diff_tiny
        assert wrong_diff > 100 * loss_diff_tiny, \
            '漏掉 decay 掩码的差应远大于数值差：%.3e vs %.3e' % (wrong_diff, loss_diff_tiny)
        print('  7. 手写 TinyLlama vs Trainer：max|Δ| = %.3e（前 10 步 %.3e / 后 10 步 %.3e，'
              'cosine 退火使它不放大）；漏掉 decay 掩码则 %.3e —— 量级差  通过'
              % (loss_diff_tiny, early, late, wrong_diff))
        assert tr_ckpt_diff <= 1e-4, 'Trainer 开关 gradient_checkpointing 的 loss 应一致：%.3e' % tr_ckpt_diff
        print('  8. Trainer gradient_checkpointing=True vs False：loss max|Δ| = %.3e'
              '（只改显存不改数学）  通过' % tr_ckpt_diff)
    else:
        for i, what in ((4, 'decay 掩码集合 == Trainer.get_decay_parameter_names'),
                        (5, 'lr 序列 vs Trainer 记录的 learning_rate'),
                        (6, '手写循环 vs Trainer 的 loss / grad_norm 对账'),
                        (7, '手写 TinyLlama vs Trainer：数值差 vs 语义差（decay 掩码）'),
                        (8, 'Trainer gradient_checkpointing 开关一致性')):
            print('  %d. %s  跳过（%s）' % (i, what, align_err))
    ratio = arms['gain']['gnorms'][SPIKE_STEP] / arms['clean']['gnorms'][SPIKE_STEP]
    assert 0.5 * SPIKE_GAIN <= ratio <= 2.0 * SPIKE_GAIN, '注入未按 gain 放大梯度：%.2f×' % ratio
    print('  9. 坏 batch 注入生效：注入步梯度范数 %.6g / 基线 %.6g = %.2f×（gain=%.0f×）  通过'
          % (arms['gain']['gnorms'][SPIKE_STEP], arms['clean']['gnorms'][SPIKE_STEP],
             ratio, SPIKE_GAIN))
    assert arms['clip']['peak_applied'] <= CLIP + 1e-5, \
        '裁剪后实际范数不应超过阈值：%.8f' % arms['clip']['peak_applied']
    assert arms['clip']['n_clipped'] > 0, '本机这一跑应该真的有步被裁到'
    assert abs(arms['clip']['peak_applied'] - CLIP) <= 1e-5, \
        '被裁的步实测范数应贴着阈值（torch 的系数含 1e-6，故略小）：%.8f' % arms['clip']['peak_applied']
    assert arms['clip']['peak_gnorm'] > CLIP, 'clip_grad_norm_ 的返回值是裁剪前范数，应 > 阈值'
    print('  10. 梯度裁剪：裁剪前峰值 %.6f（>阈值 %.1f，返回值口径），裁剪后实测峰值 %.8f；'
          '被裁 %d/%d 步  通过'
          % (arms['clip']['peak_gnorm'], CLIP, arms['clip']['peak_applied'],
             arms['clip']['n_clipped'], STEPS))
    assert arms['nan']['nan_steps'] > 0 and not math.isfinite(arms['nan']['last']), \
        'NaN 注入后应全程污染（裁剪救不了 NaN）'
    print('  11. NaN 注入 + clip=%.1f：NaN 步数 %d、末 loss 非有限 —— 裁剪救不了 NaN  通过'
          % (CLIP, arms['nan']['nan_steps']))
    assert arms['skip']['skipped'] == 1 and math.isfinite(arms['skip']['last']), '跳步应恰好一次并救回'
    assert abs(arms['skip']['last'] - arms['clip']['last']) <= 5e-2, \
        '跳步臂末 loss 应回到 clip 臂附近：%.6f vs %.6f' % (arms['skip']['last'], arms['clip']['last'])
    print('  12. 有限性跳步：skipped=%d、末 loss %.6f（clip 臂 %.6f，差 %.2e）；'
          '注入步的裁剪把 inf 梯度变成 NaN（0×inf），跳步才挡得住  通过'
          % (arms['skip']['skipped'], arms['skip']['last'], arms['clip']['last'],
             abs(arms['skip']['last'] - arms['clip']['last'])))
    assert accum['grad_rel'] <= 1e-6, '梯度累积与大 batch 不等价（相对总范数）：%.3e' % accum['grad_rel']
    assert accum['param_diff'] <= 1e-4, '一次 step 后参数偏离过大：%.3e' % accum['param_diff']
    assert abs(saved_big['elems'] / saved_small['elems'] - ACCUM_K) <= 0.05 * ACCUM_K, \
        'saved 元素应随 batch 线性放大'
    print('  13. 梯度累积 K=%d == 一次 %d×batch：梯度相对差 %.3e、参数相对差 %.3e'
          '（绝对 max|Δ| = %.3e，被近零梯度元素主导）；saved 元素 %.3f× ≈ K  通过'
          % (ACCUM_K, len(big), accum['grad_rel'], accum['param_rel'], accum['param_diff'],
             saved_big['elems'] / saved_small['elems']))
    assert ckpt_rows['none']['counters'] == {'fwd': L, 'recompute': 0}, '不检查点不应重算'
    for m in ('manual', 'official'):
        c = ckpt_rows[m]['counters']
        assert c['fwd'] == L and c['recompute'] == L, '%s 前向次数应为 L/L：%s' % (m, c)
        assert ckpt_rows[m]['grad_diff'] <= 1e-6, '%s 梯度与不检查点不一致：%.3e' % (m, ckpt_rows[m]['grad_diff'])
    assert saved_off < saved_none and saved_man < saved_none, '检查点应显著减少 saved 元素'
    print('  14. 梯度检查点：两种实现都 L→2L 次前向（L=%d），梯度 max|Δ| 手写 %.1e / 官方 %.1e；'
          'saved 元素 %d → %d / %d  通过'
          % (L, ckpt_rows['manual']['grad_diff'], ckpt_rows['official']['grad_diff'],
             saved_none, saved_man, saved_off))
    assert mp_rows[0]['fp16_overflow'] and not mp_rows[0]['bf16_overflow'], '70000 应 fp16 上溢、bf16 不上溢'
    assert mp_rows[0]['bf16'] == 70144.0, 'bf16(70000) 应为 70144：%.1f' % mp_rows[0]['bf16']
    assert mp_rows[1]['fp16'] == 65504.0 and not mp_rows[1]['fp16_overflow'], '65504 应恰为 fp16 上限'
    assert mp_rows[2]['fp16_overflow'], '65520 应 fp16 上溢'
    assert mp_rows[5]['fp16_underflow'] and not mp_rows[5]['bf16_overflow'], '1e-8 应 fp16 下溢、bf16 不下溢'
    assert mp_diff <= 5e-2, 'bf16 臂与 fp32 臂差过大：%.3e' % mp_diff
    print('  15. 混合精度：70000→fp16 inf / bf16 %.0f；65504 为 fp16 上限、65520 上溢；1e-8→fp16 '
          '下溢为 0；bf16 autocast %d 步末 loss 差 %.3e  通过' % (mp_rows[0]['bf16'], STEPS, mp_diff))
    assert win_diff(resume['good']) <= 1e-9, '正确续训应逐位复现：%.3e' % win_diff(resume['good'])
    for tag in ('no_opt', 'no_step', 'no_data'):
        assert win_diff(resume[tag]) > 100 * max(win_diff(resume['good']), 1e-12), \
            '反面教材 %s 的偏差应远大于正解' % tag
    print('  16. 断点续训：全存时恢复窗口 max|Δ| = %.1e（%d 步逐位复现）；三个反面教材偏差 '
          '%.2e / %.2e / %.2e  通过'
          % (win_diff(resume['good']), STEPS - RESUME_AT, win_diff(resume['no_opt']),
             win_diff(resume['no_step']), win_diff(resume['no_data'])))
    assert big_params == formula_params, 'meta 真数参数量 != 闭式：%d vs %d' % (big_params, formula_params)
    assert ledger['total_16P_bytes'] == 16 * big_params, '16P 口径换算错'
    assert ledger['act_saved_ratio'] > 1.5, '检查点应显著降低激活账'
    print('  17. 显存账本：7B 结构 meta 真数 %d == 闭式 %d；16P = %.1f GB；激活 saved 比 %.2f×  通过'
          % (big_params, formula_params, ledger['total_16P_gb'], ledger['act_saved_ratio']))
    assert wsd_hf['max_diff_from_warmup'] <= 1e-15, \
        'WSD 稳定段+退火段应与 HF 逐位一致：%.3e' % wsd_hf['max_diff_from_warmup']
    assert set(wsd_hf['plateau']) == {LR}, '稳定段应有 %d 步严格等于 peak' % len(wsd_hf['plateau'])
    cl, ws_, wd_ = wsd_ledger['cosine'], wsd_ledger['wsd_stable'], wsd_ledger['wsd_decay']
    assert cl['past_changed'] > 0 and cl['fut_changed'] == cl['fut_steps'], \
        'cosine 改 total：退火段历史与未走步都该被改写（past_changed=%d，fut %d/%d）' \
        % (cl['past_changed'], cl['fut_changed'], cl['fut_steps'])
    assert ws_['past_changed'] == 0 and ws_['fut_changed'] < ws_['fut_steps'], \
        'WSD 稳定段延长 total：历史与稳定段都不该被改写（fut_changed=%d/%d）' \
        % (ws_['fut_changed'], ws_['fut_steps'])
    assert wd_['past_changed'] > 0 and wd_['fut_changed'] == wd_['fut_steps'], \
        'WSD 退火段改 decay_end：历史与未走步都会被改写（past_changed=%d）' % wd_['past_changed']
    print('  18. WSD 调度器：与 HF get_wsd_schedule 稳定段+退火段 max|Δ| = %.1e（warmup 段差 %.1e '
          '来自 HF 的下限口径）；续训账 cosine 未走步 %d/%d 全变且已走步 %d/%d 也被改写、'
          'WSD 稳定段历史 %d 步零变（未走只 %d/%d 变）、WSD 退火段历史 %d/%d 被改写  通过'
          % (wsd_hf['max_diff_from_warmup'], wsd_hf['max_diff'], cl['fut_changed'], cl['fut_steps'],
             cl['past_changed'], cl['past_steps'], ws_['past_steps'], ws_['fut_changed'],
             ws_['fut_steps'], wd_['past_changed'], wd_['past_steps']))
    assert cs['norm']['inf']['n_nan'] == 2 and not cs['norm']['inf']['all_finite'], \
        '按范数裁剪应把 2 个 inf 元素变成 NaN（0×inf）'
    assert cs['value']['inf']['all_finite'] and cs['value']['inf']['max_abs'] == CLIP, \
        '按值裁剪应把 inf 直接 clamp 成 ±%.1f' % CLIP
    assert cs['norm']['nan']['n_nan'] == 8 and cs['value']['nan']['n_nan'] == 1, \
        'NaN 扩散口径：norm 传染全部 8 个元素、value 只留原来那 1 个'
    assert abs(cs['norm']['finite']['direction_cos'] - 1.0) <= 1e-6, '按范数裁剪必须严格保方向'
    assert cs['value']['finite']['direction_cos'] < 0.99, '按值裁剪应可观测地改变方向'
    print('  19. 两种裁剪对非有限值的行为：norm/inf → %d 个 NaN（系数 0×inf），value/inf → 全部有限'
          '（max|g|=%.1f）；norm/nan 传染 %d 个元素、value/nan 只留 %d 个；有限大梯度的方向余弦 '
          'norm=%.6f vs value=%.6f  通过'
          % (cs['norm']['inf']['n_nan'], cs['value']['inf']['max_abs'],
             cs['norm']['nan']['n_nan'], cs['value']['nan']['n_nan'],
             cs['norm']['finite']['direction_cos'], cs['value']['finite']['direction_cos']))
    assert abs(sub['smallest_subnormal'] - 2.0 ** -24) <= 1e-20, 'fp16 最小亚正规数应为 2^-24'
    assert sub['fp16_of_eps'] == 0.0 and sub['eps_below_subnormal'], 'AdamW 默认 eps 在 fp16 里应下溢为 0'
    assert sub['fp16_of_1e8'] == 0.0 and sub['fp16_of_6e8'] == 2.0 ** -24, '1e-8→0、6e-8→2^-24'
    assert math.isinf(sub['division_result']), 'fp16 下分母塌缩为 0 时 m̂/0 应为 inf'
    print('  20. fp16 下溢边界：最小正规 2^−14=%.4e、最小亚正规 2^−24=%.4e；eps=%.0e → fp16 %.1f，'
          '与 1e-8→0 是同一个下限（同一枚硬币两面）  通过'
          % (sub['smallest_normal'], sub['smallest_subnormal'], sub['eps_default'],
             sub['fp16_of_eps']))
    assert sc['m_state_diff'] == 0.0, '两臂进入注入步时的 m/v 必须逐位相同：%.1e' % sc['m_state_diff']
    tol = 0.01 * SPIKE_GAIN
    assert abs(sc['grad_ratio_min'] - SPIKE_GAIN) <= tol and \
           abs(sc['grad_ratio_max'] - SPIKE_GAIN) <= tol, \
        'loss×%.0f 应使每个梯度元素都落在 ×%.0f 附近：实测 [%.6f, %.6f]' % (SPIKE_GAIN, SPIKE_GAIN,
                                                                           sc['grad_ratio_min'],
                                                                           sc['grad_ratio_max'])
    assert abs(sc['l2_ratio'] - upd_ratio) <= 1e-7 * upd_ratio, \
        '两条 L2 比值算法应在浮点求和顺序意义下互证：%.8f vs %.8f' % (sc['l2_ratio'], upd_ratio)
    assert abs(sc['cold_ratio'] - (1 - 0.9) * SPIKE_GAIN / math.sqrt((1 - 0.999) * SPIKE_GAIN ** 2)) \
        <= 1e-9, '冷元素闭式算错'
    assert sc['ratio_upd_active'][0] <= sc['l2_ratio_probe'] + 1e-9 <= sc['cold_ratio'] + 1e-9, \
        '同口径三明治：探针参数自己的 L2 应夹在它自己的逐元素 p10 与冷元素闭式之间：%.4f / %.4f / %.4f' \
        % (sc['ratio_upd_active'][0], sc['l2_ratio_probe'], sc['cold_ratio'])
    assert sc['l2_ratio_probe'] <= sc['ratio_upd_active'][1] + 1e-9, \
        '按干净臂幅度加权后，探针 L2 可以低于它自己的逐元素中位数（高比值元素权重小）：%.4f vs %.4f' \
        % (sc['l2_ratio_probe'], sc['ratio_upd_active'][1])
    assert sc['share_hi_clean'] < sc['share_hi_gain'], \
        '高比值那批元素应按干净臂幅度加权后才被 L2 压住（干净臂份额 %.4f 应小于放大臂份额 %.4f）' \
        % (sc['share_hi_clean'], sc['share_hi_gain'])
    band = max(abs(sc['grad_ratio_min'] - SPIKE_GAIN), abs(sc['grad_ratio_max'] - SPIKE_GAIN)) \
        / SPIKE_GAIN * 100.0
    print('  21. 尺度不变性逐元素账：起点 m/v 差 %.1e；梯度逐元素比值 ∈ [%.4f, %.4f]（全体落在 ×%.0f '
          '的 ±%.2f%% 浮点带内，所以"只有少数参数拿到大梯度"不成立）；同口径三明治 探针 p10 %.3f× ≤ '
          '探针 L2 %.3f× ≤ 冷元素闭式 %.3f×，且探针 L2 %.3f× < 它自己的中位数 %.3f×（L2 按干净臂幅度加权）；'
          '比值 > 冷元素闭式的 %d 个元素在干净臂更新平方和里只占 %.2f%%、放大臂占 %.2f%%；全局 L2（39 个参数）%.3f× '
          '与逐元素分布不同口径  通过'
          % (sc['m_state_diff'], sc['grad_ratio_min'], sc['grad_ratio_max'], SPIKE_GAIN,
             band, sc['ratio_upd_active'][0], sc['l2_ratio_probe'], sc['cold_ratio'],
             sc['l2_ratio_probe'], sc['ratio_upd_active'][1],
             sc['n_hi_cold'], 100.0 * sc['share_hi_clean'], 100.0 * sc['share_hi_gain'],
             sc['l2_ratio']))
    assert abs(sum(ls['small']['lrs']) - sum(ls['big_lrK']['lrs'])) <= 1e-15, \
        '线性缩放后 Σlr 应精确相等'
    assert abs(r_plain * LR_SCALE_K - r_scaled) / r_scaled <= 0.1, \
        '位移比应满足"Σlr 比 × s(batch)"的乘性结构：s/%d = %.4f vs 实测 %.4f' \
        % (LR_SCALE_K, r_scaled / LR_SCALE_K, r_plain)
    assert 1.0 < r_scaled <= LR_SCALE_K, '大 batch + lr×K 应超调但不至于超过 K：%.3f' % r_scaled
    assert 1.0 / LR_SCALE_K <= r_plain < 1.0, '大 batch 不缩放 lr 应欠学习量：%.3f' % r_plain
    assert 1.0 < LR_SCALE_K / r_scaled < LR_SCALE_K, '等效缩放倍数应落在 1 与 K 之间'
    print('  22. lr 缩放账：同样数据、同样前向次数、Σlr 与小臂精确相等时，effective batch ×%d 的臂'
          '多走 %.3f×（饱和比上升）；不缩放只有 %.3f×；乘性结构 s/%d=%.4f vs 实测 %.4f（差 %.1f%%）；'
          '本机标定的等效缩放 = K/s = %.2f（介于 1 与 K=%d 之间，√K=%.1f）  通过'
          % (LR_SCALE_K, r_scaled, r_plain, LR_SCALE_K, r_scaled / LR_SCALE_K, r_plain,
             100 * abs(r_plain - r_scaled / LR_SCALE_K) / (r_scaled / LR_SCALE_K),
             LR_SCALE_K / r_scaled, LR_SCALE_K, math.sqrt(LR_SCALE_K)))
    print('\n  全部断言通过。')
    print('\n[产物] 报表 / spike_clip.tsv / ckpt_resume.json / 配图 → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
