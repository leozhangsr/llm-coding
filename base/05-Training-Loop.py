#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础篇 05 · 训练循环：一步优化器更新到底在算什么

对应文章：同目录 05-Training-Loop.md
运行：    python3 05-Training-Loop.py
依赖：    仅标准库；matplotlib 可选（缺了自动跳过画图）
产物：    ./output/05-Training-Loop/ 下的报表、optimizer_steps.tsv、float_formats.tsv、
          train_curve.json、配图

延续前四篇的方法论——**从零手写、注入已知答案、拿答案对账**。上一篇（基础篇 04）解决的是
"生成时怎么不重算"，这一篇解决"训练时那一步更新是怎么来的"：

    loss.backward() 之后，优化器拿着梯度到底做了什么？
    为什么 bf16 不是"精度更差的 fp16"，而是另一个取舍？

报表分成七节，每一节都注入"能独立算出来的已知答案"来对账：

    A. 配置与口径：本篇全部超参，以及"什么才算已知答案"
    B. 优化器四连：SGD → momentum → Adam → AdamW，每一步都能纸笔验算
    C. 学习率调度：线性 warmup + cosine 退火，三个边界值可解析算出
    D. 浮点格式的位级实现：手写 fp32→fp16 / fp32→bf16（含 RNE 舍入），逐位打印
    E. 最小训练闭环：手写 AdamW 训一个线性回归，与正规方程闭式解对账
    F. 生产三件套的机制：梯度裁剪 / 梯度累积 / 梯度检查点（只讲机制，不碰框架）
    G. 自校验：手算数值例、闭式解、整数计数、位级比对，全绿才 exit 0

对账手段只有四种：手算数值例、闭式解、整数计数、位级比对。
不靠肉眼看图，也不靠"看起来差不多"。
"""

import json
import math
import os
import random
import struct
import sys
import time

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '05-Training-Loop')

W = 70

# ---- 【B】优化器：一元二次 f(w) = a·(w − w*)²，梯度可手算
A_QUAD = 1.0
W_STAR = 0.0
W0 = 2.0
LR_A = 0.1
STEPS_A = 12
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WD = 0.1

# ---- 【C】调度
TOTAL = 80
WARMUP = 10
PEAK = 3e-3
MIN_LR_RATE = 0.1

# ---- 【E】最小训练闭环
N_SAMPLE = 48
N_FEAT = 3
TRUE_W = [1.5, -2.0, 0.75]
TRUE_B = 0.25
NOISE = 0.05
STEPS_D = 300
WARMUP_D = 20
PEAK_D = 0.05
ETA_MIN_D = 1e-4
ABL_TOL = 1.01          # 消融指标：第几步把 loss 压到闭式解的 1.01 倍以内

# ---- 【F】三件套
CLIP_DEMO_G = [3.0, 4.0]
CLIP_DEMO_THR = 1.0
ACCUM_K = 4
CKPT_L = 4
CKPT_D = 4
CKPT_H = 8


# ---------------------------------------------------------------- 通用小工具
def l2(v):
    return math.sqrt(sum(x * x for x in v))


def max_abs(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def jsonable(o):
    """把 NaN / ±inf 换成字符串，产物才是严格合法的 JSON
       （fp16 上溢那一行的值就是 inf、相对误差就是 nan，直接 dump 会写出非法字面量）。"""
    if isinstance(o, float):
        return o if math.isfinite(o) else repr(o)
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


# ---------------------------------------------------------------- 【B】优化器
def quad_grad(w, a=A_QUAD, star=W_STAR):
    """f(w) = a·(w − w*)² 的导数 = 2a·(w − w*)。整个【B】的梯度都由它给出，可纸笔验算。"""
    return 2.0 * a * (w - star)


def sgd_run(w0, steps, lr, grad_fn):
    out, w = [], w0
    for _ in range(steps):
        g = grad_fn(w)
        w = w - lr * g
        out.append(w)
    return out


def momentum_run(w0, steps, lr, grad_fn, beta=0.9):
    """经典（heavy-ball）动量：v_t = β·v_{t−1} + g_t，w_t = w_{t−1} − lr·v_t。"""
    out, w, v = [], w0, 0.0
    for _ in range(steps):
        v = beta * v + grad_fn(w)
        w = w - lr * v
        out.append(w)
    return out


def adam_step(w, g, state, t, lr, beta1=BETA1, beta2=BETA2, eps=EPS,
              wd=0.0, decoupled=False):
    """一步 Adam / AdamW。decoupled=True 即 AdamW：权重衰减不进梯度、不参与 m/v 的估计，
       而是直接作用在参数上（与 PyTorch 的顺序一致：先衰减，再走 Adam 那一步）。"""
    m, v = state
    if wd and not decoupled:                 # Adam + L2：衰减混进梯度
        g = g + wd * w
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * g * g
    state[0], state[1] = m, v
    mh = m / (1.0 - beta1 ** t)
    vh = v / (1.0 - beta2 ** t)
    if wd and decoupled:
        w = w * (1.0 - lr * wd)
    return w - lr * mh / (math.sqrt(vh) + eps), mh, vh


def adam_run(w0, steps, lr, grad_fn, wd=0.0, decoupled=False):
    out, w, state, w0s = [], w0, [0.0, 0.0], w0
    for t in range(1, steps + 1):
        w, _, _ = adam_step(w, grad_fn(w), state, t, lr, wd=wd, decoupled=decoupled)
        out.append(w)
    del w0s
    return out


def adam_first_step_hand(w0, lr, a=A_QUAD, star=W_STAR):
    """手算 Adam 的第 1 步，作为注入的已知答案。
       g = 2a(w0−w*)；m1 = (1−β1)g；v1 = (1−β2)g²
       m̂ = m1/(1−β1) = g；v̂ = v1/(1−β2) = g²  →  √v̂ = |g|
       所以 w1 = w0 − lr·g/(|g|+eps) —— 第 1 步与梯度大小无关，只取决于符号。"""
    g = quad_grad(w0, a, star)
    m1 = (1.0 - BETA1) * g
    v1 = (1.0 - BETA2) * g * g
    mh = m1 / (1.0 - BETA1)
    vh = v1 / (1.0 - BETA2)
    return dict(g=g, m1=m1, v1=v1, mh=mh, vh=vh, sqrt_vh=math.sqrt(vh),
                w1=w0 - lr * mh / (math.sqrt(vh) + EPS),
                w1_limit=w0 - lr * (1.0 if g > 0 else -1.0))


def sgd_closed_form(w0, t, lr, a=A_QUAD, star=W_STAR):
    """SGD 在 f = a(w−w*)² 上的闭式解：w_t = w* + (1 − 2·lr·a)^t · (w0 − w*)。"""
    return star + (1.0 - 2.0 * lr * a) ** t * (w0 - star)


def momentum_constant_grad(steps, lr, g0, beta=0.9):
    """常梯度 g0 下的动量位移。两个都能纸笔算出来的量：
       第 T 步的单步位移 = lr·g0·(1−β^T)/(1−β) → T 大时 = lr·g0/(1−β)
       前 T 步的累计位移 = lr·g0·[T − β(1−β^T)/(1−β)]/(1−β)
       所以"动量把有效学习率放大 1/(1−β) 倍"说的是**单步**位移；
       累计平均会因为前几步还在爬坡而略低于这个值，两个数别混。"""
    disp_series, acc, v = [], 0.0, 0.0
    for _ in range(steps):
        v = beta * v + g0
        acc += lr * v
        disp_series.append(acc)
    step_last = lr * g0 * (1.0 - beta ** steps) / (1.0 - beta)
    cum_exact = lr * g0 * (steps - beta * (1.0 - beta ** steps) / (1.0 - beta)) / (1.0 - beta)
    return disp_series, dict(limit=lr * g0 / (1.0 - beta), step_last=step_last,
                             cum_exact=cum_exact)


# ---------------------------------------------------------------- 【C】调度
def lr_at(t, warmup, total, peak, eta_min):
    """线性 warmup + cosine 退火。口径与生产篇 05 / HF Trainer 一致：
         t < warmup           lr = peak · t / warmup          （t 为 0-based，所以 lr(0) = 0）
         warmup <= t < total  lr = eta_min + (peak−eta_min)/2 · (1 + cos(π·progress))
       progress = (t − warmup) / (total − warmup)，末端 progress→1 时 lr→eta_min。"""
    if warmup > 0 and t < warmup:
        return peak * t / warmup
    prog = (t - warmup) / max(1, total - warmup)
    prog = min(1.0, max(0.0, prog))
    return eta_min + 0.5 * (peak - eta_min) * (1.0 + math.cos(math.pi * prog))


def lr_constant(t, peak):
    return peak


def lr_cosine_only(t, total, peak, eta_min):
    return lr_at(t, 0, total, peak, eta_min)


# ---------------------------------------------------------------- 【D】浮点格式
def fp32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bits_to_fp32(u):
    return struct.unpack('<f', struct.pack('<I', u & 0xFFFFFFFF))[0]


def rne_shift(v, shift):
    """把整数 v 右移 shift 位，用 round-half-to-even（银行家舍入）决定进位。
       IEEE 754 的默认舍入模式就是它：正好落在中点时，取尾数末位为 0 的那一侧。"""
    if shift <= 0:
        return v << (-shift)
    q = v >> shift
    r = v & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    if r > half or (r == half and (q & 1)):
        q += 1
    return q


def encode_fp16(x):
    """fp32 → fp16（1 符号 + 5 指数 + 10 尾数，bias=15），含次正规与上溢。"""
    u = fp32_bits(x)
    s = (u >> 31) & 1
    e = (u >> 23) & 0xFF
    m = u & 0x7FFFFF
    if e == 0xFF:
        return (s << 15) | 0x7C00 | (0x200 if m else 0)
    e16 = e - 127 + 15
    if e16 >= 31:                                  # 上溢 → ±inf
        return (s << 15) | 0x7C00
    if e16 <= 0:                                   # 次正规：把隐含的 1 补回尾数再移
        if e16 < -10:
            return s << 15                         # 连最小次正规的一半都不到 → ±0
        m |= 0x800000
        return (s << 15) + rne_shift(m, 14 - e16)
    return (s << 15) + (e16 << 10) + rne_shift(m, 13)


def decode_fp16(u):
    s = (u >> 15) & 1
    e = (u >> 10) & 0x1F
    m = u & 0x3FF
    if e == 0x1F:
        return float('nan') if m else (float('-inf') if s else float('inf'))
    val = (m / 1024.0) * 2.0 ** -14 if e == 0 else (1.0 + m / 1024.0) * 2.0 ** (e - 15)
    return -val if s else val


def encode_bf16(x):
    """fp32 → bf16（1 符号 + 8 指数 + 7 尾数，bias=127）：指数位与 fp32 完全同宽，
       所以它就是"把 fp32 的低 16 位按 RNE 舍掉"，不需要重新定标，也不会因为量级而上溢。"""
    u = fp32_bits(x)
    if ((u >> 23) & 0xFF) == 0xFF:
        return (u >> 16) | (0x40 if (u & 0x7FFFFF) else 0)
    return ((u >> 16) + rne_shift(u & 0xFFFF, 16)) & 0xFFFF


def decode_bf16(u):
    return bits_to_fp32((u & 0xFFFF) << 16)


def split_bits(u, total, exp_bits, man_bits):
    return ((u >> (total - 1)) & 1,
            (u >> man_bits) & ((1 << exp_bits) - 1),
            u & ((1 << man_bits) - 1))


def native_fp16(x):
    """Python struct 的 'e' 格式就是 IEEE 754 half。它在上溢时**抛 OverflowError**，
       而不是安静地返回 inf —— 这正好和 bf16 的行为形成对照。"""
    try:
        return struct.unpack('<H', struct.pack('<e', x))[0], None
    except OverflowError as exc:
        return None, 'OverflowError: %s' % exc


def ulp_count(fmt):
    """同一个二进制指数区间里能表示多少个值 = 2^尾数位。这是"精度"的准确定义。"""
    return {'fp32': 2 ** 23, 'fp16': 2 ** 10, 'bf16': 2 ** 7}[fmt]


# ---------------------------------------------------------------- 【E】最小闭环
def make_regression(rng, n=N_SAMPLE, d=N_FEAT):
    """合成一份线性回归数据：y = w*·x + b* + 高斯噪声。纯合成、确定性、不联网。"""
    X, y = [], []
    for _ in range(n):
        row = [rng.uniform(-1.0, 1.0) for _ in range(d)]
        X.append(row)
        y.append(sum(wi * xi for wi, xi in zip(TRUE_W, row)) + TRUE_B
                 + rng.gauss(0.0, NOISE))
    return X, y


def augment(X):
    """把 bias 当成一个恒为 1 的特征，正规方程就能一次解出 [w, b]。"""
    return [row + [1.0] for row in X]


def mse_loss(wb, X, y):
    n = len(y)
    return sum((sum(c * x for c, x in zip(wb, row)) - t) ** 2
               for row, t in zip(X, y)) / n


def grad_raw_sum(wb, X, y):
    """未归一化的梯度累加和 Σ 2·r·x。单独抽出来是为了【F】的累积对账：
       只要样本遍历顺序一致，"K 次小求和再合并"与"一次大求和"就逐位相同。"""
    d = len(wb)
    acc = [0.0] * d
    for row, t in zip(X, y):
        r = sum(c * x for c, x in zip(wb, row)) - t
        for j in range(d):
            acc[j] += 2.0 * r * row[j]
    return acc


def mse_grad(wb, X, y):
    n = len(y)
    return [v / n for v in grad_raw_sum(wb, X, y)]


def gauss_jordan(A, b):
    """带部分主元的高斯-约当消元，解 A·x = b。用来算正规方程的闭式解。"""
    n = len(A)
    M = [A[i][:] + [b[i]] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-14:
            raise ValueError('矩阵奇异，第 %d 列主元为 0' % c)
        M[c], M[p] = M[p], M[c]
        pv = M[c][c]
        M[c] = [v / pv for v in M[c]]
        for r in range(n):
            if r == c:
                continue
            f = M[r][c]
            if f:
                M[r] = [vr - f * vc for vr, vc in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def normal_equation(X, y):
    """闭式解 wb* = (XᵀX)⁻¹ Xᵀy —— 这就是【E】要拿来对账的已知答案。"""
    d = len(X[0])
    AtA = [[sum(X[k][i] * X[k][j] for k in range(len(X))) for j in range(d)]
           for i in range(d)]
    Aty = [sum(X[k][i] * y[k] for k in range(len(X))) for i in range(d)]
    return gauss_jordan(AtA, Aty)


def train_adamw(wb0, X, y, steps, warmup, peak, eta_min, wd):
    """手写 AdamW + warmup/cosine 的最小训练闭环，逐步记 loss 与 lr。"""
    total = steps
    wb = wb0[:]
    state = [[0.0] * len(wb), [0.0] * len(wb)]
    losses, lrs, wbs = [], [], []
    for t in range(total):
        lr = lr_at(t, warmup, total, peak, eta_min)
        g = mse_grad(wb, X, y)
        m, v = state
        for j in range(len(wb)):
            mj = BETA1 * m[j] + (1.0 - BETA1) * g[j]
            vj = BETA2 * v[j] + (1.0 - BETA2) * g[j] * g[j]
            m[j], v[j] = mj, vj
            mh = mj / (1.0 - BETA1 ** (t + 1))
            vh = vj / (1.0 - BETA2 ** (t + 1))
            wb[j] = wb[j] * (1.0 - lr * wd) - lr * mh / (math.sqrt(vh) + EPS)
        losses.append(mse_loss(wb, X, y))
        lrs.append(lr)
        wbs.append(wb[:])
    return dict(wb=wb, losses=losses, lrs=lrs, wbs=wbs)


def steps_to_reach(losses, thr):
    """第几步第一次把 loss 压到 thr 以下（1-based）；整轮都没压到返回 -1。
       用整数计数当指标，比"末 loss 都差不多"更能分辨调度策略的代价。"""
    for i, v in enumerate(losses):
        if v <= thr:
            return i + 1
    return -1


# ---------------------------------------------------------------- 【F】三件套
def clip_by_norm(g, thr):
    """按总范数裁剪：范数超过阈值就整体等比缩放（保方向），否则原样返回。
       返回 (裁剪后梯度, 裁剪前范数, 裁剪后范数, 是否被裁)。"""
    n = l2(g)
    if n <= thr:
        return g[:], n, n, False
    s = thr / n
    return [x * s for x in g], n, l2([x * s for x in g]), True


def split_chunks(rows, k):
    """把样本按顺序切成 K 段连续块。必须连续：累积的加法顺序要与大 batch 一致，
       跨步取样（strided）会打乱顺序，浮点上就不再逐位相同。"""
    n = len(rows)
    q, r = divmod(n, k)
    out, i = [], 0
    for j in range(k):
        sz = q + (1 if j < r else 0)
        out.append(rows[i:i + sz])
        i += sz
    return out


def accumulate_grads(wb, micros, ys, mode):
    """K 个 micro-batch 各自算梯度、累积成一份，等价于一次大 batch。
       mode='raw'  累积「未归一化」的梯度和，最后统一除一次总样本数
       mode='mean' 每个 micro 先各自除以 n_k，再除以 K 后相加
       两种写法数学上同一个式子，浮点上不一样：只有 'raw' 能与大 batch 逐位相同，
       因为它的加法顺序与大 batch 完全一致（'mean' 多了 K 次除法，舍入误差就进来了）。"""
    k = len(micros)
    acc = [0.0] * len(wb)
    n_fwd = 0
    n_tot = 0
    for X, y in zip(micros, ys):
        n_fwd += 1
        n_tot += len(y)
        if mode == 'raw':
            g = grad_raw_sum(wb, X, y)
            acc = [a + gi for a, gi in zip(acc, g)]
        else:
            g = mse_grad(wb, X, y)
            acc = [a + gi / k for a, gi in zip(acc, g)]
    if mode == 'raw':
        acc = [a / n_tot for a in acc]
    return acc, n_fwd, n_tot


def big_grad(wb, X, y):
    return mse_grad(wb, X, y), 1


# ---- 梯度检查点：一条 L 块的小前馈链，前向/反向全手写
def make_chain_params(rng, L=CKPT_L, d=CKPT_D, h=CKPT_H):
    ps = []
    for _ in range(L):
        W1 = [[rng.uniform(-1, 1) / math.sqrt(d) for _ in range(d)] for _ in range(h)]
        b1 = [0.0] * h
        W2 = [[rng.uniform(-1, 1) / math.sqrt(h) for _ in range(h)] for _ in range(d)]
        b2 = [0.0] * d
        ps.append([W1, b1, W2, b2])
    return ps


def block_forward(x, p, store):
    """一块 = dense(d→h) + ReLU + dense(h→d)。store 收集"反向需要但正向已经算过"的中间量。"""
    W1, b1, W2, b2 = p
    z = [sum(W1[i][j] * x[j] for j in range(len(x))) + b1[i] for i in range(len(W1))]
    a = [v if v > 0.0 else 0.0 for v in z]
    o = [sum(W2[i][j] * a[j] for j in range(len(a))) + b2[i] for i in range(len(W2))]
    if store is not None:
        store.append(('block_in', x[:]))
        store.append(('pre_relu', z))
        store.append(('relu_out', a))
    return o


def chain_forward(x0, params, store=None, counter=None):
    x = x0
    for p in params:
        if counter is not None:
            counter['fwd'] += 1
        x = block_forward(x, p, store)
    return x


def block_backward(go, x_in, p):
    """给定输出梯度 go 与本块输入 x_in，返回 (对输入的梯度, 本块参数梯度)。"""
    W1, b1, W2, b2 = p
    h, d = len(W1), len(W1[0])
    z = [sum(W1[i][j] * x_in[j] for j in range(d)) + b1[i] for i in range(h)]
    a = [v if v > 0.0 else 0.0 for v in z]
    gW2 = [[go[i] * a[j] for j in range(h)] for i in range(d)]
    gb2 = go[:]
    ga = [sum(W2[i][j] * go[i] for i in range(d)) for j in range(h)]
    gz = [gai if zi > 0.0 else 0.0 for gai, zi in zip(ga, z)]
    gW1 = [[gz[i] * x_in[j] for j in range(d)] for i in range(h)]
    gb1 = gz[:]
    gx = [sum(W1[i][j] * gz[i] for i in range(h)) for j in range(d)]
    return gx, [gW1, gb1, gW2, gb2]


def chain_backward_plain(x0, params, target):
    """不检查点：前向把每块的中间量全存下来，反向直接取用。"""
    store = []
    counter = {'fwd': 0, 'recompute': 0}
    out = chain_forward(x0, params, store, counter)
    go = [2.0 * (o - t) for o, t in zip(out, target)]
    grads = []
    for i in range(len(params) - 1, -1, -1):
        x_in = store[i * 3][1]
        go, gp = block_backward(go, x_in, params[i])
        grads.append(gp)
    grads.reverse()
    loss = sum((o - t) ** 2 for o, t in zip(out, target))
    stored_scalars = sum(len(v) for _, v in store)
    return dict(loss=loss, grads=grads, gx=go, counter=counter,
                stored_scalars=stored_scalars, n_blocks=len(params))


def chain_backward_ckpt(x0, params, target):
    """检查点：前向只保留每块的输入，反向时把这一块的前向重算一遍。
       代价是块前向从 L 次变成 2L 次，收益是中间量（pre_relu / relu_out）不再驻留。"""
    inputs, counter = [], {'fwd': 0, 'recompute': 0}
    x = x0
    for p in params:
        counter['fwd'] += 1
        inputs.append(x[:])                     # 只存输入，不存中间量
        x = block_forward(x, p, None)
    out = x
    go = [2.0 * (o - t) for o, t in zip(out, target)]
    grads = []
    for i in range(len(params) - 1, -1, -1):
        counter['recompute'] += 1
        x_in = inputs[i]
        go, gp = block_backward(go, x_in, params[i])
        grads.append(gp)
    grads.reverse()
    loss = sum((o - t) ** 2 for o, t in zip(out, target))
    return dict(loss=loss, grads=grads, gx=go, counter=counter,
                stored_scalars=sum(len(v) for v in inputs), n_blocks=len(params))


def grad_flat(res):
    flat = list(res['gx'])
    for gp in res['grads']:
        gW1, gb1, gW2, gb2 = gp
        for row in gW1:
            flat.extend(row)
        flat.extend(gb1)
        for row in gW2:
            flat.extend(row)
        flat.extend(gb2)
    return flat


def deep_max_abs(a, b):
    if isinstance(a, list):
        return max(deep_max_abs(x, y) for x, y in zip(a, b))
    return abs(a - b)


# ---------------------------------------------------------------- 报表
def fmt_report(cfg, opt_rows, hand, inv_rows, decay_rows, sched, floats, fsum, train, hooks):
    lines = []
    lines.append('=' * W)
    lines.append('基础篇 05 · 训练循环报表（seed=%d）' % SEED)
    lines.append('=' * W)

    lines.append('\n【A】配置与口径')
    for s in cfg:
        lines.append('    ' + s)

    lines.append('\n【B】优化器四连：f(w) = %.1f·(w − %.1f)²，w0 = %.1f，lr = %.2f'
                 % (A_QUAD, W_STAR, W0, LR_A))
    lines.append('      step        SGD     momentum        Adam      AdamW     SGD闭式')
    for r in opt_rows:
        lines.append('    %6d %10.6f %12.6f %11.6f %10.6f %11.6f'
                     % (r['t'], r['sgd'], r['mom'], r['adam'], r['adamw'], r['sgd_cf']))
    lines.append('    手算对账（Adam 第 1 步）：')
    lines.append('      g = 2a(w0−w*) = %.1f；m1 = (1−β1)g = %.4f；v1 = (1−β2)g² = %.5f'
                 % (hand['g'], hand['m1'], hand['v1']))
    lines.append('      m̂ = m1/(1−β1) = %.4f；v̂ = v1/(1−β2) = %.4f；√v̂ = %.6f'
                 % (hand['mh'], hand['vh'], hand['sqrt_vh']))
    lines.append('      w1 = w0 − lr·m̂/(√v̂+eps) = %.10f，实跑 %.10f，差 %.1e'
                 % (hand['w1'], opt_rows[0]['adam'], abs(hand['w1'] - opt_rows[0]['adam'])))
    lines.append('      √v̂ == |g| 不是巧合：第 1 步的偏差校正把 m̂ 还原成 g、v̂ 还原成 g²，'
                 '所以 w1 = w0 − lr·sign(g)')
    lines.append('      → 第 1 步位移的闭式 = lr·g/(|g| + eps)，所以 |g| >> eps 时位移就是 lr'
                 '（Adam 的尺度不变性）；|g| 落到 eps 量级时 eps 开始吃掉位移')
    lines.append('        %-10s %-16s %-16s %-10s' % ('|g|', '实跑位移', '手算 lr·g/(|g|+eps)', '差'))
    for r in inv_rows:
        lines.append('        %-10.0e %-16.12f %-16.12f %-10.1e'
                     % (r['g'], r['disp'], r['hand'], abs(r['disp'] - r['hand'])))
    lines.append('        |g| = eps = %.0e 那一行位移恰好是 lr/2 = %.2f —— '
                 'eps 不是"防除零的小数"，它是位移的下限刻度' % (EPS, LR_A / 2))
    lines.append('    SGD 闭式对账：w_t = w* + (1 − 2·lr·a)^t·(w0 − w*)，公比 = %.2f'
                 % (1.0 - 2.0 * LR_A * A_QUAD))
    lines.append('      第 %d 步：实跑 %.10f，闭式 %.10f，max|Δ| = %.1e'
                 % (STEPS_A, opt_rows[-1]['sgd'], opt_rows[-1]['sgd_cf'],
                    abs(opt_rows[-1]['sgd'] - opt_rows[-1]['sgd_cf'])))
    lines.append('    AdamW 与 Adam（L2 混进梯度）在 wd = %.2f 时从第 1 步就分叉：'
                 '第 %d 步差 %.6f' % (WD, STEPS_A,
                                    abs(opt_rows[-1]['adam'] - opt_rows[-1]['adamw'])))
    lines.append('    把分叉点放到最优点上看：常函数（梯度恒为 0）、起点 w = 1.0，'
                 '此时任何位移都纯粹来自 weight decay')
    lines.append('      %-8s %-22s %-22s' % ('wd', 'AdamW 第 1 步位移', 'Adam+L2 第 1 步位移'))
    for r in decay_rows:
        lines.append('      %-8.2f %-22.10f %-22.10f' % (r['wd'], r['disp_adamw'], r['disp_l2']))
    lines.append('      AdamW 手算 = lr·wd·w = %s，与实跑逐位相同'
                 % '、'.join('%.10f' % r['hand_adamw'] for r in decay_rows))
    lines.append('      Adam+L2 的三次位移 = %s，全部贴着 lr = %.2f（偏离 < %.1e），与 wd 无关'
                 % ('、'.join('%.10f' % r['disp_l2'] for r in decay_rows), LR_A,
                    max(abs(r['disp_l2'] - LR_A) for r in decay_rows)))
    lines.append('      → L2 混进梯度后被 √v̂ 归一化掉了：位移 = lr·(wd·w)/|wd·w| = lr，'
                 '与 wd 大小无关；AdamW 的位移才真的随 wd 缩放。这就是"解耦"。')

    lines.append('\n【C】学习率调度：warmup=%d、total=%d、peak=%.1e、eta_min=peak×%.2f'
                 % (WARMUP, TOTAL, PEAK, MIN_LR_RATE))
    lines.append('        t   warmup+cosine   cosine-only     constant')
    for r in sched['rows']:
        lines.append('    %7d %15.6e %13.6e %12.6e' % (r['t'], r['wc'], r['co'], r['ct']))
    lines.append('    边界（可解析算出，不是拟合出来的）：')
    lines.append('      lr(0) = peak·0/warmup = %.1e' % sched['l0'])
    lines.append('      lr(warmup=%d) = eta_min + (peak−eta_min)/2·(1+cos 0) = peak = %.6e'
                 % (WARMUP, sched['lw']))
    lines.append('      lr(total−1=%d) = %.6e，eta_min = %.6e（progress=%.4f，差一步到 1）'
                 % (TOTAL - 1, sched['ltail'], PEAK * MIN_LR_RATE, sched['prog_tail']))

    lines.append('\n【D】浮点格式的位级实现：手写 fp32→fp16 / fp32→bf16（RNE 舍入）')
    lines.append('    位分配：fp32 = 1+8+23、fp16 = 1+5+10（bias 15）、bf16 = 1+8+7（bias 127）')
    lines.append('    同一指数区间内的可表示值个数 = 2^尾数位：fp32 %d、fp16 %d、bf16 %d'
                 % (ulp_count('fp32'), ulp_count('fp16'), ulp_count('bf16')))
    lines.append('      fp16       符号 指数      尾数   →  值              bf16  符号 指数      尾数   →  值')
    for r in floats:
        lines.append('    %-10.8g %4d %5d %9d   →  %-15.9g %8d %5d %9d   →  %-15.9g'
                     % (r['x'], r['h_s'], r['h_e'], r['h_m'], r['hv'],
                        r['b_s'], r['b_e'], r['b_m'], r['bv']))
    lines.append('    与 Python struct 的原生 fp16（格式码 \'e\'）逐位对账：%d/%d 个值位级相同'
                 % (fsum['native_match'], fsum['native_total']))
    lines.append('    上溢对照：x = %.0f 时 struct.pack(\'<e\', x) %s，'
                 '而手写编码器按 IEEE 754 返回 inf（0x%04X）'
                 % (fsum['ovf_x'], fsum['ovf_native'], fsum['ovf_mine']))
    lines.append('      bf16 对同一个值安静地表示成 %.0f —— 指数位与 fp32 同宽，'
                 '量级再大也不会因为"格式上限"而爆掉' % fsum['ovf_bf16'])

    lines.append('\n【E】最小训练闭环：%d 维特征 + bias，%d 条合成样本，噪声 σ=%.2f'
                 % (N_FEAT, N_SAMPLE, NOISE))
    lines.append('    真值 w* = [%s]，b* = %.2f'
                 % (', '.join('%.2f' % v for v in TRUE_W), TRUE_B))
    lines.append('    闭式解（正规方程 (XᵀX)⁻¹Xᵀy）= [%s]'
                 % ', '.join('%.6f' % v for v in train['closed_form']))
    lines.append('    手写 AdamW 跑 %d 步（warmup=%d、peak=%.2f、eta_min=%.1e、wd=%.1f）= [%s]'
                 % (STEPS_D, WARMUP_D, PEAK_D, ETA_MIN_D, train['wd'],
                    ', '.join('%.6f' % v for v in train['wb'])))
    lines.append('    max|Δ| = %.3e；loss：AdamW %.8f vs 闭式解 %.8f（差 %.3e）'
                 % (train['w_diff'], train['loss'], train['loss_cf'],
                    abs(train['loss'] - train['loss_cf'])))
    lines.append('    调度消融（同数据同 seed，只换 lr 曲线；指标 = 第几步把 loss 压到闭式解的 '
                 '%.2f 倍以内）：' % ABL_TOL)
    for r in train['abl']:
        lines.append('      %-16s 达标步数 = %-6s 末 loss = %.8f'
                     % (r['name'], r['steps'] if r['steps'] > 0 else '未达标', r['loss']))
    lines.append('    warmup 在这个良性二次问题上是纯代价：第 0 步 lr = 0，前 %d 步走得比 '
                 'no warmup 慢，达标晚 %d 步'
                 % (WARMUP_D, train['abl'][0]['steps'] - train['abl'][2]['steps']))
    lines.append('      它换来的是"大模型上第 0 步不炸"，那个收益只有生产规模的模型才测得出来'
                 '（见生产篇 05 的 loss spike 三臂）')
    lines.append('    wd = %.2f 那一臂永远达不到闭式解的 %.2f 倍：weight decay 是有偏的，'
                 '它把解往 0 缩（‖w‖ %.6f vs 闭式 %.6f），末 loss %.8f > %.8f'
                 % (WD, ABL_TOL, train['norm_wd'], train['norm_cf'],
                    train['loss_wd'], train['loss_cf']))

    lines.append('\n【F】生产三件套的机制（不含框架，只数整数与对浮点）')
    lines.append('    1) 梯度裁剪：g = [%s]，‖g‖ = %.1f，阈值 = %.1f'
                 % (', '.join('%.1f' % v for v in CLIP_DEMO_G), hooks['clip_pre'],
                    CLIP_DEMO_THR))
    lines.append('       → 缩放系数 = %.1f/%.1f = %.2f，裁后 = [%s]，‖·‖ = %.10f'
                 % (CLIP_DEMO_THR, hooks['clip_pre'], hooks['clip_scale'],
                    ', '.join('%.4f' % v for v in hooks['clip_g']), hooks['clip_post']))
    lines.append('       手算答案 [%.1f, %.1f]，max|Δ| = %.1e（实跑算 3×(1/5)、手算写 3×1/5，'
                 '浮点结合律差 1 ulp）；方向不变（夹角余弦 = %.10f）'
                 % (3.0 * 0.2, 4.0 * 0.2, hooks['clip_hand_diff'], hooks['clip_cos']))
    lines.append('    2) 梯度累积：%d 条样本，一次 %d 条大 batch 的梯度 = 基准'
                 % (hooks['n_tot'], hooks['n_tot']))
    lines.append('       %-26s %-8s %-12s %s' % ('累积写法', '前向次数', '梯度 max|Δ|', '说明'))
    lines.append('       %-26s %-8d %-12.1e %s'
                 % ('K=%d，每 micro 1 条' % hooks['n_tot'], hooks['accum_one_fwd'],
                    hooks['accum_one'], '加法顺序与大 batch 完全一致 → 逐位相同'))
    lines.append('       %-26s %-8d %-12.1e %s'
                 % ('K=%d，每 micro %d 条' % (ACCUM_K, hooks['micro_size']), hooks['accum_fwd'],
                    hooks['accum_chunk'], '块内先自求和，结合顺序变了 → 残差'))
    lines.append('       %-26s %-8d %-12.1e %s'
                 % ('K=%d，各自先除 n_k' % ACCUM_K, hooks['accum_fwd'],
                    hooks['accum_mean'], '多 K 次除法，舍入误差直接进来'))
    lines.append('       %-26s %-8d %-12.1e %s'
                 % ('K=%d，跨步取样' % ACCUM_K, hooks['accum_fwd'],
                    hooks['accum_strided'], '样本顺序被打乱 → 残差同量级'))
    lines.append('       结论：累积在数学上恒等于大 batch；在浮点上要"逐位等于"只有一条路 —— '
                 '让加法顺序与大 batch 完全一致')
    lines.append('       后三种写法的残差都停在 1 ulp 量级（‖g‖ = %.6f，相对差 %.1e），'
                 '训练上不可分辨，但不要拿它当"逐位相同"去写断言'
                 % (hooks['g_big_norm'], hooks['accum_chunk'] / hooks['g_big_norm']))
    lines.append('       前向次数 %d 次 vs 大 batch %d 次 —— 累积不省算力，'
                 '省的是同一时刻驻留的激活' % (hooks['accum_fwd'], hooks['big_fwd']))
    lines.append('    3) 梯度检查点：L=%d 块的 dense(%d→%d)+ReLU+dense(%d→%d) 链'
                 % (CKPT_L, CKPT_D, CKPT_H, CKPT_H, CKPT_D))
    lines.append('       %-10s %-8s %-8s %-8s %-12s %-12s' %
                 ('模式', '块前向', '重算', '合计', '驻留标量', '梯度max|Δ|'))
    for r in hooks['ckpt_rows']:
        lines.append('       %-10s %-8d %-8d %-8d %-12d %-12.1e' %
                     (r['mode'], r['fwd'], r['recompute'], r['fwd'] + r['recompute'],
                      r['stored'], r['gdiff']))
    lines.append('       整数账：L=%d → 2L=%d 次块前向；驻留标量 %d → %d（降 %.1f%%）'
                 % (CKPT_L, 2 * CKPT_L, hooks['ckpt_rows'][0]['stored'],
                    hooks['ckpt_rows'][1]['stored'],
                    100 * (1 - hooks['ckpt_rows'][1]['stored'] / hooks['ckpt_rows'][0]['stored'])))
    lines.append('       每块驻留 d + h + h = %d+%d+%d = %d 个标量（输入 / pre-ReLU / ReLU 输出），'
                 '检查点只留输入 %d 个'
                 % (CKPT_D, CKPT_H, CKPT_H, CKPT_D + 2 * CKPT_H, CKPT_D))
    lines.append('       注意这不是"1 − 1/3"：三份中间量尺寸不同（%d/%d/%d），'
                 '降幅取决于 d 与 h 的比例，h = %d·d 时恰好是 %.0f%%'
                 % (CKPT_D, CKPT_H, CKPT_H, CKPT_H // CKPT_D,
                    100 * (1 - CKPT_D / (CKPT_D + 2 * CKPT_H))))
    lines.append('       loss 差 = %.1e，梯度逐元素 max|Δ| = %.1e —— 检查点只改显存不改数学'
                 % (hooks['ckpt_loss_diff'], hooks['ckpt_grad_diff']))
    lines.append('\n配图：fig-05-training-loop.png（四联图，与上表同源同 seed）')
    lines.append('    ① 四个优化器在同一 f(w) 上的轨迹 + SGD 闭式解散点')
    lines.append('    ② warmup+cosine / cosine-only / constant 三条 lr 曲线')
    lines.append('    ③ fp32 / fp16 / bf16 的 ULP 随指数变化（对数轴，尾数位数的直接后果）')
    lines.append('    ④ 最小闭环的 loss 曲线 + 三条消融臂 + 正规方程闭式解水平线')
    return '\n'.join(lines)


# ---------------------------------------------------------------- 配图
def draw(opt_rows, sched, train, hooks):
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

    ts = [r['t'] for r in opt_rows]
    axes[0].plot(ts, [r['sgd'] for r in opt_rows], '-o', ms=3, color='#4a9d6a', label='SGD')
    axes[0].plot(ts, [r['mom'] for r in opt_rows], '-o', ms=3, color='#d08c4a', label='momentum')
    axes[0].plot(ts, [r['adam'] for r in opt_rows], '-o', ms=3, color='#cc5a5a', label='Adam')
    axes[0].plot(ts, [r['adamw'] for r in opt_rows], '-o', ms=3, color='#8888cc', label='AdamW')
    axes[0].plot(ts, [r['sgd_cf'] for r in opt_rows], 'x', ms=5, color='#333333',
                 label='SGD 闭式 0.8^t·2')
    axes[0].axhline(W_STAR, color='#999999', ls=':', lw=1)
    axes[0].set_title('优化器轨迹：f(w) = (w − 0)²，w0 = 2')
    axes[0].set_xlabel('step')
    axes[0].set_ylabel('w')
    axes[0].legend(fontsize=8)

    st = list(range(TOTAL))
    axes[1].plot(st, [lr_at(t, WARMUP, TOTAL, PEAK, PEAK * MIN_LR_RATE) for t in st],
                 '-', color='#4a9d6a', label='warmup + cosine')
    axes[1].plot(st, [lr_cosine_only(t, TOTAL, PEAK, PEAK * MIN_LR_RATE) for t in st],
                 '--', color='#d08c4a', label='cosine only')
    axes[1].plot(st, [lr_constant(t, PEAK) for t in st], ':', color='#cc5a5a', label='constant')
    axes[1].axvline(WARMUP, color='#999999', ls='-', lw=0.8)
    axes[1].axhline(PEAK * MIN_LR_RATE, color='#8888cc', ls='--', lw=1, label='eta_min')
    axes[1].set_title('lr 调度（warmup=%d / total=%d）' % (WARMUP, TOTAL))
    axes[1].set_xlabel('step t')
    axes[1].set_ylabel('lr')
    axes[1].legend(fontsize=8)

    fmts = ['fp32', 'fp16', 'bf16']
    mant = [23, 10, 7]
    exps = list(range(-14, 16))
    for name, mb, c in zip(fmts, mant, ('#4a9d6a', '#d08c4a', '#cc5a5a')):
        axes[2].plot(exps, [2.0 ** (e - mb) for e in exps], '-o', ms=3, color=c,
                     label='%s（尾数 %d 位）' % (name, mb))
    axes[2].set_yscale('log')
    axes[2].set_title('ULP：同一量级下相邻可表示值的间隔')
    axes[2].set_xlabel('二进制指数（值 ≈ 2^指数）')
    axes[2].set_ylabel('间隔 (log)')
    axes[2].legend(fontsize=8)

    axes[3].plot(range(len(train['losses'])), train['losses'], '-', color='#4a9d6a',
                 label='AdamW + warmup/cosine')
    curves = train['abl_curves']
    axes[3].plot(range(len(curves['const'])), curves['const'], '--',
                 color='#cc5a5a', label='constant lr')
    axes[3].plot(range(len(curves['nowarm'])), curves['nowarm'], ':',
                 color='#d08c4a', label='no warmup')
    axes[3].plot(range(len(curves['wd'])), curves['wd'], '-.',
                 color='#8888cc', label='wd=%.2f' % WD)
    axes[3].axhline(train['loss_cf'], color='#333333', ls='-', lw=1.2,
                    label='正规方程闭式解 loss')
    axes[3].axvline(WARMUP_D, color='#999999', ls='-', lw=0.8)
    axes[3].set_yscale('log')
    axes[3].set_title('最小训练闭环：%d 步线性回归' % STEPS_D)
    axes[3].set_xlabel('step')
    axes[3].set_ylabel('MSE (log)')
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig-05-training-loop.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    t_main = time.perf_counter()

    cfg = [
        '仅标准库（json/math/os/random/struct/sys/time），不联网、不下载任何权重',
        '优化器：f(w) = %.1f·(w − %.1f)²，w0 = %.1f，lr = %.2f，%d 步；'
        'β1 = %.2f、β2 = %.3f、eps = %.0e、wd = %.2f'
        % (A_QUAD, W_STAR, W0, LR_A, STEPS_A, BETA1, BETA2, EPS, WD),
        '调度：warmup = %d、total = %d、peak = %.1e、eta_min = peak×%.2f'
        % (WARMUP, TOTAL, PEAK, MIN_LR_RATE),
        '浮点：手写 RNE 舍入的 fp32→fp16 / fp32→bf16，与 struct \'e\' 逐位对账',
        '闭环：%d 维特征 + bias、%d 条样本、σ = %.2f、%d 步（warmup=%d、peak=%.2f）'
        % (N_FEAT, N_SAMPLE, NOISE, STEPS_D, WARMUP_D, PEAK_D),
        '三件套：裁剪 g=[%s] 阈值 %.1f；累积 K=%d 与 K=%d 两档；检查点 L=%d 块'
        % (', '.join('%.0f' % v for v in CLIP_DEMO_G), CLIP_DEMO_THR, ACCUM_K, N_SAMPLE,
           CKPT_L),
        '已知答案来源：手算数值例 / 闭式解（正规方程、SGD 公比）/ 整数计数 / 位级比对',
        '主指标 = 上述四类；秒表只打印到终端、不落盘，保证产物逐字节可复现',
    ]

    # ---------------- 【B】优化器四连
    g_fn = lambda w: quad_grad(w)
    sgd = sgd_run(W0, STEPS_A, LR_A, g_fn)
    mom = momentum_run(W0, STEPS_A, LR_A, g_fn, beta=0.9)
    adam = adam_run(W0, STEPS_A, LR_A, g_fn, wd=0.0)
    adamw = adam_run(W0, STEPS_A, LR_A, g_fn, wd=WD, decoupled=True)
    adam_l2 = adam_run(W0, STEPS_A, LR_A, g_fn, wd=WD, decoupled=False)
    hand = adam_first_step_hand(W0, LR_A)
    opt_rows = []
    for t in range(STEPS_A):
        opt_rows.append(dict(t=t + 1, sgd=sgd[t], mom=mom[t], adam=adam[t],
                             adamw=adamw[t], adam_l2=adam_l2[t],
                             sgd_cf=sgd_closed_form(W0, t + 1, LR_A)))

    # Adam 的尺度不变性，以及它的失效边界：第 1 步位移 = lr·g/(|g| + eps)
    # |g| >> eps 时 ≈ lr（与梯度大小无关）；|g| ~ eps 时 eps 开始吃掉位移。
    inv_rows = []
    for g0 in (1e-8, 1e-6, 1e-3, 4.0, 1e3, 1e6):
        st = [0.0, 0.0]
        w1, mh, vh = adam_step(0.0, g0, st, 1, LR_A)
        inv_rows.append(dict(g=g0, w1=w1, disp=abs(w1), mh=mh, vh=vh,
                             hand=LR_A * g0 / (abs(g0) + EPS)))

    # 动量的有效学习率：常梯度下「单步」位移 / (lr·g0) → 1/(1−β)
    MOM_STEPS = 200
    disp_series, mom = momentum_constant_grad(MOM_STEPS, LR_A, 1.0, beta=0.9)
    mom_step_ratio = (disp_series[-1] - disp_series[-2]) / (LR_A * 1.0)
    mom_avg_ratio = disp_series[-1] / (MOM_STEPS * LR_A * 1.0)

    # AdamW 与 Adam+L2 的分叉点：在"最优点"上看得最清楚。
    # 取一个常函数（梯度恒为 0），起点 w = 1.0 —— loss 已经无可再降，
    # 此时任何位移都纯粹来自 weight decay，没有梯度项来打掩护。
    zero_grad = lambda w: 0.0
    decay_rows = []
    for wd_try in (0.01, WD, 0.5):
        w_dec = adam_run(1.0, 1, LR_A, zero_grad, wd=wd_try, decoupled=True)[0]
        w_l2 = adam_run(1.0, 1, LR_A, zero_grad, wd=wd_try, decoupled=False)[0]
        decay_rows.append(dict(wd=wd_try, disp_adamw=1.0 - w_dec,
                               hand_adamw=LR_A * wd_try * 1.0, disp_l2=1.0 - w_l2))

    # ---------------- 【C】调度
    sched_rows = []
    for t in (0, 1, 5, 9, 10, 11, 20, 40, 60, 78, 79):
        sched_rows.append(dict(t=t,
                               wc=lr_at(t, WARMUP, TOTAL, PEAK, PEAK * MIN_LR_RATE),
                               co=lr_cosine_only(t, TOTAL, PEAK, PEAK * MIN_LR_RATE),
                               ct=lr_constant(t, PEAK)))
    eta_min = PEAK * MIN_LR_RATE
    sched = dict(rows=sched_rows,
                 l0=lr_at(0, WARMUP, TOTAL, PEAK, eta_min),
                 lw=lr_at(WARMUP, WARMUP, TOTAL, PEAK, eta_min),
                 ltail=lr_at(TOTAL - 1, WARMUP, TOTAL, PEAK, eta_min),
                 prog_tail=(TOTAL - 1 - WARMUP) / (TOTAL - WARMUP))

    # ---------------- 【D】浮点格式
    vals = [70000.0, 65504.0, 65520.0, 1.0 + 2 ** -7, 1.0 + 2 ** -8, 1.0 + 2 ** -11,
            1e-8, 2.0 ** -24, 0.1, -3.75, 1e30, 6.1e-5]
    floats, native_ok = [], 0
    for x in vals:
        hb = encode_fp16(x)
        bb = encode_bf16(x)
        hv, bv = decode_fp16(hb), decode_bf16(bb)
        nb, nerr = native_fp16(x)
        match = (nb == hb) if nb is not None else None
        if match:
            native_ok += 1
        h_s, h_e, h_m = split_bits(hb, 16, 5, 10)
        b_s, b_e, b_m = split_bits(bb, 16, 8, 7)
        floats.append(dict(x=x, fp32_bits=fp32_bits(x), h_bits=hb, h_s=h_s, h_e=h_e, h_m=h_m,
                           hv=hv, b_bits=bb, b_s=b_s, b_e=b_e, b_m=b_m, bv=bv,
                           native_bits=nb, native_err=nerr, native_match=match,
                           h_rel=(abs(hv - x) / abs(x) if hv == hv and x else float('nan')),
                           b_rel=(abs(bv - x) / abs(x) if x else float('nan'))))
    ovf_x = 70000.0
    ovf_native_err = native_fp16(ovf_x)[1]
    fsum = dict(native_match=native_ok, native_total=len(vals),
                ovf_x=ovf_x, ovf_native=(ovf_native_err or 'ok'),
                ovf_mine=encode_fp16(ovf_x),
                ovf_bf16=decode_bf16(encode_bf16(ovf_x)))

    # ---------------- 【E】最小闭环
    Xr, yr = make_regression(rng)
    Xa = augment(Xr)
    cf = normal_equation(Xa, yr)
    loss_cf = mse_loss(cf, Xa, yr)
    wb0 = [0.0] * (N_FEAT + 1)
    run_main = train_adamw(wb0, Xa, yr, STEPS_D, WARMUP_D, PEAK_D, ETA_MIN_D, 0.0)
    run_const = train_adamw(wb0, Xa, yr, STEPS_D, 0, PEAK_D, PEAK_D, 0.0)
    run_nowarm = train_adamw(wb0, Xa, yr, STEPS_D, 0, PEAK_D, ETA_MIN_D, 0.0)
    run_wd = train_adamw(wb0, Xa, yr, STEPS_D, WARMUP_D, PEAK_D, ETA_MIN_D, WD)
    train = dict(closed_form=cf, loss_cf=loss_cf, wb=run_main['wb'],
                 w_diff=max_abs(run_main['wb'], cf), loss=run_main['losses'][-1],
                 losses=run_main['losses'], lrs=run_main['lrs'],
                 init_loss=mse_loss(wb0, Xa, yr),
                 wd=0.0, wb_wd=run_wd['wb'], loss_wd=run_wd['losses'][-1],
                 norm_main=l2(run_main['wb']), norm_wd=l2(run_wd['wb']), norm_cf=l2(cf),
                 abl=[dict(name='warmup+cosine', loss=run_main['losses'][-1],
                           steps=steps_to_reach(run_main['losses'], loss_cf * ABL_TOL)),
                      dict(name='constant lr', loss=run_const['losses'][-1],
                           steps=steps_to_reach(run_const['losses'], loss_cf * ABL_TOL)),
                      dict(name='no warmup', loss=run_nowarm['losses'][-1],
                           steps=steps_to_reach(run_nowarm['losses'], loss_cf * ABL_TOL)),
                      dict(name='wd=%.2f' % WD, loss=run_wd['losses'][-1],
                           steps=steps_to_reach(run_wd['losses'], loss_cf * ABL_TOL))],
                 abl_curves=dict(main=run_main['losses'], const=run_const['losses'],
                                 nowarm=run_nowarm['losses'], wd=run_wd['losses']))

    # ---------------- 【F】三件套
    cg, cpre, cpost, ccut = clip_by_norm(CLIP_DEMO_G, CLIP_DEMO_THR)
    hand_clip = [3.0 * CLIP_DEMO_THR / 5.0, 4.0 * CLIP_DEMO_THR / 5.0]
    cos_sim = sum(a * b for a, b in zip(cg, CLIP_DEMO_G)) / (l2(cg) * l2(CLIP_DEMO_G))
    g_small = clip_by_norm([0.3, 0.4], CLIP_DEMO_THR)[0]

    micros = split_chunks(Xa, ACCUM_K)
    ys = split_chunks(yr, ACCUM_K)
    g_big, n_fwd_big = big_grad(wb0, Xa, yr)
    # K=4×12 条：每个 micro 内部先自行求和，再合并 —— 加法结合顺序与大 batch 不同
    g_chunk, n_fwd_chunk, n_tot = accumulate_grads(wb0, micros, ys, 'raw')
    # K=4，但每个 micro 先各自除以 n_k —— 多 K 次除法，舍入误差直接进来
    g_mean, _, _ = accumulate_grads(wb0, micros, ys, 'mean')
    # K=n，每个 micro 只有 1 条：累积的加法顺序与大 batch 逐行相加完全一致 → 必然逐位相同
    g_one, n_fwd_one, _ = accumulate_grads(wb0, [[row] for row in Xa], [[t] for t in yr], 'raw')
    # K=4，跨步取样（strided）：顺序被打乱，浮点上不再相同
    g_strided, _, _ = accumulate_grads(wb0, [Xa[i::ACCUM_K] for i in range(ACCUM_K)],
                                       [[yr[i] for i in range(len(yr)) if i % ACCUM_K == k]
                                        for k in range(ACCUM_K)], 'raw')

    params = make_chain_params(random.Random(SEED + 1))
    x0 = [rng.uniform(-1, 1) for _ in range(CKPT_D)]
    target = [rng.uniform(-1, 1) for _ in range(CKPT_D)]
    plain = chain_backward_plain(x0, params, target)
    ckpt = chain_backward_ckpt(x0, params, target)
    ckpt_rows = [dict(mode='不检查点', fwd=plain['counter']['fwd'],
                      recompute=plain['counter']['recompute'],
                      stored=plain['stored_scalars'], gdiff=0.0),
                 dict(mode='检查点', fwd=ckpt['counter']['fwd'],
                      recompute=ckpt['counter']['recompute'],
                      stored=ckpt['stored_scalars'],
                      gdiff=deep_max_abs(grad_flat(plain), grad_flat(ckpt)))]
    hooks = dict(clip_g=cg, clip_pre=cpre, clip_post=cpost, clip_cut=ccut,
                 clip_scale=CLIP_DEMO_THR / cpre, clip_hand_diff=max_abs(cg, hand_clip),
                 clip_cos=cos_sim, clip_untouched=g_small,
                 accum_chunk=max_abs(g_chunk, g_big), accum_mean=max_abs(g_mean, g_big),
                 accum_one=max_abs(g_one, g_big), accum_strided=max_abs(g_strided, g_big),
                 accum_fwd=n_fwd_chunk, accum_one_fwd=n_fwd_one, big_fwd=n_fwd_big,
                 micro_size=len(micros[0]), n_tot=n_tot, g_big_norm=l2(g_big),
                 ckpt_rows=ckpt_rows,
                 ckpt_loss_diff=abs(plain['loss'] - ckpt['loss']),
                 ckpt_grad_diff=deep_max_abs(grad_flat(plain), grad_flat(ckpt)))

    text = fmt_report(cfg, opt_rows, hand, inv_rows, decay_rows, sched, floats, fsum, train, hooks)
    print('\n' + text)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')

    with open(os.path.join(OUT, 'optimizer_steps.tsv'), 'w', encoding='utf-8') as f:
        f.write('step\tsgd\tsgd_closed_form\tmomentum\tadam\tadamw\tadam_l2\n')
        for r in opt_rows:
            f.write('%d\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\t%.12e\n'
                    % (r['t'], r['sgd'], r['sgd_cf'], r['mom'], r['adam'], r['adamw'],
                       r['adam_l2']))

    with open(os.path.join(OUT, 'float_formats.tsv'), 'w', encoding='utf-8') as f:
        f.write('value\tfp32_bits\tfp16_bits\tfp16_sign\tfp16_exp\tfp16_man\tfp16_value\t'
                'fp16_relerr\tbf16_bits\tbf16_sign\tbf16_exp\tbf16_man\tbf16_value\t'
                'bf16_relerr\tnative_fp16_bits\tnative_err\tbitwise_match\n')
        for r in floats:
            f.write('%.17g\t%d\t%d\t%d\t%d\t%d\t%.17g\t%.6e\t%d\t%d\t%d\t%d\t%.17g\t%.6e\t'
                    '%s\t%s\t%s\n'
                    % (r['x'], r['fp32_bits'], r['h_bits'], r['h_s'], r['h_e'], r['h_m'],
                       r['hv'], r['h_rel'], r['b_bits'], r['b_s'], r['b_e'], r['b_m'],
                       r['bv'], r['b_rel'],
                       'NA' if r['native_bits'] is None else str(r['native_bits']),
                       'NA' if r['native_err'] is None else r['native_err'].replace('\t', ' '),
                       'NA' if r['native_match'] is None else str(r['native_match']).lower()))

    with open(os.path.join(OUT, 'train_curve.json'), 'w', encoding='utf-8') as f:
        json.dump(jsonable({'seed': SEED,
                   'optimizer': dict(a=A_QUAD, w_star=W_STAR, w0=W0, lr=LR_A, steps=STEPS_A,
                                     beta1=BETA1, beta2=BETA2, eps=EPS, wd=WD,
                                     rows=opt_rows, hand_first_step=hand,
                                     scale_invariance=inv_rows,
                                     decay_at_optimum=decay_rows,
                                     sgd_ratio=1.0 - 2.0 * LR_A * A_QUAD,
                                     momentum=dict(steps=MOM_STEPS, beta=0.9,
                                                   step_ratio=mom_step_ratio,
                                                   avg_ratio=mom_avg_ratio,
                                                   limit=1.0 / (1.0 - 0.9),
                                                   cumulative=disp_series[-1],
                                                   cumulative_closed_form=mom['cum_exact'])),
                   'schedule': dict(warmup=WARMUP, total=TOTAL, peak=PEAK,
                                    min_lr_rate=MIN_LR_RATE, rows=sched_rows,
                                    boundaries=dict(lr0=sched['l0'], lr_warmup=sched['lw'],
                                                    lr_tail=sched['ltail'],
                                                    eta_min=eta_min)),
                   'float_formats': dict(rows=floats, summary=fsum,
                                         ulp_counts={k: ulp_count(k)
                                                     for k in ('fp32', 'fp16', 'bf16')}),
                   'train_curve': dict(n_feat=N_FEAT, n_sample=N_SAMPLE, noise=NOISE,
                                       true_w=TRUE_W, true_b=TRUE_B, steps=STEPS_D,
                                       warmup=WARMUP_D, peak=PEAK_D, eta_min=ETA_MIN_D,
                                       closed_form=cf, closed_form_loss=loss_cf,
                                       adamw=run_main['wb'], adamw_loss=run_main['losses'][-1],
                                       w_diff=train['w_diff'], losses=run_main['losses'],
                                       lrs=run_main['lrs'], ablation=train['abl']),
                   'hooks': dict(clip=dict(g=CLIP_DEMO_G, thr=CLIP_DEMO_THR, out=cg,
                                           pre_norm=cpre, post_norm=cpost, clipped=ccut,
                                           hand=hand_clip, diff=hooks['clip_hand_diff'],
                                           cos=cos_sim),
                                 accumulate=dict(k=ACCUM_K, n_sample=hooks['n_tot'],
                                                 micro_size=hooks['micro_size'],
                                                 diff_one_per_micro=hooks['accum_one'],
                                                 diff_chunk=hooks['accum_chunk'],
                                                 diff_mean=hooks['accum_mean'],
                                                 diff_strided=hooks['accum_strided'],
                                                 fwd_accum=n_fwd_chunk,
                                                 fwd_one=n_fwd_one, fwd_big=n_fwd_big),
                                 checkpoint=dict(L=CKPT_L, d=CKPT_D, h=CKPT_H,
                                                 rows=ckpt_rows,
                                                 loss_diff=hooks['ckpt_loss_diff'],
                                                 grad_diff=hooks['ckpt_grad_diff']))}),
                  f, ensure_ascii=False, indent=2, allow_nan=False)

    secs = time.perf_counter() - t_main
    fig = draw(opt_rows, sched, train, hooks)
    if fig:
        print('\n[配图] %s' % fig)
    print('\n    （本机全程 %.2f 秒；秒表只打印不落盘，保证产物逐字节可复现）' % secs)

    # ---------------- 【G】自校验
    print('\n' + '=' * W)
    print('【G】自校验（手算已知答案 + 闭式解 + 整数计数 + 位级比对）')
    print('=' * W)

    print('\n  1. Adam 第 1 步可手算：w1 = w0 − lr·sign(g)，与梯度大小无关')
    assert abs(hand['mh'] - hand['g']) < 1e-12, 'm̂ 应还原成 g：%.12f vs %.12f' % (hand['mh'], hand['g'])
    assert abs(hand['sqrt_vh'] - abs(hand['g'])) < 1e-9, '√v̂ 应等于 |g|'
    assert abs(opt_rows[0]['adam'] - hand['w1']) < 1e-12, '实跑第 1 步 != 手算'
    assert abs(opt_rows[0]['adam'] - (W0 - LR_A)) < 1e-7, 'w1 应 ≈ w0 − lr'
    for r in inv_rows:
        assert abs(r['disp'] - r['hand']) < 1e-15, \
            '第 1 步位移应等于闭式 lr·g/(|g|+eps)：%.15f vs %.15f' % (r['disp'], r['hand'])
    big = [r for r in inv_rows if r['g'] >= 1e-3]
    assert max(r['disp'] for r in big) - min(r['disp'] for r in big) < 1e-5, \
        '|g| >> eps 时位移应与 |g| 无关：%s' % [r['disp'] for r in big]
    eps_row = inv_rows[0]
    assert abs(eps_row['disp'] - LR_A / 2) < 1e-15, \
        '|g| = eps 时位移应恰为 lr/2：%.15f' % eps_row['disp']
    print('     g = %.1f → m̂ = %.4f、√v̂ = %.6f，w1 = %.10f（实跑 %.10f，差 %.1e）'
          % (hand['g'], hand['mh'], hand['sqrt_vh'], hand['w1'], opt_rows[0]['adam'],
             abs(hand['w1'] - opt_rows[0]['adam'])))
    print('     第 1 步位移闭式 lr·g/(|g|+eps)：|g| = %s 六个量级全部逐位相符（最大差 %.1e）'
          % ('、'.join('%.0e' % r['g'] for r in inv_rows),
             max(abs(r['disp'] - r['hand']) for r in inv_rows)))
    print('     |g| ≥ 1e-03 时位移都是 %.10f ≈ lr（尺度不变性，极差 %.1e）；'
          '|g| = eps = %.0e 时位移掉到 %.10f = lr/2'
          % (big[0]['disp'], max(r['disp'] for r in big) - min(r['disp'] for r in big),
             EPS, eps_row['disp']))

    print('\n  2. SGD 闭式解逐位对账 + AdamW 的"解耦"是可验证的结构差异')
    sgd_cf_diff = max(abs(r['sgd'] - r['sgd_cf']) for r in opt_rows)
    assert sgd_cf_diff < 1e-12, 'SGD 与闭式解不一致：%.3e' % sgd_cf_diff
    assert abs(opt_rows[-1]['sgd_cf'] - (0.8 ** STEPS_A) * W0) < 1e-12, '公比应为 1−2·lr·a = 0.8'
    dec_diff = max(abs(a - b) for a, b in zip(adamw, adam_l2))
    assert dec_diff > 1e-3, 'AdamW 与 Adam+L2 在 wd>0 时应显著分叉：%.3e' % dec_diff
    assert abs(opt_rows[0]['adamw'] - opt_rows[0]['adam_l2']) > 0, '第 1 步就该分叉'
    mom_step_exact = (1.0 - 0.9 ** MOM_STEPS) / (1.0 - 0.9)
    assert abs(mom_step_ratio - mom_step_exact) < 1e-12, \
        '常梯度下单步位移应等于闭式 (1−β^T)/(1−β)：%.15f vs %.15f' % (mom_step_ratio, mom_step_exact)
    assert abs(mom_step_ratio - 1.0 / (1.0 - 0.9)) < 1e-8, \
        '单步位移放大应 → 1/(1−β)：%.12f' % mom_step_ratio
    assert abs(disp_series[-1] - mom['cum_exact']) < 1e-12, \
        '累计位移应等于闭式：%.12f vs %.12f' % (disp_series[-1], mom['cum_exact'])
    for r in decay_rows:
        assert abs(r['disp_adamw'] - r['hand_adamw']) < 1e-15, \
            'AdamW 在最优点的位移应精确等于 lr·wd·w：%.15f vs %.15f' \
            % (r['disp_adamw'], r['hand_adamw'])
        assert abs(r['disp_l2'] - LR_A) < 1e-6, \
            'Adam+L2 在最优点的位移应恒为 lr、与 wd 无关：%.10f' % r['disp_l2']
    print('     SGD：w_t = w* + 0.8^t·(w0−w*)，%d 步逐位 max|Δ| = %.1e' % (STEPS_A, sgd_cf_diff))
    print('     AdamW vs Adam+L2（wd=%.2f）：第 %d 步差 %.6f —— 衰减是否进 m/v 估计，结果不同'
          % (WD, STEPS_A, abs(opt_rows[-1]['adamw'] - opt_rows[-1]['adam_l2'])))
    print('     最优点上的位移（梯度恒为 0、w=1.0）：wd = %s'
          % '、'.join('%.2f' % r['wd'] for r in decay_rows))
    print('       AdamW   = %s ← 手算 lr·wd·w 逐位相同，随 wd 缩放'
          % '、'.join('%.10f' % r['disp_adamw'] for r in decay_rows))
    print('       Adam+L2 = %s ← 全部贴着 lr = %.2f（最大偏离 %.1e），与 wd 无关'
          % ('、'.join('%.10f' % r['disp_l2'] for r in decay_rows), LR_A,
             max(abs(r['disp_l2'] - LR_A) for r in decay_rows)))
    print('     momentum：常梯度 %d 步，第 %d 步的单步位移放大 %.10f× = 1/(1−β) = %.1f×；'
          '前 %d 步的平均只有 %.4f×（前几步还在爬坡）'
          % (MOM_STEPS, MOM_STEPS, mom_step_ratio, 1.0 / (1.0 - 0.9),
             MOM_STEPS, mom_avg_ratio))
    print('     累计位移 %.10f vs 闭式 lr·g0·[T − β(1−β^T)/(1−β)]/(1−β) = %.10f，差 %.1e'
          % (disp_series[-1], mom['cum_exact'], abs(disp_series[-1] - mom['cum_exact'])))

    print('\n  3. lr 调度的三个边界都可解析算出')
    assert sched['l0'] == 0.0, 'lr(0) 应为 0：%.3e' % sched['l0']
    assert abs(sched['lw'] - PEAK) < 1e-15, 'lr(warmup) 应为 peak：%.6e' % sched['lw']
    assert abs(sched['ltail'] - eta_min) < 1e-5, 'lr(total−1) 应趋近 eta_min：%.6e' % sched['ltail']
    wc = [lr_at(t, WARMUP, TOTAL, PEAK, eta_min) for t in range(TOTAL)]
    assert all(wc[i] <= wc[i + 1] for i in range(WARMUP)), 'warmup 段应单调不降'
    assert all(wc[i] >= wc[i + 1] for i in range(WARMUP, TOTAL - 1)), '退火段应单调不增'
    assert max(wc) == PEAK and min(wc) == 0.0, '极值应为 peak 与 0'
    print('     lr(0) = %.1e、lr(%d) = %.6e == peak、lr(%d) = %.6e（eta_min = %.6e，progress = %.4f）'
          % (sched['l0'], WARMUP, sched['lw'], TOTAL - 1, sched['ltail'], eta_min,
             sched['prog_tail']))
    print('     单调性：%d 步 warmup 段不降、%d 步退火段不增；max = %.1e、min = %.1e'
          % (WARMUP, TOTAL - WARMUP, max(wc), min(wc)))

    print('\n  4. fp16 / bf16 位级实现与已知答案逐位相符')
    by_x = {r['x']: r for r in floats}
    assert by_x[70000.0]['h_bits'] == 0x7C00, '70000 的 fp16 应为 inf(0x7C00)'
    assert math.isinf(by_x[70000.0]['hv']), '70000 → fp16 应为 inf'
    assert by_x[70000.0]['bv'] == 70144.0, '70000 → bf16 应为 70144：%.1f' % by_x[70000.0]['bv']
    assert by_x[65504.0]['h_bits'] == 0x7BFF, '65504 应恰为 fp16 最大正规数'
    assert by_x[65520.0]['h_bits'] == 0x7C00, '65520 是 65504 与 2^16 的中点，RNE 取偶 → 上溢'
    assert by_x[1.0 + 2 ** -7]['bv'] == 1.0 + 2 ** -7, '1+2⁻⁷ 应被 bf16 精确表示'
    assert by_x[1.0 + 2 ** -8]['bv'] == 1.0, '1+2⁻⁸ 是 bf16 的中点，RNE 取偶 → 1.0'
    assert by_x[1.0 + 2 ** -8]['hv'] == 1.0 + 2 ** -8, '1+2⁻⁸ 应被 fp16 精确表示'
    assert by_x[1.0 + 2 ** -11]['hv'] == 1.0, '1+2⁻¹¹ 是 fp16 的中点，RNE 取偶 → 1.0'
    assert by_x[1e-8]['hv'] == 0.0, '1e-8 低于 fp16 最小次正规的一半 → 下溢为 0'
    assert abs(by_x[1e-8]['bv'] - 1e-8) / 1e-8 < 2 ** -7, 'bf16 应正常表示 1e-8'
    assert by_x[-3.75]['h_s'] == 1 and by_x[-3.75]['b_s'] == 1, '符号位应保留'
    assert by_x[-3.75]['hv'] == -3.75 and by_x[-3.75]['bv'] == -3.75, '-3.75 两种格式都精确'
    assert by_x[2.0 ** -24]['hv'] == 2.0 ** -24, '2⁻²⁴ 是 fp16 最小次正规数，应精确'
    matched = [r for r in floats if r['native_match'] is True]
    mismatched = [r for r in floats if r['native_match'] is False]
    assert len(mismatched) == 0, '与 struct e 的位级结果不一致：%s' % [r['x'] for r in mismatched]
    raised = [r['x'] for r in floats if r['native_err'] is not None]
    assert 70000.0 in raised and 65520.0 in raised, 'struct e 在上溢时应抛 OverflowError'
    print('     %d/%d 个值与 struct.pack(\'<e\') 位级完全相同；%d 个上溢值 struct 抛 '
          'OverflowError（%s）' % (len(matched), len(floats), len(raised),
                                 ', '.join('%.6g' % v for v in raised)))
    print('     70000 → fp16 %s / bf16 %.0f；65504 = fp16 上限(0x7BFF)；65520 中点取偶 → inf'
          % ('inf', by_x[70000.0]['bv']))
    print('     1+2⁻⁷ bf16 精确、1+2⁻⁸ bf16 → 1.0（尾数 7 位）；1+2⁻⁸ fp16 精确、'
          '1+2⁻¹¹ fp16 → 1.0（尾数 10 位）')
    print('     1e-8 → fp16 %.1f（下溢）、bf16 %.6e（相对误差 %.2e）；ULP 个数 %d / %d / %d'
          % (by_x[1e-8]['hv'], by_x[1e-8]['bv'], by_x[1e-8]['b_rel'],
             ulp_count('fp32'), ulp_count('fp16'), ulp_count('bf16')))

    print('\n  5. 最小闭环与正规方程闭式解对账 + 消融用整数步数说话')
    assert train['w_diff'] < 1e-3, '手写 AdamW 未收敛到闭式解：%.3e' % train['w_diff']
    assert abs(train['loss'] - loss_cf) < 1e-6, 'loss 未收敛到闭式解 loss：%.3e' % abs(train['loss'] - loss_cf)
    assert train['loss'] >= loss_cf - 1e-12, '闭式解是全局最优，AdamW 的 loss 不应更低'
    st_main, st_const, st_nowarm, st_wd = [r['steps'] for r in train['abl']]
    assert st_main > st_nowarm > 0, 'warmup 在良性问题上应更晚达标：%d vs %d' % (st_main, st_nowarm)
    assert abs(train['losses'][0] - train['init_loss']) < 1e-12, \
        'warmup 的第 0 步 lr = 0，loss 应原地不动：%.10f vs %.10f' \
        % (train['losses'][0], train['init_loss'])
    assert st_wd == -1, 'wd>0 的解有偏，不应达到闭式解的 %.2f 倍：%d' % (ABL_TOL, st_wd)
    assert train['loss_wd'] > train['loss_cf'], 'wd 臂末 loss 应高于全局最优'
    assert train['norm_wd'] < train['norm_main'] < train['norm_cf'] * 1.001, \
        'weight decay 应把解往 0 缩：%.6f vs %.6f' % (train['norm_wd'], train['norm_main'])
    print('     闭式解 = [%s]，AdamW %d 步 = [%s]，max|Δ| = %.3e'
          % (', '.join('%.6f' % v for v in cf), STEPS_D,
             ', '.join('%.6f' % v for v in run_main['wb']), train['w_diff']))
    print('     loss：AdamW %.8f vs 闭式 %.8f（差 %.3e，闭式是全局最优下界）'
          % (train['loss'], loss_cf, abs(train['loss'] - loss_cf)))
    print('     达标步数（loss ≤ %.2f×闭式）：%s'
          % (ABL_TOL, '、'.join('%s=%s' % (r['name'], r['steps'] if r['steps'] > 0 else '未达标')
                                for r in train['abl'])))
    print('     warmup 的第 0 步 lr = 0 → loss 原地 %.10f；no warmup 第 0 步就用满 lr，'
          'loss 立刻降到 %.10f' % (train['losses'][0], train['abl_curves']['nowarm'][0]))
    print('     wd = %.2f：‖w‖ %.6f < 无 wd %.6f（闭式 %.6f），末 loss %.8f > %.8f —— '
          '衰减是有偏的，不是"更好的收敛"'
          % (WD, train['norm_wd'], train['norm_main'], train['norm_cf'],
             train['loss_wd'], train['loss_cf']))

    print('\n  6. 三件套：裁剪手算精确、累积对浮点、检查点整数账 L→2L')
    assert max_abs(cg, hand_clip) < 1e-15, '裁剪结果应与手算 [0.6, 0.8] 相符：%.3e' \
        % max_abs(cg, hand_clip)
    assert abs(cpost - CLIP_DEMO_THR) < 1e-12, '裁后范数应精确等于阈值：%.12f' % cpost
    assert abs(cos_sim - 1.0) < 1e-12, '裁剪应保方向：cos = %.12f' % cos_sim
    assert g_small == [0.3, 0.4], '范数低于阈值时不应改动梯度'
    assert hooks['accum_one'] == 0.0, \
        '每 micro 1 条时累积应与大 batch 逐位相同：%.3e' % hooks['accum_one']
    assert n_fwd_one == hooks['n_tot'] and n_fwd_chunk == ACCUM_K and n_fwd_big == 1, '前向次数账不对'
    acc_scale = max(l2(g_big), 1e-30)
    for key in ('accum_chunk', 'accum_mean', 'accum_strided'):
        assert hooks[key] / acc_scale < 1e-14, \
            '%s 与大 batch 的相对差应在 1e-14 以下：%.3e' % (key, hooks[key] / acc_scale)
    assert plain['counter'] == {'fwd': CKPT_L, 'recompute': 0}, '不检查点不应重算'
    assert ckpt['counter'] == {'fwd': CKPT_L, 'recompute': CKPT_L}, '检查点应为 L→2L'
    assert hooks['ckpt_grad_diff'] == 0.0, '检查点改变了梯度：%.3e' % hooks['ckpt_grad_diff']
    assert hooks['ckpt_loss_diff'] == 0.0, '检查点改变了 loss：%.3e' % hooks['ckpt_loss_diff']
    assert plain['stored_scalars'] == CKPT_L * (CKPT_D + 2 * CKPT_H), '驻留标量账不对'
    assert ckpt['stored_scalars'] == CKPT_L * CKPT_D, '检查点应只驻留每块输入'
    print('     裁剪：g=[3,4]（‖·‖=5）阈值 1 → [%s]，手算 [0.6,0.8] max|Δ| = %.1e，'
          'cos = %.10f' % (', '.join('%.10f' % v for v in cg), hooks['clip_hand_diff'], cos_sim))
    print('           这 %.1e 不是实现错了，是结合律：实跑算 3×(1/5)，手算写 3×1/5，'
          '浮点上差 1 ulp' % hooks['clip_hand_diff'])
    print('     累积：%d 条样本。每 micro 1 条（K=%d）→ max|Δ| = %.1e 精确为 0；'
          '每 micro %d 条（K=%d）→ %.1e'
          % (hooks['n_tot'], n_fwd_one, hooks['accum_one'], hooks['micro_size'], ACCUM_K,
             hooks['accum_chunk']))
    print('           各自先除 n_k → %.1e；跨步取样 → %.1e（‖g‖ = %.6f，'
          '相对差都在 1e-14 以下）'
          % (hooks['accum_mean'], hooks['accum_strided'], l2(g_big)))
    print('           前向次数：累积 %d 次 vs 大 batch %d 次 —— 不省算力，省同时驻留的激活'
          % (n_fwd_chunk, n_fwd_big))
    print('     检查点：块前向 %d→%d 次（L→2L），驻留标量 %d→%d（每块 %d 份中间量 → 1 份输入），'
          '梯度 max|Δ| = %.1e'
          % (plain['counter']['fwd'], ckpt['counter']['fwd'] + ckpt['counter']['recompute'],
             plain['stored_scalars'], ckpt['stored_scalars'], 3, hooks['ckpt_grad_diff']))

    print('\n  7. 确定性：同 seed 重跑，全部产物数值逐位一致')
    rng2 = random.Random(SEED)
    X2, y2 = make_regression(rng2)
    cf2 = normal_equation(augment(X2), y2)
    r2 = train_adamw([0.0] * (N_FEAT + 1), augment(X2), y2, STEPS_D, WARMUP_D, PEAK_D,
                     ETA_MIN_D, 0.0)
    p2 = make_chain_params(random.Random(SEED + 1))
    ck2 = chain_backward_ckpt(x0, p2, target)
    assert (Xr, yr) == (X2, y2), '数据重建不一致'
    assert cf == cf2, '闭式解重建不一致'
    assert r2['wb'] == run_main['wb'], 'AdamW 轨迹重建不一致'
    assert grad_flat(ck2) == grad_flat(ckpt), '检查点梯度重建不一致'
    assert floats[0]['h_bits'] == encode_fp16(70000.0), '编码器不确定'
    print('     数据 / 闭式解 / %d 步 AdamW 轨迹 / 检查点梯度，两次构建全部逐位相同' % STEPS_D)

    print('\n  全部断言通过。')
    print('\n[产物] 报表 / optimizer_steps.tsv / float_formats.tsv / train_curve.json / 配图 → %s'
          % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
