#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础篇 04 · 自注意力与 KV Cache：多头、因果掩码、最简 KV Cache 与纯 Python benchmark

对应文章：同目录 04-Self-Attention-KVCache.md
运行：    python3 04-Self-Attention-KVCache.py
依赖：    仅标准库；matplotlib 可选（缺了自动跳过画图）
产物：    ./output/04-Self-Attention-KVCache/ 下的报表、benchmark.tsv、kv_cache.json、配图

延续数据工程 / 分词器 / Transformer 的方法论——**从零手写、注入已知答案、拿答案对账**。
上一篇（基础篇 03）手写的是单头、无位置编码、无掩码的一块 Transformer；这一篇把注意力
拆到能数清每一次乘加，回答一个生成阶段才会暴露的问题：

    生成第 128 个词，凭什么要把前 127 个词重算一遍？

五个核心概念，全部纯 Python 可数：

    A. 多头注意力：把 d_model 切成 n_heads 份，各头独立算，再拼回；H=1 时精确退化为单头
    B. 因果掩码：位置 i 只能看 j <= i；掩码必须置 -inf 而不是 0，否则权重被静默稀释
    C. 最简 KV Cache：把已经算过的 K / V 存下来，每步只算新 token 的一行
    D. 重算 vs 缓存的等价性：两条路径共用同一个 dot / matmul 内核，输出必须逐元素相同
    E. 操作计数 benchmark：主指标是确定性整数计数器，秒表只作参考（机器相关）

自校验全部拿"能独立算出来的已知答案"对账：手算数值例、闭式求和公式、逐元素 max|Δ|。
不靠肉眼看图，也不靠"看起来差不多"。
"""

import json
import math
import os
import random
import sys
import time

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '04-Self-Attention-KVCache')

# ---- 配置锁死：朴素路径的成本是 N 的三次方级，不要调大
D_MODEL = 16
N_HEADS = 4
DH = D_MODEL // N_HEADS          # 头维度 = 4
PROMPT = 4                       # 提示词长度 P：位置 0..P-1
N_MAX = 128                      # 生成到长度 N_MAX
GRID = (8, 16, 32, 64, 128)      # 打 checkpoint 的格点（不重跑，一次跑到底）

NEG_INF = float('-inf')
W = 70


# ---------------------------------------------------------------- 极简线代（带操作计数器）
def shape(A):
    if not isinstance(A, list):
        return ()
    if A and isinstance(A[0], list):
        return (len(A), len(A[0]))
    return (len(A),)


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def matmul(A, B, cnt=None):
    """A[n,m] × B[m,p] → [n,p]。cnt['mac'] += n·m·p（解析计数，由自校验 1 证明不是循环论证）。

    刻意不做 ch03 里那个 `if a == 0.0: continue` 的跳零优化：本篇要让"计数的"和"真跑的"
    严格一致，否则因果掩码那些 0 权重会让计数变成上界而不是实测值。
    """
    n, m = len(A), len(A[0])
    p = len(B[0])
    assert m == len(B), '矩阵乘法内维不匹配：%d vs %d' % (m, len(B))
    out = zeros(n, p)
    for i in range(n):
        Ai = A[i]
        oi = out[i]
        for k in range(m):
            a = Ai[k]
            Bk = B[k]
            for j in range(p):
                oi[j] += a * Bk[j]
    if cnt is not None:
        cnt['mac'] += n * m * p
    return out


def matmul_literal(A, B, cnt):
    """逐 MAC 递增的字面计数版：只用于自校验 1，证明 matmul 的解析计数 n·m·p 是对的。"""
    n, m = len(A), len(A[0])
    p = len(B[0])
    out = zeros(n, p)
    for i in range(n):
        for k in range(m):
            a = A[i][k]
            Bk = B[k]
            for j in range(p):
                out[i][j] += a * Bk[j]
                cnt['literal_mac'] += 1
    return out


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def dot(a, b, cnt=None):
    """唯一的向量点积内核：朴素路径与缓存路径共用同一个函数——这是二者能 bitwise 对账的前提。

    一次调用记 1 个 dots（向量运算次数）和 len(a) 个 mac（标量乘加次数）。
    """
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    if cnt is not None:
        cnt['dots'] += 1
        cnt['mac'] += len(a)
    return s


def axpy_rows(w, V, cnt=None):
    """ctx 行 = Σ_j w[j]·V[j]。每 (i,j) 记一次 head_dim 维向量加权累加。

    与 dot 一样，朴素路径和缓存路径走的是同一份代码、同一个求和顺序，所以两边结果逐位相同。
    """
    dh = len(V[0])
    out = [0.0] * dh
    for j in range(len(w)):
        if cnt is not None:
            cnt['dots'] += 1
            cnt['mac'] += dh
        wj = w[j]
        Vj = V[j]
        for t in range(dh):
            out[t] += wj * Vj[t]
    return out


def row_softmax(row):
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


def masked_row_softmax(row, upto):
    """因果 softmax：把 j > upto 的位置置 -inf，再对整行归一化。

    exp(-inf - m) = 0.0，所以被掩掉的位置权重**精确为 0.0**，而且行和仍然是 1。
    """
    r = [v if j <= upto else NEG_INF for j, v in enumerate(row)]
    m = max(r)
    ex = [math.exp(v - m) for v in r]
    s = sum(ex)
    return [e / s for e in ex]


def wrong_zero_mask_softmax(row, upto):
    """反面教材：把未来位置置 0.0（而不是 -inf）再对整行 softmax。

    exp(0) = 1，未来位置照样分到概率；行和仍然是 1，不报错、不断言失败——静默错。
    """
    r = [v if j <= upto else 0.0 for j, v in enumerate(row)]
    return row_softmax(r)


def mean_std(vals):
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return mu, math.sqrt(var)


def flat(X):
    return [v for row in X for v in row]


def max_abs_diff(A, B):
    return max(abs(A[i][j] - B[i][j]) for i in range(len(A)) for j in range(len(A[0])))


def new_cnt():
    return dict(mac=0, dots=0)


# ---------------------------------------------------------------- 多头 + 因果掩码
def make_linear(rng, din, dout, scale=1.0):
    W = [[rng.uniform(-scale, scale) for _ in range(dout)] for _ in range(din)]
    return W


def make_attention_params(rng, d_model, scale=1.0):
    """q / k / v / o 四个投影，全部无 bias（与 Llama 一致，也让操作计数干净）。"""
    return dict(Wq=make_linear(rng, d_model, d_model, scale),
                Wk=make_linear(rng, d_model, d_model, scale),
                Wv=make_linear(rng, d_model, d_model, scale),
                Wo=make_linear(rng, d_model, d_model, scale))


def head_slice(W, h, dh):
    """把 [L, d_model] 按头切出第 h 份 → [L, dh]。"""
    return [row[h * dh:(h + 1) * dh] for row in W]


def concat_heads(heads):
    """把 H 个 [L, dh] 拼回 [L, d_model]，映射必须是 out[h*dh + t] = heads[h][t]。"""
    L = len(heads[0])
    dh = len(heads[0][0])
    H = len(heads)
    return [[heads[h][i][t] for h in range(H) for t in range(dh)] for i in range(L)]


def concat_heads_wrong(heads):
    """反面教材：双层循环写成 for t ... for h ...，即 out[t*H + h] = heads[h][t]。
       切头按连续块 h·dh:(h+1)·dh，拼却按交错位置放 → 与 head_slice 不再互逆。
       不抛异常、不产生 NaN，只是把头 A 的结果静默摆到头 B 的槽位上。"""
    L = len(heads[0])
    dh = len(heads[0][0])
    H = len(heads)
    return [[heads[h][i][t] for t in range(dh) for h in range(H)] for i in range(L)]


def causal_mask(S):
    """返回每行能看到的最大下标（含自己）：[0, 1, 2, ..., S-1]。"""
    return list(range(S))


def head_attention(Qh, Kh, Vh, mask, cnt):
    """单头缩放点积注意力，scores 先算满 L×L 再掩（和生产实现一致，也让计数公式干净）。"""
    L, dh = len(Qh), len(Qh[0])
    scale = 1.0 / math.sqrt(dh)
    scores = [[dot(Qh[i], Kh[j], cnt) * scale for j in range(L)] for i in range(L)]
    attn = []
    ctx = []
    for i in range(L):
        upto = (L - 1) if mask is None else mask[i]
        w = masked_row_softmax(scores[i], upto)
        attn.append(w)
        ctx.append(axpy_rows(w, Vh, cnt))
    return ctx, attn, scores


def multi_head_attention(X, params, n_heads, mask, cnt):
    """多头因果自注意力：投影 → 切头 → 各头独立算 → 拼回 → 输出投影。"""
    L, d = len(X), len(X[0])
    dh = d // n_heads
    assert d % n_heads == 0, '维度不匹配：d_model=%d 不能被 n_heads=%d 整除' % (d, n_heads)
    Q = matmul(X, params['Wq'], cnt)          # L·d²
    K = matmul(X, params['Wk'], cnt)
    V = matmul(X, params['Wv'], cnt)
    ctx_heads, attn_heads = [], []
    for h in range(n_heads):
        c, a, _ = head_attention(head_slice(Q, h, dh), head_slice(K, h, dh),
                                 head_slice(V, h, dh), mask, cnt)
        ctx_heads.append(c)
        attn_heads.append(a)
    ctx = concat_heads(ctx_heads)
    o = matmul(ctx, params['Wo'], cnt)        # L·d²
    return o, attn_heads, ctx


# ---------------------------------------------------------------- 最简 KV Cache
def kv_cache_new(n_heads, dh):
    return dict(K=[[] for _ in range(n_heads)], V=[[] for _ in range(n_heads)],
                n_heads=n_heads, dh=dh)


def kv_cache_append(cache, k_row, v_row):
    """把新 token 的 K / V 按头切片追加进缓存。切片只搬运引用值，浮点位完全不变。"""
    dh = cache['dh']
    for h in range(cache['n_heads']):
        cache['K'][h].append(k_row[h * dh:(h + 1) * dh])
        cache['V'][h].append(v_row[h * dh:(h + 1) * dh])


def kv_cache_scalars(cache):
    """缓存里存了多少个标量：K 与 V 各 n_heads × L × dh → 合计 2·d_model·L。"""
    return sum(len(cache['K'][h]) * cache['dh'] + len(cache['V'][h]) * cache['dh']
               for h in range(cache['n_heads']))


def project_qkv(xrow, params, cnt):
    """一个 token 的 q / k / v 投影：三次 [1,d]×[d,d]，各 d² 次乘加。"""
    q = matmul([xrow], params['Wq'], cnt)[0]
    k = matmul([xrow], params['Wk'], cnt)[0]
    v = matmul([xrow], params['Wv'], cnt)[0]
    return q, k, v


def cached_attention(q, cache, params, cnt):
    """缓存路径的一步：只有 1 行 query，对 L 长的缓存算 1×L 的分数，再输出投影。"""
    dh = cache['dh']
    scale = 1.0 / math.sqrt(dh)
    ctx_heads = []
    for h in range(cache['n_heads']):
        qh = q[h * dh:(h + 1) * dh]
        Kh, Vh = cache['K'][h], cache['V'][h]
        L = len(Kh)
        scores = [dot(qh, Kh[j], cnt) * scale for j in range(L)]   # L 次点积
        w = masked_row_softmax(scores, L - 1)                      # 最后一个位置：全可见
        ctx_heads.append(axpy_rows(w, Vh, cnt))                    # L 次加权累加
    ctx = [0.0] * (cache['n_heads'] * dh)
    for h in range(cache['n_heads']):
        for t in range(dh):
            ctx[h * dh + t] = ctx_heads[h][t]
    return matmul([ctx], params['Wo'], cnt)[0]


# ---------------------------------------------------------------- 两条解码路径
def decode_naive(X, params, n_heads, n_max, grid):
    """无缓存：每生成一个位置，都把 0..L-1 整段重新投影、重新算 L×L 注意力。"""
    cnt = new_cnt()
    ck = {}
    per_step = []
    outs = []
    t0 = time.time()
    for t in range(PROMPT, n_max):
        L = t + 1
        before = (cnt['mac'], cnt['dots'])
        o, _, _ = multi_head_attention(X[:L], params, n_heads, causal_mask(L), cnt)
        outs.append(o[L - 1])
        per_step.append((L, cnt['mac'] - before[0], cnt['dots'] - before[1]))
        if L in grid:
            ck[L] = dict(mac=cnt['mac'], dots=cnt['dots'], wall=time.time() - t0)
    return outs, ck, per_step, cnt


def decode_cached(X, params, n_heads, n_max, grid):
    """有缓存：预填充一次 P 个 token，之后每步只算新 token 的一行。

    口径：预填充的开销单独记在 pcnt 里，不进生成阶段计数——它是个不随 N 增长的常数，
    但报表里会连同它一起给"含预填充"的加速比，避免把一次性成本藏起来。
    """
    dh = DH
    pcnt = new_cnt()
    # Prefill（预填充）：完整跑一遍 P 个 token 的注意力。真实推理要靠最后一行的输出
    # 采样出第一个生成 token；这里 teacher forcing 不回喂，但计数必须忠实。
    Qp = matmul(X[:PROMPT], params['Wq'], pcnt)
    Kp = matmul(X[:PROMPT], params['Wk'], pcnt)
    Vp = matmul(X[:PROMPT], params['Wv'], pcnt)
    ctx_heads = []
    for h in range(n_heads):
        c, _, _ = head_attention(head_slice(Qp, h, dh), head_slice(Kp, h, dh),
                                 head_slice(Vp, h, dh), causal_mask(PROMPT), pcnt)
        ctx_heads.append(c)
    prefill_out = matmul(concat_heads(ctx_heads), params['Wo'], pcnt)[PROMPT - 1]

    cache = kv_cache_new(n_heads, dh)
    for t in range(PROMPT):
        kv_cache_append(cache, Kp[t], Vp[t])

    cnt = new_cnt()
    ck = {}
    per_step = []
    outs = []
    t0 = time.time()
    for t in range(PROMPT, n_max):                 # Decode（解码）
        L = t + 1
        before = (cnt['mac'], cnt['dots'])
        q, k, v = project_qkv(X[t], params, cnt)
        kv_cache_append(cache, k, v)
        outs.append(cached_attention(q, cache, params, cnt))
        per_step.append((L, cnt['mac'] - before[0], cnt['dots'] - before[1]))
        if L in grid:
            ck[L] = dict(mac=cnt['mac'], dots=cnt['dots'], wall=time.time() - t0,
                         scalars=kv_cache_scalars(cache))
    return outs, ck, per_step, cnt, pcnt, cache, prefill_out


# ---------------------------------------------------------------- 闭式公式（已知答案）
def sum_range(a, b):
    """Σ_{i=a}^{b} i，整数精确。"""
    return (b - a + 1) * (a + b) // 2


def sum_sq_range(a, b):
    """Σ_{i=a}^{b} i²，整数精确。"""
    def pre(n):
        return n * (n + 1) * (2 * n + 1) // 6
    return pre(b) - pre(a - 1)


def theory_marginal(d, n_heads, L):
    """生成长度为 L 的那一步的边际开销。

    朴素：q/k/v/o 四个投影各作用在 L 行上 → 4·L·d²；scores 与 ctx 各 L²·d → 2·L²·d
    缓存：四个投影只作用在 1 行上 → 4·d²；scores 与 ctx 各 L·d/H 每头 → 2·L·d
    """
    return dict(naive_mac=4 * L * d * d + 2 * L * L * d,
                cached_mac=4 * d * d + 2 * L * d,
                naive_dots=2 * n_heads * L * L,
                cached_dots=2 * n_heads * L)


def theory_cumulative(d, n_heads, P, N):
    """从 L=P+1 累加到 N 的闭式和。S1 = ΣL，S2 = ΣL²。"""
    S1 = sum_range(P + 1, N)
    S2 = sum_sq_range(P + 1, N)
    marg = theory_marginal(d, n_heads, P)
    return dict(S1=S1, S2=S2,
                naive_mac=4 * d * d * S1 + 2 * d * S2,
                cached_mac=4 * d * d * (N - P) + 2 * d * S1,
                naive_dots=2 * n_heads * S2,
                cached_dots=2 * n_heads * S1,
                prefill_mac=marg['naive_mac'], prefill_dots=marg['naive_dots'])


# ---------------------------------------------------------------- 参照实现（对账用）
def ref_single_head_unmasked(X, params):
    """无掩码单头参照实现：最直白的写法独立算一遍，用来对账 H=1 时的多头主路径。"""
    S, d = len(X), len(X[0])
    Q = matmul(X, params['Wq'])
    K = matmul(X, params['Wk'])
    V = matmul(X, params['Wv'])
    scale = 1.0 / math.sqrt(d)
    scores = [[sum(Q[i][t] * K[j][t] for t in range(d)) * scale for j in range(S)]
              for i in range(S)]
    attn = [row_softmax(r) for r in scores]
    ctx = [[sum(attn[i][j] * V[j][t] for j in range(S)) for t in range(d)] for i in range(S)]
    return matmul(ctx, params['Wo']), attn, ctx


# ---------------------------------------------------------------- 报表
def fmt_report(cfg_lines, trace_lines, causal, worked, headmap, equiv, bench):
    W = 70
    lines = []
    lines.append('=' * W)
    lines.append('基础篇 04 · 自注意力与 KV Cache 报表（seed=%d）' % SEED)
    lines.append('=' * W)

    lines.append('\n【A】配置与口径')
    for s in cfg_lines:
        lines.append('    ' + s)

    lines.append('\n【B】多头注意力的维度流动（以 L=%d 为例）' % N_MAX)
    for s in trace_lines:
        lines.append('    ' + s)

    lines.append('\n【C】因果掩码、能手算的数值例与多头切拼')
    for s in causal:
        lines.append('    ' + s)
    lines.append('    数值例（Wq=Wk=Wv=Wo=I，投影是单位阵，分数行可以纸笔算出来）：')
    for s in worked:
        lines.append('      ' + s)
    lines.append('    多头切/拼索引互逆（反面教材）：')
    for s in headmap:
        lines.append('      ' + s)

    lines.append('\n【D】KV Cache 等价性：缓存路径 == 重算路径？')
    for s in equiv:
        lines.append('    ' + s)

    lines.append('\n【E】KV Cache 存储账（换来的代价）')
    lines.append('      N    缓存标量数    闭式 2·d·N    朴素累计重算 MAC')
    for r in bench['rows']:
        lines.append('    %5d %12d %13d %19d'
                     % (r['N'], r['c_scalars'], r['c_scalars_theory'], r['n_mac']))
    lines.append('    → 计算量从 O(t²) 累计降到 O(t) 累计，存储反过来从 O(1) 涨到 O(t)')

    lines.append('\n【F】benchmark：生成阶段累计操作数（P=%d → N）' % PROMPT)
    lines.append('      N     朴素MAC       缓存MAC     加速比    朴素dots     缓存dots   加速比')
    for r in bench['rows']:
        lines.append('    %5d %11d %11d %9.2f× %11d %11d %8.2f×'
                     % (r['N'], r['n_mac'], r['c_mac'], r['ratio_mac'],
                        r['n_dots'], r['c_dots'], r['ratio_dots']))
    lines.append('    闭式对账（ΣL=%d、ΣL²=%d @N=%d）：'
                 % (bench['theory']['S1'], bench['theory']['S2'], N_MAX))
    lines.append('      朴素 = 4d²·ΣL + 2d·ΣL² = %d   实测 = %d   %s'
                 % (bench['theory']['naive_mac'], bench['rows'][-1]['n_mac'],
                    '一致' if bench['theory']['naive_mac'] == bench['rows'][-1]['n_mac'] else '不一致'))
    lines.append('      缓存 = 4d²·(N−P) + 2d·ΣL = %d   实测 = %d   %s'
                 % (bench['theory']['cached_mac'], bench['rows'][-1]['c_mac'],
                    '一致' if bench['theory']['cached_mac'] == bench['rows'][-1]['c_mac'] else '不一致'))
    lines.append('    预填充（一次性，不计入生成阶段）：MAC=%d、dots=%d'
                 % (bench['prefill_mac'], bench['prefill_dots']))
    lines.append('    含预填充的缓存总量 = %d，加速比 %.2f×'
                 % (bench['total_mac'], bench['ratio_total']))
    lines.append('    秒表（机器相关）只打印到终端、不落盘，以保证产物逐字节可复现')

    return '\n'.join(lines)


# ---------------------------------------------------------------- 配图
def draw(attn8, bench, per_step_n, per_step_c, cache_scalars):
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

    # ① 因果注意力权重热力图（上三角精确为 0）
    im = axes[0].imshow(attn8, cmap='Greens', vmin=0, vmax=1)
    axes[0].set_title('因果注意力权重（第 0 头，L=8）\n上三角被 -inf 掩掉，精确为 0.00')
    axes[0].set_xlabel('被看的位置 j')
    axes[0].set_ylabel('发起的位置 i')
    for i in range(len(attn8)):
        for j in range(len(attn8[0])):
            axes[0].text(j, i, '%.2f' % attn8[i][j], ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    # ② 累计操作数 vs N：实测点必须落在闭式理论线上
    ns = list(range(PROMPT + 1, N_MAX + 1))
    th_n = [theory_cumulative(D_MODEL, N_HEADS, PROMPT, n)['naive_mac'] for n in ns]
    th_c = [theory_cumulative(D_MODEL, N_HEADS, PROMPT, n)['cached_mac'] for n in ns]
    axes[1].plot(ns, th_n, '--', color='#cc5a5a', lw=1.2, label='朴素 闭式 4d²ΣL+2dΣL²')
    axes[1].plot(ns, th_c, '--', color='#4a9d6a', lw=1.2, label='缓存 闭式 4d²(N−P)+2dΣL')
    axes[1].plot([r['N'] for r in bench['rows']], [r['n_mac'] for r in bench['rows']],
                 'o', color='#cc5a5a', ms=7, label='朴素 实测计数')
    axes[1].plot([r['N'] for r in bench['rows']], [r['c_mac'] for r in bench['rows']],
                 's', color='#4a9d6a', ms=7, label='缓存 实测计数')
    axes[1].set_yscale('log')
    axes[1].set_title('生成阶段累计乘加数 vs 序列长度\n实测点全部落在闭式理论线上（整数相等）')
    axes[1].set_xlabel('生成到的序列长度 N')
    axes[1].set_ylabel('累计 MAC（对数轴）')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # ③ 每步边际开销 + 加速比
    ls = [p[0] for p in per_step_n]
    axes[2].plot(ls, [p[1] for p in per_step_n], color='#cc5a5a', lw=1.6, label='朴素 每步 MAC')
    axes[2].plot(ls, [p[1] for p in per_step_c], color='#4a9d6a', lw=1.6, label='缓存 每步 MAC')
    axes[2].set_yscale('log')
    axes[2].set_title('每步边际开销：朴素随 L 平方涨，缓存只随 L 线性涨')
    axes[2].set_xlabel('当前序列长度 L')
    axes[2].set_ylabel('该步 MAC（对数轴）')
    ax3 = axes[2].twinx()
    ratio = [pn[1] / pc[1] for pn, pc in zip(per_step_n, per_step_c)]
    ax3.plot(ls, ratio, color='#8888cc', lw=1.2, ls=':', label='单步加速比')
    ax3.set_ylabel('单步加速比（右轴，恒等于 L）', color='#8888cc')
    ax3.tick_params(axis='y', labelcolor='#8888cc')
    axes[2].legend(fontsize=8, loc='upper left')

    # ④ 省下的计算 vs 换来的存储
    axes[3].plot([r['N'] for r in bench['rows']], cache_scalars, 'o-', color='#4a9d6a',
                 lw=1.8, label='缓存标量数 = 2·d·N（左轴）')
    axes[3].set_xlabel('序列长度 N')
    axes[3].set_ylabel('KV Cache 标量数', color='#4a9d6a')
    axes[3].tick_params(axis='y', labelcolor='#4a9d6a')
    axes[3].set_title('KV Cache 的代价：存储随 N 线性增长\n（右轴：朴素累计重算量，平方级）')
    ax4 = axes[3].twinx()
    ax4.plot([r['N'] for r in bench['rows']], [r['n_mac'] for r in bench['rows']],
             's--', color='#d08c4a', lw=1.6, label='朴素累计 MAC（右轴）')
    ax4.set_yscale('log')
    ax4.set_ylabel('朴素累计 MAC（对数轴）', color='#d08c4a')
    ax4.tick_params(axis='y', labelcolor='#d08c4a')

    path = os.path.join(OUT, 'fig-04-attention-kvcache.png')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    # teacher forcing：输入序列事先固定，不做 argmax 回喂。
    # 否则两条路径哪怕只差一个浮点末位，也会被放大成 token 分叉，对账就失去意义。
    X = [[rng.uniform(-1, 1) for _ in range(D_MODEL)] for _ in range(N_MAX)]
    params = make_attention_params(rng, D_MODEL)

    cfg_lines = [
        'd_model=%d  n_heads=%d  head_dim=%d  投影无 bias' % (D_MODEL, N_HEADS, DH),
        '提示词长度 P=%d  生成到 N=%d  checkpoint 格点 %s' % (PROMPT, N_MAX, list(GRID)),
        '解码方式：teacher forcing（X 事先固定，不回喂 argmax）',
        'benchmark 口径：只统计生成阶段（N−P=%d 步）；预填充单独报，不随 N 增长' % (N_MAX - PROMPT),
        '主指标 = 确定性整数操作计数器（MAC / dots），秒表只作参考',
    ]
    trace_lines = [
        '输入 X               shape=%s   （L × d_model）' % ((N_MAX, D_MODEL),),
        'Q = X·Wq             shape=%s   （Wq: d_model×d_model，形状不变，L·d² 次乘加）' % ((N_MAX, D_MODEL),),
        'K = X·Wk / V = X·Wv  shape=%s   （同上，各 L·d²）' % ((N_MAX, D_MODEL),),
        '切成 %d 个头          shape=%s × %d   （head_dim = d_model / n_heads）'
        % (N_HEADS, (N_MAX, DH), N_HEADS),
        'scores = Qh·Khᵀ/√dh  shape=%s   （L² 个 %d 维点积，先算满再掩）' % ((N_MAX, N_MAX), DH),
        'attn = 因果 softmax  shape=%s   （每行 j>i 处置 -inf → 权重精确 0）' % ((N_MAX, N_MAX),),
        'ctx_h = attn·Vh      shape=%s   （L² 次向量加权累加）' % ((N_MAX, DH),),
        'concat_heads         shape=%s   （%d 个头拼回 d_model）' % ((N_MAX, D_MODEL), N_HEADS),
        'out = ctx·Wo         shape=%s   （输出投影，L·d²）' % ((N_MAX, D_MODEL),),
    ]

    # ---------------- 【C】因果掩码：小序列 + 能手算的数值例
    S_DEMO = 8
    Xs = X[:S_DEMO]
    o8, attn8, ctx8 = multi_head_attention(Xs, params, N_HEADS, causal_mask(S_DEMO), None)

    upper_max = max(attn8[h][i][j] for h in range(N_HEADS)
                    for i in range(S_DEMO) for j in range(S_DEMO) if j > i)
    rowsum_err = max(abs(sum(attn8[h][i]) - 1.0) for h in range(N_HEADS) for i in range(S_DEMO))

    # 只改位置 t=5：位置 <5 的输出必须一字不变，位置 >=5 必须变
    PERT = 5
    Xp = [row[:] for row in Xs]
    Xp[PERT] = [v + 0.5 for v in Xp[PERT]]
    _, _, ctxp = multi_head_attention(Xp, params, N_HEADS, causal_mask(S_DEMO), None)
    before_diff = max(max(abs(ctx8[i][j] - ctxp[i][j]) for j in range(D_MODEL))
                      for i in range(PERT))
    after_diff = min(max(abs(ctx8[i][j] - ctxp[i][j]) for j in range(D_MODEL))
                     for i in range(PERT, S_DEMO))

    # 反面教材：置 0 而不是 -inf
    demo_row = [0.3, -0.2, 0.8, 1.1, 0.4, 0.0, -0.5, 0.2]
    good = masked_row_softmax(demo_row, 2)
    bad = wrong_zero_mask_softmax(demo_row, 2)
    causal = [
        '因果掩码 = 位置 i 只能看 j <= i（含自己），掩掉的位置置 -inf 再 softmax',
        '全部 %d 个头的上三角权重最大值 = %.1e（应精确为 0）' % (N_HEADS, upper_max),
        '每行权重和最大偏差 = %.2e（应为 1）' % rowsum_err,
        '只改位置 %d：位置 <%d 的 ctx max|Δ| = %.1e（应精确为 0）' % (PERT, PERT, before_diff),
        '            位置 ≥%d 的 ctx min L∞ 变化 = %.4f（应 > 0）' % (PERT, after_diff),
        '反面教材（同一行分数 %s，upto=2）：' % ['%.1f' % v for v in demo_row],
        '  置 -inf：权重 = %s  ← 未来位置精确 0' % ['%.4f' % v for v in good],
        '  置  0.0：权重 = %s  ← 未来位置分走了 %.1f%% 概率，静默错'
        % (['%.4f' % v for v in bad], 100.0 * sum(bad[3:])),
    ]

    I2 = [[1.0, 0.0], [0.0, 1.0]]
    p_id = dict(Wq=I2, Wk=I2, Wv=I2, Wo=I2)
    # 单头用正交 token（Q=K=X，scores 就是 X·Xᵀ/√2，能手算）
    Xo = [[1.0, 0.0], [0.0, 1.0]]
    _, a1, _ = multi_head_attention(Xo, p_id, 1, causal_mask(2), None)
    # 双头用 [[2,0],[1,1]]：dh=1、scale=1 时，两个头的分数行分别是 [2,1] 与 [0,1]，
    # softmax 后正好互为镜像 [e/(e+1), 1/(e+1)] 与 [1/(e+1), e/(e+1)]
    Xm = [[2.0, 0.0], [1.0, 1.0]]
    _, a2, _ = multi_head_attention(Xm, p_id, 2, causal_mask(2), None)
    eh = math.e / (math.e + 1.0)
    eo = 1.0 / (1.0 + math.exp(1.0 / math.sqrt(2.0)))
    worked = [
        '单头 X=[[1,0],[0,1]]（d=2, scale=1/√2）：',
        '  attn[0] = [%.4f, %.4f]（只能看自己，上三角精确为 0）' % (a1[0][0][0], a1[0][0][1]),
        '  attn[1] = [%.4f, %.4f]（手算 softmax([0, 1/√2]) = [%.4f, %.4f]）'
        % (a1[0][1][0], a1[0][1][1], eo, 1 - eo),
        '双头 X=[[2,0],[1,1]]（dh=1, scale=1，两个头各看一个维度）：',
        '  头0 attn[1] = [%.4f, %.4f]（分数行 [2,1] → softmax = [e/(e+1), 1/(e+1)]）'
        % (a2[0][1][0], a2[0][1][1]),
        '  头1 attn[1] = [%.4f, %.4f]（分数行 [0,1] → softmax = [1/(e+1), e/(e+1)]）'
        % (a2[1][1][0], a2[1][1][1]),
        '→ 同样两个 token、同一套单位投影，两个头看出的关系互为镜像：',
        '  这就是"多头看不同关系"最便宜的手算证据（e/(e+1) = %.4f）' % eh,
    ]

    # 反面教材：切与拼的下标顺序不互逆（同一份 heads，两种拼法）
    Qd, Kd, Vd = matmul(Xs, params['Wq']), matmul(Xs, params['Wk']), matmul(Xs, params['Wv'])
    heads_ok = [head_attention(head_slice(Qd, h, DH), head_slice(Kd, h, DH),
                               head_slice(Vd, h, DH), causal_mask(S_DEMO), None)[0]
                for h in range(N_HEADS)]
    ctx_ok, ctx_sw = concat_heads(heads_ok), concat_heads_wrong(heads_ok)
    ok_rt = all(head_slice(ctx_ok, h, DH) == heads_ok[h] for h in range(N_HEADS))
    bad_rt = all(head_slice(ctx_sw, h, DH) == heads_ok[h] for h in range(N_HEADS))
    swap_diff = max_abs_diff(ctx_ok, ctx_sw)
    n_diff = sum(1 for i in range(S_DEMO) for j in range(D_MODEL)
                 if ctx_ok[i][j] != ctx_sw[i][j])
    headmap = [
        '切头是连续块：head_slice(W, h, dh) = W[:, h·dh:(h+1)·dh]，'
        '所以拼头必须用 out[h·dh+t] = heads[h][t] 才互逆',
        '正确拼法 round-trip：head_slice(concat_heads(heads), h) == heads[h] 对全部 %d 个头 → %s'
        % (N_HEADS, ok_rt),
        '写反拼法（for t … for h …）round-trip：%s；此时切出的"第 0 头·第 0 行" = %s'
        % (bad_rt, ['%.4f' % v for v in head_slice(ctx_sw, 0, DH)[0]]),
        '真实"第 0 头·第 0 行"应当是：%s' % (['%.4f' % v for v in heads_ok[0][0]]),
        '同一份 heads，两种拼法逐元素 max|Δ| = %.6f，%d/%d 个位置的值不同'
        % (swap_diff, n_diff, S_DEMO * D_MODEL),
        '  剩下 %d 个位置恰好相同：h·dh+t == t·H+h 只在 h==t 成立（dh=H=%d 的对角线）'
        % (S_DEMO * D_MODEL - n_diff, DH),
        '第 0 行前 8 个标量 正确：%s' % (['%.4f' % v for v in ctx_ok[0][:8]]),
        '第 0 行前 8 个标量 写反：%s' % (['%.4f' % v for v in ctx_sw[0][:8]]),
        '→ 形状不变、不抛异常、不出 NaN，只是头 A 的值静默落进头 B 的槽位；'
        '生产篇 04 的 repeat 头序错配是同一类坑的工业版本',
    ]

    # ---------------- 【D】两条解码路径 + 等价性
    t_start = time.time()
    outs_n, ck_n, per_step_n, cnt_n = decode_naive(X, params, N_HEADS, N_MAX, GRID)
    t_naive = time.time() - t_start
    t_start = time.time()
    outs_c, ck_c, per_step_c, cnt_c, pcnt, cache, prefill_out = \
        decode_cached(X, params, N_HEADS, N_MAX, GRID)
    t_cached = time.time() - t_start

    equiv_diff = max(abs(outs_n[i][j] - outs_c[i][j])
                     for i in range(len(outs_n)) for j in range(D_MODEL))
    equiv = [
        '朴素重算 vs KV Cache，%d 个生成位置的输出逐元素比对：' % len(outs_n),
        '  max|Δ| = %.1e' % equiv_diff,
        '  两条路径共用同一个 dot / axpy_rows / matmul 内核，求和顺序一致 → 逐位相同',
        '  朴素累计 MAC = %d、dots = %d' % (cnt_n['mac'], cnt_n['dots']),
        '  缓存累计 MAC = %d、dots = %d（另有预填充 MAC = %d）'
        % (cnt_c['mac'], cnt_c['dots'], pcnt['mac']),
    ]

    # ---------------- 【E】【F】benchmark 表 + 闭式对账
    rows = []
    for L in GRID:
        th = theory_cumulative(D_MODEL, N_HEADS, PROMPT, L)
        rows.append(dict(N=L,
                         n_mac=ck_n[L]['mac'], c_mac=ck_c[L]['mac'],
                         n_dots=ck_n[L]['dots'], c_dots=ck_c[L]['dots'],
                         ratio_mac=ck_n[L]['mac'] / ck_c[L]['mac'],
                         ratio_dots=ck_n[L]['dots'] / ck_c[L]['dots'],
                         n_wall=ck_n[L]['wall'], c_wall=ck_c[L]['wall'],
                         c_scalars=ck_c[L]['scalars'],
                         c_scalars_theory=2 * D_MODEL * L,
                         th_n_mac=th['naive_mac'], th_c_mac=th['cached_mac'],
                         th_n_dots=th['naive_dots'], th_c_dots=th['cached_dots']))
    theory_full = theory_cumulative(D_MODEL, N_HEADS, PROMPT, N_MAX)
    total_mac = theory_full['prefill_mac'] + ck_c[N_MAX]['mac']
    bench = dict(rows=rows, theory=theory_full,
                 prefill_mac=pcnt['mac'], prefill_dots=pcnt['dots'],
                 total_mac=total_mac,
                 ratio_total=ck_n[N_MAX]['mac'] / total_mac)

    report = fmt_report(cfg_lines, trace_lines, causal, worked, headmap, equiv, bench)
    print()
    print(report)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')

    # 秒表只打印、不落盘：机器相关，写进产物会破坏逐字节可复现
    print('\n    参考秒表（机器相关，只报不断言，不写入产物）：')
    for r in rows:
        print('      N=%3d  朴素 %.3fs   缓存 %.4fs' % (r['N'], r['n_wall'], r['c_wall']))

    with open(os.path.join(OUT, 'benchmark.tsv'), 'w', encoding='utf-8') as f:
        f.write('N\tnaive_mac\tcached_mac\tratio_mac\tnaive_dots\tcached_dots\t'
                'ratio_dots\tcached_scalars\ttheory_naive_mac\ttheory_cached_mac\n')
        for r in rows:
            f.write('%d\t%d\t%d\t%.6f\t%d\t%d\t%.6f\t%d\t%d\t%d\n'
                    % (r['N'], r['n_mac'], r['c_mac'], r['ratio_mac'],
                       r['n_dots'], r['c_dots'], r['ratio_dots'], r['c_scalars'],
                       r['th_n_mac'], r['th_c_mac']))

    with open(os.path.join(OUT, 'kv_cache.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'd_model': D_MODEL, 'n_heads': N_HEADS, 'head_dim': DH,
                   'prompt': PROMPT, 'n_max': N_MAX,
                   'equivalence_max_abs_diff': equiv_diff,
                   'checkpoints': [{k: v for k, v in r.items() if not k.endswith('_wall')}
                                   for r in rows],
                   'prefill': {'mac': pcnt['mac'], 'dots': pcnt['dots']},
                   'cached_scalars_at': {str(L): 2 * D_MODEL * L for L in GRID},
                   'causal_demo': {'upper_max': upper_max, 'rowsum_err': rowsum_err,
                                   'before_diff': before_diff, 'after_diff': after_diff,
                                   'leak_zero_mask': sum(bad[3:])},
                   'head_map': {'correct_roundtrip_exact': ok_rt,
                                'swapped_roundtrip_exact': bad_rt,
                                'swap_max_abs_diff': swap_diff,
                                'positions_differing': n_diff,
                                'positions_total': S_DEMO * D_MODEL},
                   'per_step_naive_mac': [p[1] for p in per_step_n],
                   'per_step_cached_mac': [p[1] for p in per_step_c]},
                  f, ensure_ascii=False, indent=2)

    fig = draw(attn8[0], bench, per_step_n, per_step_c,
               [r['c_scalars'] for r in rows])
    if fig:
        print('\n[配图] %s' % fig)

    # ---------------- 自校验
    print('\n' + '=' * W)
    print('【G】自校验（对账 + benchmark 闭式公式）')
    print('=' * W)

    print('\n  1. 计数器自证：解析计数 == 逐次递增的字面计数（matmul / dot / axpy_rows）')
    A = [[rng.uniform(-1, 1) for _ in range(5)] for _ in range(3)]
    B = [[rng.uniform(-1, 1) for _ in range(7)] for _ in range(5)]
    ca, cl = new_cnt(), dict(literal_mac=0)
    Ya = matmul(A, B, ca)
    Yl = matmul_literal(A, B, cl)
    assert ca['mac'] == cl['literal_mac'] == 3 * 5 * 7, \
        '计数不自证：解析 %d vs 字面 %d' % (ca['mac'], cl['literal_mac'])
    assert max_abs_diff(Ya, Yl) == 0.0, '两种 matmul 结果不一致'
    cd = new_cnt()
    dv = dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], cd)
    assert (cd['dots'], cd['mac'], dv) == (1, 3, 32.0), 'dot 计数或数值错误：%s' % cd
    cx = new_cnt()
    xv = axpy_rows([0.5, 0.25], [[1.0, 2.0], [3.0, 4.0]], cx)
    assert (cx['dots'], cx['mac']) == (2, 4), 'axpy_rows 计数错误：%s' % cx
    assert xv == [1.25, 2.0], 'axpy_rows 数值错误：%s' % xv
    print('     [3,5]×[5,7]：解析 %d == 字面 %d == 3·5·7，结果 max|Δ| = %.1e'
          % (ca['mac'], cl['literal_mac'], max_abs_diff(Ya, Yl)))
    print('     dot：1 次向量运算 / 3 次乘加，值 %.1f；axpy_rows：2 次 / 4 次乘加，值 %s'
          % (dv, ['%.2f' % v for v in xv]))

    print('\n  2. 因果性：上三角精确为 0、行和为 1、未来不影响过去')
    assert upper_max == 0.0, '上三角权重不为 0：%.2e' % upper_max
    assert rowsum_err < 1e-12, '行和偏离 1：%.2e' % rowsum_err
    assert before_diff == 0.0, '改动位置 %d 却影响了更早的位置：%.2e' % (PERT, before_diff)
    assert after_diff > 1e-6, '改动位置 %d 未影响之后的位置：%.2e' % (PERT, after_diff)
    assert sum(bad[3:]) > 0.05 and sum(good[3:]) == 0.0, '置 0 掩码应分走概率、置 -inf 应为 0'
    print('     上三角 max = %.1e；行和偏差 %.2e；改位置 %d → 之前 max|Δ| = %.1e、之后 %.4f'
          % (upper_max, rowsum_err, PERT, before_diff, after_diff))
    print('     置 -inf：未来权重合计 %.1e；置 0.0：未来权重合计 %.4f（分走 %.1f%%）'
          % (sum(good[3:]), sum(bad[3:]), 100.0 * sum(bad[3:])))

    print('\n  3. 手算数值例 + 切拼互逆：单位投影可对纸笔；拼反下标顺序不报错但结果全错')
    assert abs(a1[0][0][0] - 1.0) < 1e-12 and abs(a1[0][0][1]) < 1e-12, '第 0 行应只看自己'
    assert abs(a1[0][1][0] - eo) < 1e-9, '单头 attn[1][0] 应为 %.6f' % eo
    assert abs(a2[0][1][0] - eh) < 1e-9, '头0 attn[1][0] 应为 e/(e+1)'
    assert abs(a2[1][1][0] - (1.0 - eh)) < 1e-9, '头1 attn[1][0] 应为 1/(e+1)'
    mirror = max(abs(a2[0][1][0] - a2[1][1][1]), abs(a2[0][1][1] - a2[1][1][0]))
    assert mirror < 1e-12, '两头应互为镜像：%.2e' % mirror
    assert ok_rt, '正确拼法应与 head_slice 精确互逆'
    assert not bad_rt, '写反的拼法不该互逆却通过了 round-trip，说明 demo 失效'
    assert swap_diff > 1e-6, '两种拼法应当明显不同：%.2e' % swap_diff
    assert all(math.isfinite(v) for row in ctx_sw for v in row), '写反拼法不应产生 NaN/inf'
    assert n_diff == S_DEMO * D_MODEL - S_DEMO * DH, \
        '不同的位置数应正好去掉 h==t 的对角线：实测 %d' % n_diff
    print('     单头 attn[1] = [%.6f, %.6f]，手算 softmax([0, 1/√2]) = [%.6f, %.6f]'
          % (a1[0][1][0], a1[0][1][1], eo, 1 - eo))
    print('     头0 attn[1] = [%.6f, %.6f]、头1 attn[1] = [%.6f, %.6f]，手算 e/(e+1) = %.6f'
          % (a2[0][1][0], a2[0][1][1], a2[1][1][0], a2[1][1][1], eh))
    print('     互为镜像的偏差 max|Δ| = %.1e' % mirror)
    print('     切/拼 round-trip：正确 %s、写反 %s；两种拼法 max|Δ| = %.6f，%d/%d 个位置不同'
          % (ok_rt, bad_rt, swap_diff, n_diff, S_DEMO * D_MODEL))

    print('\n  4. H=1 退化：多头主路径 == 无掩码单头参照实现')
    S4 = 6
    X4 = X[:S4]
    o4, _, ctx4 = multi_head_attention(X4, params, 1, None, None)
    r4, _, rctx4 = ref_single_head_unmasked(X4, params)
    d4 = max(max_abs_diff(o4, r4), max_abs_diff(ctx4, rctx4))
    assert shape(o4) == (S4, D_MODEL), 'H=1 形状应守恒'
    assert d4 < 1e-12, 'H=1 未退化为单头：max|Δ|=%.2e' % d4
    print('     shape %s 守恒，out 与 ctx 的 max|Δ| = %.1e' % (shape(o4), d4))

    print('\n  5. KV Cache 等价性：缓存路径 == 重算路径（逐元素）')
    assert equiv_diff < 1e-12, '两条路径输出不一致：%.2e' % equiv_diff
    for r in rows:
        assert r['c_scalars'] == r['c_scalars_theory'], \
            '缓存标量数 %d != 闭式 %d @N=%d' % (r['c_scalars'], r['c_scalars_theory'], r['N'])
    assert pcnt['mac'] == theory_full['prefill_mac'], '预填充计数 != 闭式'
    assert pcnt['dots'] == theory_full['prefill_dots'], '预填充 dots != 闭式'
    assert all(math.isfinite(v) for v in prefill_out), '预填充输出出现非有限值'
    print('     %d 个生成位置 max|Δ| = %.1e；缓存标量数 == 2·d·N（5 个格点整数精确）'
          % (len(outs_n), equiv_diff))
    print('     预填充实测 MAC = %d == 闭式 4·P·d² + 2·P²·d = %d；dots = %d == 2H·P² = %d'
          % (pcnt['mac'], theory_full['prefill_mac'], pcnt['dots'], theory_full['prefill_dots']))
    print('     预填充最后一行输出 L2 = %.4f（真实推理靠它采样出第一个生成 token）'
          % math.sqrt(sum(v * v for v in prefill_out)))

    print('\n  6. benchmark 闭式对账：实测计数 == 闭式求和（整数相等，无容差）')
    for r in rows:
        assert r['n_mac'] == r['th_n_mac'], 'N=%d 朴素 MAC %d != 闭式 %d' % (r['N'], r['n_mac'], r['th_n_mac'])
        assert r['c_mac'] == r['th_c_mac'], 'N=%d 缓存 MAC %d != 闭式 %d' % (r['N'], r['c_mac'], r['th_c_mac'])
        assert r['n_dots'] == r['th_n_dots'], 'N=%d 朴素 dots 不符' % r['N']
        assert r['c_dots'] == r['th_c_dots'], 'N=%d 缓存 dots 不符' % r['N']
    for L, nm, nd in per_step_n:
        tm = theory_marginal(D_MODEL, N_HEADS, L)
        assert nm == tm['naive_mac'] and nd == tm['naive_dots'], 'L=%d 朴素边际不符' % L
    for L, cm, cd in per_step_c:
        tm = theory_marginal(D_MODEL, N_HEADS, L)
        assert cm == tm['cached_mac'] and cd == tm['cached_dots'], 'L=%d 缓存边际不符' % L
    for (L, nm, _), (_, cm, _) in zip(per_step_n, per_step_c):
        assert abs(nm / cm - L) <= 1e-12, 'L=%d 单步加速比应恰好等于 L：%s' % (L, nm / cm)
    print('     5 个格点 + 全部 %d 个单步边际，MAC 与 dots 均整数相等' % len(per_step_n))
    print('     单步加速比 == 当前序列长度 L（全部 %d 步逐一断言，整数精确）' % len(per_step_n))
    print('     N=%d：朴素 %d MAC、缓存 %d MAC → 加速比 %.2f×（含预填充 %.2f×）'
          % (N_MAX, ck_n[N_MAX]['mac'], ck_c[N_MAX]['mac'],
             rows[-1]['ratio_mac'], bench['ratio_total']))
    print('     秒表（机器相关，只报不断言）：朴素 %.3fs、缓存 %.4fs' % (t_naive, t_cached))

    print('\n  7. 确定性：同 seed 重建，缓存路径输出逐元素一致')
    rng2 = random.Random(SEED)
    X2 = [[rng2.uniform(-1, 1) for _ in range(D_MODEL)] for _ in range(N_MAX)]
    params2 = make_attention_params(rng2, D_MODEL)
    outs2, _, _, _, _, _, _ = decode_cached(X2, params2, N_HEADS, N_MAX, ())
    assert X == X2, 'X 重建不一致'
    d7 = max(abs(outs_c[i][j] - outs2[i][j])
             for i in range(len(outs_c)) for j in range(D_MODEL))
    assert d7 == 0.0, '重建不确定：max|Δ|=%.2e' % d7
    print('     两次构建 %d 个位置输出 max|Δ| = %.1e' % (len(outs_c), d7))

    print('\n  全部断言通过。')
    print('\n[产物] 报表 / benchmark.tsv / kv_cache.json / 配图 → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
