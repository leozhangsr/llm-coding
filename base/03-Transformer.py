#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础篇 03 · Transformer：从零手写一个单头 Post-Norm 解码器块

对应文章：同目录 03-Transformer.md
运行：    python3 03-Transformer.py
依赖：    仅标准库；matplotlib 可选（缺了自动跳过画图）
产物：    ./output/03-Transformer/ 下的报表、维度流 trace、注意力权重、配图

延续数据工程 / 分词器的方法论——**从零手写、注入已知答案、拿答案对账**。这一篇把
Transformer 最小的一块（单头自注意力 + 前馈 MLP + 残差 + 层归一化，Post-Norm 写法）
用纯 Python 搭出来，重点不在训模型，而在：

    A. 维度流动：一个 [seq, d_model] 的张量，走完一块后每一步的形状怎么变，全程可追
    B. 组件机理：softmax / 缩放点积注意力 / LayerNorm / ReLU-MLP 各自在算什么
    C. 数值例：用"单位投影 + 正交 token"这种能手算的场景，对账注意力权重与输出
    D. 全局混合：自注意力让任意两个位置直接交互——改一个 token，全序列输出都变
    E. Post-Norm：先残差相加再归一化（原始 Transformer 写法），与生产篇 Pre-Norm 对照

自校验全部拿"能独立算出来的已知答案"对账，不靠肉眼看图。
"""

import json
import math
import os
import random
import sys

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '03-Transformer')

EPS = 1e-5


# ---------------------------------------------------------------- 极简线代（纯标准库）
def shape(A):
    if not isinstance(A, list):
        return ()
    if A and isinstance(A[0], list):
        return (len(A), len(A[0]))
    return (len(A),)


def zeros(r, c):
    return [[0.0] * c for _ in range(r)]


def matmul(A, B):
    """A[n,m] × B[m,p] → [n,p]。"""
    n, m = len(A), len(A[0])
    p = len(B[0])
    assert m == len(B), '矩阵乘法内维不匹配：%d vs %d' % (m, len(B))
    out = zeros(n, p)
    for i in range(n):
        Ai = A[i]
        oi = out[i]
        for k in range(m):
            a = Ai[k]
            if a == 0.0:
                continue
            Bk = B[k]
            for j in range(p):
                oi[j] += a * Bk[j]
    return out


def transpose(A):
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def row_softmax(row):
    """数值稳定 softmax：先减最大值再取指数。"""
    m = max(row)
    ex = [math.exp(v - m) for v in row]
    s = sum(ex)
    return [e / s for e in ex]


def layernorm(X, gamma, beta, eps=EPS):
    """逐行 LayerNorm：减均值、除标准差，再仿射 (×gamma + beta)。"""
    out = []
    for row in X:
        n = len(row)
        mu = sum(row) / n
        var = sum((v - mu) ** 2 for v in row) / n
        denom = math.sqrt(var + eps)
        out.append([(row[j] - mu) / denom * gamma[j] + beta[j] for j in range(n)])
    return out


def relu(X):
    return [[max(0.0, v) for v in row] for row in X]


# ---------------------------------------------------------------- 组件
def make_linear(rng, din, dout, scale=0.5):
    W = [[rng.uniform(-scale, scale) for _ in range(dout)] for _ in range(din)]
    b = [rng.uniform(-scale, scale) for _ in range(dout)]
    return W, b


def linear(X, W, b):
    Y = matmul(X, W)
    return [[Y[i][j] + b[j] for j in range(len(b))] for i in range(len(X))]


def single_head_attention(X, Wq, Wk, Wv, Wo):
    """单头缩放点积自注意力（本文不加因果掩码——那是生产篇 04 的主题）。
       scores = QK^T / sqrt(d)；attn = softmax(scores)；ctx = attn·V；out = ctx·Wo。"""
    S, d = len(X), len(X[0])
    Q, K, V = matmul(X, Wq), matmul(X, Wk), matmul(X, Wv)
    scale = 1.0 / math.sqrt(d)
    Kt = transpose(K)
    scores = [[sum(Q[i][t] * Kt[t][j] for t in range(d)) * scale for j in range(S)]
              for i in range(S)]
    attn = [row_softmax(r) for r in scores]
    ctx = matmul(attn, V)
    out = matmul(ctx, Wo)
    return out, attn, scores


def feed_forward(X, W1, b1, W2, b2):
    """两层 ReLU 前馈：out = relu(X·W1 + b1)·W2 + b2。"""
    H = relu(linear(X, W1, b1))
    return linear(H, W2, b2)


def post_norm_block(X, P):
    """Post-Norm 解码器块（原始 Transformer 写法）：
         h   = LayerNorm(X + Attn(X))
         out = LayerNorm(h + FFN(h))
       残差先相加、再归一化——与生产篇的 Pre-Norm（先归一化再进子层）正好相反。"""
    attn, aw, _ = single_head_attention(X, P['Wq'], P['Wk'], P['Wv'], P['Wo'])
    h = layernorm(add(X, attn), P['g1'], P['b1'])
    ffn = feed_forward(h, P['W1'], P['bb1'], P['W2'], P['bb2'])
    out = layernorm(add(h, ffn), P['g2'], P['b2'])
    return out, aw, h


def make_block_params(rng, d_model, d_ff):
    # 注意力投影用较大初始化（scale=1.0）拉开打分差异；FFN 用较小初始化（scale=0.5）稳住输出尺度。
    # gamma 初始化为 1、beta 为 0：此时 LayerNorm 的仿射是恒等，不改变归一化结果（真实模型会学 gamma/beta）。
    Wq, _ = make_linear(rng, d_model, d_model, scale=1.0)
    Wk, _ = make_linear(rng, d_model, d_model, scale=1.0)
    Wv, _ = make_linear(rng, d_model, d_model, scale=1.0)
    Wo, _ = make_linear(rng, d_model, d_model, scale=1.0)
    W1, bb1 = make_linear(rng, d_model, d_ff)
    W2, bb2 = make_linear(rng, d_ff, d_model)
    g1 = [1.0] * d_model
    b1 = [0.0] * d_model
    g2 = [1.0] * d_model
    b2 = [0.0] * d_model
    return dict(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, W1=W1, bb1=bb1, W2=W2, bb2=bb2,
                g1=g1, b1=b1, g2=g2, b2=b2)


# ---------------------------------------------------------------- 参考实现（对账用）
def ref_attention_identity(X):
    """已知答案：Wq=Wk=Wv=Wo=I（单位阵）时，Q=K=V=X，Wo 不改变结果。
       于是 scores[i][j] = dot(X[i],X[j]) / sqrt(d)，attn = softmax(scores)，ctx = attn·X。
       这个函数用最直白的写法独立算一遍，用来对账主路径。"""
    S, d = len(X), len(X[0])
    scale = 1.0 / math.sqrt(d)
    scores = [[sum(X[i][t] * X[j][t] for t in range(d)) * scale for j in range(S)]
              for i in range(S)]
    attn = [row_softmax(r) for r in scores]
    ctx = [[sum(attn[i][j] * X[j][t] for j in range(S)) for t in range(d)] for i in range(S)]
    return attn, ctx


# ---------------------------------------------------------------- 统计
def mean_std(vals):
    n = len(vals)
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / n
    return mu, math.sqrt(var)


def flat(X):
    return [v for row in X for v in row]


# ---------------------------------------------------------------- 报表
def fmt_report(d_model, seq, d_ff, X, P, trace_lines, worked, invariants, mix):
    W = 70
    lines = []
    lines.append('=' * W)
    lines.append('基础篇 03 · Transformer 报表（单头 Post-Norm，seed=%d）' % SEED)
    lines.append('=' * W)

    lines.append('\n【A】配置')
    lines.append('    d_model=%d  seq_len=%d  d_ff=%d  注意力头数=1（多头见生产篇 03/04）'
                 % (d_model, seq, d_ff))
    lines.append('    写法：Post-Norm（残差相加后再 LayerNorm，原始 Transformer 顺序）')

    lines.append('\n【B】维度流动 trace（一个张量走完一块，每步形状）')
    for s in trace_lines:
        lines.append('    ' + s)

    lines.append('\n【C】数值例：单位投影 + 正交 token（能手算的已知答案）')
    lines.append('    输入 X = [[1,0],[0,1]]（两个正交单位向量），Wq=Wk=Wv=Wo=I')
    lines.append('    dot/sqrt(d) 后 scores 对角=%.4f、非对角=%.4f'
                 % (worked['diag'], worked['off']))
    lines.append('    softmax 权重（每行）：[%.4f, %.4f]（两行对称）'
                 % (worked['attn00'], worked['attn01']))
    lines.append('    ctx[0] = [%.4f, %.4f]（主路径 vs 参考实现 max|Δ|=%.2e）'
                 % (worked['ctx00'], worked['ctx01'], worked['maxdiff']))

    lines.append('\n【D】组件不变量（对账，非肉眼）')
    lines.append('    softmax 每行和 = 1：最大偏差 %.2e' % invariants['softmax_err'])
    lines.append('    注意力权重非负：最小值 %.4f（应 ≥ 0）' % invariants['attn_min'])
    lines.append('    LayerNorm 每行均值≈0：最大 |mean| = %.2e' % invariants['ln_mean_err'])
    lines.append('    LayerNorm 每行方差≈1：最大 |var-1| = %.2e' % invariants['ln_var_err'])

    lines.append('\n【E】全局混合：扰动 token0，看每个位置输出的变化量')
    lines.append('    （自注意力让任意两位置直接交互，改一处、全序列都变）')
    for i, delta in enumerate(mix):
        lines.append('    位置 %d 输出变化 L2 = %.4f' % (i, delta))
    lines.append('    （数值依赖固定 seed 的随机初始化，换 seed 会变；但"所有位置均 > 0"这一性质不变）')

    lines.append('\n【F】Post-Norm 块前后激活统计')
    lines.append('    输入  X    mean=%.4f  std=%.4f'
                 % (invariants['x_mean'], invariants['x_std']))
    lines.append('    残差 h     mean=%.4f  std=%.4f'
                 % (invariants['h_mean'], invariants['h_std']))
    lines.append('    输出 out   mean=%.4f  std=%.4f'
                 % (invariants['o_mean'], invariants['o_std']))
    lines.append('    → LayerNorm 把每行拉回零均值、单位方差，是数值稳定的关键')

    return '\n'.join(lines)


# ---------------------------------------------------------------- 配图
def draw(X, P, attn, mix, h):
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

    im = axes[0].imshow(attn, cmap='Greens', vmin=0, vmax=1)
    axes[0].set_title('注意力权重矩阵（每行 softmax，和为 1）')
    axes[0].set_xlabel('被看的位置 j')
    axes[0].set_ylabel('发起的位置 i')
    for i in range(len(attn)):
        for j in range(len(attn[0])):
            axes[0].text(j, i, '%.2f' % attn[i][j], ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    rowsums = [sum(r) for r in attn]
    axes[1].bar(range(len(rowsums)), rowsums, color='#4a9d6a')
    axes[1].axhline(1.0, color='#cc5a5a', ls='--', lw=1)
    axes[1].set_ylim(0.9, 1.05)
    axes[1].set_title('softmax 每行和（应恒为 1）')
    axes[1].set_xlabel('行 i')
    axes[1].set_ylabel('行和')

    means = [sum(r) / len(r) for r in h]
    vars_ = [sum((v - sum(r) / len(r)) ** 2 for v in r) / len(r) for r in h]
    axes[2].scatter(means, vars_, color='#8888cc', s=60)
    axes[2].axhline(1.0, color='#d08c4a', ls='--', lw=1)
    axes[2].axvline(0.0, color='#d08c4a', ls='--', lw=1)
    axes[2].set_title('LayerNorm 后：均值≈0、方差≈1')
    axes[2].set_xlabel('每行均值')
    axes[2].set_ylabel('每行方差')

    axes[3].bar(range(len(mix)), mix, color='#d08c4a')
    axes[3].set_title('扰动 token0 → 各位置输出变化')
    axes[3].set_xlabel('输出位置')
    axes[3].set_ylabel('L2 变化量')

    path = os.path.join(OUT, 'fig-03-transformer.png')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    d_model, seq, d_ff = 8, 4, 16
    X = [[rng.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq)]
    P = make_block_params(rng, d_model, d_ff)

    # ---------------- B. 维度流动 trace
    Q = matmul(X, P['Wq'])
    attn_out, attn, scores = single_head_attention(X, P['Wq'], P['Wk'], P['Wv'], P['Wo'])
    out, _, h = post_norm_block(X, P)
    trace_lines = [
        '输入 X                shape=%s   （seq_len × d_model）' % (shape(X),),
        'Q = X·Wq             shape=%s   （Wq: d_model×d_model，形状不变）' % (shape(Q),),
        'scores = QKᵀ/√d      shape=%s   （seq × seq：每个位置对每个位置的注意力打分）' % (shape(scores),),
        'attn = softmax       shape=%s   （每行归一化，和为 1）' % (shape(attn),),
        'ctx = attn·V         shape=%s   （加权求和，回到 seq × d_model）' % (shape(matmul(attn, matmul(X, P['Wv']))),),
        'attn_out = ctx·Wo    shape=%s   （输出投影，形状不变）' % (shape(attn_out),),
        'h = LN(X + attn_out) shape=%s   （残差 + Post-Norm，形状不变）' % (shape(h),),
        'ffn 中间层           shape=%s   （d_model → d_ff，先升维）' % (shape(matmul(h, P['W1'])),),
        'out = LN(h + FFN(h)) shape=%s   （走完一块，形状与输入一致）' % (shape(out),),
    ]

    # ---------------- C. 数值例：单位投影 + 正交 token（能手算）
    Xo = [[1.0, 0.0], [0.0, 1.0]]
    I2 = [[1.0, 0.0], [0.0, 1.0]]
    ref_attn, ref_ctx = ref_attention_identity(Xo)
    # 主路径：把 Wq=Wk=Wv=Wo=I、d=2 塞进 single_head_attention
    main_out, main_attn, main_scores = single_head_attention(Xo, I2, I2, I2, I2)
    worked = dict(
        diag=main_scores[0][0], off=main_scores[0][1],
        attn00=main_attn[0][0], attn01=main_attn[0][1],
        ctx00=main_out[0][0], ctx01=main_out[0][1],
        maxdiff=max(abs(main_out[i][j] - ref_ctx[i][j]) for i in range(2) for j in range(2)))

    # ---------------- D. 组件不变量
    softmax_err = max(abs(sum(r) - 1.0) for r in attn)
    attn_min = min(v for r in attn for v in r)
    ln_mean_err = max(abs(sum(r) / len(r)) for r in h)
    ln_var_err = max(abs(sum((v - sum(r) / len(r)) ** 2 for v in r) / len(r) - 1.0) for r in h)
    x_mean, x_std = mean_std(flat(X))
    h_mean, h_std = mean_std(flat(h))
    o_mean, o_std = mean_std(flat(out))
    invariants = dict(softmax_err=softmax_err, attn_min=attn_min,
                      ln_mean_err=ln_mean_err, ln_var_err=ln_var_err,
                      x_mean=x_mean, x_std=x_std, h_mean=h_mean, h_std=h_std,
                      o_mean=o_mean, o_std=o_std)

    # ---------------- E. 全局混合：扰动 token0，看所有位置输出变化
    Xp = [row[:] for row in X]
    Xp[0] = [v + 0.5 for v in Xp[0]]
    out_p, _, _ = post_norm_block(Xp, P)
    mix = [math.sqrt(sum((out[i][j] - out_p[i][j]) ** 2 for j in range(d_model)))
           for i in range(seq)]

    # ---------------- 报表 / 落盘
    report = fmt_report(d_model, seq, d_ff, X, P, trace_lines, worked, invariants, mix)
    print()
    print(report)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(os.path.join(OUT, 'attention.json'), 'w', encoding='utf-8') as f:
        json.dump({'d_model': d_model, 'seq': seq, 'd_ff': d_ff,
                   'attn': attn, 'worked_example': worked}, f, ensure_ascii=False, indent=2)

    fig = draw(X, P, attn, mix, h)
    if fig:
        print('\n[配图] %s' % fig)

    # ---------------- 自校验：拿能独立算出的已知答案对账
    print('\n' + '=' * 70)
    print('【G】自校验（对账，不是靠肉眼看图）')
    print('=' * 70)

    print('\n  1. 维度流动：一块 Transformer 前后形状不变')
    assert shape(X) == shape(out) == (seq, d_model), '块前后形状应一致'
    print('     输入 %s → 输出 %s（seq_len × d_model 守恒）' % (shape(X), shape(out)))

    print('\n  2. softmax：每行和为 1、全部非负')
    assert softmax_err < 1e-9, 'softmax 行和偏离 1：%.2e' % softmax_err
    assert attn_min >= 0.0, 'softmax 出现负值'
    print('     %d 行，行和最大偏差 %.2e，最小权重 %.4f' % (len(attn), softmax_err, attn_min))

    print('\n  3. LayerNorm：每行零均值、单位方差')
    assert ln_mean_err < 1e-6, 'LN 均值偏离 0：%.2e' % ln_mean_err
    assert ln_var_err < 1e-4, 'LN 方差偏离 1：%.2e' % ln_var_err
    print('     最大 |mean|=%.2e，最大 |var-1|=%.2e' % (ln_mean_err, ln_var_err))

    print('\n  4. 数值例对账：单位投影 + 正交 token，主路径 == 参考实现')
    assert worked['maxdiff'] < 1e-12, '主路径与参考实现不一致 %.2e' % worked['maxdiff']
    # 手算：scores 对角 = 1/√2 ≈ 0.7071，非对角 = 0
    assert abs(worked['diag'] - 1 / math.sqrt(2)) < 1e-9, '对角打分错误'
    assert abs(worked['off']) < 1e-12, '正交 token 非对角打分应为 0'
    print('     scores 对角=%.4f(=1/√2)、非对角=%.1e；ctx[0]=[%.4f, %.4f]；max|Δ|=%.1e'
          % (worked['diag'], worked['off'], worked['ctx00'], worked['ctx01'], worked['maxdiff']))

    print('\n  5. 全局混合：扰动一个 token，所有位置输出都改变')
    assert all(d > 1e-6 for d in mix), '存在未受扰动影响的位置，注意力未全局混合'
    print('     改 token0 后各位置输出 L2 变化：%s（均 > 0）'
          % ', '.join('%.3f' % d for d in mix))

    print('\n  6. 确定性：同 seed 重建，输出逐元素一致')
    rng2 = random.Random(SEED)
    X2 = [[rng2.uniform(-1, 1) for _ in range(d_model)] for _ in range(seq)]
    P2 = make_block_params(rng2, d_model, d_ff)
    out2, _, _ = post_norm_block(X2, P2)
    maxd = max(abs(out[i][j] - out2[i][j]) for i in range(seq) for j in range(d_model))
    assert X == X2 and maxd < 1e-12, '重建不确定，max|Δ|=%.2e' % maxd
    print('     两次构建输出 max|Δ| = %.1e（当前纯 Python 实现下逐元素精确一致；'
          '换向量化实现后应用容差比较）' % maxd)

    print('\n  全部断言通过。')
    print('\n[产物] 报表 / 注意力权重 / 配图 → %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
