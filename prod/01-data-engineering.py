#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产篇 01 · 数据工程：从手写 demo 到企业级流水线

对应文章：同目录 01-数据工程.md（建议先读 基础篇/01-数据工程.py 与 .md）
运行：    python3 01-数据工程.py
依赖：    numpy + torch（CPU 即可，本机真跑）；datatrove / text-dedup 为可选工业库，
          装了就跑、没装就打印"未在本机验证"并给出等价调用示例（见文末说明）。
产物：    ./output/生产篇01-数据工程/ 下的报表、loss 曲线图、去重结果 jsonl

基础篇把机理讲透、demo 跑通；生产篇从"上得了规模、扛得住线上"的角度重做四件事：

    A. 手写 MinHash + LSH —— 生产上大规模网页去重的主力，与基础篇的 SimHash 对照
    B. SimHash 向量化加速 —— 把基础篇的纯 Python 双重循环换成 numpy 批量位运算，验证等价并测速
    C. 去重前后 loss 对比 —— 用一个小字符级语言模型，硬碰硬地证明"去重到底有没有用"
    D. datatrove / text-dedup 工业库调用版 —— 企业级流水线长什么样（能 import 就真跑）
    E. 线上问题清单 —— 重复污染 / 配比漂移 / 流式中断恢复（含一个可跑的断点续跑 demo）

哪些是本机真跑验证的、哪些只是"工业库调用示例未验证"，运行结束会在报表里逐条标明。
"""

import hashlib
import json
import os
import random
import re
import struct
import sys
import time
from collections import Counter, defaultdict

import numpy as np

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '生产篇01-数据工程')

# 运行期逐项登记：这一步是"本机真跑验证"还是"工业库示例未验证"
VERIFY_LOG = []


def log_verify(item, status, note=''):
    VERIFY_LOG.append((item, status, note))
    print('  [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))


# ==================================================================
#  A. 合成语料（制造"重复污染"，供 MinHash 去重与 loss 对比使用）
# ==================================================================
# 复用基础篇的思路：句模板 + 槽位 + 每篇独有"指纹尾"。区别是这一版故意造大量
# 近似重复（同一篇被多个网站转载、各改几个字），因为生产上去重的主要对象就是它们。

TEMPLATES = [
    '在{a}场景下，{b}往往决定了{c}的上限，工程上通常先{d}再谈优化。',
    '很多团队低估了{a}的成本，等{c}压测出来才发现瓶颈在{b}。',
    '{a}看起来是参数问题，实际是{b}问题，动手前先把{c}量一遍。',
    '我们踩过的坑是{d}没做，结果{c}在长尾上抖得厉害，后来靠{b}兜住。',
    '离线指标涨了不代表线上能用，{c}要单独看，尤其是{d}之后。',
]
SLOT_A = ['分布式训练', '在线推理', '数据清洗', '检索增强', '模型压缩', '批量离线']
SLOT_B = ['显存分片', '批大小', '缓存命中率', '算子融合', '数据配比', '调度策略']
SLOT_C = ['吞吐', '首 token 时延', '单位成本', '收敛速度', '稳定性', '显存占用']
SLOT_D = ['压测到 P99', '把日志打全', '量一遍真实分布', '拆成两阶段', '加一层缓存']


def _fill(rng, tpl):
    return (tpl.replace('{a}', rng.choice(SLOT_A)).replace('{b}', rng.choice(SLOT_B))
               .replace('{c}', rng.choice(SLOT_C)).replace('{d}', rng.choice(SLOT_D)))


def make_doc(rng):
    sents = [_fill(rng, t) for t in rng.sample(TEMPLATES, rng.randint(3, 5))]
    sig = hashlib.blake2b(str(rng.random()).encode(), digest_size=5).hexdigest().upper()
    sents.append('文档指纹 %s；序号 %s。' % (sig, ' '.join(str(rng.randint(1000, 9999)) for _ in range(4))))
    return ''.join(sents)


def make_corpus(n_unique=400, dup_rounds=3, seed=SEED):
    """造一份带"重复污染"的语料。

    返回 (docs, groups)：
      docs    每篇 {'id','text','uid'}，uid 是它真正来自哪篇唯一文档
      groups  真值重复簇 {uid: [id,...]}，用于给去重算法打分

    污染方式：每篇唯一文档，除了自己，还派生 dup_rounds 篇近似副本
    （改标点 + 加不同来源行），模拟"一稿多发、各网站改几个字转载"。
    """
    rng = random.Random(seed)
    docs, groups = [], defaultdict(list)

    def add(text, uid):
        did = 'd%05d' % len(docs)
        docs.append({'id': did, 'text': text, 'uid': uid})
        groups[uid].append(did)
        return did

    for u in range(n_unique):
        uid = 'u%04d' % u
        base = make_doc(rng)
        add(base, uid)
        for r in range(dup_rounds):
            # 近似副本：把句号换成逗号（改动前 r+1 处）+ 追加不同来源行
            variant = base.replace('。', '，', r + 1)
            variant += rng.choice(['（来源：网络综合整理）', '（转载自公开渠道）',
                                   '（编辑：站点管理员）', '（版权归原作者所有）'])
            add(variant, uid)

    rng.shuffle(docs)          # 打乱顺序，id/uid 不变
    return docs, dict(groups)


# ==================================================================
#  B. 手写 MinHash + LSH —— 生产上大规模去重的主力
# ==================================================================
# SimHash 用汉明距离近似"加权特征向量的余弦相似度"（符号投票），MinHash 估计的是"集合的
# Jaccard 相似度"。网页去重里 MinHash + LSH 更常用，因为它对"集合重叠比例"的估计无偏、
# 且天然适配分布式（每篇文档的签名可独立计算，签名再分桶）。

# 用 2^31-1（梅森素数，M2147483647）作模数：a、b、shingle 哈希都压到 < 2^31，
# 于是 a*x+b < 2^62，既落在 uint64 内不溢出，又能做"真"模运算（Mersenne 折叠）。
# 为什么不用 2^61-1：要让 (a*x+b) mod p 成为真正的全排列哈希，a 必须取满 [1,p)。
# 若 a 取满 2^61，a*x（x<2^32）会溢出 uint64，只能像 datasketch 那样用"位与取低 61 位"
# 近似（那其实是 mod 2^61，不是 mod p）；缩到 31 位空间就能干净地做真模，无需近似。
MERSENNE_PRIME = (1 << 31) - 1
MAX_HASH = MERSENNE_PRIME
_SHIFT = np.uint64(31)
_MASK = np.uint64(MERSENNE_PRIME)


def shingles(text, k=3):
    """字符 k-shingle 集合。字符级对中文友好；英文/多语言生产常用词级 5-gram
    （datatrove 默认 n_grams=5 即词级，能避开字符级噪声）。"""
    return set(text[i:i + k] for i in range(max(0, len(text) - k + 1)))


def _hash_shingle(token):
    """shingle → [0, 2^31-1) 的整数（先 blake2b 到 32 位，再压进 31 位空间）。"""
    v = struct.unpack('<I', hashlib.blake2b(token.encode('utf-8'), digest_size=4).digest())[0]
    return v % MERSENNE_PRIME


def _mersenne_mod(x, p=np.uint64(MERSENNE_PRIME)):
    """对梅森素数 p=2^31-1 做真取模：折叠 x = 低位 + 溢出高位，重复几次直到收敛，再条件减 p。

    这是真 mod p（不是 `x & p` 的 mod 2 幂近似），保证 (a*x+b) mod p 是均匀全排列、
    Jaccard 估计严格无偏。输入 x < 2^62（a、b、x 均 < 2^31 保证不溢出）。
    """
    for _ in range(4):
        x = (x & p) + (x >> _SHIFT)
    x = np.where(x >= p, x - p, x)
    return np.where(x >= p, x - p, x)


class MinHasher:
    """num_perm 个哈希函数，用通用哈希 (a*x+b) mod p（p=2^31-1）生成，一次算出一篇的签名。

    a∈[1,p)、b∈[0,p)、x<p ⇒ a*x+b < 2^62：真模 p 且 uint64 不溢出。
    """

    def __init__(self, num_perm=128, seed=SEED):
        rng = random.Random(seed)
        self.num_perm = num_perm
        self.a = np.array([rng.randint(1, MERSENNE_PRIME - 1) for _ in range(num_perm)], dtype=np.uint64)
        self.b = np.array([rng.randint(0, MERSENNE_PRIME - 1) for _ in range(num_perm)], dtype=np.uint64)

    def signature(self, shingle_set):
        """返回长度 num_perm 的签名：第 i 位 = 第 i 个哈希函数作用下所有 shingle 的最小值。"""
        if not shingle_set:
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        hs = np.array([_hash_shingle(s) for s in shingle_set], dtype=np.uint64)   # [S]
        # perm = (a*x + b) mod p → [num_perm, S]，每行取 min（真 Mersenne 模，非位与近似）
        perm = _mersenne_mod(self.a[:, None] * hs[None, :] + self.b[:, None])
        return perm.min(axis=1)


def jaccard_from_sig(s1, s2):
    """两条签名相同位的比例，就是 Jaccard 相似度的无偏估计。

    估计标准差 ≈ sqrt(J(1-J)/num_perm)，上界 0.5/sqrt(num_perm)（J=0.5 时取到）。
    num_perm=128 → 上界约 0.044；要分辨 0.05 的 Jaccard 差异，num_perm 需上到数百。
    """
    return float((s1 == s2).mean())


def true_jaccard(set1, set2):
    return len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0.0


def minhash_lsh(sigs, ids, threshold=0.7, bands=None):
    """LSH 分段：把签名切成 bands 段，任一段完全相同即候选对。

    阈值与段数的关系（经典 LSH S 曲线）：近似 threshold ≈ (1/bands)^(1/rows)。
    这里给定 threshold 反推一个合理的 bands，让 S 曲线的陡升点落在 threshold 附近。
    """
    num_perm = len(next(iter(sigs.values())))
    if bands is None:
        # 在 num_perm 的约数里挑一个让 (1/b)^(1/(num_perm/b)) 最接近 threshold 的分法
        best, best_gap = 1, 9e9
        for b in range(1, num_perm + 1):
            if num_perm % b:
                continue
            rows = num_perm // b
            t = (1.0 / b) ** (1.0 / rows)
            if abs(t - threshold) < best_gap:
                best, best_gap = b, abs(t - threshold)
        bands = best
    rows = num_perm // bands

    buckets = defaultdict(list)
    for i in ids:
        sig = sigs[i]
        for bidx in range(bands):
            key = (bidx, tuple(sig[bidx * rows:(bidx + 1) * rows].tolist()))
            buckets[key].append(i)

    cand = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        members = sorted(set(members))
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                cand.add((members[x], members[y]))
    return cand, bands, rows


def run_minhash_dedup(docs, threshold=0.7, num_perm=128):
    """完整跑一遍 MinHash-LSH 去重：建签名 → LSH 候选 → Jaccard 确认 → 并查集聚簇。"""
    hasher = MinHasher(num_perm=num_perm)
    shingle_sets = {d['id']: shingles(d['text']) for d in docs}
    sigs = {d['id']: hasher.signature(shingle_sets[d['id']]) for d in docs}
    ids = [d['id'] for d in docs]

    cand, bands, rows = minhash_lsh(sigs, ids, threshold)

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in cand:
        if jaccard_from_sig(sigs[a], sigs[b]) >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    clusters = defaultdict(list)
    for d in docs:
        clusters[find(d['id'])].append(d)
    keep, drop = [], []
    for members in clusters.values():
        members.sort(key=lambda d: (-len(d['text']), d['id']))
        keep.append(members[0])
        drop.extend(members[1:])
    return {'keep': keep, 'drop': drop, 'bands': bands, 'rows': rows,
            'cand': len(cand), 'sigs': sigs, 'shingles': shingle_sets,
            'threshold': threshold, 'num_perm': num_perm}


def score_against_truth(docs, result, groups):
    """拿真值簇给去重结果打分：召回 = 簇塌成一篇的比例；误删 = 被错并的不同 uid。"""
    kept_ids = {d['id'] for d in result['keep']}
    collapsed = sum(1 for uid, ids in groups.items() if sum(i in kept_ids for i in ids) <= 1)
    recall = collapsed / len(groups)

    # 误删：同一簇里本该只留一篇，但如果两篇不同 uid 被并到一起就是误删
    kept_by_uid = defaultdict(int)
    for d in docs:
        if d['id'] in kept_ids:
            kept_by_uid[d['uid']] += 1
    # 每个 uid 应恰好留 1 篇；留 0 篇说明它被错误并进了别的簇
    lost_uid = sum(1 for uid in groups if kept_by_uid[uid] == 0)
    fpr = lost_uid / len(groups)
    return {'recall': recall, 'fpr': fpr, 'collapsed': collapsed,
            'n_groups': len(groups), 'kept': len(result['keep'])}


# ==================================================================
#  C. SimHash 向量化加速 —— 把纯 Python 双重循环换成 numpy 批量位运算
# ==================================================================
def simhash_pure(text, bits=64, n=3):
    """基础篇的纯 Python 版本：每篇文档做 特征数 × bits 次加法。"""
    feats = Counter(text[i:i + n] for i in range(max(0, len(text) - n + 1)))
    v = [0] * bits
    for gram, w in feats.items():
        h = int.from_bytes(hashlib.blake2b(gram.encode('utf-8'), digest_size=8).digest(), 'big')
        for b in range(bits):
            v[b] += w if (h >> b) & 1 else -w
    fp = 0
    for b in range(bits):
        if v[b] > 0:
            fp |= 1 << b
    return fp


def _gram_sign(gram, bits=64):
    """一个 gram → 长度 bits 的 ±1 向量（第 b 位对应哈希的第 b 比特），与纯 Python 版位序一致。"""
    h = int.from_bytes(hashlib.blake2b(gram.encode('utf-8'), digest_size=8).digest(), 'big')
    bidx = np.arange(bits, dtype=np.uint64)
    bitsarr = ((np.uint64(h) >> bidx) & np.uint64(1)).astype(np.int8)
    return bitsarr * 2 - 1                          # 1 → +1，0 → -1


def simhash_batch(texts, bits=64, n=3):
    """numpy 批量向量化版本：这才是生产上真正的加速点。

    两处提速：① 全局 gram 哈希缓存——语料里大量 gram 是重复的，只哈希一次；
    ② 每篇文档的"加权符号求和"用一次矩阵乘法 w @ M 算完 64 位，取代 Python 内层循环。
    与纯 Python 版逐篇产生完全相同的指纹（主流程会断言校验）。
    """
    cache = {}
    counters = []
    for t in texts:
        c = Counter(t[i:i + n] for i in range(max(0, len(t) - n + 1)))
        counters.append(c)
        for g in c:
            if g not in cache:
                cache[g] = _gram_sign(g, bits)
    fps = []
    for c in counters:
        if not c:
            fps.append(0)
            continue
        grams = list(c)
        M = np.stack([cache[g] for g in grams])                 # [G, bits]
        w = np.array([c[g] for g in grams], dtype=np.float64)   # [G]
        v = w @ M                                               # [bits] 加权求和
        fp = 0
        pos = np.nonzero(v > 0)[0]
        for b in pos:
            fp |= 1 << int(b)
        fps.append(fp)
    return fps


# ==================================================================
#  D. 去重前后 loss 对比 —— 用一个小字符级语言模型证明"去重有没有用"
# ==================================================================
def build_char_vocab(texts):
    vocab = {}
    for t in texts:
        for ch in t:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    vocab.setdefault('<unk>', len(vocab))
    return vocab


def encode(texts, vocab, seq_len):
    """把文本拼成一条长 token 流，再切成 seq_len 的窗口（next-char 预测）。"""
    ids = [vocab.get(ch, vocab['<unk>']) for t in texts for ch in t]
    xs, ys = [], []
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        xs.append(ids[i:i + seq_len])
        ys.append(ids[i + 1:i + seq_len + 1])
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def train_char_lm(train_x, train_y, val_x, val_y, vocab_size, steps=300,
                  hidden=128, lr=3e-3, batch=32, seed=SEED):
    """训练一个小 GRU 字符级语言模型，返回 (train_loss 曲线, val_loss 曲线, 最终 val loss)。"""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.head = nn.Linear(hidden, vocab_size)

        def forward(self, x):
            o, _ = self.gru(self.emb(x))
            return self.head(o)

    model = TinyLM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    tx = torch.from_numpy(train_x)
    ty = torch.from_numpy(train_y)
    vx = torch.from_numpy(val_x)
    vy = torch.from_numpy(val_y)

    train_curve, val_curve = [], []
    n = len(tx)
    for step in range(steps):
        model.train()
        idx = torch.randint(0, n, (batch,))
        xb, yb = tx[idx], ty[idx]
        logits = model(xb)
        loss = lossf(logits.reshape(-1, vocab_size), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_curve.append(float(loss.detach()))

        if (step + 1) % 20 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl = float(lossf(model(vx).reshape(-1, vocab_size), vy.reshape(-1)))
            val_curve.append((step + 1, vl))
    return train_curve, val_curve, val_curve[-1][1]


def loss_experiment(docs, groups, seq_len=64, steps=300):
    """对照实验：同样 token 预算下，"污染语料"（少量唯一文档重复填满）vs "去重语料"（更多唯一文档）。

    两者用同一个 held-out 干净验证集评估。预期：去重语料见过更多不同内容，泛化更好、val loss 更低。
    """
    import torch  # noqa: F401  (确保可用)
    rng = random.Random(SEED + 1)

    # held-out 验证集：全新唯一文档，训练时绝不出现
    val_docs = [make_doc(rng) for _ in range(120)]

    # 去重语料 = 每个 uid 只留一篇（按 uid 取代表，与 MinHash 结果的口径一致）
    seen, dedup_texts = set(), []
    for d in docs:
        if d['uid'] not in seen:
            seen.add(d['uid'])
            dedup_texts.append(d['text'])

    # 污染语料：故意只取 1/4 唯一文档、靠重复填满，把 token 量对齐到去重语料。
    # 这是"放大记忆化信号"的极端设定，不是生产 75% 污染率的直接映射（生产 75% 污染
    # 通常多样性仍在 400 篇，只是多了副本）。这里压到 1/4 是为了让 val loss 差异肉眼可见。
    target_chars = sum(map(len, dedup_texts))
    n_polluted_unique = max(20, len(dedup_texts) // 4)
    polluted_pool = dedup_texts[:n_polluted_unique]
    polluted_texts = []
    acc = 0
    while acc < target_chars:
        doc = rng.choice(polluted_pool)
        if acc + len(doc) > target_chars:            # 末尾按 token 预算精确截断，不超调
            polluted_texts.append(doc[:target_chars - acc])
            break
        polluted_texts.append(doc)
        acc += len(doc)

    # 统一词表（用去重语料 + 验证集，避免污染语料引入词表差异）
    vocab = build_char_vocab(dedup_texts + val_docs)
    vs = len(vocab)

    ded_x, ded_y = encode(dedup_texts, vocab, seq_len)
    pol_x, pol_y = encode(polluted_texts, vocab, seq_len)
    val_x, val_y = encode(val_docs, vocab, seq_len)

    print('    去重语料：%d 篇唯一文档，%d 个训练窗口' % (len(dedup_texts), len(ded_x)))
    print('    污染语料：%d 篇唯一文档重复填充，%d 个训练窗口' % (n_polluted_unique, len(pol_x)))
    print('    验证集：  %d 篇 held-out 唯一文档，%d 个窗口' % (len(val_docs), len(val_x)))

    t0 = time.time()
    ded_train, ded_val, ded_final = train_char_lm(ded_x, ded_y, val_x, val_y, vs, steps=steps)
    ded_time = time.time() - t0
    t0 = time.time()
    pol_train, pol_val, pol_final = train_char_lm(pol_x, pol_y, val_x, val_y, vs, steps=steps)
    pol_time = time.time() - t0

    return {
        'dedup': {'train': ded_train, 'val': ded_val, 'final_val': ded_final,
                  'uniq': len(dedup_texts), 'time': ded_time, 'train_windows': len(ded_x)},
        'polluted': {'train': pol_train, 'val': pol_val, 'final_val': pol_final,
                     'uniq': n_polluted_unique, 'time': pol_time, 'train_windows': len(pol_x)},
        'val_windows': len(val_x),
    }


def draw_loss_curves(exp):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [提示] 未安装 matplotlib，跳过 loss 曲线画图')
        return None
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    axes[0].plot(exp['dedup']['train'], color='#4a9d6a', label='去重语料 train loss')
    axes[0].plot(exp['polluted']['train'], color='#d08c4a', label='污染语料 train loss')
    axes[0].set_title('训练损失（污染语料因重复而更快"背下来"）')
    axes[0].set_xlabel('训练步')
    axes[0].set_ylabel('loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    dv = exp['dedup']['val']
    pv = exp['polluted']['val']
    axes[1].plot([s for s, _ in dv], [l for _, l in dv], 'o-', color='#4a9d6a', label='去重语料 val loss')
    axes[1].plot([s for s, _ in pv], [l for _, l in pv], 'o-', color='#d08c4a', label='污染语料 val loss')
    axes[1].set_title('验证损失（held-out 干净文档，去重更低=泛化更好）')
    axes[1].set_xlabel('训练步')
    axes[1].set_ylabel('val loss')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig-loss-compare.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ==================================================================
#  E. datatrove / text-dedup 工业库调用版（能 import 就真跑，否则给示例并标注未验证）
# ==================================================================
# datatrove 版本以 0.10.x 为准；不同小版本 Stage 类名/参数可能调整，使用前对照官方文档。
DATATROVE_SNIPPET = '''# ---- datatrove 0.10.x：HuggingFace 开源的大规模数据流水线（分布式、可续跑）----
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.writers import JsonlWriter
from datatrove.pipeline.filters import GopherQualityFilter
from datatrove.pipeline.dedup.minhash import (MinhashConfig, MinhashDedupSignature,
                                              MinhashDedupBuckets, MinhashDedupCluster,
                                              MinhashDedupFilter)
from datatrove.executor import LocalPipelineExecutor

cfg = MinhashConfig(n_grams=5, num_buckets=14, hashes_per_bucket=8)   # 总排列=112，词级 5-gram

# 1) 质量过滤：Gopher 规则，等价于基础篇手写规则层，但工业级、可配置、天然并行
#    tasks=总分片任务数，workers=并发进程数（workers<=tasks；按 CPU 核数设）
LocalPipelineExecutor(
    pipeline=[JsonlReader("data/raw"),
              GopherQualityFilter(min_avg_word_length=3, max_symbol_word_ratio=0.1),
              JsonlWriter("data/filtered")],
    tasks=128, workers=32, logging_dir="logs/filter").run()

# 2) MinHash 去重：签名→分桶→聚簇→过滤，四个 Stage 各起一次 Executor（任务数不同）
#    注意：MinhashDedupSignature 默认 language=english，走 spacy 词形还原，
#    需先 pip install spacy 及 en 语言模型——这正是本模块 B 手写 MinHash-LSH 替代验证的原因。
LocalPipelineExecutor(pipeline=[MinhashDedupSignature(output_folder="data/sig", config=cfg)],
                      tasks=128, workers=32, logging_dir="logs/sig").run()
LocalPipelineExecutor(pipeline=[MinhashDedupBuckets(input_folder="data/sig",
                                                    output_folder="data/bkt",
                                                    index_folder="data/index", config=cfg)],
                      tasks=16, workers=16, logging_dir="logs/bkt").run()
LocalPipelineExecutor(pipeline=[MinhashDedupCluster(input_folder="data/bkt",
                                                    output_folder="data/clu", config=cfg)],
                      tasks=16, workers=16, logging_dir="logs/clu").run()
# Filter 读"签名目录"（定位本行属于哪条签名）+ exclusion_writer（把被删的单独落盘），
# 干净输出走 pipeline 末尾的 JsonlWriter，而不是 Filter 自己的 output_folder。
LocalPipelineExecutor(pipeline=[JsonlReader("data/filtered"),
                                MinhashDedupFilter(input_folder="data/sig",
                                                   exclusion_writer=JsonlWriter("data/removed")),
                                JsonlWriter("data/deduped")],
                      tasks=128, workers=32, logging_dir="logs/filter_dup").run()
'''

TEXTDEDUP_SNIPPET = '''# ---- text-dedup：专注去重的工业库（MinHash / SimHash / SuffixArray 多种算法）----
# 命令行最省事（直接吃 HuggingFace datasets）：
#   python -m text_dedup.minhash --data <hf_dataset> --column text \\
#       --threshold 0.7 --num_perm 128 --output data/deduped
'''


def try_run_industrial():
    """datatrove / text-dedup：能 import 就核对 API 骨架，并给出可运行的调用示例。

    说明：datatrove 的 MinHash 签名阶段内部用 spacy 分词器，完整端到端跑还需要额外
    安装 spacy 及语言模型（属重依赖）。按本系列约定，这类"重依赖工业库"以
    "库已安装 + API 骨架已核对 + 等价手写实现本机真跑"的方式交付，端到端调用示例
    如实标注为未在本机验证。手写 MinHash-LSH（模块 B）就是它的等价可跑实现。
    """
    print('\n' + '=' * 70)
    print('【D】datatrove / text-dedup 工业库（库可用性 + API 骨架核对）')
    print('=' * 70)
    checks = []
    try:
        import datatrove  # noqa: F401
        import inspect
        from datatrove.pipeline.readers import JsonlReader            # noqa: F401
        from datatrove.pipeline.writers import JsonlWriter            # noqa: F401
        from datatrove.pipeline.filters import GopherQualityFilter    # noqa: F401
        from datatrove.pipeline.dedup.minhash import (MinhashConfig, MinhashDedupSignature,
                                                      MinhashDedupBuckets, MinhashDedupCluster,
                                                      MinhashDedupFilter)
        from datatrove.executor import LocalPipelineExecutor          # noqa: F401
        # 逐个核对关键 API 形状（本机实测过的方法/参数名），保证示例骨架不是凭记忆写的
        cfg_params = set(inspect.signature(MinhashConfig.__init__).parameters)
        assert {'n_grams', 'num_buckets', 'hashes_per_bucket'} <= cfg_params, 'MinhashConfig 参数不符'
        sig_params = set(inspect.signature(MinhashDedupSignature.__init__).parameters)
        assert {'output_folder', 'config', 'language'} <= sig_params, 'Signature 参数不符'
        bkt_params = set(inspect.signature(MinhashDedupBuckets.__init__).parameters)
        assert {'input_folder', 'output_folder', 'config'} <= bkt_params, 'Buckets 参数不符'
        clu_params = set(inspect.signature(MinhashDedupCluster.__init__).parameters)
        assert {'input_folder', 'output_folder', 'config'} <= clu_params, 'Cluster 参数不符'
        flt_params = set(inspect.signature(MinhashDedupFilter.__init__).parameters)
        assert {'input_folder', 'exclusion_writer'} <= flt_params, 'Filter 参数不符'
        checks.append('datatrove 0.10.x 已安装，MinhashConfig + 四阶段 Stage 的 __init__ 签名逐一核对通过')
        log_verify('datatrove 库与 API 骨架', '真跑验证',
                   'import + MinhashConfig/Signature/Buckets/Cluster/Filter 参数核对通过')
        log_verify('datatrove MinHash 端到端去重', '未验证',
                   '签名阶段默认 language=english 依赖 spacy 词形还原（重依赖），'
                   '以手写 MinHash-LSH 等价实现本机真跑替代')
    except Exception as e:
        checks.append('datatrove 不可用（%s: %s）' % (type(e).__name__, e))
        log_verify('datatrove', '未验证', '本机不可用：%s' % type(e).__name__)
    try:
        import text_dedup  # noqa: F401
        checks.append('text-dedup 已安装（提供 MinHash / SimHash / SuffixArray 多种去重）')
        log_verify('text-dedup 库', '真跑验证', 'import 成功')
    except Exception as e:
        checks.append('text-dedup 不可用（%s）' % type(e).__name__)
        log_verify('text-dedup', '未验证', '本机不可用：%s' % type(e).__name__)

    for c in checks:
        print('  - ' + c)
    print('\n  等价调用示例（datatrove 四阶段 MinHash 去重）：')
    print(DATATROVE_SNIPPET)
    print(TEXTDEDUP_SNIPPET)
    return checks


# ==================================================================
#  F. 线上问题三：流式处理中断恢复（断点续跑 demo，真跑）
# ==================================================================
def streaming_with_resume(docs, ckpt_path, batch_size=200, crash_at=None):
    """演示流式处理的断点续跑：处理进度写进 checkpoint，中断后从断点继续，不重不漏。

    crash_at：模拟在第几个 batch 崩溃（None 表示不崩）。返回 (处理总数, 是否发生过恢复, 处理过的 id 列表)。
    把已处理 id 也记进 checkpoint，续跑结束后才能校验"每篇恰好处理一次"。
    """
    state = {'offset': 0, 'processed': 0, 'done_ids': []}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            state = json.load(f)

    resumed = state['offset'] > 0
    i = state['offset']
    n_batches = 0
    while i < len(docs):
        batch = docs[i:i + batch_size]
        # ... 这里对 batch 做过滤/去重/写库 ...
        state['processed'] += len(batch)
        state['done_ids'].extend(d['id'] for d in batch)
        i += len(batch)
        state['offset'] = i
        n_batches += 1
        with open(ckpt_path, 'w') as f:
            json.dump(state, f)                    # 每批落一次盘：崩溃也只丢当前批
        if crash_at is not None and n_batches >= crash_at:
            raise RuntimeError('模拟崩溃于 offset=%d' % i)
    return state['processed'], resumed, state['done_ids']


# ==================================================================
#  报表
# ==================================================================
def report_minhash(docs, result, score, groups):
    L = []
    naive = len(docs) * (len(docs) - 1) // 2
    L.append('=' * 70)
    L.append('生产篇 01 · 数据工程报表（合成污染语料，seed=%d）' % SEED)
    L.append('=' * 70)
    L.append('')
    L.append('【A】语料与重复污染')
    L.append('    唯一文档 %d 篇，注入近似副本后共 %d 篇（污染率 %.1f%%）'
             % (len(groups), len(docs), 100 * (1 - len(groups) / len(docs))))
    L.append('    真值重复簇 %d 个（每簇 1 篇原文 + 若干近似副本）' % len(groups))
    L.append('')
    L.append('【B】手写 MinHash + LSH 去重')
    L.append('    num_perm=%d，阈值=%.2f，LSH 分段 bands=%d × rows=%d'
             % (result['num_perm'], result['threshold'], result['bands'], result['rows']))
    L.append('    两两比较 %d 组 → LSH 候选 %d 组（压到 %.1f%%，候选/文档比 %.2f）'
             % (naive, result['cand'], 100 * result['cand'] / naive, result['cand'] / len(docs)))
    L.append('    （bands 越多召回越高但候选对膨胀；生产要监控"候选对/文档数"比值，防 O(n²) 候选爆炸）')
    L.append('    去重后 %d 篇 → 留下 %d 篇，删除 %d 篇'
             % (len(docs), len(result['keep']), len(result['drop'])))
    L.append('    （注：1600→400 恰好等于唯一文档数，是因为这份合成语料的 Jaccard 两极分化'
             '（副本 80%+、无关近 0）；生产网页 Jaccard 连续分布，阈值附近有海量半重复，不会这么干净）')
    L.append('    （并查集聚簇在生产上应加簇大小上限/与簇代表二次校验，防传递性把巨量正常文档连成一簇）')
    L.append('    对照真值：召回 %.1f%%（%d/%d 簇塌成一篇），误删率 %.2f%%'
             % (100 * score['recall'], score['collapsed'], score['n_groups'], 100 * score['fpr']))
    return '\n'.join(L)


def report_minhash_quality(result, sample_n=2000, seed=SEED):
    """验证 MinHash 签名估计 Jaccard 的准确度：签名估计 vs 真实 Jaccard。"""
    rng = random.Random(seed)
    ids = list(result['sigs'])
    errs = []
    for _ in range(sample_n):
        a, b = rng.sample(ids, 2)
        est = jaccard_from_sig(result['sigs'][a], result['sigs'][b])
        tru = true_jaccard(result['shingles'][a], result['shingles'][b])
        errs.append(abs(est - tru))
    errs = np.array(errs)
    return {'mae': float(errs.mean()), 'p95': float(np.percentile(errs, 95)),
            'max': float(errs.max()), 'n': len(errs)}


# ==================================================================
#  主流程
# ==================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    print('[1/6] 造污染语料…')
    docs, groups = make_corpus()
    print('      %d 篇文档，%d 个真值重复簇' % (len(docs), len(groups)))

    print('[2/6] 手写 MinHash + LSH 去重…')
    result = run_minhash_dedup(docs, threshold=0.7, num_perm=128)
    score = score_against_truth(docs, result, groups)
    jq = report_minhash_quality(result)
    print('      召回 %.1f%%，误删 %.2f%%，Jaccard 估计 MAE=%.4f'
          % (100 * score['recall'], 100 * score['fpr'], jq['mae']))

    print('[3/6] SimHash 向量化加速对照…')
    sample = [d['text'] for d in docs[:400]]
    t0 = time.time()
    fp_pure = [simhash_pure(t) for t in sample]
    t_pure = time.time() - t0
    t0 = time.time()
    fp_vec = simhash_batch(sample)
    t_vec = time.time() - t0
    identical = sum(1 for a, b in zip(fp_pure, fp_vec) if a == b)
    speedup = t_pure / max(1e-9, t_vec)
    print('      %d 篇：纯 Python %.3fs vs numpy 批量 %.3fs（%.2fx），指纹一致 %d/%d'
          % (len(sample), t_pure, t_vec, speedup, identical, len(sample)))

    print('[4/6] 流式中断恢复 demo…')
    ckpt = os.path.join(OUT, 'stream_ckpt.json')
    if os.path.exists(ckpt):
        os.remove(ckpt)
    try:
        streaming_with_resume(docs, ckpt, batch_size=200, crash_at=3)   # 第 3 批崩溃
    except RuntimeError as e:
        print('      第一次运行崩溃：%s' % e)
    processed, resumed, done_ids = streaming_with_resume(docs, ckpt, batch_size=200)  # 续跑
    print('      续跑完成：resumed=%s，累计处理 %d 篇（应等于 %d）' % (resumed, processed, len(docs)))

    print('[5/6] 去重前后 loss 对比实验（torch CPU 小字符级 LM）…')
    exp = loss_experiment(docs, groups, seq_len=64, steps=300)
    print('      去重语料 final val loss = %.4f（%d 篇唯一文档）'
          % (exp['dedup']['final_val'], exp['dedup']['uniq']))
    print('      污染语料 final val loss = %.4f（%d 篇唯一文档）'
          % (exp['polluted']['final_val'], exp['polluted']['uniq']))

    print('[6/6] 工业库调用版…')
    dt_result = try_run_industrial()

    # ---------------- 落盘
    with open(os.path.join(OUT, 'dedup_kept.jsonl'), 'w', encoding='utf-8') as f:
        for d in result['keep']:
            f.write(json.dumps({k: d[k] for k in ('id', 'text', 'uid')}, ensure_ascii=False) + '\n')
    fig = draw_loss_curves(exp)

    # ---------------- 报表
    lines = [report_minhash(docs, result, score, groups), '']
    lines.append('【C】SimHash 向量化加速对照（%d 篇样本）' % len(sample))
    lines.append('    纯 Python 双重循环  %.4fs' % t_pure)
    lines.append('    numpy 批量（全局 gram 缓存 + 矩阵乘）  %.4fs（%.2fx）' % (t_vec, speedup))
    lines.append('    两种实现指纹完全一致：%d/%d 篇' % (identical, len(sample)))
    lines.append('')
    lines.append('【D】MinHash 签名估计 Jaccard 的准确度（%d 对样本）' % jq['n'])
    lines.append('    平均绝对误差 MAE = %.4f，P95 = %.4f，最大 = %.4f'
                 % (jq['mae'], jq['p95'], jq['max']))
    lines.append('    （MinHash 估计标准差上界 = 0.5/sqrt(num_perm) = 0.5/sqrt(%d) ≈ %.3f；'
                 'MAE 与上界同量级）' % (result['num_perm'], 0.5 / np.sqrt(result['num_perm'])))
    lines.append('')
    lines.append('【E】去重前后 loss 对比（同一 held-out 干净验证集，%d 个验证窗口）'
                 % exp['val_windows'])
    lines.append('    去重语料 %d 个训练窗口 / 污染语料 %d 个训练窗口（token 预算对齐）'
                 % (exp['dedup']['train_windows'], exp['polluted']['train_windows']))
    ded_tr = float(np.mean(exp['dedup']['train'][-50:]))
    pol_tr = float(np.mean(exp['polluted']['train'][-50:]))
    lines.append('    %-12s %10s %14s %10s' % ('语料', '唯一文档', 'train(后50均)', 'val'))
    lines.append('    %-12s %10d %14.4f %10.4f'
                 % ('去重', exp['dedup']['uniq'], ded_tr, exp['dedup']['final_val']))
    lines.append('    %-12s %10d %14.4f %10.4f'
                 % ('污染', exp['polluted']['uniq'], pol_tr, exp['polluted']['final_val']))
    gap = exp['polluted']['final_val'] - exp['dedup']['final_val']
    lines.append('    → 去重语料的 val loss 比污染语料低 %.4f（%s）'
                 % (abs(gap), '去重改善泛化' if gap > 0 else '本 run 差异很小/反向，见文章讨论'))
    lines.append('    （注：单次运行、单种子，仅演示方法与方向；生产应多种子重复报告均值±方差）')
    lines.append('')
    lines.append('【F】流式中断恢复')
    lines.append('    第 3 批模拟崩溃 → 从 checkpoint 续跑 → 累计处理 %d/%d 篇，每篇恰好一次（id 集合校验）'
                 % (processed, len(docs)))
    lines.append('')
    lines.append('=' * 70)
    lines.append('【验证状态清单】哪些是本机真跑、哪些是未验证示例')
    lines.append('=' * 70)
    for item, status, note in VERIFY_LOG:
        lines.append('    [%s] %s%s' % (status, item, ('  —— ' + note) if note else ''))

    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text + '\n')
    if fig:
        print('\n[配图] %s' % fig)

    # ---------------- 自校验
    print('\n' + '=' * 70)
    print('【自校验】')
    print('=' * 70)
    assert score['recall'] >= 0.95, 'MinHash 召回过低 %.3f' % score['recall']
    print('  1. MinHash-LSH 召回 %.1f%% ≥ 95%%  通过' % (100 * score['recall']))
    assert score['fpr'] <= 0.02, 'MinHash 误删过高 %.3f' % score['fpr']
    print('  2. MinHash-LSH 误删率 %.2f%% ≤ 2%%  通过' % (100 * score['fpr']))
    assert identical == len(sample), 'SimHash 向量化与纯 Python 结果不一致'
    print('  3. SimHash 向量化与纯 Python 指纹逐篇一致（%d/%d）  通过' % (identical, len(sample)))
    assert jq['mae'] < 1.5 / np.sqrt(result['num_perm']), 'Jaccard 估计误差过大 %.4f' % jq['mae']
    print('  4. MinHash Jaccard 估计 MAE %.4f < 理论量级 %.4f（1.5/sqrt(num_perm)）  通过'
          % (jq['mae'], 1.5 / np.sqrt(result['num_perm'])))
    assert processed == len(docs), '流式续跑处理数不对'
    all_ids = [d['id'] for d in docs]
    assert len(done_ids) == len(set(done_ids)) == len(all_ids), '流式续跑有重处理或漏处理'
    assert set(done_ids) == set(all_ids), '处理过的 id 集合与语料不一致'
    print('  5. 流式中断恢复：崩溃后续跑处理 %d/%d 篇，每篇恰好一次（id 集合校验通过）  通过'
          % (processed, len(docs)))
    kept_uids = Counter(d['uid'] for d in result['keep'])
    assert max(kept_uids.values()) == 1, '有 uid 留下多篇'
    print('  6. 去重后每个 uid 恰好留 1 篇  通过')
    print('\n  全部断言通过。')
    print('\n[产物] %s' % OUT)
    return 0


if __name__ == '__main__':
    sys.exit(main())
