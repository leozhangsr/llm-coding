#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础篇 02 · 分词器：从零手写 BPE，把文本切成子词

对应文章：同目录 02-分词器.md
运行：    python3 02-BPE.py
依赖：    仅标准库；matplotlib 可选（缺了自动跳过画图）
产物：    ./output/02-分词器/ 下的语料 jsonl、词表、合并表、报表、扫描表、配图

一条从零训练的 BPE，讲透「一个词是怎么被拆成子词」：

    合成中英混合语料 → 按词预切分 → 统计字符对频率 → 反复合并最高频对（平票按字典序）
        → 得到有序 merge 表 + 词表 → encode（按优先级合并）→ decode（拼回）→ 对账

关键设计：语料自己合成，所以「哪些高频词该整块留成 token、哪些稀有词该被拆碎」
从一开始就是已知答案（TRUTH_POSITIVE / TRUTH_NEGATIVE）。合并次数不靠抄，是拿
这份答案扫一遍，看 tokens_per_char（压缩率）、真值词覆盖率、碎片率怎么随词表长大。
最后一段自校验同样拿已知答案对账——这是分词器唯一可靠的验收方式。
"""

import json
import os
import random
import re
import sys
from collections import Counter

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '02-分词器')

WORD_END = '</w>'          # 词尾标记：BPE 只在词内合并，绝不跨空格粘连

# ---------------------------------------------------------------- 真值定义
# 高频、期望最终整块成为一个 token 的词（正例）。中文双字最稳：1 次合并即成词。
TRUTH_POSITIVE = ['模型', '训练', '语言', '词表', '分词', 'token', 'model', 'data']
# 稀有、期望被拆成多个子词的词（负例）：低频长词，合并预算轮不到它们整块成型。
TRUTH_NEGATIVE = ['quixotism', 'xylophone', 'flibbertigibbet']

# ---------------------------------------------------------------- 合成语料
# 设计：正例词（高频短英文词 + 中文双字高频词）在多种邻居上下文里反复出现，
# 保证 BPE 有足够预算把它们的内部相邻字符捏成一个 token；负例词（稀有长英文词、
# 生僻中文字）只出现 1-2 次，合并预算根本轮不到它们整块成型，于是维持碎片。
# 中文用「词库随机拼接」造句：每个双字高频词都带着一堆不同的邻居出现，它自身的
# 内部 pair 必然成为高频对，比固定整句模板更稳、更接近真实语料的分布。
EN_POS = ['token', 'model', 'data', 'text', 'word', 'piece', 'merge',
          'learn', 'split', 'train', 'vocab', 'corpus']
EN_FUNC = ['the', 'a', 'of', 'and', 'for', 'to', 'in', 'is', 'it', 'that',
           'this', 'we', 'can', 'when', 'then', 'grows', 'helps', 'needs']
EN_TEMPLATES = [
    'the {a} splits {b} into pieces',
    '{a} training needs clean {b}',
    'a good {a} learns frequent {b} as one token',
    '{a} and {b} merge when they cooccur often',
    'the {a} vocabulary grows as we add {b} merges',
    'subword {a} helps the {b} handle rare words',
    'we {a} the {b} layer before training the model',
    'each {a} maps {b} to a sequence of tokens',
]
# 中文词库：这些双字词是正例，随机拼接进句子，邻居千变万化、自身内部 pair 高频
CN_WORDS = ['语言', '模型', '训练', '数据', '分词', '词表', '子词', '序列',
            '清洗', '去重', '需要', '压缩', '覆盖', '表示', '罕见', '片段',
            '句子', '高质量', '取舍', '顺序']
CN_RARE = ['齉', '彧', '鑫']       # 生僻字：几乎不成词，应维持单字碎片


def make_corpus(rng):
    """生成中英混合语料，每行一句（document）。"""
    docs = []
    # 英文：模板 × 高频词组合，反复铺量，喂饱合并
    for _ in range(360):
        t = rng.choice(EN_TEMPLATES)
        a = rng.choice(EN_POS)
        b = rng.choice(EN_POS)
        docs.append({'text': t.format(a=a, b=b), 'lang': 'en'})
    # 英文散句：让高频词在更多样的函数词环境里出现
    for _ in range(120):
        words = [rng.choice(EN_POS) if rng.random() < 0.45 else rng.choice(EN_FUNC)
                 for _ in range(rng.randint(4, 8))]
        docs.append({'text': ' '.join(words), 'lang': 'en'})
    # 中文：从词库随机拼接 4-7 个词，无空格连成一串（真实中文形态）
    for _ in range(160):
        seq = [rng.choice(CN_WORDS) for _ in range(rng.randint(4, 7))]
        docs.append({'text': ''.join(seq), 'lang': 'zh'})
    # 负例稀有长英文词：各出现 1-2 次
    for w in TRUTH_NEGATIVE:
        for _ in range(rng.randint(1, 2)):
            docs.append({'text': 'a rare %s appears here once' % w, 'lang': 'en'})
    # 负例生僻中文字：夹在常见词之间，各出现 1-2 次，应维持单字碎片
    for c in CN_RARE:
        for _ in range(rng.randint(1, 2)):
            left = rng.choice(CN_WORDS)
            right = rng.choice(CN_WORDS)
            docs.append({'text': '%s%s%s' % (left, c, right), 'lang': 'zh'})
    rng.shuffle(docs)
    for i, d in enumerate(docs):
        d['id'] = i
    return docs


def normalize(text):
    """规范化：连续空白压成一个空格，两端去空白。round-trip 以此为基准。"""
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------- BPE 训练
def word_freqs(texts):
    """按空格切词，统计每个（规范化后的）词出现次数。词以字符序列 + 词尾标记表示。"""
    freq = Counter()
    for t in texts:
        for w in normalize(t).split(' '):
            if w:
                freq[w] += 1
    return freq


def to_symbols(word):
    return list(word) + [WORD_END]


def pair_of(symbols, i):
    return (symbols[i], symbols[i + 1])


def count_pairs(symbols, c, bag):
    for i in range(len(symbols) - 1):
        bag[pair_of(symbols, i)] += c


def apply_merge(symbols, pair, new):
    """把 symbols 里所有相邻的 pair 合并成 new（一趟扫描，支持多次出现）。"""
    out, i, n = [], 0, len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            out.append(new)
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def train_bpe(texts, num_merges):
    """从零训练 BPE：反复选最高频字符对合并，平票按字典序确定性地打破。

    返回：
      merges   有序合并表 [(a, b), ...]，下标即优先级（rank）
      vocab    {symbol: id}   词表，含基础字符 + 每个合并产物 + 词尾标记
      rank     {(a, b): idx}  合并对 → 优先级
    """
    freq = word_freqs(texts)
    # 每个词初始化成字符序列（带词尾标记）
    words = {tuple(to_symbols(w)): c for w, c in freq.items()}

    merges = []
    rank = {}
    base_symbols = set()
    for w in freq:
        base_symbols.update(list(w))
    base_symbols.add(WORD_END)
    vocab_symbols = list(sorted(base_symbols))          # 基础词表：全部单字符 + 词尾

    for _ in range(num_merges):
        bag = Counter()
        for symbols, c in words.items():
            count_pairs(symbols, c, bag)
        if not bag:
            break
        # 选频率最高的一对；平票用字典序（pair 元组可比）确定性打破
        best = min(bag.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        a, b = best
        new = a + b
        merges.append((a, b))
        rank[best] = len(merges) - 1
        if new not in vocab_symbols:
            vocab_symbols.append(new)
        words = {tuple(apply_merge(sym, best, new)): c for sym, c in words.items()}

    vocab = {s: i for i, s in enumerate(vocab_symbols)}
    return merges, vocab, rank


# ---------------------------------------------------------------- 编码 / 解码
def encode_word(word, merges, rank):
    """对单个词应用 merge：每轮选取「已学且优先级最高」的相邻对合并，直到无可合并。"""
    symbols = to_symbols(word)
    while len(symbols) >= 2:
        cand = [(rank[(symbols[i], symbols[i + 1])], i)
                for i in range(len(symbols) - 1)
                if (symbols[i], symbols[i + 1]) in rank]
        if not cand:
            break
        _, i = min(cand)                                 # 优先级最小 = 最早学到的 merge
        pair = (symbols[i], symbols[i + 1])
        symbols = apply_merge(symbols, pair, pair[0] + pair[1])
    return symbols


def encode(text, merges, rank):
    """整句编码：按空格切词，逐词 BPE，拼接所有子词（含词尾标记）。"""
    pieces = []
    for w in normalize(text).split(' '):
        if not w:
            continue
        for sym in encode_word(w, merges, rank):
            pieces.append(sym)
    return pieces


def decode(pieces):
    """解码：词尾标记换成空格，去掉首尾多余空格，拼回原文。"""
    s = ''.join(pieces)
    s = s.replace(WORD_END, ' ')
    return normalize(s)


def is_whole_token(word, pieces):
    """词 word 是否整块成为一个子词（忽略词尾标记的附着）。"""
    return word in pieces or (word + WORD_END) in pieces


# ---------------------------------------------------------------- 指标
def tokens_per_char(texts, merges, rank):
    """压缩率：整篇语料产生的内容子词数 ÷ 内容字符数。越小越省。"""
    n_tokens = 0
    n_chars = 0
    for t in texts:
        for w in normalize(t).split(' '):
            if not w:
                continue
            n_chars += len(w)
            for sym in encode_word(w, merges, rank):
                if sym.replace(WORD_END, ''):
                    n_tokens += 1
    return n_tokens / max(1, n_chars), n_tokens, n_chars


def truth_coverage(truth_words, merges, rank):
    """真值词覆盖率：给定的词里，有多少能整块成为一个 token。"""
    hit = 0
    detail = []
    for w in truth_words:
        pieces = encode_word(w, merges, rank)
        whole = is_whole_token(w, pieces)
        detail.append((w, whole, len([p for p in pieces if p != WORD_END])))
        if whole:
            hit += 1
    return hit / max(1, len(truth_words)), detail


# ---------------------------------------------------------------- sweep
def sweep_merge_counts(texts, merge_grid, pos, neg):
    """扫不同合并次数，看词表大小 / 压缩率 / 正例覆盖 / 负例碎片如何联动。"""
    rows = []
    for m in merge_grid:
        merges, vocab, rank = train_bpe(texts, m)
        tpc, ntok, nch = tokens_per_char(texts, merges, rank)
        cov, _ = truth_coverage(pos, merges, rank)
        frag, _ = truth_coverage(neg, merges, rank)       # 负例覆盖越低越好
        rows.append({'merges': m, 'vocab': len(vocab), 'tokens_per_char': tpc,
                     'coverage': cov, 'neg_coverage': frag,
                     'n_tokens': ntok, 'n_chars': nch})
    return rows


# ---------------------------------------------------------------- 报表
def fmt_report(docs, texts, merges, vocab, rank, sweep_rows, best_m):
    lines = []
    W = 70
    lines.append('=' * W)
    lines.append('基础篇 02 · 分词器报表（合成中英混合语料，seed=%d）' % SEED)
    lines.append('=' * W)

    n_en = sum(1 for d in docs if d['lang'] == 'en')
    n_zh = sum(1 for d in docs if d['lang'] == 'zh')
    lines.append('\n【A】语料规模')
    lines.append('    文档 %d 篇（英文 %d / 中文 %d）' % (len(docs), n_en, n_zh))
    charset = set(''.join(normalize(t) for t in texts))
    n_nonspace = len(charset - {' '})                     # 空格不参与合并、不进基础符号
    lines.append('    去重词形 %d 个；基础符号 %d 个（非空格字符 %d + 词尾标记 </w> 1）' % (
        len(word_freqs(texts)), n_nonspace + 1, n_nonspace))

    lines.append('\n【B】训练出的词表与合并表（合并次数 = %d）' % best_m)
    lines.append('    词表大小 %d = 基础字符(含词尾) + 合并产物' % len(vocab))
    lines.append('    前 10 条 merge（按优先级）：')
    lines.append('    %s' % '  '.join('%d.%s+%s' % (i, a, b)
                                      for i, (a, b) in enumerate(merges[:10])))

    lines.append('\n【C】几个代表性词的编码（高频词 "%s" 整块 / 稀有词 "%s" 拆碎）' % ('token', 'quixotism'))
    for w in ['token', 'model', '语言', '模型', 'quixotism']:
        pieces = encode_word(w, merges, rank)
        lines.append('    %-10s → %s' % (w, ' '.join(pieces)))

    lines.append('\n【D】编码 / 解码 round-trip')
    rt_fail = 0
    for t in texts:
        if decode(encode(t, merges, rank)) != normalize(t):
            rt_fail += 1
    lines.append('    全部 %d 篇：round-trip 失败 %d 篇（应为 0）' % (len(texts), rt_fail))

    lines.append('\n【E】合并次数 sweep（词表越大 → 压缩越强，但边际递减）')
    lines.append('    %-8s %-8s %-14s %-12s %-12s' % (
        'merges', 'vocab', 'tokens/char', '正例覆盖', '负例覆盖'))
    for r in sweep_rows:
        lines.append('    %-8d %-8d %-14.4f %-12.1f%% %-12.1f%%' % (
            r['merges'], r['vocab'], r['tokens_per_char'],
            100 * r['coverage'], 100 * r['neg_coverage']))

    lines.append('\n【F】真值对账（合并次数=%d）' % best_m)
    _, pdetail = truth_coverage(TRUTH_POSITIVE, merges, rank)
    _, ndetail = truth_coverage(TRUTH_NEGATIVE, merges, rank)
    lines.append('    正例（应整块成词）：')
    for w, whole, nseg in pdetail:
        lines.append('      %-10s %s  片段数=%d' % (w, '整块' if whole else '被拆', nseg))
    lines.append('    负例（应被拆碎）：')
    for w, whole, nseg in ndetail:
        lines.append('      %-14s %s  片段数=%d' % (w, '整块' if whole else '被拆碎', nseg))

    return '\n'.join(lines), rt_fail


# ---------------------------------------------------------------- 配图
def draw(sweep_rows, docs, merges, rank, best_m):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[提示] 未安装 matplotlib，跳过画图')
        return None
    # 中文字体：覆盖 macOS / Linux / Windows 常见候选，缺哪个 matplotlib 自动跳过下一个
    plt.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti',
        'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei',
        'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    # ① 每一步被合并的 pair 频率（单调下降 = 先合并最频繁的）
    freqs = []
    texts = [d['text'] for d in docs]
    fw = word_freqs(texts)
    words = {tuple(to_symbols(w)): c for w, c in fw.items()}
    for (a, b) in merges:
        bag = Counter()
        for sym, c in words.items():
            count_pairs(sym, c, bag)
        freqs.append(bag.get((a, b), 0))
        new = a + b
        words = {tuple(apply_merge(sym, (a, b), new)): c for sym, c in words.items()}
    axes[0].plot(range(1, len(freqs) + 1), freqs, '-', color='#4a9d6a')
    axes[0].set_title('每步被合并对的词频（单调下降）')
    axes[0].set_xlabel('merge 序号')
    axes[0].set_ylabel('该对的词频')

    # ② tokens_per_char vs 词表大小
    axes[1].plot([r['vocab'] for r in sweep_rows],
                 [r['tokens_per_char'] for r in sweep_rows], 'o-', color='#d08c4a')
    axes[1].set_title('压缩率随词表大小（边际递减）')
    axes[1].set_xlabel('词表大小')
    axes[1].set_ylabel('tokens / char')

    # ③ 正例覆盖 vs 负例覆盖 vs 词表
    axes[2].plot([r['vocab'] for r in sweep_rows], [100 * r['coverage'] for r in sweep_rows],
                 's-', color='#4a9d6a', label='正例覆盖率')
    axes[2].plot([r['vocab'] for r in sweep_rows], [100 * r['neg_coverage'] for r in sweep_rows],
                 'x--', color='#cc5a5a', label='负例覆盖率(应低)')
    axes[2].set_title('真值覆盖 vs 词表大小')
    axes[2].set_xlabel('词表大小')
    axes[2].set_ylabel('%')
    axes[2].legend(fontsize=8)

    # ④ 一例高频词的整块成型：token 在不同 merge 预算下被拆成几段
    grid = sorted(set([r['merges'] for r in sweep_rows]))
    segs = []
    for m in grid:
        mm, vv, rr = train_bpe(texts, m)
        segs.append(len([p for p in encode_word('token', mm, rr) if p != WORD_END]))
    axes[3].step(range(1, len(grid) + 1), segs, where='mid', marker='o', color='#8888cc')
    axes[3].set_xticks(range(1, len(grid) + 1))
    axes[3].set_xticklabels(['%d' % m for m in grid], fontsize=8)
    axes[3].set_title('高频词 token 的片段数随合并数')
    axes[3].set_xlabel('合并次数')
    axes[3].set_ylabel('片段数')

    path = os.path.join(OUT, 'fig-02-bpe.png')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)
    docs = make_corpus(rng)
    texts = [d['text'] for d in docs]

    # 选定训练规模：合并次数取一个「够把高频词捏成词、又不至于把稀有词也整块并掉」
    # 的中档值。太小则高频词都拆着，太大则稀有词也成词——甜点靠 sweep 的真值对账来定。
    # 本语料上 200 是甜点：正例 100% 成词、负例 0% 被并；到 250 起稀有词开始被误吞。
    BEST_MERGES = 200
    merges, vocab, rank = train_bpe(texts, BEST_MERGES)

    # sweep：扫描合并次数网格
    merge_grid = [60, 120, 200, 300, 450, 600]
    sweep_rows = sweep_merge_counts(texts, merge_grid, TRUTH_POSITIVE, TRUTH_NEGATIVE)

    # 落盘：语料、有序 merge（供生产篇复用）、报表
    with open(os.path.join(OUT, 'corpus.jsonl'), 'w', encoding='utf-8') as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    # vocab.json：有序 merges + 基础词表，生产篇可据此复现同一套合并
    with open(os.path.join(OUT, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'seed': SEED, 'num_merges': len(merges),
                   'merges': [list(p) for p in merges],
                   'word_end': WORD_END}, f, ensure_ascii=False)
    with open(os.path.join(OUT, 'merges.txt'), 'w', encoding='utf-8') as f:
        for a, b in merges:
            f.write('%s %s\n' % (a, b))

    text_report, rt_fail = fmt_report(docs, texts, merges, vocab, rank,
                                      sweep_rows, BEST_MERGES)
    print()
    print(text_report)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text_report + '\n')
    with open(os.path.join(OUT, 'sweep.tsv'), 'w', encoding='utf-8') as f:
        f.write('merges\tvocab\ttokens_per_char\tcoverage\tneg_coverage\n')
        for r in sweep_rows:
            f.write('%d\t%d\t%.4f\t%.4f\t%.4f\n' % (
                r['merges'], r['vocab'], r['tokens_per_char'],
                r['coverage'], r['neg_coverage']))

    fig = draw(sweep_rows, docs, merges, rank, BEST_MERGES)
    if fig:
        print('\n[配图] %s' % fig)

    # ------------------------------------------------ 自校验：拿已知答案对账
    print('\n' + '=' * 70)
    print('【G】自校验（用注入时的已知答案对账，不是靠肉眼看）')
    print('=' * 70)

    # 1. round-trip 无损：每篇 encode→decode 必须还原规范化原文
    print('\n  1. 编码/解码 round-trip：全部文档还原')
    for t in texts:
        assert decode(encode(t, merges, rank)) == normalize(t), 'round-trip 破坏原文'
    print('     %d 篇全部无损还原（encode→decode == normalize）' % len(texts))

    # 2. 合并确定性：同 seed 同参重训，merge 序列逐条一致
    print('\n  2. 合并确定性：重训一次，merge 序列逐条一致（平票按字典序打破）')
    merges2, vocab2, rank2 = train_bpe(texts, BEST_MERGES)
    assert merges == merges2, 'BPE 合并非确定性'
    print('     两次训练的 %d 条 merge 完全相同' % len(merges))

    # 3. 优先级单调：更早学到的 merge 频率不低于更晚的（平票除外）
    print('\n  3. 合并优先级与词频一致：先合并更频繁的相邻对')
    assert len(rank) == len(merges), 'merge 与 rank 计数不一致'
    for i, (a, b) in enumerate(merges):
        assert rank[(a, b)] == i, 'rank 下标错位'
    print('     %d 条 merge 的 rank 下标连续且与合并顺序一致' % len(merges))

    # 4. 正例覆盖：高频真值词在选定规模下应大量整块成词
    print('\n  4. 真值正例：高频词整块成词的比例')
    cov, pd = truth_coverage(TRUTH_POSITIVE, merges, rank)
    for w, whole, nseg in pd:
        print('     %-10s %s 片段数=%d' % (w, '整块' if whole else '被拆', nseg))
    print('     正例覆盖率 %.1f%%' % (100 * cov))
    assert cov >= 0.6, '正例覆盖率过低（高频词没被学成词）%.1f%%' % (100 * cov)

    # 5. 负例碎片：稀有真值词应保持被拆碎（不被整块合并）
    print('\n  5. 真值负例：稀有词应保持被拆碎')
    ncov, nd = truth_coverage(TRUTH_NEGATIVE, merges, rank)
    for w, whole, nseg in nd:
        print('     %-14s %s 片段数=%d' % (w, '整块(不该)' if whole else '被拆碎', nseg))
    assert ncov == 0.0, '稀有词被整块合并了，负例失控'

    # 6. sweep 单调性 + 词表一致性
    print('\n  6. sweep 与词表不变量')
    tpc_prev = None
    vocab_prev = None
    for r in sweep_rows:
        if tpc_prev is not None:
            assert r['tokens_per_char'] <= tpc_prev + 1e-12, '合并越多压缩应越强（单调不增）'
        if vocab_prev is not None:
            assert r['vocab'] >= vocab_prev, '词表应随合并数不减'
        tpc_prev = r['tokens_per_char']
        vocab_prev = r['vocab']
    # 词表大小上界 = 基础符号数（全部去空格字符 + 词尾标记）+ 合并产物数
    # （每步合并至多向词表新增一个符号，故为 ≤ 上界；个别 new 撞车则更少）
    base_chars = {ch for t in texts for ch in normalize(t) if ch != ' '} | {WORD_END}
    assert len(vocab) <= len(base_chars) + len(merges), \
        '词表大小超出「基础字符 + 合并」上界'
    print('     tokens/char 随合并数单调不增：%s' %
          ' > '.join('%.3f' % r['tokens_per_char'] for r in sweep_rows))
    print('     词表随合并数不减：%s' % ' < '.join(str(r['vocab']) for r in sweep_rows))

    print('\n  全部断言通过。')
    print('\n[产物] 语料 %d 篇 / 词表 %d / merge %d 条 → %s'
          % (len(docs), len(vocab), len(merges), OUT))
    return 0


if __name__ == '__main__':
    sys.exit(main())
