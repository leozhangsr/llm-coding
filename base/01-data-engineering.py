#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""基础篇 01 · 数据工程：规则过滤 + SimHash 去重 + 配比

对应文章：同目录 01-数据工程.md
运行：    python3 01-数据工程.py
依赖：    仅标准库；matplotlib 可选（缺了自动跳过画图）
产物：    ./output/01-数据工程/ 下的语料 jsonl、统计报表、参数扫描表、配图

一条流水线的五道工序，每道都记账（进多少、出多少、为什么被拦）：

    合成原始语料 → 规范化 → 规则过滤 → 精确去重 → SimHash 近似去重 → 领域配比 → 落盘

关键设计：语料是自己合成的，所以"注入了哪些脏数据、哪几篇互为重复"是已知答案。
SimHash 的阈值 k 和分段数 bands 不靠抄，而是拿这份已知答案扫一遍参数选出来的。
最后一段自校验同样拿已知答案对账——这是数据工程唯一可靠的验收方式。
"""

import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

SEED = 20261016
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'output', '01-数据工程')

# ---------------------------------------------------------------- 合成语料
# 句模板 + 槽位词拼出可读的中文段落，每篇再追加一句"独有细节"
# （压测数字 / 日期 / 版次），让无关文档之间确实不相似。
# 少了这句独有细节，合成语料会互相过度相似，SimHash 的标定就失真了。

CLAUSES = {
    '技术': [
        '在分布式训练里，{a}直接决定{b}的上限，工程上一般先{c}再{d}。',
        '推理服务的成本大头是{a}，把{c}做对，单机吞吐能翻一倍以上。',
        '{a}看起来是个参数问题，实际是{b}问题，动手之前要先量一遍分布。',
        '我们踩过的坑是{c}没做，结果{b}在长尾请求上抖得厉害。',
        '离线指标涨了不代表线上能用，{b}要单独看，尤其是{d}之后。',
        '{a}和{c}是一对矛盾，取舍标准是业务对{b}的容忍度。',
        '同样的模型，换了{a}之后{b}的变化比换硬件还明显。',
        '复盘的时候发现，真正拖慢进度的是{c}没人负责。',
    ],
    '生活': [
        '周末去{a}走了两公里，回来的路上顺路买了{c}。',
        '{a}这事看着小，坚持三个月之后{b}确实不一样。',
        '家里的{c}用久了会响，拆开清理一次就安静了。',
        '早上六点起床先{c}，一整天{b}都稳一些。',
        '楼下新开的{a}排队很长，味道一般，不太值得等。',
        '换季的时候把{c}收起来，柜子里腾出的空间比想象中大。',
        '{a}人少的时候最好去，{b}会明显好一些。',
        '把{c}这件事交给别人做之后，才发现自己一直高估了它的难度。',
    ],
    '新闻': [
        '据{a}消息，该市{c}项目已进入验收阶段，涉及{b}。',
        '报告显示，今年第三季度{b}同比增长，主要由{c}拉动。',
        '有关部门就{a}发布通知，要求各地在{d}前完成排查。',
        '会上提到，{c}的试点范围将扩大到{b}相关领域。',
        '记者从{a}获悉，{c}将于下月启动，预计覆盖{b}。',
        '多位受访者表示，{a}的变化对{b}影响最直接。',
        '统计口径调整之后，{b}的数据与去年不直接可比。',
        '{c}的推进节奏，取决于{a}与地方财政的衔接。',
    ],
}

SLOTS = {
    '技术': {
        'a': ['通信带宽', '显存分片策略', '批大小', '梯度累积', 'KV Cache 命中率',
              '算子融合', '采样温度', '数据配比'],
        'b': ['吞吐', '收敛速度', '首 token 时延', '单位 token 成本', '显存占用', '稳定性'],
        'c': ['把调度改对', '量一遍真实分布', '加一层缓存', '压测到 P99', '拆成两阶段',
              '把日志打全'],
        'd': ['谈优化', '换硬件', '加并发', '上量化', '扩集群'],
    },
    '生活': {
        'a': ['公园', '菜市场', '健身房', '图书馆', '河边步道', '旧货市场'],
        'b': ['精神', '睡眠', '体力', '心情', '专注度'],
        'c': ['那台旧风扇', '冬天的被子', '跑步这件事', '阳台的花', '厨房的刀具', '书桌上那盏灯'],
        'd': ['中午', '下班后', '周末'],
    },
    '新闻': {
        'a': ['市发改委', '行业协会', '当地媒体', '统计部门', '住建部门'],
        'b': ['公共交通', '新能源', '养老服务', '义务教育', '制造业', '老旧小区改造'],
        'c': ['改造升级', '专项检查', '试点扩面', '数据互通', '标准制定'],
        'd': ['月底', '季度末', '年底前'],
    },
}


def make_doc(rng, category):
    """拼一段 4-7 句的中文段落，句模板不重复抽样，末尾追加一句独有细节。"""
    slot = SLOTS[category]
    sents = []
    for tpl in rng.sample(CLAUSES[category], rng.randint(4, 7)):
        for k, pool in slot.items():
            tpl = tpl.replace('{%s}' % k, rng.choice(pool))
        sents.append(tpl)
    if category == '技术':
        sents.append('压测记录：并发 %d，P99 %d 毫秒，吞吐 %d token/s。'
                     % (rng.randint(1, 256), rng.randint(80, 4000), rng.randint(200, 9000)))
    elif category == '生活':
        sents.append('记于 %d 月 %d 日，那天气温 %d 度。'
                     % (rng.randint(1, 12), rng.randint(1, 28), rng.randint(-5, 38)))
    else:
        sents.append('（发稿时间 %d 月 %d 日，第 %d 版）'
                     % (rng.randint(1, 12), rng.randint(1, 28), rng.randint(1, 16)))
    return ''.join(sents)


def make_signature(rng):
    """给每篇正常文档盖一个独一无二的"指纹尾"：随机十六进制串 + 随机序号。

    为什么需要它：正常文档是用同一批句模板拼的，彼此天然共享大量 3-gram，
    不加区分的话两篇无关文档的汉明距离可能低到 6-9，会和真重复混在一起，
    逼你把阈值 k 调小、召回跟着崩。盖上这段高熵尾巴后，无关文档被拉到距离 ≥16，
    而重复文档因为整段照抄（尾巴也一样）依旧贴得很近——分离度就是这么造出来的。
    """
    parts = [hashlib.blake2b(str(rng.random()).encode('utf-8'), digest_size=5)
             .hexdigest().upper() for _ in range(3)]
    nums = ' '.join(str(rng.randint(1000, 9999)) for _ in range(6))
    return '文档指纹 %s；序列 %s。' % (' '.join(parts), nums)


def make_corpus(n_clean=900, seed=SEED):
    """造一份带已知脏数据的语料。

    返回 (docs, dup_pairs)：
      docs       每篇带 truth 字段，记录它的真实身份
      dup_pairs  注入的重复关系（源 id, 副本 id），用来构造真值簇做验收
    """
    rng = random.Random(seed)
    docs = []
    dup_pairs = []

    def add(text, category, truth):
        docs.append({'id': 'd%05d' % len(docs), 'text': text,
                     'category': category, 'truth': truth})
        return docs[-1]['id']

    clean_ids = []
    for _ in range(n_clean):
        cat = rng.choices(['技术', '生活', '新闻'], weights=[5, 3, 2])[0]
        clean_ids.append(add(make_doc(rng, cat) + make_signature(rng), cat, 'normal'))

    body = {d['id']: d['text'] for d in docs}

    for _ in range(60):                       # 脏数据一：过短
        add(rng.choice(['好文，转了。', '这个说法有意思。', '收藏备用。', '顶一下。', '已读。']),
            '生活', 'too_short')

    for _ in range(40):                       # 脏数据二：HTML 与脚本残留
        add('<div class="article">%s</div><script>var t=1;</script>' % make_doc(rng, '技术'),
            '技术', 'html')

    for _ in range(50):                       # 脏数据三：广告与导流
        tail = rng.choice(['点击关注领取资料', '加微信进群免费学习', '版权所有，转载必究',
                           '限时优惠立即抢购', '扫码关注获取更多'])
        add(make_doc(rng, '生活') + tail, '生活', 'ad')

    for _ in range(40):                       # 脏数据四：乱码与符号噪声
        junk = ''.join(rng.choice('★☆※◆▲●■□◇○@#$%^&*~`|\\/<>[]{}+=') for _ in range(200))
        add(junk[:120] + make_doc(rng, '新闻')[:40] + junk[120:], '新闻', 'garbled')

    for _ in range(30):                       # 脏数据五：复读机
        add(make_doc(rng, '技术')[:40] * 8, '技术', 'repetition')

    for _ in range(70):                       # 脏数据六：完全重复（转载稿一字不改）
        src = rng.choice(clean_ids)
        dup_pairs.append((src, add(body[src], '技术', 'exact_dup')))

    for _ in range(70):                       # 脏数据七：近似重复（改标点 + 加来源行）
        src = rng.choice(clean_ids)
        variant = body[src].replace('。', '，', 2) + '（来源：网络综合整理）'
        dup_pairs.append((src, add(variant, '技术', 'near_dup')))

    rng.shuffle(docs)          # 只打乱顺序，id 不重排：dup_pairs 里的 id 必须一直有效
    return docs, dup_pairs


def truth_groups(dup_pairs):
    """把重复关系并成簇，返回 [set(id), ...]。每簇最终只允许留下一篇。"""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in dup_pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = defaultdict(set)
    for x in list(parent):
        groups[find(x)].add(x)
    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------- 工序一：规范化
def normalize(text):
    """统一空白与全角空格，不动语义。放在最前面，是为了让后面的规则和哈希可比。"""
    text = text.replace('\u3000', ' ').replace('\xa0', ' ')
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n', text)
    return text.strip()


# ---------------------------------------------------------------- 工序二：规则过滤
AD_WORDS = ('点击关注', '免费领取', '加微信', '进群', '版权所有', '限时优惠',
            '立即抢购', '扫码关注')
HTML_RE = re.compile(r'<[a-zA-Z/!][^>]{0,120}>')
CJK_RE = re.compile(r'[\u4e00-\u9fff]')
ALLOWED_RE = re.compile(r'[\u4e00-\u9fff0-9A-Za-z\s，。、；：？！“”‘’（）《》—….,;:?!()\-/]')


def split_sentences(text):
    return [s for s in re.split(r'[。！？；\n]', text) if s.strip()]


def rule_filter(text):
    """返回 (是否保留, 拦截原因)。规则从便宜到贵排：先长度，再正则，最后统计量。"""
    n = len(text)
    if n < 50:
        return False, '过短'
    if HTML_RE.search(text):
        return False, 'HTML 残留'
    if any(w in text for w in AD_WORDS):
        return False, '广告导流'
    if len(CJK_RE.findall(text)) / n < 0.4:
        return False, '中文占比过低'
    if len(ALLOWED_RE.findall(text)) / n < 0.8:
        return False, '符号噪声'

    sents = split_sentences(text)
    if not sents:
        return False, '无有效句子'
    if len(sents) >= 3 and Counter(sents).most_common(1)[0][1] / len(sents) > 0.4:
        return False, '复读'
    if n / len(sents) > 160:
        return False, '断句异常'
    return True, ''


# ---------------------------------------------------------------- 工序三：精确去重
def exact_key(text):
    return hashlib.sha1(text.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------- 工序四：SimHash
def char_ngrams(text, n=3):
    return [text[i:i + n] for i in range(max(0, len(text) - n + 1))]


def simhash(text, bits=64, n=3):
    """字符 n-gram → 64 位指纹。

    每一位的取值由"落在这位上的特征权重之和"的符号决定：
    哈希位为 1 的特征加权重，为 0 的减权重，最后看总和的正负。
    两篇文档共有的 n-gram 越多，符号被翻转的位就越少，汉明距离就越小。

    这个双重循环是纯 Python 的瓶颈（每篇文档 特征数×64 次加法）。
    生产上第一件事就是把它向量化或换成 C 实现——见生产篇 01。
    """
    feats = Counter(char_ngrams(text, n))
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


def hamming(a, b):
    return bin(a ^ b).count('1')


def band_keys(fp, bits, bands):
    """把指纹切成 bands 段：只要有一段完全相同就成为候选对。

    这一步是 SimHash 能上规模的关键——两两比较是 O(n²)，分段建桶后只比较同桶文档。
    代价是漏检：汉明距离 d 的重复对，只有在"至少有一段没被这 d 个差异位命中"时才会同桶。

    段数怎么定？鸽笼原理给了确定答案：要让"距离 ≤ k 的对一定同桶"，就把 64 位切成
    k+1 段——k 个差异位最多污染 k 段，剩下至少一段干净。所以本文一律取 bands = k + 1。

    注意别贪多切段：段数越多每段越窄（bands=16 时每段才 4 位、只有 16 种取值），
    900 篇文档会把桶塞到平均 50 多篇，桶一"热"就要跳过，反而把真重复漏掉。
    bands = k+1 是召回与桶大小之间的那个甜点，下面的扫描会把这个权衡量出来。
    """
    width = bits // bands          # 64//9=7，只用低 63 位分段，最高 1 位不参与
    # 未覆盖的位不影响鸽笼召回：距离 ≤k 是"全部 64 位"上的距离，它在上表用的 63 位上
    # 只会更小（≤k 依然成立），所以"差异位最多污染 k 段、至少留一段干净"的结论不变。
    mask = (1 << width) - 1
    return [(i, (fp >> (i * width)) & mask) for i in range(bands)]


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


HOT_BUCKET = 256         # 超过这个大小的桶直接跳过，并计数（生产上的热点桶问题）


def near_dedup(docs, fps, threshold, bands=None, bits=64):
    """分段建桶 → 候选对算汉明距离 → 并查集聚簇 → 每簇留最长的一篇。

    bands 默认取 threshold+1（鸽笼原理），保证距离 ≤ threshold 的对一定进同一桶。
    """
    if bands is None:
        bands = threshold + 1
    buckets = defaultdict(list)
    for d in docs:
        for key in band_keys(fps[d['id']], bits, bands):
            buckets[key].append(d['id'])

    cand, hot = set(), 0
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        if len(ids) > HOT_BUCKET:
            hot += 1
            continue
        ids = sorted(set(ids))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                cand.add((ids[i], ids[j]))

    dsu = DSU()
    for a, b in cand:
        if hamming(fps[a], fps[b]) <= threshold:
            dsu.union(a, b)

    clusters = defaultdict(list)
    for d in docs:
        clusters[dsu.find(d['id'])].append(d)

    keep, drop = [], []
    for members in clusters.values():
        members.sort(key=lambda d: (-len(d['text']), d['id']))
        keep.append(members[0])
        drop.extend(members[1:])
    keep.sort(key=lambda d: d['id'])
    return keep, drop, len(cand), hot


def dedup_score(docs, fps, groups, threshold, bands=None):
    """拿真值簇给一组参数打分。

    两个召回口径要分清（审稿指出：95.4% 若含精确去重已塌的簇，会高估 SimHash 的贡献）：
      - recall      ：端到端簇塌陷率，把"完全重复"（已被上一道 SHA1 塌成 1 篇）也算进去；
      - near_recall ：SimHash 的边际召回，只统计"进入本工序时仍 ≥2 篇"的簇（即近似重复簇），
                      这才是 SimHash 自己抓回来的比例。
    """
    if bands is None:
        bands = threshold + 1
    keep, drop, n_cand, hot = near_dedup(docs, fps, threshold, bands)
    kept = {d['id'] for d in keep}
    present = {d['id'] for d in docs}
    collapsed = sum(1 for g in groups if len([i for i in g if i in kept]) <= 1)
    recall = collapsed / len(groups)

    active = [g for g in groups if len([i for i in g if i in present]) >= 2]
    near_collapsed = sum(1 for g in active if len([i for i in g if i in kept]) <= 1)
    near_recall = near_collapsed / len(active) if active else 1.0

    in_group = {i for g in groups for i in g}
    eligible = [d['id'] for d in docs if d['truth'] == 'normal' and d['id'] not in in_group]
    wrong = sum(1 for i in eligible if i not in kept)
    fpr = wrong / max(1, len(eligible))
    return {'k': threshold, 'bands': bands, 'recall': recall, 'near_recall': near_recall,
            'n_active': len(active), 'fpr': fpr,
            'cand': n_cand, 'hot': hot, 'kept': len(keep), 'drop': drop, 'keep': keep}


def sweep(docs, fps, groups, ks=(4, 6, 8, 10, 12, 14)):
    """扫阈值 k，段数按鸽笼原理取 bands=k+1，选 F1 最高的组合。有真值标签才敢这么选。

    这里 k 越大召回越高（更多真重复被抓住），但误删也越高（少数正常文档被错并）——
    因为合成语料里"重复对"与"正常对"的汉明距离分布有重叠，没有免费的完美阈值。
    """
    rows = []
    for k in ks:
        r = dedup_score(docs, fps, groups, k)          # bands 自动 = k+1
        p = 1 - r['fpr']
        r['f1'] = 2 * p * r['recall'] / (p + r['recall']) if p + r['recall'] else 0
        rows.append(r)
    rows.sort(key=lambda r: (-r['f1'], r['cand']))
    return rows


# ---------------------------------------------------------------- 工序五：配比
def rebalance(docs, target, rng):
    """按目标比例下采样超额领域。只降不升：

    小领域数据不够时不复制补齐——复制等于人为造重复，会被上一道工序删掉，
    即使删不掉也会让模型把这几篇背下来。
    """
    by_cat = defaultdict(list)
    for d in docs:
        by_cat[d['category']].append(d)
    total = sum(target.values())
    scales = [len(by_cat[c]) / (w / total) for c, w in target.items() if by_cat.get(c)]
    if not scales:
        return docs
    scale = min(scales)

    out = []
    for cat, want in target.items():
        pool = by_cat.get(cat, [])
        n = min(len(pool), int(round(scale * want / total)))
        out.extend(rng.sample(pool, n) if n < len(pool) else pool)
    out.sort(key=lambda d: d['id'])
    return out


# ---------------------------------------------------------------- 报表与画图
def fmt_report(stage, rule_reasons, dedup_reasons, cat_before, cat_after,
               calib, best, sweep_rows):
    L = []
    base = stage[0][1][0]
    L.append('=' * 70)
    L.append('基础篇 01 · 数据工程流水线报表（合成语料，seed=%d）' % SEED)
    L.append('=' * 70)
    L.append('')
    L.append('【一】各工序进出账')
    L.append('    %-16s %8s %8s %8s %9s' % ('工序', '进入', '留下', '拦掉', '累计留存'))
    for name, (inp, outp) in stage:
        L.append('    %-16s %8d %8d %8d %8.1f%%'
                 % (name, inp, outp, inp - outp, 100.0 * outp / base))
    L.append('')
    L.append('【二】规则过滤：拦截原因分布（合计 %d 篇）' % sum(rule_reasons.values()))
    for reason, c in rule_reasons.most_common():
        L.append('    %-14s %5d 篇  %5.1f%%' % (reason, c, 100.0 * c / base))
    L.append('')
    L.append('【三】去重：两类重复各自的量（合计 %d 篇）' % sum(dedup_reasons.values()))
    for reason, c in dedup_reasons.most_common():
        L.append('    %-14s %5d 篇  %5.1f%%' % (reason, c, 100.0 * c / base))
    L.append('')
    L.append('【四】SimHash 参数扫描（拿注入的真值簇打分，选 F1 最高的组合）')
    L.append('    %4s %6s %8s %10s %8s %8s %10s %6s'
             % ('k', 'bands', '端到端召回', 'SimHash召回', '误删率', 'F1', '候选对', '热点桶'))
    for r in sweep_rows[:8]:
        L.append('    %4d %6d %9.1f%% %10.1f%% %7.2f%% %8.4f %10d %6d'
                 % (r['k'], r['bands'], 100 * r['recall'], 100 * r['near_recall'],
                    100 * r['fpr'], r['f1'], r['cand'], r['hot']))
    L.append('    → 选定 k=%d, bands=%d' % (best['k'], best['bands']))
    L.append('    口径说明：端到端召回含"完全重复"（已被上一道 SHA1 塌成 1 篇）；')
    L.append('             SimHash 召回只统计进入本工序时仍 ≥2 篇的近似重复簇（共 %d 组），才是 SimHash 的边际贡献；'
             % best['n_active'])
    L.append('             F1 = 端到端召回与(1-误删率)的调和平均，是教学口径下的综合分，非标准 TP/(TP+FP)。')
    L.append('')
    L.append('【五】SimHash 汉明距离分布（阈值 k=%d）' % best['k'])
    for name, arr in calib:
        if not arr:
            L.append('    %-18s 无样本' % name)
            continue
        s = sorted(arr)
        q = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1)))]
        L.append('    %-18s n=%-6d P5=%-3d 中位=%-3d P95=%-3d 最大=%d'
                 % (name, len(s), q(0.05), q(0.5), q(0.95), s[-1]))
    L.append('')
    L.append('【六】领域配比（目标 技术:生活:新闻 = 5:3:2）')
    for cat in ('技术', '生活', '新闻'):
        b, a = cat_before.get(cat, 0), cat_after.get(cat, 0)
        pb = 100.0 * b / max(1, sum(cat_before.values()))
        pa = 100.0 * a / max(1, sum(cat_after.values()))
        L.append('    %-4s %6d 篇 (%4.1f%%)  →  %6d 篇 (%4.1f%%)' % (cat, b, pb, a, pa))
    return '\n'.join(L)


def draw(stage, rule_reasons, calib, threshold, sweep_rows):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[提示] 未安装 matplotlib，跳过画图')
        return None
    plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))

    names = [s[0] for s in stage]
    vals = [s[1][1] for s in stage]
    axes[0].bar(range(len(names)), vals, color='#4a9d6a')
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=18, ha='right', fontsize=9)
    axes[0].set_title('各工序留下的文档数')
    for i, v in enumerate(vals):
        axes[0].text(i, v, str(v), ha='center', va='bottom', fontsize=9)

    items = rule_reasons.most_common()[::-1]
    axes[1].barh([r for r, _ in items], [c for _, c in items], color='#d08c4a')
    axes[1].set_title('规则过滤：拦截原因分布')
    for i, (_, c) in enumerate(items):
        axes[1].text(c, i, ' %d' % c, va='center', fontsize=9)

    for name, arr in calib:
        if arr:
            axes[2].hist(arr, bins=range(0, 46, 2), alpha=0.62,
                         label='%s (n=%d)' % (name, len(arr)))
    axes[2].axvline(threshold, color='red', ls='--', label='阈值 k=%d' % threshold)
    axes[2].set_title('SimHash 汉明距离分布')
    axes[2].set_xlabel('汉明距离（0=完全一致，32=毫不相关）')
    axes[2].legend(fontsize=8)

    rs = sorted(sweep_rows, key=lambda r: r['k'])
    axes[3].plot([r['k'] for r in rs], [r['recall'] for r in rs], 'o-',
                 color='#4a9d6a', label='召回（抓到的真重复）')
    axes[3].plot([r['k'] for r in rs], [1 - r['fpr'] for r in rs], 'x--',
                 color='#d08c4a', label='精确（1 - 误删率）')
    axes[3].plot([r['k'] for r in rs], [r['f1'] for r in rs], 's-',
                 color='#8888cc', alpha=0.8, label='F1')
    axes[3].axvline(threshold, color='red', ls='--', alpha=0.6, label='选定 k=%d' % threshold)
    axes[3].set_title('阈值 k 的权衡（段数 = k+1）')
    axes[3].set_xlabel('汉明距离阈值 k')
    axes[3].set_ylabel('召回 / 精确 / F1')
    axes[3].set_ylim(0, 1.02)
    axes[3].legend(fontsize=8)
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT, 'fig-01-pipeline.png')
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(SEED)

    docs, dup_pairs = make_corpus()
    truth = {d['id']: d['truth'] for d in docs}
    groups = truth_groups(dup_pairs)
    base = len(docs)
    stage = [('0 原始语料', (base, base)), ('1 规范化', (base, base))]

    for d in docs:
        d['norm'] = normalize(d['text'])

    # 工序二：规则过滤
    rule_reasons, kept_docs = Counter(), []
    for d in docs:
        ok, why = rule_filter(d['norm'])
        if ok:
            kept_docs.append(d)
        else:
            d['drop_stage'], d['drop_reason'] = '规则过滤', why
            rule_reasons[why] += 1
    stage.append(('2 规则过滤', (base, len(kept_docs))))

    # 工序三：精确去重
    dedup_reasons = Counter()
    seen, after_exact = {}, []
    for d in kept_docs:
        key = exact_key(d['norm'])
        if key in seen:
            d['drop_stage'], d['drop_reason'] = '精确去重', '完全重复'
            dedup_reasons['完全重复'] += 1
        else:
            seen[key] = d['id']
            after_exact.append(d)
    stage.append(('3 精确去重', (len(kept_docs), len(after_exact))))

    # 工序四：SimHash 近似去重（先用真值扫参数，再用选出的参数跑）
    fps = {d['id']: simhash(d['norm']) for d in docs}
    print('[标定] 计算 %d 篇文档的 64 位指纹…' % len(docs))

    sweep_rows = sweep(after_exact, fps, groups)
    best = sweep_rows[0]
    print('[标定] 扫描 %d 组参数，选定 k=%d bands=%d（召回 %.1f%%，误删 %.2f%%）'
          % (len(sweep_rows), best['k'], best['bands'],
             100 * best['recall'], 100 * best['fpr']))

    keep, drop, n_cand, hot = near_dedup(after_exact, fps, best['k'], best['bands'])
    for d in drop:
        d['drop_stage'], d['drop_reason'] = 'SimHash 去重', '近似重复'
        dedup_reasons['近似重复'] += 1
    stage.append(('4 SimHash 去重', (len(after_exact), len(keep))))
    naive = len(after_exact) * (len(after_exact) - 1) // 2
    print('[去重] 分段索引把 %d 组两两比较压到 %d 组候选（%.1f%%），跳过大桶 %d 个'
          % (naive, n_cand, 100.0 * n_cand / naive, hot))

    # 工序五：领域配比
    cat_before = Counter(d['category'] for d in keep)
    final = rebalance(keep, {'技术': 5, '生活': 3, '新闻': 2}, rng)
    cat_after = Counter(d['category'] for d in final)
    stage.append(('5 领域配比', (len(keep), len(final))))

    # ------------------------------------------------ 落盘
    def dump(name, rows):
        path = os.path.join(OUT, name)
        with open(path, 'w', encoding='utf-8') as f:
            for d in rows:
                f.write(json.dumps({k: v for k, v in d.items() if k != 'norm'},
                                   ensure_ascii=False) + '\n')
        return path

    p_raw = dump('corpus_raw.jsonl', docs)
    p_drop = dump('corpus_dropped.jsonl', [d for d in docs if 'drop_reason' in d])
    p_clean = dump('corpus_clean.jsonl', final)

    # 5.3 的距离分布必须用"真正无关的正常文档对"来标定，不能混入脏数据/重复副本，
    # 否则"正常对 P5"会被脏数据拉低、与扫描表里的误删率对不上（审稿指出的关键问题）。
    near_d = [hamming(fps[a], fps[b]) for a, b in dup_pairs]
    in_group = {i for g in groups for i in g}
    normal_ids = [d['id'] for d in docs
                  if d['truth'] == 'normal' and d['id'] not in in_group]
    rand_d = [hamming(fps[a], fps[b]) for a, b in
              (rng.sample(normal_ids, 2) for _ in range(3000))]
    calib = [('重复对（注入）', near_d), ('正常对（无关文档）', rand_d)]

    text_report = fmt_report(stage, rule_reasons, dedup_reasons, cat_before, cat_after,
                             calib, best, sweep_rows)
    print()
    print(text_report)
    with open(os.path.join(OUT, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(text_report + '\n')
    with open(os.path.join(OUT, 'sweep.tsv'), 'w', encoding='utf-8') as f:
        f.write('k\tbands\trecall\tfpr\tf1\tcand\thot\tkept\n')
        for r in sorted(sweep_rows, key=lambda x: (x['bands'], x['k'])):
            f.write('%d\t%d\t%.4f\t%.5f\t%.4f\t%d\t%d\t%d\n'
                    % (r['k'], r['bands'], r['recall'], r['fpr'], r['f1'],
                       r['cand'], r['hot'], r['kept']))

    fig = draw(stage, rule_reasons, calib, best['k'], sweep_rows)
    if fig:
        print('\n[配图] %s' % fig)

    # ------------------------------------------------ 自校验：拿已知答案对账
    print('\n' + '=' * 70)
    print('【七】自校验（用注入时的已知答案对账，不是靠肉眼看）')
    print('=' * 70)

    kept_ids = {d['id'] for d in keep}
    rule_dropped = {d['id'] for d in docs if d.get('drop_stage') == '规则过滤'}

    print('\n  1. 规则层：各类脏数据拦截率')
    for t in ('too_short', 'html', 'ad', 'garbled', 'repetition'):
        tid = [i for i in truth if truth[i] == t]
        hit = sum(1 for i in tid if i in rule_dropped)
        print('     %-12s %6.1f%%  (%d/%d 篇)' % (t, 100.0 * hit / len(tid), hit, len(tid)))
        assert hit / len(tid) >= 0.95, '%s 拦截率过低' % t

    print('\n  2. 去重层：真值簇塌陷情况（用注入的重复关系对账）')
    alive_counts = Counter(len([i for i in g if i in kept_ids]) for g in groups)
    collapsed = alive_counts.get(0, 0) + alive_counts.get(1, 0)
    measured_recall = collapsed / len(groups)
    print('     真值簇 %d 个；存活篇数分布 %s'
          % (len(groups), dict(sorted(alive_counts.items()))))
    print('     塌成 ≤1 篇的簇 %d 个，实测召回 %.1f%%（扫描预测 %.1f%%）'
          % (collapsed, 100 * measured_recall, 100 * best['recall']))
    # LSH 用了鸽笼段数 bands=k+1，实测召回必须与扫描一致，且没有簇留下 3 篇以上
    assert abs(measured_recall - best['recall']) < 1e-9, '实测召回与扫描不一致'
    assert max(alive_counts) <= 2, '有簇留下 3 篇以上，去重基本失效'

    print('\n  3. 误删率：不属于任何簇的正常文档，过了规则层就必须活到去重后')
    in_group = {i for g in groups for i in g}
    normal_alive = [i for i in truth if truth[i] == 'normal'
                    and i not in in_group and i not in rule_dropped]
    killed = [i for i in normal_alive if i not in kept_ids]
    print('     应存活 %d 篇，被误删 %d 篇，误删率 %.2f%%'
          % (len(normal_alive), len(killed), 100.0 * len(killed) / max(1, len(normal_alive))))
    assert len(killed) / max(1, len(normal_alive)) <= 0.01, '误删率超过 1%'

    print('\n  4. 输出不变量')
    keys = Counter(exact_key(d['norm']) for d in final)
    dup_left = sum(1 for v in keys.values() if v > 1)
    print('     最终语料 %d 篇，完全重复残留 %d 组' % (len(final), dup_left))
    assert dup_left == 0, '最终语料里还有完全重复'

    close = sum(1 for a, b in (rng.sample(final, 2) for _ in range(4000))
                if hamming(fps[a['id']], fps[b['id']]) <= best['k'])
    print('     随机抽取 4000 对（每对取自不同文档），汉明距离 ≤ k 的 %d 对' % close)
    assert close == 0, '最终语料里还有近似重复'

    print('\n  5. 配比与账目一致性')
    tot = sum(cat_after.values())
    for cat, want in (('技术', 5), ('生活', 3), ('新闻', 2)):
        print('     %-4s 目标 %4.1f%%  实际 %4.1f%%'
              % (cat, 10.0 * want, 100.0 * cat_after[cat] / tot))
        assert abs(100.0 * cat_after[cat] / tot - 10.0 * want) < 1.0, '配比偏离目标'
    assert len(final) == tot == sum(cat_after.values()), '配比后计数不一致'
    assert stage[-1][1][1] == len(final)
    assert all(os.path.getsize(p) > 0 for p in (p_raw, p_drop, p_clean))
    assert len(final) < len(keep) <= len(after_exact) < len(kept_docs) < base, '工序未递减'
    # 去重不是无损的：距离分布有重叠，F1 最优点必然在召回与误删之间取舍。
    # 这里卡一个诚实的底线，而不是假装能做到 100%。
    assert best['recall'] >= 0.85, 'F1 最优参数召回过低（%.1f%%）' % (100 * best['recall'])
    assert best['fpr'] <= 0.03, 'F1 最优参数误删率过高（%.2f%%）' % (100 * best['fpr'])
    assert best['f1'] >= 0.9, 'F1 最优值过低（%.4f）' % best['f1']

    print('\n  全部断言通过。')
    print('\n[产物] 原始 %d 篇 → %s\n       拦掉 %d 篇 → %s\n       可用 %d 篇 → %s'
          % (base, p_raw, base - len(keep), p_drop, len(final), p_clean))
    return 0


if __name__ == '__main__':
    sys.exit(main())
