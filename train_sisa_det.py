# -*- coding: utf-8 -*-
"""
train_sisa_det.py -- training entry point for SCDE.

SCDE = SISA (subspace-constrained view generation) + SDC (supervised offset
contrast) for single-source cross-scene hyperspectral classification.

    source patch --(SISA)--> displaced view, along the offset subspace
                 --(SDC)--> contrastive constraint on the view pair
    total loss   = CE(X) + lambda_sisa * CE(X') + lambda_sdc * L_SDC(X, X')

The three published arms are configured in scripts/exp_common.sh. A single arm
can also be run directly:

  python train_sisa_det.py --source Houston13 --target Houston18 \
      --data_path ./data/ --patch_size 13 --epochs 200 --batch_size 4096 \
      --no_lr_decay --gpu_resident 1 --num_workers 0 \
      --proj_init Identity --nsd --nsd_mode swa --nsd_start 50 \
      --re_ratio 5 --lr 0.01 --wd 0.0001 --eval_batch_size 4096 \
      --sisa_alpha 2.0 --sisa_blocks 8 --sisa_classwise \
      --sisa_lambda 0.5 --sdc_lambda 1.0 --sdc_temp 0.1 --sdc_max_n 1024 \
      --seeds 333,444,555,666,777

This file is the research training script. Besides the SCDE path it still
carries options from earlier experiments (alternative projection
initialisations, directional controls, spatial masks, extra regularisers).
All of them are inactive under the flags above and are scheduled for removal
in the cleaned-up release.
"""
import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data

from datasets import HyperX, get_dataset


class _SigmaAdaptState:
    """[M3 2026-10-08] 逐方向幅度 σ 在线重估的全局状态。

    设计上刻意做成「一个模块级可变单例」而不是新增函数参数：
    改 sisa_augment 的签名会牵动 11 个调用点，且任何一个漏改都会在
    异质/classwise 分支上炸出更难查的错。默认 factor=1.0 ⇒ 逐位等价历史行为。
    """
    __slots__ = ('factor', 'hist', 'enabled')

    def __init__(self):
        self.factor = 1.0
        self.hist = []
        self.enabled = False


_SIG_ADAPT_STATE = _SigmaAdaptState()

from utils_HSI import metrics, sample_gt, seed_worker
from vca_dres import build_einit_cnn


# ============================================================
# 种子初始化
# ============================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id, base_seed):
    seed_worker(base_seed * 10000 + worker_id)



def gpu_augment(data_batch, p_flip=0.5, p_rad=0.5):
    """GPU batch-level flip+radiation augmentation.
    data_batch: (B, C, H, W) on GPU
    """
    B = data_batch.size(0)
    # Horizontal flip (dim 3 = W)
    mask_h = torch.rand(B, device=data_batch.device) < p_flip
    if mask_h.any():
        data_batch[mask_h] = data_batch[mask_h].flip(dims=[3])
    # Vertical flip (dim 2 = H)
    mask_v = torch.rand(B, device=data_batch.device) < p_flip
    if mask_v.any():
        data_batch[mask_v] = data_batch[mask_v].flip(dims=[2])
    # Radiation noise
    mask_rad = torch.rand(B, device=data_batch.device) < p_rad
    if mask_rad.any():
        alpha = 0.9 + torch.rand(B, 1, 1, 1, device=data_batch.device) * 0.2
        noise = torch.randn_like(data_batch)
        aug_data = alpha * data_batch + (1.0 / 25.0) * noise
        idx = mask_rad.nonzero(as_tuple=True)[0]
        data_batch[idx] = aug_data[idx]
    return data_batch


def spectral_shift_augment(data_batch, strength=0.1):
    """CDSP: physically-grounded HSI cross-domain spectral shift simulation (R11-R15).

    Mimics the characterized cross-sensor / cross-time domain gap by applying:
      1) per-band radiometric scaling (cross-sensor gain differences),
      2) additive sensor noise,
      3) a small circular band shift (band misregistration / spectral offset, R11).
    The learnable projection is then trained to keep classes discriminable under
    this shift -> "preserve discriminative cross-domain information".
    data_batch: (B, C, H, W) on GPU. Returns shifted tensor (same shape).
    """
    x = data_batch
    C = x.size(1)
    dev = x.device
    # 1) Band-wise radiometric scaling (broadcast over B,H,W)
    band_scale = 1.0 + strength * torch.randn(C, 1, 1, device=dev)
    x = x * band_scale
    # 2) Additive sensor noise
    x = x + (strength * 0.15) * torch.randn_like(x)
    # 3) Circular band shift (small, preserves tensor shape)
    if C > 4:
        shift = int(torch.randint(-2, 3, (1,)).item())
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=1)
    return x


def _push_one(model, Xs, Vc, eps):
    """有限差分把一组光谱方向 Vc (r,B) 推到特征空间: J v_i ≈ [feat(x+eps·v_i)-feat(x)]/eps
    返回 JV (r,d) —— 已含网络对各方向的敏感性。"""
    dev = Xs.device
    with torch.no_grad():
        f0 = model.forward_backbone(Xs)
        if isinstance(f0, (tuple, list)):
            f0 = f0[0]
        cols = []
        for i in range(Vc.shape[0]):
            v = Vc[i].to(dev).to(Xs.dtype).view(1, -1, 1, 1)
            f1 = model.forward_backbone(Xs + eps * v)
            if isinstance(f1, (tuple, list)):
                f1 = f1[0]
            cols.append(((f1 - f0) / eps).mean(dim=0))
    return torch.stack(cols, dim=0)                    # (r, d)


def push_subspace_to_feature(model, X, y, V, sigma, eps=1e-2, n_max=2048, classes=None):
    """M1b: 把偏移子空间从【光谱空间】推到【特征空间】。

    V     : (r,B) 全局 或 (C,r,B) classwise
    sigma : (r,)  或 (C,r)
    返回 (W, sigma_J, scale):
      W       : (r,d) 或 (C,r,d) —— JV 经 QR 正交化后的特征侧支撑
      sigma_J : 双重拉伸后的尺度 = sigma_i * ||J v_i||
      scale   : ||J v_i||（网络敏感性，仅作诊断打印）
    """
    dev = X.device
    eps = float(eps)

    def _one(Vc, sc, Xs):
        n = min(int(n_max), Xs.shape[0])
        Xs = Xs[:n].detach()
        JV = _push_one(model, Xs, Vc, eps)             # (r, d)
        scale = JV.norm(dim=1)                         # (r,)
        # 🔴 数值保护: 某方向若被网络压平到 0, QR 会退化, 用单位向量兜底
        if not torch.isfinite(JV).all() or float(scale.max()) <= 0:
            d = JV.shape[1]
            Wc = torch.eye(min(Vc.shape[0], d), d, device=dev, dtype=JV.dtype)
            return Wc, torch.zeros(Wc.shape[0], device=dev, dtype=JV.dtype), scale
        Q, _ = torch.linalg.qr(JV.t())                 # Q: (d, r)
        Wc = Q.t().contiguous()                        # (r, d)
        return Wc, (sc.to(dev) * scale[:Wc.shape[0]]).clamp(min=1e-12), scale

    if V.dim() == 3:
        Ws, sJs, scs = [], [], []
        for c in range(V.shape[0]):
            if y is not None and int((y == c).sum()) >= 2:
                Xc = X[y == c]
            else:
                Xc = X
            Wc, sJc, sc = _one(V[c], sigma[c], Xc)
            Ws.append(Wc)
            sJs.append(sJc)
            scs.append(sc)
        return torch.stack(Ws), torch.stack(sJs), torch.stack(scs)

    W, sJ, sc = _one(V, sigma, X)
    return W, sJ, sc


def estimate_shift_subspace(img, gt, n_blocks=8, rank=5, min_px=10, hetero=False,
                            classwise=False):
    """SISA: estimate the *cross-domain shift subspace* from SOURCE data only.

    Rationale (R14/R15): the real cross-domain spectral shift is not an arbitrary
    hand-designed perturbation -- it lives in a low-dimensional subspace that is
    already visible as *spatial heterogeneity inside the source domain*
    (different regions / acquisition conditions shift class spectra coherently).

    We therefore:
      1) split the source scene into a spatial block grid,
      2) per block, average the class-conditional deviation from the global class
         mean  ->  isolates the block(=condition) effect from class composition,
      3) SVD over blocks  ->  top-r shift directions V (r,B) and their scales sigma.

    Uses NO target-domain data or labels (fully DG-legal). Auto-adapts to each
    dataset: strong globally-consistent shift (Houston) yields one dominant
    direction with large sigma; heterogeneous shift (Pavia) yields several
    directions; weak shift (HyRANK) yields small sigma -> mild augmentation.

    hetero=True: 每块单独 SVD, 返回 (V_blocks (n_blk,r,B), sigma_blocks (n_blk,r))。
    块内标注样本不足(<min_px 或 <2 类)的块 fallback 到全局 V/sigma。
    (针对 Pavia 的异质偏移: 全局单一方向概括不了块级不同偏移方向)

    Returns:
      hetero=False: V (rank, B) float32, sigma (rank,) float32
      hetero=True : V_blocks (n_blk, rank, B) float32, sigma_blocks (n_blk, rank) float32
    """
    H, W, B = img.shape
    X = img.reshape(-1, B).astype(np.float64)
    y = gt.reshape(-1)

    classes = [c for c in np.unique(y) if c > 0]
    gmean = {}
    for c in classes:
        m = X[y == c]
        if len(m) >= min_px:
            gmean[c] = m.mean(axis=0)

    bh = max(1, H // n_blocks)
    bw = max(1, W // n_blocks)
    n_col = int(np.ceil(W / bw))
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bid = ((ii // bh) * n_col + (jj // bw)).reshape(-1)
    n_blk = int(bid.max()) + 1

    rows = []
    for b in np.unique(bid):
        sel = (bid == b)
        devs = []
        for c, gm in gmean.items():
            m = X[sel & (y == c)]
            if len(m) >= min_px:
                devs.append(m.mean(axis=0) - gm)
        if devs:
            rows.append(np.mean(devs, axis=0))

    if len(rows) < 2:  # fallback: unconditional block means
        rows = []
        for b in np.unique(bid):
            sel = (bid == b) & (y > 0)
            if sel.sum() >= min_px:
                rows.append(X[sel].mean(axis=0))
    if len(rows) < 2:  # last resort: global spectral covariance
        sel = y > 0
        D = X[sel] - X[sel].mean(axis=0, keepdims=True)
    else:
        D = np.stack(rows)
        D = D - D.mean(axis=0, keepdims=True)

    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    r = int(min(rank, Vt.shape[0]))
    V = Vt[:r]
    sigma = S[:r] / np.sqrt(max(1, D.shape[0]))

    if not hetero and not classwise:
        return V.astype(np.float32), sigma.astype(np.float32)

    # ---- 异质: 每块单独 SVD ----
    if hetero:
        V_blocks = np.tile(V.astype(np.float32)[None], (n_blk, 1, 1))       # (n_blk,r,B) 默认全局
        sigma_blocks = np.tile(sigma.astype(np.float32)[None], (n_blk, 1))  # (n_blk,r)
        for b in range(n_blk):
            sel = (bid == b)
            devs = []
            for c, gm in gmean.items():
                m = X[sel & (y == c)]
                if len(m) >= min_px:
                    devs.append(m.mean(axis=0) - gm)
            if len(devs) >= 2:
                Db = np.stack(devs)
                Db = Db - Db.mean(axis=0, keepdims=True)
                _, Sb, Vtb = np.linalg.svd(Db, full_matrices=False)
                rb = int(min(r, Vtb.shape[0]))
                V_blocks[b, :rb] = Vtb[:rb].astype(np.float32)
                sigma_blocks[b, :rb] = (Sb[:rb] / np.sqrt(max(1, Db.shape[0]))).astype(np.float32)
        return V_blocks, sigma_blocks

    # ---- 类条件: 每个【类】单独估偏移方向 ----
    # 动机: 全局单一方向隐含假设"域偏移对所有地物都一样", 这个假设太粗。
    #   诊断已证 (diag_classwise_shift.py) 类间方向两两 |cos|:
    #     Pavia 0.783 / Houston 0.690 / HyRANK 0.474 (都远低于 1, 随机基线仅 .06~.12)
    #   -> 植被 / 建筑 / 水体 / 裸土 的光谱偏移方向确实不同。
    # 比空间 hetero 稳得多: hetero 崩在欠采样 (单空间块仅几百像素, Houston 83% 的块
    #   退回全局); 而"类"是有语义的划分, 每类样本数千, 方向估计可靠。
    #
    # 返回 (n_cls, r, B): 第 ci 行对应【0-based 类索引 ci】(= 1-based 标签 ci+1),
    #   因此训练时可直接用 label_batch (已是 0-based) 索引。
    # 块数不足 (<3) 的类 fallback 到全局 V/sigma。
    # n_cls 必须覆盖【所有】正标签, 不能只取 gmean 的键:
    #   gmean 过滤掉了样本不足(min_px)的类, 若 n_cls 偏小, 训练时 label_batch
    #   (0-based, 最大可到 max_label-1) 索引 V_cls 会越界。
    #   样本不足的类保留全局 V 作为 fallback, 不影响正确性。
    _all_cls = [int(c) for c in np.unique(y) if c > 0]
    n_cls = max(_all_cls) if _all_cls else 0
    # [2026-09-19 加固] 训练时用 label_batch (0-based) 直接索引 V_cls，故 n_cls 必须严格等于
    # 最大 0-based 标签 + 1。spec 上要求标签连续取 1..K（无空洞），否则越界。此处显式断言，
    # 而不是依赖「max(1-based) == max(0-based)+1」这一碰巧成立的关系。
    assert n_cls == len(_all_cls), (
        f"classwise: 标签存在空洞（{len(_all_cls)} 个类但最大标签为 {n_cls}）；"
        f"V_cls 行数会小于 label_batch 的最大索引，训练时将越界。"
        f"请先把标签重映射为连续的 1..{len(_all_cls)}。")
    V_cls = np.tile(V.astype(np.float32)[None], (n_cls, 1, 1))        # (n_cls,r,B) 默认全局
    sigma_cls = np.tile(sigma.astype(np.float32)[None], (n_cls, 1))   # (n_cls,r)
    n_ok = 0
    for c, gm in gmean.items():
        ci = int(c) - 1                        # 1-based 标签 -> 0-based 行索引
        if ci < 0 or ci >= n_cls:
            continue
        rows_c = []
        for b in np.unique(bid):
            m = X[(bid == b) & (y == c)]
            if len(m) >= min_px:
                rows_c.append(m.mean(axis=0) - gm)
        if len(rows_c) >= 3:                   # 至少 3 个块才估方向, 否则保留全局
            Dc = np.stack(rows_c)
            Dc = Dc - Dc.mean(axis=0, keepdims=True)
            _, Sc, Vtc = np.linalg.svd(Dc, full_matrices=False)
            rc = int(min(r, Vtc.shape[0]))
            V_cls[ci, :rc] = Vtc[:rc].astype(np.float32)
            sigma_cls[ci, :rc] = (Sc[:rc] / np.sqrt(max(1, Dc.shape[0]))).astype(np.float32)
            n_ok += 1
    _n_fallback = len(gmean) - n_ok
    if _n_fallback > 0:
        # [2026-09-19] 显式列出静默回退的类：这些类【不会】用逐类方向，而是沿用全局 V/sigma。
        # 实测每个数据集恰有 1 个类触发（Houston c4 / HyRANK-Loukia c1 / HyRANK-Dioni c3）。
        _fb = [int(c) for c, gm in gmean.items()
               if sum(1 for b in np.unique(bid) if ((bid == b) & (y == c)).sum() >= min_px) < 3]
        print(f"[SISA] classwise FALLBACK to global for classes {sorted(_fb)} "
              f"(<3 blocks with >= {min_px} px each -> direction not estimated)")
    print(f"[SISA] classwise: {n_ok}/{len(gmean)} classes own shift direction "
          f"(rest fall back to global)")
    return V_cls, sigma_cls


def make_control_directions(mode, V, img_src, gt_src, seed=0):
    """A5 系列：保持秩、幅度与形状不变，只替换【不变性轴的方向】。

    动机（2026-09-14 判定实验 tools/houston_reconcile.py）：`--sdc_dir random` 用的是
    **白噪声子空间**，它与任何"光谱平滑"的方向几乎正交，因此"估计轴 >> 随机轴"可能只是
    "平滑 vs 白噪声"，而不是"选对了轴"。必须换成【结构化】负对照。

    mode:
      'smooth'  光谱高斯平滑后的随机方向（真实、但完全不来自数据）
                -> 检验"只要扰动够平滑/真实就行？"
      'pca'     源域有标签像素的前 r 个主方向（= 领先数据子空间，最强竞争者）
                -> 检验"S 是不是只是领先数据子空间？"
      'cap'     classwise 专用：保持每类 (V,sigma) 不变，只【打乱"类 -> 方向"的指派】
                -> 检验"类条件"是否真的必要（错的类配对，错的轴）
      'intra'   **类内** PCA 顶 r 方向（每像素减类均值后 PCA）= 去掉类间差异后的数据主成分
                -> C-56：检验"`V̂` 的身份是否有用"（生产 `V̂` 有 81–84% 能量落在它里面）
      'orth'    E2 正交控制：同维随机方向**约束在 `⊥span(V̂)`**（project-wide 对照纪律）
                -> 生产 `random` 是**未约束**同维随机，弱于纪律要求；本支把它补齐
    """
    rng = np.random.RandomState(int(seed))
    B = int(V.shape[-1])
    r = int(V.shape[-2]) if V.ndim == 3 else int(V.shape[0])

    if mode == 'cap':
        if V.ndim != 3:
            raise SystemExit('sdc_dir=cap 只对 --sisa_classwise 有意义')
        return V[rng.permutation(V.shape[0])].copy()

    if mode == 'ms1':
        # P0 (2026-09-15) capacity-ceiling proxy axis: span{ms, 1}.
        # ms = mean spectrum of labelled source pixels; 1 = constant (all-band) vector.
        # These are exactly the two directions the U0 diagnosis flagged as systematically
        # excluded from S (dm: 2.14% inside span(S) vs 38.3% inside span{ms,1}).
        # row0/row1 span{ms,1} after the QR below; rows 2..r-1 are random orthonormal
        # complements so rank / sigma / tensor shape stay identical to the estimated
        # axis (R3: same alpha = same perturbation budget). Additive; DG-legal
        # (source labels only, no target data used).
        Xs = np.asarray(img_src).reshape(-1, B).astype(np.float64)
        ys = np.asarray(gt_src).reshape(-1)
        if int((ys > 0).sum()) > 100:
            ms = Xs[ys > 0].mean(0)
        else:
            ms = Xs.mean(0)
        rows = [ms, np.ones(B, dtype=np.float64)]
        for _ in range(max(0, r - 2)):
            rows.append(rng.standard_normal(B))
        M = np.stack(rows)
    elif mode == 'smooth':
        k = np.exp(-0.5 * ((np.arange(B) - B // 2) / 3.0) ** 2)
        k = k / k.sum()
        M = np.stack([np.convolve(rng.standard_normal(B), k, mode='same') for _ in range(r)])
    elif mode == 'pca':
        X = np.asarray(img_src).reshape(-1, B).astype(np.float64)
        y = np.asarray(gt_src).reshape(-1)
        Xl = X[y > 0] if int((y > 0).sum()) > 100 else X
        _, _, Vt = np.linalg.svd(Xl - Xl.mean(0, keepdims=True), full_matrices=False)
        M = Vt[:r]
    elif mode == 'intra':
        # C-56（2026-09-23）：**类内** PCA 顶-r 方向 = 去掉类间差异后的数据主成分
        #   （= 每个像素减其【类均值】后做 PCA）。这正是 #33 探针里 `intra_frac`
        #   所用子空间：生产 `V̂` 有 81–84% 的能量落在它里面。
        #   保持秩/形状/Σ/α 不变（只换方向），用于在生产通道上判定「`V̂` 的身份是否有用」
        #   —— 若 intra 的 OA 与 shift 不显著不同 ⇒ 身份无关，SISA 收益来自「沿数据主方差增广」；
        #      若显著更差 ⇒ 身份有用，有用分量就在「类内 PCA 的正交补」里。DG-legal（只用源域标签）。
        #
        # 🔴 2026-09-23 修正（classwise 3-D）：Houston 生产臂**必带 `--sisa_classwise`**，
        #   此时 `_V` 是 (n_cls, r, B) 的**逐类**方向（类间 |cos| 0.690）。
        #   旧写法落到函数尾部的 `np.tile(Vb[None], ...)` ⇒ 会把逐类方向**塌缩成单一全局方向**，
        #   于是"类内 PCA"的效应与"丢了逐类结构"的效应混在一起，**结论不可用**。
        #   现按 `V.ndim == 3` 分流：逐类时用【该类自己的】主方差方向（保持逐类结构不变）。
        X = np.asarray(img_src).reshape(-1, B).astype(np.float64)
        y = np.asarray(gt_src).reshape(-1)
        if V.ndim == 3:
            n_cls = int(V.shape[0])
            out3 = np.empty((n_cls, r, B), dtype=np.float32)
            for ci in range(n_cls):
                m = (y == (ci + 1))
                if int(m.sum()) > r + 1:
                    Xc = X[m] - X[m].mean(0, keepdims=True)
                    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                    rr = int(min(r, Vt.shape[0]))
                    M3 = Vt[:rr]
                    if rr < r:
                        M3 = np.concatenate(
                            [M3, rng.standard_normal((r - rr, B))], axis=0)
                else:
                    M3 = rng.standard_normal((r, B))
                Q3, _ = np.linalg.qr(M3.T)
                out3[ci] = np.ascontiguousarray(Q3.T).astype(np.float32)
            _al = [float(abs(np.dot(out3[i, 0], V[i, 0]))) for i in range(n_cls)]
            print(f"[intra] classwise 3-D: per-class intra-PCA axis, "
                  f"|cos(row0 vs shift row0)| median={np.median(_al):.4f} "
                  f"(rank={r}, n_cls={n_cls})")
            return out3
        mask = (y > 0) if int((y > 0).sum()) > 100 else np.ones_like(y, dtype=bool)
        yl, Xl = y[mask], X[mask]
        cls = [int(c) for c in np.unique(yl) if c > 0]
        gmean = np.stack([Xl[yl == c].mean(0) for c in cls])
        Xc = np.empty_like(Xl)
        for i, c in enumerate(cls):
            Xc[yl == c] = Xl[yl == c] - gmean[i]
        _, _, Vt = np.linalg.svd(Xc - Xc.mean(0, keepdims=True), full_matrices=False)
        M = Vt[:r]
    elif mode == 'orth':
        # E2（2026-09-28）：**正交控制**（`⊥span(V̂)` 的同维随机） —— 与本项目的对照纪律一致
        #   （`MEMORY` §1.3：「对照选正交控制 ⊥span(V5)，不要各向同性 iso」）。
        #   生产里的 `random` 是【未约束】同维随机（QR(gaussian)，见本文件 random 分支）
        #   ⇒ 生产阶梯一直在用**弱于纪律要求**的对照；本支把它补齐，让生产阶梯与探针同级。
        #   做法：随机方向先投影掉估计轴 V̂ 的张成，再取前 r 个正交行。
        #   ⇒ 秩 / 形状 / Σ 不变（只换方向），DG-legal（只用源域，不读目标域）。
        if V.ndim == 3:
            # classwise：逐类投影掉【该类自己的】V̂ 行空间，保持逐类结构不变。
            # （若落到函数尾部 np.tile 会把逐类方向塌缩成单一全局方向，见 C-56 记录）
            n_cls = int(V.shape[0])
            out3 = np.empty((n_cls, r, B), dtype=np.float32)
            for ci in range(n_cls):
                Vu = V[ci].astype(np.float64)
                G = rng.standard_normal((B, max(r, 1)))
                G = G - Vu.T @ (Vu @ G)          # 逐类投影掉该类 V̂ 的行空间
                Q, _ = np.linalg.qr(G)
                out3[ci] = np.ascontiguousarray(Q.T[:r]).astype(np.float32)
            _al = [float(abs(np.dot(out3[i, 0], V[i, 0]))) for i in range(n_cls)]
            print(f"[orth] classwise 3-D: per-class axis ⊥ span(V̂), "
                  f"|cos(row0 vs shift row0)| median={np.median(_al):.4f} "
                  f"(rank={r}, n_cls={n_cls})")
            return out3
        Vu = np.asarray(V, np.float64)
        if Vu.shape[0] >= B:
            print(f"[orth] ⚠️ rank(V̂)={Vu.shape[0]} >= B={B} ⇒ 无正交补，退回未约束 random")
            M = rng.standard_normal((r, B))
        else:
            G = rng.standard_normal((B, r))
            G = G - Vu.T @ (Vu @ G)              # 投影掉 V̂ 的行空间
            M = G.T
    else:
        raise SystemExit('unknown sdc_dir mode: %s' % mode)

    # 行正交归一化（QR 的列 -> 转置成行）
    Q, _ = np.linalg.qr(np.asarray(M, np.float64).T)
    Vb = np.ascontiguousarray(Q.T).astype(np.float32)
    if V.ndim == 3:
        return np.tile(Vb[None], (V.shape[0], 1, 1)).copy()
    return Vb


def reshape_sigma_gamma(sigma, gamma):
    """[STRETCH] 重塑 SISA 度量 Lambda=diag(sigma^2) 的**形状**，保持预算不变。

    sigma_i' = sigma_i^gamma * c,  c = sqrt(sum(sigma^2) / sum(sigma'^2))

    即 sum(sigma_i'^2) == sum(sigma_i^2)  =>  E||delta||^2 不变，alpha 无需改动即可公平比较。

      gamma = 1.0 : 现役（跨域偏移的相对强度 == 源域类内异质性的相对强度）
      gamma = 0.0 : 各向同性（集合内等权，最大熵先验）
      gamma < 1   : 向各向同性插值

    动机（2026-09-20 E1 实测，tools/_e1_stretch_shape.py）：
      真实跨域偏移在 M(c) 各轴上的能量份额 e_true 与 sigma^2 的份额，
      在 Pavia / HyRANK 上余弦 0.97（假设成立），但在 Houston 上只有 0.52
      （而各向同性为 0.94）=> 现役假设在 Houston 上不成立。

    支持 sigma 为 (r,) 或 (n_cls, r)（classwise 逐类重塑，逐行保持预算）。
    """
    if gamma is None or abs(float(gamma) - 1.0) < 1e-12:
        return sigma
    s = np.asarray(sigma, dtype=np.float64)
    out = np.empty_like(s)
    if s.ndim == 1:
        rows = [s]
        dst = [out]
    else:
        rows = [s[i] for i in range(s.shape[0])]
        dst = [out[i] for i in range(s.shape[0])]
    for src, dst_row in zip(rows, dst):
        s2 = float((src ** 2).sum())
        g = np.power(np.maximum(src, 1e-30), float(gamma))
        denom = float((g ** 2).sum())
        c = np.sqrt(s2 / (denom + 1e-30)) if denom > 0 else 0.0
        dst_row[:] = g * c
    return out.astype(np.float32)


def class_mean_spectra(img, gt, n_cls, min_px=10):
    """[XG 2026-09-21] 逐类平均光谱 mu_c: (n_cls, B) float32, 行序 = 0-based 类索引。
    gt 为 1-indexed (0=背景); 样本不足的类退化为全局均值。
    """
    H, W, B = img.shape
    X = img.reshape(-1, B).astype(np.float64)
    y = gt.reshape(-1)
    gl = X[y > 0].mean(axis=0) if (y > 0).any() else np.zeros(B)
    out = np.tile(gl[None], (max(1, int(n_cls)), 1))
    for c in range(1, int(n_cls) + 1):
        m = X[y == c]
        if len(m) >= min_px:
            out[c - 1] = m.mean(axis=0)
    return out.astype(np.float32)


def xg_relevance(x, V, sigma=None, y=None, mu_full=None, block_ids=None,
                 mode='coupled', tau=1.0, beta=1.0):
    """[XG] 实例级"方向相关性" g: (N, r)。

    图像空间 -> 偏移空间: p_i = V_sel^T (x_i - mu_c(i)) 是样本自身在 S 内的坐标;
    g 由 |p|^tau 归一化而来 —— 只回答"这 r 个方向里当前样本占哪几个",
    不给新方向(重估 S 已死)。

    🔴 能量守恒归一化(不是"均值 1"): 末尾按 sigma 加权, 把每个样本的
      E||delta||^2 ∝ Σ_k (sigma_k g_k)^2 拉回 == Σ_k sigma_k^2。
      否则 g 的方差会抬高扰动能量, 四臂比较就混进"能量变大"的混淆,
      不再是纯方向实验(球面幅度退化的直接推论)。

    mode: coupled(原样) / perm(batch 内打乱, 只破坏实例对应) / anti(点亮最不占的方向)
    beta<1: 与平坦 g=1 混合(混合后再做能量守恒归一化)。
    """
    n = x.size(0)
    xm = x.detach().mean(dim=(2, 3)).float().cpu()      # (N,B) 偏移在 patch 内空间恒定
    if mu_full is not None and y is not None:
        _mu = mu_full.detach().float().cpu()
        dev = xm - _mu[y.detach().cpu().long()]         # (N,B)
    else:
        dev = xm
    Vc = V.detach().float().cpu()
    idx = None
    if Vc.dim() == 3:
        idx = block_ids if block_ids is not None else y
        if idx is None:
            return None
        idx = idx.detach().cpu().long()
        Vsel = Vc[idx]                                  # (N,r,B)
    else:
        Vsel = Vc.unsqueeze(0).expand(n, -1, -1)        # (N,r,B)
    p = torch.einsum('nrb,nb->nr', Vsel, dev)           # (N,r)
    g_raw = p.abs().clamp_min(1e-8) ** float(tau)
    g = g_raw / (g_raw.mean(dim=1, keepdim=True) + 1e-12)
    if float(beta) < 1.0:
        g = (1.0 - float(beta)) + float(beta) * g
        g = g / (g.mean(dim=1, keepdim=True) + 1e-12)
    if mode == 'perm':
        g = g[torch.randperm(n)]
    elif mode == 'anti':
        # 🔴 修正(2026-09-21): 反序必须作用在【beta 混合之后】的 g 上。
        #    原实现用 g_raw, 导致 anti 完全忽略 --xg_beta (b0.5 与 b1.0 逐位相同,
        #    12 个 run 变成 6 个重复)。现改 g => beta 生效。
        ga = g.max(dim=1, keepdim=True).values - g + 1e-8
        g = ga / (ga.mean(dim=1, keepdim=True) + 1e-12)
    # ---- 能量守恒: 把 Σ_k (sigma_k g_k)^2 拉回 Σ_k sigma_k^2 ----
    if sigma is not None:
        sc = sigma.detach().float().cpu()
        if sc.dim() >= 2:
            if idx is None:
                return g
            sc = sc[idx]                                # (N,r)
        else:
            sc = sc.reshape(1, -1).expand(n, -1)        # (N,r)
        w2 = sc ** 2
        num = w2.sum(dim=1, keepdim=True)
        den = (w2 * g ** 2).sum(dim=1, keepdim=True) + 1e-12
        g = g * torch.sqrt(num / den)
    return g


def _spatial_masks(ps, radius, device):
    """(center, ring) 两个 (ps,ps) 0/1 掩码; center = 切比雪夫半径 <= radius。"""
    c = ps // 2
    ar = torch.arange(ps, device=device)
    ii, jj = torch.meshgrid(ar, ar, indexing='ij')
    d = torch.maximum((ii - c).abs(), (jj - c).abs()).float()
    center = (d <= float(radius)).float()
    return center, 1.0 - center


def sisa_augment(x, V, sigma, alpha=1.0, block_ids=None, mult_strength=0.0, path_strength=0.0,
                 xg_mode='off', xg_tau=1.0, xg_beta=1.0, xg_mu=None, y=None,
                 mask_mode='none', mask_radius=2, mask_energy=True):
    """SISA: perturb each sample along the estimated source-internal shift subspace.

    x: (N, B, H, W); V: (r, B) 或 (n_blk, r, B); sigma: (r,) 或 (n_blk, r).
    block_ids: (N,) 每个样本的空间块 id (仅异质模式传入); 为 None 时走全局 V。
    Shift is spatially constant within a patch (a domain shift is a per-scene/
    per-condition spectral offset, not noise)  —— 当 mask_mode='none' 时成立（生产口径）。

    [空间掩码 2026-09-21] mask_mode != 'none' 时，偏移**不再空间一致**：
      'center' 只作用于中心 (2*radius+1)^2 区域; 'ring' 只作用于外环;
      'mix'    逐样本随机取 {全图, 中心, 外环} 之一;
      'incons' 全图覆盖但中心与外环用**两次独立抽样**的 delta（空间不一致但处处有扰动）。
    mask_energy=True 时按 1/sqrt(覆盖比例) 缩放，使扰动 **Frobenius 能量与全图一致**
      （R3: 只改"空间分布"，扰动预算逐位不变）。'incons' 天然等能量，无需缩放。

    U0 (radiative-transfer-correct perturbation, 2026-09-10):
      mult_strength > 0 -> MULTIPLICATIVE (illumination/geometry) shift:
          x' = x * (1 + mult_strength * delta)
      path_strength > 0 -> additive PATH-RADIATION (atmospheric) term:
          x' = x' + path_strength * eps0 * 1   (eps0 ~ N(0,1), 1 = all-band vector)
      mult_strength == 0 -> legacy ADDITIVE shift x' = x + delta (backward compatible).
    """
    n = x.size(0)
    # [M3 2026-10-08] 逐方向幅度 σ 在线重估（C4）。
    # 语义：不是换方向（那是 DSSD/dssd_period），而是把「沿每个方向放多大」的
    # 乘子从数据侧重估 —— 减掉当前 batch 在该方向上【已经自然存在】的那一部分，
    # 剩下的才是"需要补"的幅度。符号沿用原 σ（只在幅度上收紧）。
    # off / 非正修正因子时逐位等价历史行为。
    _m3 = float(getattr(_SIG_ADAPT_STATE, 'factor', 1.0))
    if _m3 != 1.0:
        sigma = (sigma * _m3) if not torch.is_tensor(sigma) else (sigma * _m3)
    _xg = None
    if xg_mode != 'off':
        _xg = xg_relevance(x, V, sigma=sigma, y=y, mu_full=xg_mu, block_ids=block_ids,
                           mode=xg_mode, tau=xg_tau, beta=xg_beta)

    def _draw_delta():
        """抽一次 (N,B) 的 CPU delta。"""
        if block_ids is not None and V.dim() == 3:
            # 异质: 每个样本用其所属块的 V/sigma
            # 与非 hetero 分支一致: 在 CPU 生成确定性噪声, 最后再搬回 x.device
            Vsel = V[block_ids].detach().cpu()                                # (N, r, B)
            sigsel = (sigma[block_ids].unsqueeze(-1) * alpha).detach().cpu()  # (N, r, 1)
            eps = torch.randn(n, V.size(1), 1) * sigsel * (_xg.unsqueeze(-1) if _xg is not None else 1.0)
            return torch.einsum('nrb,nrd->nbd', Vsel, eps)                    # (N, B, 1) CPU
        r = V.size(0)
        _sig_cpu = (sigma.unsqueeze(0) * alpha).detach().cpu()
        _V_cpu = V.detach().cpu()
        eps = torch.randn(n, r) * _sig_cpu * (_xg if _xg is not None else 1.0)
        return (eps @ _V_cpu)                                                 # (N, B) CPU

    _d = _draw_delta()
    if _d.dim() == 3:
        _d = _d.squeeze(-1)

    if mask_mode == 'none':
        delta = _d.to(x.device).view(n, -1, 1, 1)                # (N, B, 1, 1)
    else:
        ps = x.shape[-1]
        ctr, rng_ = _spatial_masks(ps, mask_radius, x.device)
        ones = torch.ones_like(ctr)
        if mask_mode == 'incons':
            # 中心与外环两次独立抽样，覆盖全图 => Frobenius 能量天然与全图一致
            _d2 = _draw_delta()
            if _d2.dim() == 3:
                _d2 = _d2.squeeze(-1)
            delta = (_d.to(x.device).view(n, -1, 1, 1) * rng_
                     + _d2.to(x.device).view(n, -1, 1, 1) * ctr)     # (N,B,ps,ps)
        else:
            if mask_mode == 'center':
                m = ctr.unsqueeze(0).expand(n, -1, -1)
            elif mask_mode == 'ring':
                m = rng_.unsqueeze(0).expand(n, -1, -1)
            elif mask_mode == 'mix':
                pick = torch.randint(0, 3, (n,))                      # 0 全图 / 1 中心 / 2 外环
                stack = torch.stack([ones, ctr, rng_], dim=0)          # (3,ps,ps)
                m = stack[pick]                                        # (n,ps,ps)
            else:
                raise ValueError('unknown mask_mode: %s' % mask_mode)
            frac = m.mean(dim=(1, 2)).to(x.device)                    # (n,)
            delta = _d.to(x.device).view(n, -1, 1, 1) * m.unsqueeze(1)  # (N,B,ps,ps)
            if mask_energy:
                delta = delta / torch.sqrt(frac.clamp(min=1e-6)).view(n, 1, 1, 1)

    if mult_strength > 0:
        # 乘性光照/几何变异 (radiative-transfer first-order): 光谱沿方向缩放
        x_shift = x * (1.0 + mult_strength * delta)
    else:
        x_shift = x + delta
    if path_strength > 0:
        # 加性路径辐射: 全谱恒定偏移 (大气散射), 与场景内容无关
        eps0 = torch.randn(n, 1, 1, 1, device=x.device) * path_strength
        x_shift = x_shift + eps0.expand(-1, x.shape[1], -1, -1)
    return x_shift


def model_sensitivity_subspace(model, x_loader, device, rank, n_classes=None, n_max=2048):
    """[DSSP-D 2026-09-21] 用【当前模型】的输入梯度二阶矩 top-r 方向构造偏移子空间。

    与 estimate_shift_subspace 的唯一区别是"什么时候问谁"：
      后者从**原始像素**的块间/类间异质性出发，训练前估一次即冻结；
      本函数从**当前模型的决策敏感度**出发，可随训练重估（动态偏移子空间）。
    两者都只用源域数据、都不需要方向先验、都不碰目标域。

    返回 V：与 sisa_V 同构 —— (r,B) 或 (n_cls,r,B)，**行正交归一**。
    ⚠️ 行正交归一 + σ/α 沿用冻结值 ⇒ E||delta||^2 = α^2 Σσ^2 与冻结臂完全相同，
       即"只换子空间取向、扰动预算逐位不变"（R3）。
    """
    was_training = model.training
    model.eval()
    Gc = {}
    seen = 0
    # [FIX 2026-10-08] n_max 必须在【梯度计算之前】生效。
    #   旧写法是"先跑完整个 batch 的前向+反向，再判断 seen>=n_max 才 break"
    #   ⇒ 若 loader 的一个 batch 就 >= n_max（HyRANK 留出集 3682 / eval_batch_size 4096
    #     恰好一个 batch），截断完全失效，每次都白算 3682 个样本的带梯度前反向（~24 s）。
    #   新写法：先按剩余预算把 batch 切到 <= (n_max - seen)，再做前反向。
    #   🔴 数值语义保持：旧行为取的是"前 n_max 个样本"（同一 batch 内的前若干行，
    #     因为原来也是 accumulate 后 break），新的切片取的同样是前 n_max 个，顺序一致。
    for xb, yb in x_loader:
        remain = n_max - seen
        if remain <= 0:
            break
        if xb.size(0) > remain:
            xb = xb[:remain]
            yb = yb[:remain]
        x = xb.to(device).detach().clone().requires_grad_(True)
        # [FIX 2026-10-08] 标签索引口径随 loader 类型而变：
        #   普通 DataLoader 的 HyperX 给 1-indexed 标签 ⇒ 需 -1；
        #   GPUResidentLoader 在构造时已经整体 -1 ⇒ 不能再减（否则变 -1 ⇒ CUDA assert）。
        #   🔴 这里必须显式判断、绝不能默认，否则静默错位或直接崩。
        _y = yb.to(device)
        if getattr(x_loader, 'labels', None) is None:
            _y = (_y - 1)
        y = _y.long()
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        g = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        g = g.mean(dim=(2, 3)).detach()             # (N,B) 偏移在 patch 内空间恒定
        for c in torch.unique(y).tolist():
            Gc.setdefault(int(c), []).append(g[y == c].double())
        seen += g.size(0)
        if seen >= n_max:
            break
    model.train(was_training)

    def _top_r(rows_list, r):
        if not rows_list:
            return None
        A = torch.cat(rows_list, 0)
        if A.size(0) < 2:
            return None
        w, Q = torch.linalg.eigh((A.T @ A) / A.size(0))
        idx = torch.argsort(w, descending=True)[:r]
        return Q[:, idx].T.contiguous().float().cpu().numpy()   # (r,B) 正交归一行

    rows_all = [t for v in Gc.values() for t in v]
    Vg = _top_r(rows_all, rank)
    if Vg is None:
        return None
    if n_classes is None:
        return Vg.astype(np.float32)
    n_cls = int(n_classes)
    V_out = np.tile(Vg[None], (n_cls, 1, 1))        # (n_cls,r,B) 默认全局, 与估计器 fallback 同构
    for c, rs in Gc.items():
        if 0 <= c < n_cls:
            Vc = _top_r(rs, rank)
            if Vc is not None:
                V_out[c] = Vc
    return V_out.astype(np.float32)


def subspace_drift(V_old, V_new):
    """[DSSP-D] 子空间取向漂移：把一个子空间的每行投影到另一个子空间，
    返回 (mean |cos|, 最大主角度/度, 投影后损失能量份额)。仅诊断用。"""
    try:
        A = np.asarray(V_old, dtype=np.float64).reshape(-1, V_old.shape[-1])
        B = np.asarray(V_new, dtype=np.float64).reshape(-1, V_new.shape[-1])
        Qa = np.linalg.qr(A.T)[0]
        Qb = np.linalg.qr(B.T)[0]
        s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
        s = np.clip(s, 0.0, 1.0)
        return float(s.mean()), float(np.degrees(np.arccos(s.min()))), float((s ** 2).sum() / len(s))
    except Exception:
        return float('nan'), float('nan'), float('nan')


def _sigma_self_energy(model, loader, device, V, sigma, n_max=2048):
    """[M3 2026-10-08] 逐方向"自然残余能量"：源域留出集特征沿每个 V 行的方差。

    返回 (r,) 的 np 数组。语义：模型在训练中途已经自发把多少能量放到了
    第 j 个偏移方向上 —— 多则该方向"已激活"，少则仍需补充。
    只在 --sisa_sigma_adaptive 开启时调用；异常由调用方兜住（fail-soft）。
    """
    _V = V.detach().cpu().numpy().astype(np.float64)
    if _V.ndim == 3:                      # classwise: 取各类的均值行（诊断口径）
        _V = _V.mean(0)
    _r = _V.shape[0]
    _acc = []
    _n = 0
    model.eval()
    with torch.no_grad():
        for _b in loader:
            # [FIX 2026-10-08] 与 model_sensitivity_subspace 同因：截断必须在【前向之前】。
            #   本函数只有前向（no_grad），代价远小于前者，但同属"先算后截"缺陷，一并修。
            _remain = n_max - _n
            if _remain <= 0:
                break
            _x = _b[0]
            if _x.size(0) > _remain:
                _x = _x[:_remain]
            _x = _x.to(device)
            _f = model(_x)
            if isinstance(_f, (tuple, list)):
                _f = _f[0]
            _f = _f.reshape(_f.size(0), -1).float().cpu().numpy().astype(np.float64)
            _f = _f - _f.mean(0, keepdims=True)
            _acc.append(_f)
            _n += _f.shape[0]
            if _n >= n_max:
                break
    model.train()
    if not _acc:
        return np.ones(_r, dtype=np.float64)
    _F = np.concatenate(_acc, 0)[:n_max]           # (n, d)
    _proj = _F @ _V.T                              # (n, r) 沿每个偏移方向的坐标
    return (_proj ** 2).mean(0)                    # (r,) 方差口径


def blend_subspaces(V_old, V_new, mix):
    """[DSSP-D] 在两个行正交归一子空间之间按 mix 插值（主角度意义下的加权混合）。

    mix=1 取 V_new，mix=0 取 V_old，中间取加权 Gram 的 top-r 主子空间。
    输出仍为**行正交归一** ⇒ 与冻结的 σ/α 组合后扰动预算逐位不变（R3），
    因此"换子空间取向"这件事不会顺带改变扰动幅度。
    """
    mix = float(mix)
    if mix >= 1.0 - 1e-12:
        return np.asarray(V_new, dtype=np.float32)
    if mix <= 1e-12:
        return np.asarray(V_old, dtype=np.float32)
    A = np.asarray(V_old, dtype=np.float64)
    B = np.asarray(V_new, dtype=np.float64)
    if A.shape != B.shape or A.ndim not in (2, 3):
        # 形状不一致时 fail-closed：退回新子空间（上层会打印告警）
        return np.asarray(V_new, dtype=np.float32)

    def _mix2d(a, b):
        G = (1.0 - mix) * (a.T @ a) + mix * (b.T @ b)
        w, Q = np.linalg.eigh(G)
        idx = np.argsort(w)[::-1][:a.shape[0]]
        return Q[:, idx].T

    if A.ndim == 2:
        return _mix2d(A, B).astype(np.float32)
    out = np.empty_like(A, dtype=np.float32)
    for i in range(A.shape[0]):
        out[i] = _mix2d(A[i], B[i]).astype(np.float32)
    return out


def fisher_criterion(z, labels, n_classes, eps=1e-4):
    """DSSP: discriminative Fisher scatter on the spectral projection z (N,k).

    Minimise within-class scatter and maximise between-class scatter -> keeps the
    class-discriminative spectral directions (the cross-domain discriminative info
    of §0) intact in the projection, complementary to SISA's shift suppression.
    Differentiable w.r.t. z (-> vca_proj weights). Returns Sw/(Sb+eps) (smaller=better).
    """
    mu = z.mean(0, keepdim=True)                 # (1, k)
    Sw = torch.zeros((), device=z.device, dtype=z.dtype)
    Sb = torch.zeros((), device=z.device, dtype=z.dtype)
    for c in range(n_classes):
        mask = (labels == c)
        nc = int(mask.sum())
        if nc < 2:
            continue
        zc = z[mask]
        muc = zc.mean(0, keepdim=True)
        Sw = Sw + ((zc - muc) ** 2).sum()
        Sb = Sb + nc * ((muc - mu) ** 2).sum()
    return Sw / (Sb + eps)



def jacobian_shift_reg(forward_fn, x, V_shift, sigma_shift, eps_fd=1e-3):
    """tr(J^T J Sigma_S) = sum_k sigma_k^2 ||J v_k||^2.

    forward_fn: callable x -> features (N, d)
    V_shift: (r, B) right singular vectors of the shift subspace
    sigma_shift: (r,) singular values (shift magnitudes)
    eps_fd: finite-difference step for the directional derivative

    Uses central differences (2 forward passes per direction) to avoid
    double backprop overhead. The shift is spatially constant (a domain
    shift is a per-scene spectral offset), so v_k is broadcast over H, W.
    """
    r = V_shift.shape[0]
    B = V_shift.shape[1]
    loss_inv = x.new_zeros(())
    for k in range(r):
        v_k = V_shift[k].view(1, B, 1, 1).expand_as(x)
        h_plus = forward_fn(x + eps_fd * v_k)
        h_minus = forward_fn(x - eps_fd * v_k)
        jvp_k = (h_plus - h_minus) / (2 * eps_fd)
        loss_inv = loss_inv + sigma_shift[k].pow(2) * jvp_k.pow(2).sum(dim=1).mean()
    return loss_inv


def jacobian_isotropic_reg(forward_fn, x, n_dirs=5, eps_fd=1e-3):
    """Shui-style isotropic ‖J‖_F² estimated via random projections.

    Draws n_dirs random directions and averages ||J v||^2 to approximate
    the full Frobenius norm. Same cost as jacobian_shift_reg.
    """
    B = x.shape[1]
    loss_iso = x.new_zeros(())
    for _ in range(n_dirs):
        v = torch.randn(B, device=x.device)
        v = v / (v.norm() + 1e-12)
        v_4d = v.view(1, B, 1, 1).expand_as(x)
        h_plus = forward_fn(x + eps_fd * v_4d)
        h_minus = forward_fn(x - eps_fd * v_4d)
        jvp = (h_plus - h_minus) / (2 * eps_fd)
        loss_iso = loss_iso + jvp.pow(2).sum(dim=1).mean()
    return loss_iso * B / n_dirs



def coral_loss(feat_src, feat_tgt):
    """CORAL: align second-order statistics (covariance) of two feature sets.

    feat_src: (N, d) source domain features
    feat_tgt: (M, d) target domain features (or shift-augmented proxy)
    """
    d = feat_src.size(1)
    mean_s = feat_src.mean(0, keepdim=True)
    mean_t = feat_tgt.mean(0, keepdim=True)
    cs = feat_src - mean_s
    ct = feat_tgt - mean_t
    cov_s = (cs.t() @ cs) / max(1, feat_src.size(0) - 1)
    cov_t = (ct.t() @ ct) / max(1, feat_tgt.size(0) - 1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d)


class GRL(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial domain adaptation."""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    """Simple domain discriminator for DANN."""
    def __init__(self, feat_dim=512, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)


def supcon_shift_loss(z1, z2, labels, temperature=0.1, max_n=1024):
    """SDC (Shift-Discriminative Consistency): supervised contrastive on shift views.

    z1 = features of the ORIGINAL view (N, d)
    z2 = features of the SHIFT-augmented view (N, d)
    The two views differ ONLY by a perturbation inside the estimated domain-shift
    subspace S.

    Positives = same class (across both views)  -> forces the representation to be
        INVARIANT along the shift directions while KEEPING those directions.
    Negatives = other classes                   -> builds/keeps DISCRIMINATIVE structure.

    Unlike domain-invariant / alignment methods (DANN, CORAL, IRM) that REMOVE
    shift-correlated directions and thereby throw away whatever discriminative
    signal lives in them, we retain S and only enforce invariance inside it.
    """
    if z1.size(0) > max_n:
        idx = torch.randperm(z1.size(0)).to(z1.device)[:max_n]
        z1 = z1[idx]
        z2 = z2[idx]
        labels = labels[idx]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)                    # (2N, d)
    N2 = z.size(0)
    y = torch.cat([labels, labels], dim=0)            # (2N,)
    sim = (z @ z.t()) / temperature                   # (2N, 2N)
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    eye = torch.eye(N2, device=z.device, dtype=torch.bool)
    neg = (~eye).float()
    exp_sim = torch.exp(sim) * neg
    log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    same = (y.unsqueeze(0) == y.unsqueeze(1)).float()
    pos = same * neg
    n_pos = pos.sum(dim=1)
    log_prob = sim - log_denom
    loss = -(pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
    keep = n_pos > 0
    if int(keep.sum()) == 0:
        return z1.sum() * 0.0
    return loss[keep].mean()


# ============================================================
# CI: Multi-head classifier wrapper
# ============================================================
class CIModel(nn.Module):
    """Wrapper: backbone shared, K independent Linear(512→7) classifier heads."""

    def __init__(self, base_model, K, n_classes, seed):
        super().__init__()
        self.base = base_model
        self.K = K
        self.classifiers = nn.ModuleList([
            nn.Linear(512, n_classes, bias=False) for _ in range(K)
        ])
        # Different Kaiming init per head (seeded by seed + k)
        for k, cls_head in enumerate(self.classifiers):
            torch.manual_seed(seed + 1000 + k)
            nn.init.kaiming_normal_(cls_head.weight, a=0, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        feat = self.base.forward_backbone(x)                 # (N, 512)
        logits = torch.stack([h(feat) for h in self.classifiers], dim=0)  # (K, N, 7)
        features = self.base.fc_embed(feat)                  # (N, 128)
        return logits, features

    def get_classifier_params(self):
        """Return classifier head parameters (for optimizer, GBTC, MTTC)."""
        params = []
        for h in self.classifiers:
            params.extend(list(h.parameters()))
        return params

    def get_backbone_params(self):
        """Return backbone parameters (exclude classifier heads)."""
        return [p for n, p in self.base.named_parameters()
                if 'classifier' not in n and 'fc_embed' not in n]


# ============================================================
# 单种子训练
# ============================================================


class GPUResidentLoader:
    """Drop-in replacement for DataLoader: yields preloaded GPU tensors.
    Zero IPC overhead. Use when model << GPU and DataLoader is the bottleneck."""
    def __init__(self, patches, labels, batch_size, shuffle=True, block_ids=None):
        self.patches = patches   # (N, C, H, W) on GPU
        self.labels = labels     # (N,) raw class ids (1-based) on GPU
        self.bs = batch_size
        self.N = len(labels)
        self.shuffle = shuffle
        self.block_ids = block_ids  # (N,) on GPU, 可选(异质偏移用)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.N).to(self.patches.device)
        else:
            perm = torch.arange(self.N, device=self.patches.device)
        for b in range(0, self.N, self.bs):
            idx = perm[b:b+self.bs]
            if self.block_ids is not None:
                yield self.patches[idx], self.labels[idx], self.block_ids[idx]
            else:
                yield self.patches[idx], self.labels[idx]


def train_one_seed(args, seed):
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[seed={seed}] device={device}")

    # ---- 精度探针初始化（2026-09-14 用户授权，默认【完全不碰】后端开关）----
    # 🔴 本机实测事实（torch 2.13.0+cu132 / RTX 5090D sm_120）：
    #      torch.backends.cuda.matmul.allow_tf32 = False   <- 框架默认（matmul 走纯 FP32）
    #      torch.backends.cudnn.allow_tf32       = True    <- 框架默认（★ 卷积走 TF32！）
    #   ⇒ 历史 150 runs 的所谓 "FP32 基线"，其实是【cuDNN 卷积 TF32 + matmul FP32】的混合精度。
    #   ⇒ 故默认路径【一个开关都不改】，以保证与历史 runs 逐位可比。
    #      --tf32        : 额外打开 matmul 的 TF32（cudnn 本来就是 on）→ 真正的"全 TF32"
    #      --strict_fp32 : 两边都关 → 真正的纯 FP32（会明显变慢，仅用于诊断）
    _tf32 = bool(getattr(args, 'tf32', False)) and device.type == 'cuda'
    _strict_fp32 = bool(getattr(args, 'strict_fp32', False)) and device.type == 'cuda'
    if _tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif _strict_fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    # else: 保持框架默认（cudnn tf32=on, matmul tf32=off）—— 与历史 runs 完全一致

    _amp = bool(getattr(args, 'amp', False))
    _amp_full = bool(getattr(args, 'amp_full', False))
    _amp_eval = bool(getattr(args, 'amp_eval', False))
    _prec_on = (_amp or _amp_full) and device.type == 'cuda'
    _amp_eval = _amp_eval and device.type == 'cuda'

    _amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if _prec_on else None
    _amp_eval_ctx = (torch.autocast(device_type='cuda', dtype=torch.float16)
                     if (_amp_eval or _prec_on) else None)
    _scaler = torch.amp.GradScaler('cuda', enabled=_prec_on)

    print(f"[precision] matmul_tf32={torch.backends.cuda.matmul.allow_tf32} "
          f"cudnn_tf32={torch.backends.cudnn.allow_tf32} "
          f"autocast={'fp16' if _prec_on else 'off'} scaler={'on' if _prec_on else 'off'} "
          f"amp_eval={_amp_eval} strict_fp32={_strict_fp32}")

    # 互斥保护：GD / HGR 直接改写 p.grad，与 GradScaler 的梯度缩放冲突。
    if _prec_on:
        if getattr(args, 'gd_alpha', 0) > 0:
            raise SystemExit('[precision] --amp 与 --gd_alpha(GD) 不兼容：GD 直接改写 p.grad，'
                             '会与 GradScaler 的梯度缩放冲突。请勿同时开启。')

    # ---- 加载数据 ----
    img_src, gt_src, label_values, _, ignored_labels, _, _ = get_dataset(
        args.source, target_folder=args.data_path)

    # Subsample: randomly select N valid pixels from source
    if args.subsample is not None:
        rng = np.random.RandomState(seed)
        valid_mask = gt_src > 0
        valid_idx = np.where(valid_mask.reshape(-1))[0]
        N = min(args.subsample, len(valid_idx))
        chosen = rng.choice(valid_idx, N, replace=False)
        sub_mask = np.zeros_like(gt_src.reshape(-1), dtype=bool)
        sub_mask[chosen] = True
        gt_src = np.where(sub_mask.reshape(gt_src.shape), gt_src, 0)
        print(f"[seed={seed}] Subsample: {N} / {len(valid_idx)} pixels")
    img_tgt, gt_tgt, _, _, _, _, _ = get_dataset(
        args.target, target_folder=args.data_path)

    n_bands = img_src.shape[2]
    n_classes = len(label_values)

    print(f"[seed={seed}] {args.source}→{args.target}: bands={n_bands}, classes={n_classes}"
          f", src_pixels={np.count_nonzero(gt_src)}, tgt_pixels={np.count_nonzero(gt_tgt)}")

    # ---- 源域采样 ----
    # 注意: sample_gt 返回 (train_gt, test_gt, train_set, test_set); 第二个返回值
    # 就是 20% 分层留出集(random_state=23 固定), 用作【源域验证集】来做超参选择,
    # 完全不碰目标域 -> 保证 target-oracle 协议的公平性(与 LDGnet 同款选点)。
    train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, 0.8, mode='random')
    test_gt_tgt, _, _, _ = sample_gt(gt_tgt, 1, mode='random')

    # ---- 构建模型 ----
    res_channels = tuple(map(int, args.res_channels.split(',')))

    vca_mode = 'Kaiming' if args.no_vca else args.proj_init
    # Set pca_k: for mix it sets module attribute; for pca proj it overrides --vca_k
    k_for_proj = args.vca_k
    if args.proj_init == 'mix' and args.pca_k is not None:
        import vca_dres
        vca_dres._prepare_projection.pca_k = args.pca_k
    if args.unified_r0 is not None:
        args.proj_init = 'Unified'
        import vca_dres
        vca_dres._prepare_projection.unified_r0 = args.unified_r0
        args.proj_frozen = True  # Unified forces frozen
    if args.proj_init in ('pca', 'PCA') and args.pca_k is not None:
        k_for_proj = args.pca_k
    if args.proj_init == 'SDP':
        import vca_dres
        vca_dres._prepare_projection.sdp_tau = args.sdp_tau
        vca_dres._prepare_projection.sdp_gamma = args.sdp_gamma
        vca_dres._prepare_projection.sdp_blocks = args.sdp_blocks
        vca_dres._prepare_projection.sdp_delta = args.sdp_delta
        vca_dres._prepare_projection.sdp_ortho = (not args.sdp_no_ortho)
        if args.pca_k is not None:
            k_for_proj = args.pca_k
    # Parse aug_param
    aug_param = args.aug_param
    if aug_param is not None:
        try:
            if ',' in aug_param:
                pass  # keep as string for scale tuples
            elif '.' in aug_param:
                aug_param = float(aug_param)
            else:
                aug_param = int(aug_param)
        except (ValueError, TypeError):
            pass

    base_model, k_endmembers = build_einit_cnn(
        img_src, gt_src, n_bands, n_classes, seed=seed, k_endmembers=k_for_proj, res_channels=res_channels,
                     no_vca=args.no_vca, proj_init=args.proj_init, proj_frozen=args.proj_frozen,
                     aug_type=args.aug_type, aug_param=aug_param,
                     proj_ortho_lambda=args.proj_ortho_lambda)

    ci_mode = args.ci_heads > 1
    if ci_mode:
        model = CIModel(base_model, args.ci_heads, n_classes, seed)
        mode_tag = f'CI-K{args.ci_heads}'
    else:
        model = base_model
        if args.no_vca:
            mode_tag = 'Kaiming'
        elif args.proj_init == 'centroid':
            mode_tag = 'CCP'
        elif args.proj_init == 'VCA':
            mode_tag = 'M1'
        else:
            mode_tag = args.proj_init

    model = model.to(device)

    # === [极限探针] torch.compile：融合内核、削减 Python/launch 开销（FP32，默认关） ===
    if getattr(args, 'torch_compile', False):
        try:
            _cmode = getattr(args, 'compile_mode', 'default') or 'default'
            model = torch.compile(model, mode=_cmode, dynamic=False)
            print(f"[seed={seed}] torch.compile ENABLED (FP32, mode={_cmode})")
        except Exception as e:
            print(f"[seed={seed}] torch.compile FAILED ({type(e).__name__}: {e}); 退回普通模型")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    init_tag = vca_mode
    frozen_tag = 'frz' if args.proj_frozen else 'lrn'
    print(f"[seed={seed}] {mode_tag}: {init_tag}_{frozen_tag}_Res2DCNN k={k_endmembers}, params={n_params / 1e6:.3f}M")

    # ---- 超参 ----
    hyperparams = {
        'patch_size': args.patch_size,
        'ignored_labels': ignored_labels,
        'flip_augmentation': True,
        'radiation_augmentation': True,
        'mixture_augmentation': False,
        'center_pixel': True,
        'supervision': 'full',
    }

    # ---- re_ratio: 源域数据重复 ----
    r = args.patch_size // 2 + 1
    img_src_pad = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    train_gt_pad = np.pad(train_gt_src, ((r, r), (r, r)), 'constant', constant_values=0)
    img_src_tiled = np.concatenate([img_src_pad] * args.re_ratio, axis=0)
    train_gt_tiled = np.concatenate([train_gt_pad] * args.re_ratio, axis=0)

    # ---- DataLoader ----
    # Eval passes are per-sample independent (BN running stats + no dropout), so a
    # larger eval batch is a pure throughput win with bit-identical predictions.
    ev_bs = getattr(args, 'eval_batch_size', None) or args.batch_size
    train_dataset = HyperX(img_src_tiled, train_gt_tiled, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        pin_memory=True, num_workers=args.num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, seed))

    test_dataset = HyperX(img_tgt, test_gt_tgt, **hyperparams)
    test_loader = data.DataLoader(
        test_dataset, batch_size=ev_bs, shuffle=False,
        pin_memory=True, num_workers=args.num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, seed))

    # 源域留出验证集 (选参用, 只用源域, 不碰目标域)
    val_dataset = HyperX(img_src, val_gt_src, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset, batch_size=ev_bs, shuffle=False,
        pin_memory=True, num_workers=args.num_workers,
        worker_init_fn=lambda wid: worker_init_fn(wid, seed))

    # ---- 优化器 ----

    # sisa_hetero 必须在 GPU-resident 块之前定义, 否则 526 行 `if sisa_hetero` 触发 UnboundLocalError
    sisa_hetero = getattr(args, 'sisa_hetero', False)
    sisa_classwise = getattr(args, 'sisa_classwise', False)

    # --- GPU-resident: preload patches, replace loaders ---
    if args.gpu_resident:
        print(f"[seed={seed}] GPU-resident: preloading patches to GPU...")
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        tp = torch.from_numpy(np.asarray(train_dataset.cached_patches)).to(device)
        tl = torch.tensor([l - 1 for l in train_dataset.labels], dtype=torch.long, device=device)
        vp = torch.from_numpy(np.asarray(test_dataset.cached_patches)).to(device)
        vl = torch.tensor([l - 1 for l in test_dataset.labels], dtype=torch.long, device=device)
        _blk = None
        if sisa_hetero:
            # 每个 patch 的空间块 id (与 estimate_shift_subspace 同一划分公式)
            _bh = max(1, img_src.shape[0] // args.sisa_blocks)
            _bw = max(1, img_src.shape[1] // args.sisa_blocks)
            _ncol = int(np.ceil(img_src.shape[1] / _bw))
            _Hpad = img_src.shape[0] + 2 * r
            _bids = []
            for (x, y) in train_dataset.indices:
                xo = int(np.clip((x % _Hpad) - r, 0, img_src.shape[0] - 1))
                yo = int(np.clip(y - r, 0, img_src.shape[1] - 1))
                _bids.append((xo // _bh) * _ncol + (yo // _bw))
            _blk = torch.tensor(_bids, dtype=torch.long, device=device)
        train_loader = GPUResidentLoader(tp, tl, args.batch_size, shuffle=True, block_ids=_blk)
        test_loader = GPUResidentLoader(vp, vl, ev_bs, shuffle=False)
        # [FIX 2026-10-08] val_loader 也必须 GPU 化 —— 这是 `--dssd_period` 24 s/次 的【真根因】。
        #   上面只替换了 train/test，`val_loader` 是全流程里唯一还在走 CPU DataLoader 的
        #   （num_workers=0 ⇒ 逐 batch 现场调 HyperX.__getitem__ 切 patch）。
        #   DSSP-D 每 period 个 epoch 就完整迭代一次 val_loader ⇒ 每次 ~24 s（实测）。
        #   实测对照：同样的 3682 个 patch，模型前反向仅 0.018 s、H2D 仅 0.041 s
        #   ⇒ 24 s 里 99.7% 是 CPU 侧的 patch 切片/组装开销。
        #   ⚠️ 只影响 DSSP-D / M3 这类"迭代留出集"的路径；
        #      普通评估走的是 test_loader（本来就是 GPUResident），故历史结果不受影响。
        vvp = torch.from_numpy(np.asarray(val_dataset.cached_patches)).to(device)
        vvl = torch.tensor([l - 1 for l in val_dataset.labels], dtype=torch.long, device=device)
        val_loader = GPUResidentLoader(vvp, vvl, ev_bs, shuffle=False)
        del tp, tl, vp, vl, vvp, vvl  # keep loaders holding refs
        print(f"[seed={seed}] GPU-resident ready: train={len(train_loader.labels)}, "
              f"test={len(test_loader.labels)}, val={len(val_loader.labels)}")

    # ---- SISA: estimate the cross-domain shift subspace from SOURCE data only ----
    sisa_V = sisa_sigma = None
    xg_mu = None
    sisa_sigma_global = None      # [DIAG] --sisa_global_delta 用: 全局 sigma（与 sisa_V 同一次估计产生）
    # 🔴 LR-DRO 也需要 sisa_V: 若只看 sisa/sdc lambda, A2 臂(LR-DRO 单独, sisa=sdc=0)
    #    会让 sisa_V 保持 None => LR-DRO 分支静默跳过且不报错。必须一并纳入。
    _need_subspace = (args.sisa_lambda > 0 or args.remove_shift
                      or args.sdc_lambda > 0
                      or getattr(args, 'lrdro_lambda', 0.0) > 0
                      # M1b 特征侧同样需要光谱子空间(推送 JV 的原料), 漏了会静默跳过
                      or getattr(args, 'lrdro_feat_lambda', 0.0) > 0)
    if _need_subspace:
        _dev = next(model.parameters()).device
        # ---- S4/C1 (2026-09-19): 外部注入偏移子空间（跨城可移植性实验）----
        # 目的：用【另一座城】估出的 S 训本城模型，检验 S 是场景无关量还是场景属性。
        # 只替换 _V/_sig 这一对，后续 _dir_mode / calibrate / 张量化全部照旧。
        # 缺省 (None) 时行为与历史逐位一致。
        _ext = getattr(args, 'sisa_shift_pkl', None)
        if _ext:
            _ext = os.path.abspath(_ext)
            if not os.path.isfile(_ext):
                raise SystemExit(f"[S4] --sisa_shift_pkl 不存在: {_ext}")
            try:
                _blob = torch.load(_ext, map_location='cpu', weights_only=False)
            except Exception as _e:
                raise SystemExit(f"[S4] 载入失败 {_ext}: {type(_e).__name__}: {_e}")
            for _k in ('V', 'sigma'):
                if _k not in _blob:
                    raise SystemExit(f"[S4] pkl 缺键 '{_k}': {_ext} (keys={list(_blob.keys())})")
            _V = np.asarray(_blob['V'], dtype=np.float32)
            _sig = np.asarray(_blob['sigma'], dtype=np.float32)
            # ---- fail-closed 校验 ----
            # [D-18b 2026-09-23] 原实现只接受 2-D V，并因此把 --sisa_classwise 一律拒掉。
            #   但 classwise 的 _V 是 3-D (n_cls, r, B)（见 L1271 逐类 QR 分支），
            #   ⇒ 「2-D 注入 + classwise」确实是形状/语义不匹配（拒得对）。
            #   现在**显式支持 3-D 逐类注入**：允许 ndim==3 **且** classwise 同时开启，
            #   逐类正交性逐类校验。hetero 的 3-D 语义不同（块 id，见 L1602），仍一律拒绝。
            _is_cls3d = (_V.ndim == 3)
            if _V.ndim not in (2, 3):
                raise SystemExit(f"[S4] V 必须是 2-D (r,B) 或 3-D (n_cls,r,B)，实得 shape={_V.shape}")
            if _is_cls3d:
                if sisa_hetero:
                    raise SystemExit("[S4] --sisa_shift_pkl(3-D) 与 --sisa_hetero 互斥（块级 3-D 语义不同）")
                if not sisa_classwise:
                    raise SystemExit(
                        f"[S4] V 是 3-D {_V.shape} ⇒ 必须同时带 --sisa_classwise（逐类载体）")
                if _V.shape[0] != n_classes:
                    raise SystemExit(
                        f"[S4] V.shape[0]={_V.shape[0]} 与数据集类别数 n_classes={n_classes} 不一致")
                if _V.shape[2] != img_src.shape[2]:
                    raise SystemExit(
                        f"[S4] 波段不匹配: V.shape={_V.shape} 但 img_src 有 "
                        f"{img_src.shape[2]} 波段 (跨传感器 S 无定义，禁做)")
                if _sig.ndim != 2 or _sig.shape != (_V.shape[0], _V.shape[1]):
                    raise SystemExit(
                        f"[S4] classwise 要求 sigma 为 2-D (n_cls,r)={( _V.shape[0], _V.shape[1])}，"
                        f"实得 shape={_sig.shape}")
                _Vn = _V.astype(np.float64)
                _eye = np.eye(_V.shape[1])
                _orth_err = float(max(np.abs(_Vn[_ci] @ _Vn[_ci].T - _eye).max()
                                      for _ci in range(_Vn.shape[0])))
                # 🔴 硬闸 = **行范数必须为 1**（σ 的语义载体：σ_j 是「沿第 j 个单位方向的幅度」）。
                _nrm_err = float(max(abs(np.linalg.norm(_Vn[_ci], axis=1) - 1.0).max()
                                     for _ci in range(_Vn.shape[0])))
                if _nrm_err > 1e-4:
                    raise SystemExit(f"[S4] 逐类行范数不为 1（max_err={_nrm_err:.3e}）")
                # 正交性缺陷**报告但不拦**：生产 classwise 估计器在「某类可用块 < rank」时
                #   会把 SVD 的前 rc 行与全局 V 的旧行混在一类里 ⇒ ||VVᵀ−I||∞ ≠ 0
                #   （实测 Houston c1 = 0.5980）。若此处一律拒绝，生产核自身都无法注入。
                if _orth_err > 1e-4:
                    _bad = [i + 1 for i in range(_Vn.shape[0])
                            if float(np.abs(_Vn[i] @ _Vn[i].T - _eye).max()) > 1e-4]
                    print(f"[seed={seed}] [S4] ⚠️ 逐类行不正交 max_err={_orth_err:.4f}，"
                          f"缺陷类={_bad}（复刻生产 classwise 行为，非注入错误）")
                # 🔴 注入与「方向替换」不能同时用：sdc_dir ∈ {random, noclass, disc,
                #     smooth, pca, cap, ms1, intra, orth}
                #    会在下面 L1299–1344 把 _V 整个换掉 ⇒ 注入被静默丢弃（跑出来的是对照，不是换核）。
                #    只有 'shift'（默认）是「不替换方向」⇒ 注入才有效。此处 fail-closed。
                if str(getattr(args, 'sdc_dir', 'shift')) != 'shift':
                    raise SystemExit(
                        f"[S4] 3-D 注入要求 --sdc_dir shift（不替换方向）；实得 {args.sdc_dir} "
                        "⇒ 注入会被方向替换覆盖，禁止")
            else:
                if _sig.ndim != 1:
                    raise SystemExit(f"[S4] sigma 必须是 1-D (r,)，实得 shape={_sig.shape}")
                if _V.shape[1] != img_src.shape[2]:
                    raise SystemExit(
                        f"[S4] 波段不匹配: V.shape={_V.shape} 但 img_src 有 "
                        f"{img_src.shape[2]} 波段 (跨传感器 S 无定义，禁做)")
                if _sig.shape[0] != _V.shape[0]:
                    raise SystemExit(f"[S4] sigma.shape={_sig.shape} 与 V.shape={_V.shape} 不一致")
                _Vn = _V.astype(np.float64)
                _G = _Vn @ _Vn.T
                _orth_err = float(np.abs(_G - np.eye(_Vn.shape[0])).max())
                if _orth_err > 1e-4:
                    raise SystemExit(f"[S4] V 行不正交: ||VV^T - I||_inf = {_orth_err:.3e} > 1e-4")
                if sisa_hetero or sisa_classwise:
                    raise SystemExit("[S4] --sisa_shift_pkl(2-D) 与 --sisa_hetero/--sisa_classwise 互斥"
                                     "（classwise 需要 3-D 逐类 V；hetero 与外部注入不兼容）")
            if _is_cls3d:
                # 便于事后从日志一眼核对：走的是 classwise 3-D 通道、方向模式为 random
                print(f"[seed={seed}] [S4] channel=classwise(3-D) n_cls={_V.shape[0]} "
                      f"rank={_V.shape[1]} mode={getattr(args, 'sdc_dir', 'shift')}")
            _src_tag = _blob.get('source', '?')
            print(f"[seed={seed}] [S4] EXTERNAL SHIFT SUBSPACE: {os.path.basename(_ext)} "
                  f"(donor={_src_tag}, V={_V.shape}, sigma={np.array2string(_sig, precision=6)}, "
                  f"orth_err={_orth_err:.2e})")
        else:
            _V, _sig = estimate_shift_subspace(img_src, train_gt_src,
                                               n_blocks=args.sisa_blocks, rank=args.sisa_rank,
                                               min_px=args.sisa_min_px,
                                               hetero=sisa_hetero,
                                               classwise=sisa_classwise)
            # [STRETCH] 重塑度量形状（gamma=1 恒等，不影响任何历史读数）
            _g = float(getattr(args, 'sisa_sigma_gamma', 1.0))
            if abs(_g - 1.0) >= 1e-12:
                _before = np.asarray(_sig, dtype=np.float64)
                _sig = reshape_sigma_gamma(_sig, _g)
                _after = np.asarray(_sig, dtype=np.float64)
                _b2 = float((_before ** 2).sum())
                _a2 = float((_after ** 2).sum())
                print(f"[seed={seed}] [STRETCH] sigma_gamma={_g}: "
                      f"sum(sigma^2) {_b2:.6e} -> {_a2:.6e} (预算保持, rel diff "
                      f"{abs(_a2 - _b2) / (_b2 + 1e-30):.2e})")
        # [DIAG] 另取一次【全局】估计，仅用于 --sisa_global_delta 的幅度来源。
        # 只在显式开启时才多跑（避免影响历史读数与耗时）。
        if getattr(args, 'sisa_global_delta', False):
            _Vg, _sigg = estimate_shift_subspace(img_src, train_gt_src,
                                                 n_blocks=args.sisa_blocks, rank=args.sisa_rank,
                                                 min_px=args.sisa_min_px,
                                                 hetero=False, classwise=False)
            # 🔴 [2026-09-19 修复] 形状契约: 全局估计给出 sigma (r,) —— 但 sisa_augment 在
            # V 为 3-D 时会走 `sigma[block_ids]` 分支 (block_ids=label_batch, 取值 0..n_cls-1)。
            # 直接把 (r,) 传进去 -> `index out of bounds for dimension 0 with size r`
            # (r=5 而 label 最大 13/14)，即 pool 里 5 个 run 全部 rc=1 的真因。
            # 逐类分支要求 sigma 与 V_cls 行对齐 => 必须 tile 成 (n_cls, r)。
            _n_cls_g = int(_V_cls_ncls if '_V_cls_ncls' in dir() else 0)
            if _n_cls_g <= 0:
                _ys_g = np.asarray(train_gt_src).reshape(-1)
                _allc_g = [int(c) for c in np.unique(_ys_g) if c > 0]
                _n_cls_g = max(_allc_g) if _allc_g else 0
            sisa_sigma_global = torch.as_tensor(
                np.tile(np.asarray(_sigg, dtype=np.float32)[None], (_n_cls_g, 1)),
                dtype=torch.float32, device=_dev)
            print(f"[DIAG] --sisa_global_delta ON: 逐类子空间 + 全局 sigma（解耦诊断） "
                  f"sigma_global.shape={tuple(sisa_sigma_global.shape)} "
                  f"(tiled from ({_sigg.shape[0]},) to align with V_cls rows)")
        if not sisa_hetero:
            _dir_mode = str(getattr(args, 'sdc_dir', 'random'))
            if _dir_mode == 'random':
                # A3 control: same rank & magnitude, RANDOM orthonormal directions.
                # QR of a (B, r) gaussian -> Q is (B, r) with orthonormal COLUMNS;
                # transpose to (r, B) so rows (the shift directions) stay orthonormal.
                if sisa_classwise:
                    # classwise: _V is (n_cls, r, B). Randomized the DIRECTIONS per class
                    # while keeping the per-class sigma (= "same magnitude, only axis swapped").
                    # (fix 2026-09-14: previously this branch collapsed the 3-D classwise _V
                    #  into a 2-D global random matrix -> IndexError at the classwise print
                    #  and a shape mismatch inside sisa_augment.)
                    _n_cls, _r, _B = _V.shape
                    _V_rand = np.empty_like(_V)
                    for _ci in range(_n_cls):
                        _G = torch.randn(_B, _r)
                        _Q, _ = torch.linalg.qr(_G)
                        _V_rand[_ci] = _Q.T.numpy().astype(np.float32)
                    _V = _V_rand
                else:
                    _G = torch.randn(_V.shape[1], _V.shape[0])
                    _Q, _ = torch.linalg.qr(_G)
                    _V = _Q.T.numpy().astype(np.float32)
            elif _dir_mode == 'noclass':
                # S2 (2026-09-15): 候选**新轴**，不是对照。
                #   V_new = orth(S) 去掉【类判别成分】（V_d = 源域类均值行空间）。
                #   保持秩 / 形状不变；sigma 由下面原样沿用 ⇒ 同 α = 同扰动预算（R3）。
                #   理论：收益 = 覆盖 − 损伤，若损伤来自"类判别被抹掉"，则该轴应更优。
                #   ⚠️ 必须保持 --sisa_calibrate=False，否则 σ 被重标定、逐峰比较不公平。
                #   --noclass_k=<int>（S2 剂量-反应，2026-09-15 补）：只投影掉 V_d 的前 k 维；
                #   缺省 None = 全量移除（＝原始行为）。
                from noclass_axis import make_noclass_axis as _mk_noclass
                _nk = getattr(args, 'noclass_k', None)
                _V, _ninfo = _mk_noclass(_V, img_src, train_gt_src, seed=int(seed) + 9973, k=_nk)
                print(f"[seed={seed}] SDC dir CANDIDATE AXIS: noclass "
                      f"(class-discriminative component projected out; V_d dim={_ninfo['dim_Vd']}, "
                      f"k_used={_ninfo['k_used']}, {_ninfo['note']}, "
                      f"rank_after={_ninfo['rank_after']}) "
                      f"shape={np.asarray(_V).shape}")
            elif _dir_mode == 'disc':
                # E1（2026-09-28）：noclass 的**对偶**候选新轴，不是对照。
                #   V_new = orth(把 S 的每一行投影到 V_d 上后剩下的部分)
                #         = (V · Vdᵀ) · Vd，即**只留**类判别成分。
                #   动机：生产的「增广方向阶梯」已有「去判别」的两个末端
                #   （noclass 46 run / intra 20 run），唯独「只留判别」这一端没有。
                #   C-62 的零 GPU `disc` 臂判为**最有害**（−0.2508 / −0.6490 / −0.2350）
                #   ⇒ 本支是它的**生产版**。
                #   保持秩 / 形状不变；sigma 由下面原样沿用 ⇒ 同 α = 同扰动预算（R3）。
                #   ⚠️ 必须保持 --sisa_calibrate=False，否则 σ 被重标定、逐峰比较不公平。
                from disc_axis import make_disc_axis as _mk_disc
                _dk = getattr(args, 'disc_k', None)
                _V, _dinfo = _mk_disc(_V, img_src, train_gt_src, seed=int(seed) + 9973, k=_dk)
                print(f"[seed={seed}] SDC dir CANDIDATE AXIS: disc "
                      f"(ONLY class-discriminative component kept; V_d dim={_dinfo['dim_Vd']}, "
                      f"k_used={_dinfo['k_used']}, {_dinfo['note']}, "
                      f"purity_before={np.array2string(np.array(_dinfo['purity_before']), precision=4)}, "
                      f"rank_after={_dinfo['rank_after']}) "
                      f"shape={np.asarray(_V).shape}")
            elif _dir_mode in ('smooth', 'pca', 'cap', 'ms1', 'intra', 'orth'):
                # A5 structured controls (2026-09-14): the white-noise `random` axis is a
                # straw-man baseline (orthogonal to any smooth spectral direction). These
                # replace the axis with a *realistic* but WRONG subspace while keeping the
                # rank / per-direction sigma / tensor shape identical to the estimate.
                _V = make_control_directions(_dir_mode, _V, img_src, train_gt_src,
                                             seed=int(seed) + 9973)
                print(f"[seed={seed}] SDC dir control: {_dir_mode} "
                      f"(rank & sigma kept, axis replaced) shape={np.asarray(_V).shape}")
            # ---- DSSP: calibrate the perturbation magnitude (fixes Pavia collapse) ----
            if args.sisa_calibrate:
                Xf = img_src.reshape(-1, img_src.shape[2]).astype(np.float64)
                proj = Xf @ _V.T                                   # (N, r) signal along each shift dir
                sigstd = proj.std(0)                              # (r,) signal scale per dir
                rel = _sig / (sigstd + 1e-12)                     # sigma/|x| per dir
                scale = np.clip(np.array(args.sisa_target_rel, dtype=np.float64) / (rel + 1e-12), 0.0, 1.0)
                _sig = (_sig * scale).astype(np.float32)
                print(f"[seed={seed}] SISA calibrated: target_rel={args.sisa_target_rel}, "
                      f"scale={np.array2string(scale, precision=3)}")
        # ---- [STRETCH-D1] oracle 真实方向 e_true 作 Λ 形状（预算匹配, α 不变）----
        if getattr(args, 'e_true_as_sigma', False):
            _shape_file = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'tools', '_e1_stretch_shape.json')
            if not os.path.isfile(_shape_file):
                raise SystemExit("[E1-D1] shape file missing: %s" % _shape_file)
            _sh = json.load(open(_shape_file, encoding='utf-8'))
            _sk = args.source.lower()
            _tag = ('H' if 'houston' in _sk else
                    'Y' if ('loukia' in _sk or 'dioni' in _sk) else
                    'P' if 'pavia' in _sk else '?')
            if _tag == '?':
                raise SystemExit("[E1-D1] cannot map source=%s to H/Y/P" % args.source)
            _et = np.asarray(_sh['%s_s333' % _tag]['e_true'], dtype=np.float64)
            _et = _et / (_et.sum() + 1e-30)              # 已为概率分布(Σ=1)
            _sig = np.asarray(_sig, dtype=np.float64)
            if _sig.ndim == 2:                          # classwise: (n_cls, r)，每类套同一真实方向形状
                _new = np.zeros_like(_sig)
                for _c in range(_sig.shape[0]):
                    _E = float((_sig[_c] ** 2).sum() + 1e-30)
                    _new[_c] = np.sqrt(_et) * np.sqrt(_E)
                _sig = _new.astype(np.float32)
            else:                                       # 全局: (r,)
                _E = float((_sig ** 2).sum() + 1e-30)
                _sig = (np.sqrt(_et) * np.sqrt(_E)).astype(np.float32)
            print(f"[E1-D1] injected e_true as sigma shape (tag={_tag}); "
                  f"e_true={np.array2string(_et, precision=3)}", flush=True)
        # ---- [STRETCH-D3] 任意信号作 Λ 形状（预算匹配, α 不变）----
        # 对比「偏移信号来自哪里」: sigma_ref(方差) / e_true(能量) / w(脆弱性) / s(决策敏感度=e_true⊙w)
        # 与 --e_true_as_sigma 互斥; sigma_ref 等价于不覆盖(生产默认)。
        _ov_path = getattr(args, 'sigma_override', '')
        _ov_key = getattr(args, 'sigma_override_key', '')
        if _ov_path and _ov_key:
            if _ov_key == 'sigma_ref':
                print(f"[D3] sigma_override=sigma_ref ⇒ 不覆盖(生产默认 Λ)", flush=True)
            else:
                if not os.path.isfile(_ov_path):
                    raise SystemExit("[D3] override json missing: %s" % _ov_path)
                _ov = json.load(open(_ov_path, encoding='utf-8'))
                _sk = args.source.lower()
                _tag = ('H' if 'houston' in _sk else
                        'Y' if ('loukia' in _sk or 'dioni' in _sk) else
                        'P' if 'pavia' in _sk else '?')
                if _tag == '?':
                    raise SystemExit("[D3] cannot map source=%s to H/Y/P" % args.source)
                _ov_seed = str(getattr(args, 'sigma_override_seed', '333'))
                _vec = np.asarray(
                    _ov[_tag]['seeds'][_ov_seed][_ov_key], dtype=np.float64)
                _vec = _vec / (_vec.sum() + 1e-30)          # 归一成概率分布(Σ=1), 与 e_true 同口径
                _sig = np.asarray(_sig, dtype=np.float64)
                if _sig.ndim == 2:                          # classwise: (n_cls, r)
                    _new = np.zeros_like(_sig)
                    for _c in range(_sig.shape[0]):
                        _E = float((_sig[_c] ** 2).sum() + 1e-30)
                        _new[_c] = np.sqrt(_vec) * np.sqrt(_E)   # 预算匹配: Σσ² 不变
                    _sig = _new.astype(np.float32)
                else:                                       # 全局: (r,)
                    _E = float((_sig ** 2).sum() + 1e-30)
                    _sig = (np.sqrt(_vec) * np.sqrt(_E)).astype(np.float32)
                print(f"[D3] injected signal='{_ov_key}' (tag={_tag}, seed={_ov_seed}) "
                      f"as sigma shape; alloc={np.array2string(_vec, precision=3)}", flush=True)
        sisa_V = torch.from_numpy(_V).to(_dev)
        sisa_sigma = torch.from_numpy(_sig).to(_dev)

        # ---- [XG] 逐类平均光谱: 交互指导需要"样本相对类均值"的偏差 ----
        if getattr(args, 'xg_mode', 'off') != 'off':
            xg_mu = torch.from_numpy(
                class_mean_spectra(img_src, train_gt_src, n_classes)).float()
            print(f"[XG] ACTIVE mode={args.xg_mode} tau={args.xg_tau} beta={args.xg_beta} "
                  f"mu={tuple(xg_mu.shape)}", flush=True)
        # ---- [空间掩码] R9b 生效性探针：不打这行就说明开关没生效 ----
        if getattr(args, 'sisa_mask_mode', 'none') != 'none':
            _ps = getattr(args, 'patch_size', 13)
            _r = int(getattr(args, 'sisa_mask_radius', 2))
            _nc = (2 * _r + 1) ** 2
            print(f"[SISA MASK] ACTIVE mode={args.sisa_mask_mode} radius={_r} "
                  f"center={_nc}/{_ps*_ps}px energy_match={bool(getattr(args, 'sisa_mask_energy', 1))}",
                  flush=True)

        # ---- [STRETCH] 脆弱性探针：w_i = E_x |v_i^T grad_x L| （模型侧敏感度）----
        # 与 sigma（数据侧：源域内部异质性）正交，是"伸缩该对准什么"的第二个候选信号。
        # 只在显式给出 --vuln_probe_ckpt 时执行；算完存 json 并退出（不训练）。
        _vck = getattr(args, 'vuln_probe_ckpt', None)
        if _vck:
            _sd = torch.load(_vck, map_location=_dev)
            model.load_state_dict(_sd)
            model.eval()
            print(f"[VULN] loaded ckpt: {os.path.basename(_vck)}", flush=True)
            _Vn = np.asarray(_V, dtype=np.float64)          # (r,B) 或 (n_cls,r,B)
            _acc = None
            _cnt = 0
            _chunk = int(os.environ.get('VULN_CHUNK', '512'))
            for _xb, _yb in train_loader:
                _yb = _yb.to(_dev)
                _lb_all = _yb.detach().cpu().numpy()      # gpu_resident=1 => 已是 0-based
                _n = int(_xb.shape[0])
                for _s0 in range(0, _n, _chunk):
                    _e0 = min(_n, _s0 + _chunk)
                    # detach().clone() 保证是 leaf（GPU-resident 的切片不是 leaf）
                    _xc = _xb[_s0:_e0].detach().clone().requires_grad_(True)
                    _logits = model(_xc)
                    if isinstance(_logits, tuple):
                        _logits = _logits[0]
                    _loss = F.cross_entropy(_logits, _yb[_s0:_e0], reduction='sum')
                    model.zero_grad(set_to_none=True)
                    (_g,) = torch.autograd.grad(_loss, _xc)
                    # delta 在 patch 内空间恒定 => 对该共享光谱偏移的梯度 = 空间求和
                    _gs = _g.sum(dim=(2, 3)).detach()      # (n, B)
                    _lb = _lb_all[_s0:_e0]
                    if _Vn.ndim == 3:                      # classwise: 逐类累积（0-based）
                        if _acc is None:
                            _acc = np.zeros((_Vn.shape[0], _Vn.shape[1]))
                        for _ci in range(_Vn.shape[0]):
                            _sel = (_lb == _ci)
                            if _sel.sum() == 0:
                                continue
                            _proj = _gs[_sel].float().cpu().numpy() @ _Vn[_ci].T
                            _acc[_ci] += np.abs(_proj).sum(0)
                    else:
                        _proj = _gs.float().cpu().numpy() @ _Vn.T
                        _acc = (np.abs(_proj).sum(0) if _acc is None
                                else _acc + np.abs(_proj).sum(0))
                    del _xc, _g, _gs, _logits, _loss
                _cnt += _n
            _w = _acc / max(1, _cnt)
            _sn = np.asarray(_sig, dtype=np.float64)
            _sig_avg = _sn.mean(0) if _sn.ndim == 2 else _sn
            _w_avg = _w.mean(0) if _w.ndim == 2 else _w
            _es = (_sig_avg ** 2) / (_sig_avg ** 2).sum()
            _ew = (_w_avg ** 2) / (_w_avg ** 2).sum()
            _cos = float(_es @ _ew / (np.linalg.norm(_es) * np.linalg.norm(_ew) + 1e-30))
            print(f"[VULN] n_samples={_cnt}")
            print(f"[VULN] w (vulnerability)  = {np.array2string(_w_avg, precision=6)}")
            print(f"[VULN] sigma (heterogeneity)= {np.array2string(_sig_avg, precision=6)}")
            print(f"[VULN] e_w   = {np.array2string(_ew, precision=4)}")
            print(f"[VULN] e_sig = {np.array2string(_es, precision=4)}")
            print(f"[VULN] cos(e_w, e_sigma) = {_cos:.4f}")
            _out = getattr(args, 'vuln_probe_out', None) or 'vuln_probe.json'
            json.dump(dict(source=args.source, target=args.target, n=_cnt,
                           w=_w.tolist(), sigma=_sn.tolist(),
                           e_w=_ew.tolist(), e_sigma=_es.tolist(), cos=_cos),
                      open(_out, 'w', encoding='utf-8'), indent=1)
            print(f"[VULN] saved -> {_out}")
            return None
        if getattr(args, 'lrdro_lambda', 0.0) > 0:
            print(f"[seed={seed}] LR-DRO ON: lambda={args.lrdro_lambda:g}, rho={args.lrdro_rho:g}, "
                  f"V={tuple(sisa_V.shape)}, sigma={np.array2string(_sig, precision=6)}", flush=True)
        if sisa_hetero:
            print(f"[seed={seed}] SISA hetero: {_V.shape[0]} blocks x rank={_V.shape[1]}/{_V.shape[2]}bands")
        elif sisa_classwise:
            _rel_c = _sig / (np.abs(img_src).mean() + 1e-12)
            print(f"[seed={seed}] SISA classwise: {_V.shape[0]} classes x rank={_V.shape[1]}"
                  f"/{_V.shape[2]}bands, per-class sigma[0]="
                  f"{np.array2string(_sig[:, 0], precision=4)}, sigma0/|x|="
                  f"{np.array2string(_rel_c[:, 0], precision=3)}")
        else:
            _rel = _sig / (np.abs(img_src).mean() + 1e-12)
            print(f"[seed={seed}] SISA subspace: rank={_V.shape[0]}/{_V.shape[1]}bands, "
                  f"sigma={np.array2string(_sig, precision=4)}, sigma/|x|={np.array2string(_rel, precision=4)}")

    # === A4 baseline: REMOVE the shift subspace (domain-alignment behaviour) ===
    if args.remove_shift:
        Xf = img_src.reshape(-1, img_src.shape[2]).astype(np.float64)
        Xc = Xf - Xf.mean(0).reshape(1, -1)
        St = (Xc.T @ Xc) / max(len(Xc) - 1, 1)
        Pperp = np.eye(St.shape[0]) - _V.T @ _V                # projector onto S-perp
        Cp = Pperp @ St @ Pperp
        Cp = (Cp + Cp.T) / 2.0
        _w, _U = np.linalg.eigh(Cp)
        Q = _U[:, ::-1][:, :k_endmembers]                      # (bands, k) top-k dirs in S-perp
        _proj = base_model.vca_proj
        with torch.no_grad():
            _proj.weight.data.copy_(
                torch.from_numpy(np.ascontiguousarray(Q.T)).float()
                .to(_proj.weight.device).view_as(_proj.weight))
        for _pp in _proj.parameters():
            _pp.requires_grad = False                          # keep it strictly inside S-perp
        print(f"[seed={seed}] REMOVE-SHIFT baseline: proj rows confined to S-perp "
              f"(r={_V.shape[0]}, k={k_endmembers}), projection FROZEN")

    base_lr = args.lr
    clr = args.classifier_lr if args.classifier_lr is not None else base_lr

    # Param grouping for separate lr / MTTC
    if ci_mode:
        backbone_params = model.get_backbone_params()
        classifier_params = model.get_classifier_params()
    else:
        backbone_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if 'classifier' in name or 'fc_embed' in name:
                classifier_params.append(param)
            else:
                backbone_params.append(param)

    # HGR 需要独立的 classifier param group 才能单独调它的 lr（M1 默认单 group）
    hgr_mode = args.hgr_target > 0 or args.hgr_fixed_m > 0

    if clr != base_lr or args.mttc_beta > 0 or hgr_mode:
        optimizer = torch.optim.Adam([
            {'params': backbone_params, 'lr': base_lr},
            {'params': classifier_params, 'lr': clr},
        ], weight_decay=args.wd)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.wd)

    if clr != base_lr:
        print(f"[seed={seed}] A2 mode: backbone_lr={base_lr}, classifier_lr={clr}")
    else:
        print(f"[seed={seed}] M1 mode: lr={base_lr}")

    gbtc_mode = args.gbtc_lambda > 0
    cp_mode = args.cp_epsilon > 0
    mttc_mode = args.mttc_beta > 0
    acp_mode = args.acp_lambda > 0 and args.acp_h_target > 0
    gd_mode = args.gd_alpha > 0

    if gbtc_mode:
        print(f"[seed={seed}] GBTC: lambda={args.gbtc_lambda}")
    if cp_mode:
        print(f"[seed={seed}] CP: epsilon={args.cp_epsilon}")
    if mttc_mode:
        print(f"[seed={seed}] MTTC: beta={args.mttc_beta}, tau={args.mttc_tau}")
    if acp_mode:
        print(f"[seed={seed}] ACP: H_target={args.acp_h_target}, lambda={args.acp_lambda}")
    if gd_mode:
        print(f"[seed={seed}] GD: alpha={args.gd_alpha}")

    # === HGR: Head-Backbone Gradient Ratio 闭环控制 ===
    # 直接把 per-parameter RMS 梯度比 rho_bb/rho_cls 维持在 r*，分类头 lr 倍率自适应。
    # A2 退化为 "m 恒等于 0.1" 的特例。
    # 注意 backbone_params / classifier_params 里 classifier_params 含 fc_embed，
    # 而 fc_embed 与 logits 并联、纯 CE 下 grad 恒为 None（实测确认），
    # HGRController 内部按 grad 非 None 的参数统计，会自动排除它。
    hgr = None
    hgr_r_log = []
    if hgr_mode:
        from hgr import HGRController
        fixed = args.hgr_fixed_m > 0
        hgr = HGRController(
            optimizer, cls_group_idx=1, base_cls_lr=clr,
            r_target=1.0 if fixed else args.hgr_target,
            gain=0.0 if fixed else args.hgr_gain,      # 固定倍率模式关掉控制器
            adjust_every=args.hgr_every,
            warmup_steps=args.hgr_warmup,
            init_m=args.hgr_fixed_m if fixed else 1.0)
        print(f"[seed={seed}] HGR: {'FIXED m=' + str(args.hgr_fixed_m)
                                    if fixed else 'r*=' + str(args.hgr_target)
                                    + ' gain=' + str(args.hgr_gain)
                                    + ' every=' + str(args.hgr_every)}")

    criterion = nn.CrossEntropyLoss()

    # ---- MTTC state ----
    mttc_snapshot = None

    # ---- 训练循环 ----
    best_oa = 0.0
    best_state = None  # best-ckpt fix
    best_epoch = 0
    best_aa = 0.0
    best_val_oa = 0.0  # 源域留出集 OA (选参用)
    best_vote_oa = 0.0  # CI voting eval
    oa_hist = []  # per-epoch target OA (mean_stable / m_last100)

    # ---- NSD (Null-Space Drift suppression) v1: post-saturation SWA ----
    # 训练后期对权重做滑动平均(SWA)，等价抵消零空间随机游走，使 last≈best 且不依赖目标标签选点。
    nsd_oa = None
    nsd_aa = None
    swa_state = None
    swa_n = 0

    # === DSSP-D: 动态偏移子空间（周期重估 V，σ/α 冻结）===
    _dssd_period = int(getattr(args, 'dssd_period', 0) or 0)
    _dssd_on = _dssd_period > 0
    _dssd_warmup = int(getattr(args, 'dssd_warmup', 0) or 0)
    _dssd_mix = float(getattr(args, 'dssd_mix', 1.0))
    _dssd_nmax = int(getattr(args, 'dssd_nmax', 2048) or 2048)
    _dssd_log = []
    if _dssd_on:
        if sisa_V is None:
            raise SystemExit("[DSSP-D] --dssd_period>0 但 SISA 子空间未构建"
                             "（需同时给 --sisa_lambda>0 或 --sdc_lambda>0）")
        if sisa_hetero:
            raise SystemExit("[DSSP-D] 暂不支持 --sisa_hetero（块级 3-D V 语义不同，需先定义块 id 对齐）")
        if not (0.0 <= _dssd_mix <= 1.0):
            raise SystemExit(f"[DSSP-D] --dssd_mix 须在 [0,1]，实得 {_dssd_mix}")
        if _dssd_mix <= 1e-12:
            raise SystemExit("[DSSP-D] mix=0 等于冻结基线，请直接用 baseline 臂而不是开 DSSP-D")
        print(f"[seed={seed}] [DSSP-D] 动态偏移子空间 ON: period={_dssd_period} "
              f"warmup={_dssd_warmup} mix={_dssd_mix} nmax={_dssd_nmax} "
              f"(方向不可知 / 只用源域留出集 / σ,α 冻结 ⇒ 扰动预算逐位不变)", flush=True)

    t0 = time.time()

    # [M2 2026-10-08] 幅度自适应状态: 每 epoch 末按留出集指标更新 alpha。
    #   v 为 None ⇒ 尚未评估(第一 epoch 用 --sisa_alpha 的值); 之后用自适应值。
    _adapt_alpha_state = {'v': None, 'hist': []}
    _alpha_adapt_on = (getattr(args, 'sisa_alpha_adaptive', 'off') != 'off'
                       and not args.sisa_calibrate)

    # [M3 2026-10-08] 逐方向幅度 σ 在线重估（C4）状态与调度。
    #   与 DSSP-D 同构（warmup 起算 + 周期触发），但作用对象是【幅度】不是【方向】。
    _sig_adapt_on = bool(getattr(args, 'sisa_sigma_adaptive', False))
    _sig_adapt_period = int(getattr(args, 'sisa_sigma_adapt_period', 0) or 0)
    _sig_adapt_warmup = int(getattr(args, 'sisa_sigma_adapt_warmup', 0) or 0)
    _sig_adapt_nmax = int(getattr(args, 'sisa_sigma_adapt_nmax', 2048) or 2048)
    _SIG_ADAPT_STATE.enabled = False
    _SIG_ADAPT_STATE.factor = 1.0
    _SIG_ADAPT_STATE.hist = []
    if _sig_adapt_on:
        if sisa_V is None:
            raise SystemExit("[M3] --sisa_sigma_adaptive 需要 SISA 子空间"
                             "（需同时给 --sisa_lambda>0 或 --sdc_lambda>0）")
        if sisa_hetero:
            raise SystemExit("[M3] 暂不支持 --sisa_hetero（块级 σ 语义不同，需先定义块 id 对齐）")
        if _sig_adapt_period <= 0:
            raise SystemExit("[M3] --sisa_sigma_adapt_period 须 > 0（否则永不触发）")
        if args.sisa_calibrate:
            raise SystemExit("[M3] 与 --sisa_calibrate 互斥（后者把幅度钉死，重估会被覆盖）")
        _SIG_ADAPT_STATE.enabled = True
        print(f"[seed={seed}] [M3] 逐方向幅度自适应 ON: period={_sig_adapt_period} "
              f"warmup={_sig_adapt_warmup} lambda={args.sisa_sigma_adapt_lambda} "
              f"floor={args.sisa_sigma_adapt_floor} ceil={args.sisa_sigma_adapt_ceil} "
              f"(V 不动；σ 幅度随训练变化 ⇒ 预算改变，这是 C4 的目的)", flush=True)

    for epoch in range(1, args.epochs + 1):
        # LR schedule
        if args.no_lr_decay:
            current_lr = base_lr
        else:
            current_lr = base_lr / math.pow(1 + 10 * (epoch - 1) / args.epochs, 0.75)

        # Update backbone lr in optimizer
        for g in optimizer.param_groups:
            if len(optimizer.param_groups) == 1 or len(classifier_params) == 0:
                g['lr'] = current_lr
            elif not any(p is q for p in g['params'] for q in classifier_params):
                g['lr'] = current_lr

        # === NSD: 训练期直接抑制零空间游走（freeze / lrd），非事后平均 ===
        if args.nsd and args.nsd_mode in ('freeze', 'lrd') and epoch >= args.nsd_start:
            for g in optimizer.param_groups:
                is_backbone = (len(optimizer.param_groups) == 1 or len(classifier_params) == 0
                               or not any(p is q for p in g['params'] for q in classifier_params))
                if is_backbone:
                    g['lr'] = 0.0 if args.nsd_mode == 'freeze' else current_lr * args.nsd_decay

        # ---- 训练 ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # === DSSP-D: 周期重估偏移子空间（方向不可知；只用源域留出集；σ/α 冻结）===
        # 🔴 触发点从 warmup 起算：(epoch - warmup) % period == 0。
        #    原先写 epoch % period == 0 会让 "warmup=20, period=1000" 永远不触发
        #    ⇒ "只估一次然后冻结"那一臂静默退化成基线（首轮实测已撞：结果与 F 逐位相同）。
        if (_dssd_on and sisa_V is not None and epoch >= _dssd_warmup
                and ((epoch - _dssd_warmup) % _dssd_period == 0)):
            _Vn = model_sensitivity_subspace(
                model, val_loader, _dev, args.sisa_rank,
                n_classes=(n_classes if sisa_classwise else None),
                n_max=_dssd_nmax)
            if _Vn is None:
                print(f"[seed={seed}] [DSSP-D] epoch {epoch}: 敏感度估计样本不足，保持原 V",
                      flush=True)
            else:
                _Vo_np = sisa_V.detach().cpu().numpy()
                _d_mean, _d_ang, _d_share = subspace_drift(_Vo_np, _Vn)
                _Vb = blend_subspaces(_Vo_np, _Vn, _dssd_mix)
                if np.asarray(_Vb).shape != np.asarray(_Vo_np).shape:
                    print(f"[seed={seed}] [DSSP-D] epoch {epoch}: 形状不符 "
                          f"{np.asarray(_Vb).shape} vs {np.asarray(_Vo_np).shape} ⇒ 跳过",
                          flush=True)
                else:
                    # 🔴 fail-closed: 行必须正交归一，否则 E||delta||^2 != alpha^2 sum(sigma^2)，
                    # "预算不变"被破坏 ⇒ 与冻结臂的比较不再公平（R3）。宁可炸也不要静默跑完 200 轮。
                    _Vb64 = np.asarray(_Vb, dtype=np.float64)
                    if _Vb64.ndim == 3:
                        _I = np.eye(_Vb64.shape[1])[None]
                        _orth = float(np.abs(_Vb64 @ _Vb64.transpose(0, 2, 1) - _I).max())
                    else:
                        _orth = float(np.abs(_Vb64 @ _Vb64.T
                                             - np.eye(_Vb64.shape[0])).max())
                    if not np.isfinite(_orth) or _orth > 1e-4:
                        raise SystemExit(
                            f"[DSSP-D] epoch {epoch}: 重估后 V 行不正交 "
                            f"||VV^T - I||_inf = {_orth:.3e} > 1e-4 ⇒ 终止"
                            f"（否则扰动预算被破坏，与冻结臂不可比）")
                    sisa_V = torch.from_numpy(
                        np.ascontiguousarray(_Vb, dtype=np.float32)).to(_dev)
                    _dssd_log.append(dict(
                        epoch=int(epoch),
                        mean_abs_cos=float(_d_mean),
                        max_principal_angle_deg=float(_d_ang),
                        captured_energy_share=float(_d_share),
                        orth_err=float(_orth)))
                    print(f"[seed={seed}] [DSSP-D] epoch {epoch}: V 重估 mix={_dssd_mix} "
                          f"drift mean|cos|={_d_mean:.4f} max_angle={_d_ang:.2f}deg "
                          f"share={_d_share:.4f} orth_err={_orth:.1e} "
                          f"(σ/α 未动 ⇒ 预算不变)", flush=True)

        # === [M3 2026-10-08] 逐方向幅度 σ 在线重估（C4）===
        # 与 DSSP-D 的区别：DSSP-D 换的是【方向 V】（取向），σ/α 冻结 ⇒ 预算逐位不变；
        # 这里换的是【每个方向放多大的乘子 σ】（幅度），V 不动。
        # 判据：用源域留出集的特征，看每个方向上"模型已经自发产生"的能量占比。
        #   ratio_j = clip(1 - λ * self_j / mean(self), floor, ceil)
        #   self_j 大 ⇒ 该方向模型本来就自发变化 ⇒ 少补；self_j 小 ⇒ 补足。
        #   这是"减掉自然残余"的口径，factor 可上可下（回应用户 12:01「可上可下」）。
        # 预算会被改变 ⇒ 与冻结臂不可比，这是 C4 的**实验目的本身**（不是缺陷）。
        if (_SIG_ADAPT_STATE.enabled and sisa_sigma is not None
                and epoch >= _sig_adapt_warmup
                and ((epoch - _sig_adapt_warmup) % _sig_adapt_period == 0)):
            try:
                _self = _sigma_self_energy(model, val_loader, _dev, sisa_V,
                                           sisa_sigma, n_max=_sig_adapt_nmax)
                _mean_self = float(_self.mean()) if float(_self.mean()) > 1e-30 else 1.0
                _ratio = 1.0 - float(args.sisa_sigma_adapt_lambda) * (_self / _mean_self)
                _ratio = np.clip(_ratio, float(args.sisa_sigma_adapt_floor),
                                 float(args.sisa_sigma_adapt_ceil))
                _SIG_ADAPT_STATE.factor = float(np.mean(_ratio))
                _SIG_ADAPT_STATE.hist.append(dict(
                    epoch=int(epoch),
                    factor=float(_SIG_ADAPT_STATE.factor),
                    ratio=[float(v) for v in _ratio],
                    self_energy=[float(v) for v in _self]))
                print(f"[seed={seed}] [M3] epoch {epoch}: σ 逐方向重估 "
                      f"factor={_SIG_ADAPT_STATE.factor:.4f} "
                      f"ratio={np.array2string(_ratio, precision=3)} "
                      f"(V 未动；预算随之变化，这是 C4 的目的)", flush=True)
            except Exception as _e_m3:      # fail-soft：诊断臂不该拖垮主训练
                print(f"[seed={seed}] [M3] epoch {epoch}: 重估失败({type(_e_m3).__name__}: "
                      f"{_e_m3}) ⇒ 保持 factor={_SIG_ADAPT_STATE.factor}", flush=True)

        for data_batch, label_batch, *_rest in train_loader:
            block_batch = _rest[0] if _rest else None  # 异质模式的块 id (N,)
            data_batch = data_batch.to(device)
            label_batch = label_batch.to(device) - (0 if args.gpu_resident else 1)

            if args.gpu_aug:
                data_batch = gpu_augment(data_batch)

            optimizer.zero_grad()

            # === CP: classifier perturbation ===
            cp_weight_orig = None
            if cp_mode:
                if ci_mode:
                    cp_weight_orig = [h.weight.data.clone() for h in model.classifiers]
                    with torch.no_grad():
                        for h in model.classifiers:
                            noise = torch.randn_like(h.weight) * args.cp_epsilon
                            h.weight.data.add_(noise)
                else:
                    cp_weight_orig = model.classifier.weight.data.clone()
                    with torch.no_grad():
                        noise = torch.randn_like(model.classifier.weight) * args.cp_epsilon
                        model.classifier.weight.data.add_(noise)

            # === Forward ===
            _feat_mode = (getattr(args, 'lrdro_feat_lambda', 0.0) > 0 and not ci_mode)
            if _feat_mode:
                # M1b: 需要 backbone 输出 feat，使特征侧梯度只反传到 feat（不反传卷积层）
                if _amp_ctx is not None:
                    raise SystemExit(
                        "[M1b] --lrdro_feat_lambda 不支持 AMP 组合（需在 feat 上取精确梯度）")
                feat_main = model.forward_backbone(data_batch)
                logits = model.classifier(feat_main)
                feats = model.fc_embed(feat_main)
            elif _amp_ctx is not None:
                if _amp_full:
                    # 激进模式：autocast 作用域覆盖整个 loss 计算块，在 backward 前退出。
                    _amp_ctx.__enter__()
                    logits, feats = model(data_batch)
                else:
                    # 稳妥模式：仅模型前向在 FP16，输出立即转回 FP32 -> 下游 loss/SISA/SDC 逻辑逐字节不变。
                    with _amp_ctx:
                        logits, feats = model(data_batch)
                    if isinstance(logits, (list, tuple)):
                        logits = type(logits)(t.float() if torch.is_tensor(t) else t for t in logits)
                    else:
                        logits = logits.float() if torch.is_tensor(logits) else logits
                    if isinstance(feats, (list, tuple)):
                        feats = type(feats)(t.float() if torch.is_tensor(t) else t for t in feats)
                    else:
                        feats = feats.float() if torch.is_tensor(feats) else feats
            else:
                logits, feats = model(data_batch)
                feat_main = None

            # === GBTC: gradient balance ===
            if gbtc_mode:
                if ci_mode:
                    # CI + GBTC: average CE loss across heads, then balance
                    losses_ci = [criterion(logits[k], label_batch) for k in range(args.ci_heads)]
                    loss_ce = sum(losses_ci) / len(losses_ci)
                else:
                    loss_ce = criterion(logits, label_batch)

                # Compute per-group gradient norms (create_graph for second-order)
                grad_bb = torch.autograd.grad(loss_ce, backbone_params,
                                              create_graph=True, retain_graph=True,
                                              allow_unused=True)
                grad_cls = torch.autograd.grad(loss_ce, classifier_params,
                                               create_graph=True,
                                               allow_unused=True)

                norm_bb_sq = sum((g.pow(2).sum() if g is not None else torch.tensor(0.0, device=device))
                                 for g in grad_bb)
                norm_cls_sq = sum((g.pow(2).sum() if g is not None else torch.tensor(0.0, device=device))
                                  for g in grad_cls)
                # Guard against zero gradient (e.g., at epoch start)
                norm_bb = torch.sqrt(norm_bb_sq + 1e-12)
                norm_cls = torch.sqrt(norm_cls_sq + 1e-12)

                L_gbtc = (torch.log(norm_bb) - torch.log(norm_cls)).pow(2)
                L_total = loss_ce + args.gbtc_lambda * L_gbtc
                # === ACP: adaptive confidence penalty ===
                if acp_mode:
                    avg_logits = logits.mean(dim=0) if ci_mode else logits
                    probs = F.softmax(avg_logits, dim=-1)
                    log_probs = F.log_softmax(avg_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    acp_penalty = torch.clamp(args.acp_h_target - entropy, min=0).pow(2)
                    L_total = L_total + args.acp_lambda * acp_penalty
                if _amp_full:
                    _amp_ctx.__exit__(None, None, None)
                _scaler.scale(L_total).backward()

                # Record gradient ratio
                with torch.no_grad():
                    r_t = (norm_bb / (norm_cls + 1e-12)).item()
            else:
                if ci_mode:
                    loss = sum(criterion(logits[k], label_batch) for k in range(args.ci_heads)) / args.ci_heads
                else:
                    loss = criterion(logits, label_batch)
                # === ACP: adaptive confidence penalty ===
                if acp_mode:
                    avg_logits = logits.mean(dim=0) if ci_mode else logits
                    probs = F.softmax(avg_logits, dim=-1)
                    log_probs = F.log_softmax(avg_logits, dim=-1)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()
                    acp_penalty = torch.clamp(args.acp_h_target - entropy, min=0).pow(2)
                    loss = loss + args.acp_lambda * acp_penalty

                # === Dual-view consistency (Part C) ===
                if hasattr(args, 'l_con') and args.l_con > 0 and args.con_aug != 'none':
                    logits_pert, feats_pert = model(data_batch, tmp_aug=(args.con_aug, args.con_aug_param))
                    l_con_val = F.mse_loss(feats.detach(), feats_pert) * args.l_con
                    loss = loss + l_con_val

                # === Soft orthogonal regularization on projection rows ===
                if args.proj_ortho_lambda > 0:
                    ortho_val = base_model.ortho_penalty()
                    if ortho_val is not None:
                        loss = loss + ortho_val

                # === CDSP: Cross-Domain Spectral Projection consistency ===
                if args.cdsp_lambda > 0 and args.cdsp_shift != 'none':
                    x_shift = spectral_shift_augment(data_batch, strength=args.cdsp_strength)
                    logits_shift, _ = model(x_shift)
                    if ci_mode:
                        loss_shift = sum(criterion(logits_shift[k], label_batch)
                                         for k in range(args.ci_heads)) / args.ci_heads
                    else:
                        loss_shift = criterion(logits_shift, label_batch)
                    loss = loss + args.cdsp_lambda * loss_shift

                # === SISA: source-internal shift-subspace augmentation (also builds SDC positives) ===
                if (args.sisa_lambda > 0 or args.sdc_lambda > 0) and sisa_V is not None:
                    _alpha = 1.0 if args.sisa_calibrate else args.sisa_alpha
                    # [M2 2026-10-08] 幅度自适应: 按模型当前状态(留出集 loss/conf)动态调 alpha。
                    #   off(默认) ⇒ 与历史逐位一致; loss/conf ⇒ 每 epoch 由 _adapt_alpha 更新。
                    if getattr(args, 'sisa_alpha_adaptive', 'off') != 'off' \
                            and not args.sisa_calibrate and _adapt_alpha_state.get('v') is not None:
                        _alpha = float(_adapt_alpha_state['v'])
                    if getattr(args, 'sisa_alpha_marginalize', False) and not args.sisa_calibrate:
                        import math as _math
                        _lo, _hi = args.sisa_alpha_min, args.sisa_alpha_max
                        _alpha = _math.exp(torch.empty(1).uniform_(_math.log(_lo), _math.log(_hi)).item())
                    # [M1 2026-10-08] 关增广但保留 SISA/SDC 损失项（消融格子）。
                    #   no_augment=True ⇒ 不扰动, 但后续 sisa_lambda/sdc_lambda 分支照常执行
                    #   (此时 x_sisa == data_batch ⇒ CE 项退化为原样本 CE, 两视图 SDC 退化为自对比)。
                    #   默认 False ⇒ 与历史 150 runs 逐位一致。
                    if getattr(args, 'sisa_no_augment', False):
                        x_sisa = data_batch
                        logits_sisa, feats_sisa = model(x_sisa)
                    elif sisa_hetero and block_batch is not None:
                        x_sisa = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha, block_ids=block_batch, mult_strength=args.sisa_mult, path_strength=args.sisa_path, xg_mode=args.xg_mode, xg_tau=args.xg_tau, xg_beta=args.xg_beta, xg_mu=xg_mu, y=label_batch, mask_mode=args.sisa_mask_mode, mask_radius=args.sisa_mask_radius, mask_energy=bool(args.sisa_mask_energy))
                    elif sisa_classwise:
                        # 类条件: 每个样本沿【它自己那个类】的偏移方向扰动。
                        # label_batch 已是 0-based 类索引, 与 V_cls 的行序一一对应。
                        # [DIAG] --sisa_global_delta: 子空间仍逐类, 但幅度退回全局 sigma,
                        #        用于解耦"逐类子空间"与"delta 类绑定"两个副作用。
                        _sig = sisa_sigma_global if getattr(args, 'sisa_global_delta', False) else sisa_sigma
                        x_sisa = sisa_augment(data_batch, sisa_V, _sig, alpha=_alpha, block_ids=label_batch, mult_strength=args.sisa_mult, path_strength=args.sisa_path, xg_mode=args.xg_mode, xg_tau=args.xg_tau, xg_beta=args.xg_beta, xg_mu=xg_mu, y=label_batch, mask_mode=args.sisa_mask_mode, mask_radius=args.sisa_mask_radius, mask_energy=bool(args.sisa_mask_energy))
                    else:
                        x_sisa = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha, mult_strength=args.sisa_mult, path_strength=args.sisa_path, xg_mode=args.xg_mode, xg_tau=args.xg_tau, xg_beta=args.xg_beta, xg_mu=xg_mu, y=label_batch, mask_mode=args.sisa_mask_mode, mask_radius=args.sisa_mask_radius, mask_energy=bool(args.sisa_mask_energy))
                    logits_sisa, feats_sisa = model(x_sisa)
                    if args.sisa_lambda > 0:
                        if getattr(args, 'u4_kl_sisa', False):
                            T = args.u4_kl_temp
                            logits_orig, _ = model(data_batch)
                            p_src = F.softmax(logits_orig / T, dim=1)
                            p_aug = F.softmax(logits_sisa / T, dim=1)
                            loss_sisa = F.kl_div(p_aug.log(), p_src, reduction='batchmean') * T * T
                        elif ci_mode:
                            loss_sisa = sum(criterion(logits_sisa[k], label_batch)
                                            for k in range(args.ci_heads)) / args.ci_heads
                        else:
                            loss_sisa = criterion(logits_sisa, label_batch)
                        loss = loss + args.sisa_lambda * loss_sisa
                    if args.sdc_lambda > 0:
                        loss_sdc = supcon_shift_loss(feats, feats_sisa, label_batch,
                                                     temperature=args.sdc_temp,
                                                     max_n=args.sdc_max_n)
                        loss = loss + args.sdc_lambda * loss_sdc

                # === U1: Jacobian shift regularization ===
                if args.u1_lambda > 0 and sisa_V is not None:
                    _fwd = lambda inp: model(inp)[1] if not ci_mode else model.base.fc_embed(model.base.forward_backbone(inp))
                    if args.u1_mode == 'shift':
                        loss_u1 = jacobian_shift_reg(_fwd, data_batch, sisa_V, sisa_sigma)
                    elif args.u1_mode == 'isotropic':
                        loss_u1 = jacobian_isotropic_reg(_fwd, data_batch, n_dirs=sisa_V.shape[0])
                    else:
                        V_rand = torch.randn_like(sisa_V)
                        V_rand = torch.linalg.qr(V_rand.T)[0].T
                        loss_u1 = jacobian_shift_reg(_fwd, data_batch, V_rand, sisa_sigma)
                    loss = loss + args.u1_lambda * loss_u1

                # === M1 / LR-DRO: rank-constrained DRO over the offset subspace ===
                # U_S(rho) = {delta = V z : z^T Lambda^{-1} z <= rho^2},  Lambda = diag(sigma^2)
                # 一阶展开下内层 max 闭式: delta* = rho * V Lambda V^T g / ||Lambda^{1/2} V^T g||_2
                # 🔴 g 必须是【逐样本】梯度 (reduction='none' + grad_outputs=ones)；
                #    用 batch-mean loss 的梯度会差 N 倍且因归一化而看似正常(见 M1 预注册 §1)。
                if getattr(args, 'lrdro_lambda', 0.0) > 0 and sisa_V is not None:
                    # 🔴 R9 形状契约（2026-09-20 扩展：支持 classwise 逐类子空间）
                    #   2-D V = (r, B)      全局子空间          sigma = (r,)
                    #   3-D V = (C, r, B)  classwise 逐类子空间 sigma = (C, r)
                    #   两者统一成"每个样本配一个 (r, B) 子空间"后走同一条闭式解，
                    #   因此 classwise 下内层 max 就是【逐类最坏情况】，与 M(c) 的物理事实一致。
                    if sisa_V.dim() == 3:
                        if sisa_sigma.dim() != 2 or tuple(sisa_sigma.shape) != tuple(sisa_V.shape[:2]):
                            raise SystemExit(
                                f"[M1] R9 契约违反: classwise 下 sigma 应为 (C,r)={tuple(sisa_V.shape[:2])}, "
                                f"实际 {tuple(sisa_sigma.shape)}")
                        if int(label_batch.max().item()) >= sisa_V.shape[0]:
                            raise SystemExit(
                                f"[M1] R9 契约违反: label max={int(label_batch.max().item())} "
                                f"越界 (C={sisa_V.shape[0]})")
                    elif sisa_V.dim() == 2:
                        if sisa_sigma.dim() != 1 or sisa_sigma.shape[0] != sisa_V.shape[0]:
                            raise SystemExit(
                                f"[M1] R9 契约违反: 全局下 sigma 应为 (r,)={sisa_V.shape[0]}, "
                                f"实际 {tuple(sisa_sigma.shape)}")
                    else:
                        raise SystemExit(
                            f"[M1] V.dim()={sisa_V.dim()} 不支持 (需 2-D 全局 或 3-D classwise)")
                    _rho = float(args.lrdro_rho)
                    _frac = float(getattr(args, 'lrdro_grad_frac', 1.0))
                    _N0 = int(data_batch.size(0))
                    _gf = _N0 if _frac >= 1.0 else max(1, int(_N0 * _frac))

                    # per-sample 梯度：分块反传以控制显存（--lrdro_grad_frac）
                    _chunks = []
                    for _s in range(0, _N0, _gf):
                        _xe = data_batch[_s:_s + _gf].detach().clone().requires_grad_(True)
                        _lg = model(_xe)
                        _lg = _lg[0] if isinstance(_lg, (tuple, list)) else _lg
                        _lab = label_batch[_s:_s + _gf]
                        _ell = F.cross_entropy(_lg, _lab, reduction='none')
                        (_g,) = torch.autograd.grad(
                            _ell, _xe, grad_outputs=torch.ones_like(_ell))
                        # shift 在 patch 内空间恒定 => 沿 H,W 聚合, 得 (N, B)
                        _gc = _g.sum(dim=(2, 3))
                        if sisa_V.dim() == 3:
                            _Vs = sisa_V[_lab]            # (N, r, B) 逐类子空间
                            _sg = sisa_sigma[_lab]        # (N, r)
                        else:
                            _n = _gc.shape[0]
                            _Vs = sisa_V.unsqueeze(0).expand(_n, -1, -1)
                            _sg = sisa_sigma.unsqueeze(0).expand(_n, -1)
                        _proj = torch.einsum('nrb,nb->nr', _Vs, _gc)      # (N, r)  V^T g
                        _wproj = _proj * _sg                              # Λ^{1/2} V^T g
                        _den = _wproj.norm(dim=1, keepdim=True).clamp(min=1e-12)
                        _gx = (_sg ** 2) * _proj                          # Λ V^T g
                        _chunks.append(
                            _rho * torch.einsum('nr,nrb->nb', _gx, _Vs) / _den)   # (N, B)
                    _delta = torch.cat(_chunks, dim=0).detach()   # (N, B)
                    del _chunks

                    # 可观测性：首个 batch 打印扰动量级，用于核对是否与 SISA 的 ||delta|| 同阶
                    if not getattr(args, '_lrdro_dbg_done', False):
                        with torch.no_grad():
                            _dnrm = _delta.norm(dim=1)
                            _bnrm = data_batch.detach().flatten(1).norm(dim=1)
                        print(f"[seed={seed}] LR-DRO first batch: ||delta*|| mean={_dnrm.mean().item():.6f} "
                              f"max={_dnrm.max().item():.6f} | ||x|| mean={_bnrm.mean().item():.4f} "
                              f"| rel={(_dnrm / (_bnrm + 1e-12)).mean().item():.6f}", flush=True)
                        args._lrdro_dbg_done = True

                    x_lrdro = (data_batch + _delta.view(_N0, -1, 1, 1)).detach()
                    logits_lrdro, _ = model(x_lrdro)
                    if ci_mode:
                        loss_lrdro = sum(criterion(logits_lrdro[k], label_batch)
                                         for k in range(args.ci_heads)) / args.ci_heads
                    else:
                        loss_lrdro = criterion(logits_lrdro, label_batch)
                    loss = loss + args.lrdro_lambda * loss_lrdro

                # === M1b / LR-DRO-F: 特征空间的秩约束最坏情况 ===
                # 与光谱侧同一闭式解，但支撑换成 V 经网络推送后的 JV（QR 正交化），
                # 度量 = 物理能量 σ ⊗ 网络敏感性 ||J v_i||（双重拉伸）。
                # 🔑 成本关键：per-sample 梯度只反传到 feat（classifier 是线性层），
                #    不反传卷积层 ⇒ 显存/时间远低于光谱侧
                #    （Pavia 实测：光谱侧 18.3→21.7 GB、2.5→68 s/ep，慢 27×，系显存 thrashing）。
                if getattr(args, 'lrdro_feat_lambda', 0.0) > 0 and sisa_V is not None:
                    if feat_main is None:
                        raise SystemExit(
                            "[M1b] feat_main 为 None：前向未走 feat 路径（ci_mode/AMP?）")
                    _fk = '_featW_s%d' % seed
                    if not hasattr(args, _fk):
                        _W, _sJ, _scale = push_subspace_to_feature(
                            model, data_batch.detach(), label_batch, sisa_V, sisa_sigma,
                            eps=args.lrdro_feat_eps, n_max=args.lrdro_feat_push_n)
                        setattr(args, _fk, (_W, _sJ))
                        print(f"[seed={seed}] M1b LR-DRO-F ON: lambda={args.lrdro_feat_lambda:g}, "
                              f"rho={args.lrdro_feat_rho:g}, W={tuple(_W.shape)}, "
                              f"||Jv||={np.array2string(_scale.detach().cpu().numpy().reshape(-1), precision=5)}, "
                              f"sigma_J={np.array2string(_sJ.detach().cpu().numpy().reshape(-1)[:6], precision=6)}",
                              flush=True)
                    feat_W, feat_sigma_J = getattr(args, _fk)
                    _rho_f = float(args.lrdro_feat_rho)
                    if feat_W.dim() == 3:
                        _Wf = feat_W[label_batch]            # (N, r, d) 逐类支撑
                        _sf = feat_sigma_J[label_batch]      # (N, r)
                    else:
                        _nf = feat_main.shape[0]
                        _Wf = feat_W.unsqueeze(0).expand(_nf, -1, -1)
                        _sf = feat_sigma_J.unsqueeze(0).expand(_nf, -1)
                    _ell_f = F.cross_entropy(logits, label_batch, reduction='none')
                    (_g_f,) = torch.autograd.grad(
                        _ell_f, feat_main, grad_outputs=torch.ones_like(_ell_f),
                        retain_graph=True)
                    _proj_f = torch.einsum('nrb,nb->nr', _Wf, _g_f)          # W^T g
                    _wproj_f = _proj_f * _sf                                 # Λ^{1/2} W^T g
                    _den_f = _wproj_f.norm(dim=1, keepdim=True).clamp(min=1e-12)
                    _gx_f = (_sf ** 2) * _proj_f                             # Λ W^T g
                    _delta_f = _rho_f * torch.einsum('nr,nrb->nb', _gx_f, _Wf) / _den_f
                    if not getattr(args, '_m1b_dbg_s%d' % seed, False):
                        with torch.no_grad():
                            _dn = _delta_f.norm(dim=1)
                            _fn = feat_main.detach().norm(dim=1)
                        print(f"[seed={seed}] M1b first batch: ||d_feat*|| mean={_dn.mean().item():.6f} "
                              f"max={_dn.max().item():.6f} | ||feat|| mean={_fn.mean().item():.4f} "
                              f"| rel={(_dn / (_fn + 1e-12)).mean().item():.6f}", flush=True)
                        setattr(args, '_m1b_dbg_s%d' % seed, True)
                    logits_fadv = model.classifier(feat_main + _delta_f.detach())
                    loss = loss + args.lrdro_feat_lambda * criterion(logits_fadv, label_batch)

                # === CORAL baseline: covariance alignment ===
                if args.coral_lambda > 0 and sisa_V is not None:
                    _alpha_c = 1.0 if args.sisa_calibrate else args.sisa_alpha
                    if sisa_classwise:
                        x_coral = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha_c, block_ids=label_batch, mult_strength=args.sisa_mult, path_strength=args.sisa_path)
                    else:
                        x_coral = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha_c, mult_strength=args.sisa_mult, path_strength=args.sisa_path)
                    _, feats_coral = model(x_coral)
                    loss_coral = coral_loss(feats, feats_coral)
                    loss = loss + args.coral_lambda * loss_coral

                # === DANN baseline: adversarial domain adaptation ===
                if args.dann_lambda > 0 and sisa_V is not None:
                    _alpha_d = 1.0 if args.sisa_calibrate else args.sisa_alpha
                    if sisa_classwise:
                        x_dann = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha_d, block_ids=label_batch, mult_strength=args.sisa_mult, path_strength=args.sisa_path)
                    else:
                        x_dann = sisa_augment(data_batch, sisa_V, sisa_sigma, alpha=_alpha_d, mult_strength=args.sisa_mult, path_strength=args.sisa_path)
                    _, feats_dann = model(x_dann)
                    feat_dim = feats.size(1)
                    if 'dann_disc' not in dir():
                        dann_disc = DomainDiscriminator(feat_dim).to(device)
                        optimizer.add_param_group({'params': dann_disc.parameters(), 'lr': args.classifier_lr})
                    p = epoch / args.epochs
                    grl_alpha = args.dann_alpha * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
                    feat_src_gRL = GRL.apply(feats, grl_alpha)
                    feat_tgt_gRL = GRL.apply(feats_dann, grl_alpha)
                    pred_src = dann_disc(feat_src_gRL)
                    pred_tgt = dann_disc(feat_tgt_gRL)
                    bce_src = F.binary_cross_entropy_with_logits(pred_src, torch.zeros_like(pred_src))
                    bce_tgt = F.binary_cross_entropy_with_logits(pred_tgt, torch.ones_like(pred_tgt))
                    loss_dann = bce_src + bce_tgt
                    loss = loss + args.dann_lambda * loss_dann

                # === DSSP: discriminative Fisher regularization on the spectral projection ===
                if args.dssp_lambda > 0:
                    _proj = model.base.vca_proj if hasattr(model, 'base') else model.vca_proj
                    z = _proj(data_batch.unsqueeze(1)).squeeze(2)  # (N,k,13,13)
                    z_vec = z.mean(dim=(-2, -1))                   # (N,k)
                    loss_d = fisher_criterion(z_vec, label_batch, n_classes)
                    loss = loss + args.dssp_lambda * loss_d

                if _amp_full:
                    _amp_ctx.__exit__(None, None, None)
                _scaler.scale(loss).backward()

            # === CP: restore original weights ===
            if cp_mode:
                with torch.no_grad():
                    if ci_mode:
                        for h, orig in zip(model.classifiers, cp_weight_orig):
                            h.weight.data.copy_(orig)
                    else:
                        model.classifier.weight.data.copy_(cp_weight_orig)

            # === GD: gradient decorrelation ===
            if gd_mode:
                with torch.no_grad():
                    grad_bb_list = []
                    grad_cls_list = []
                    for n, p in model.named_parameters():
                        if p.grad is None:
                            continue
                        if 'classifier' in n or 'fc_embed' in n:
                            grad_cls_list.append(p.grad.view(-1))
                        else:
                            grad_bb_list.append(p.grad.view(-1))
                    if grad_bb_list and grad_cls_list:
                        g_bb = torch.cat(grad_bb_list)
                        g_cls = torch.cat(grad_cls_list)
                        # Truncate g_bb to match g_cls size (g_bb >> g_cls in small-data DG)
                        g_bb_trunc = g_bb[:g_cls.numel()]
                        proj = (g_cls @ g_bb_trunc) / (g_bb_trunc @ g_bb_trunc + 1e-8)
                        g_cls_proj = g_cls - args.gd_alpha * proj * g_bb_trunc
                        offset = 0
                        for n, p in model.named_parameters():
                            if p.grad is not None and ('classifier' in n or 'fc_embed' in n):
                                n_p = p.numel()
                                p.grad.data = g_cls_proj[offset:offset+n_p].view_as(p.grad)
                                offset += n_p

            # === HGR: 观测梯度比并调整分类头 lr 倍率（必须在 optimizer.step() 之前）===
            if hgr is not None:
                r_t = hgr.step(backbone_params, classifier_params)
                hgr_r_log.append(r_t)

            # 精度探针：启用 AMP 时用 GradScaler 驱动 step/update（enabled=False 时等价于原生 step）
            if _prec_on:
                _scaler.step(optimizer)
                _scaler.update()
            else:
                optimizer.step()

            # === MTTC: post-step repulsion ===
            if mttc_mode and mttc_snapshot is not None:
                with torch.no_grad():
                    for p in classifier_params:
                        p.data.add_(args.mttc_beta * (p.data - mttc_snapshot[p]))

            # Stats
            with torch.no_grad():
                if ci_mode:
                    loss_val = sum(criterion(logits[k], label_batch).item()
                                   for k in range(args.ci_heads)) / args.ci_heads
                    _, pred = logits[0].max(1)
                elif gbtc_mode:
                    loss_val = loss_ce.item()
                    _, pred = logits.max(1)
                else:
                    loss_val = loss.item()
                    _, pred = logits.max(1)

            train_loss += loss_val * data_batch.size(0)
            train_total += label_batch.size(0)
            train_correct += pred.eq(label_batch).sum().item()

        train_acc = 100. * train_correct / train_total

        # ---- [极限探针] eval_every: 每 N 轮评估一次（默认1=与原版完全一致） ----
        do_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs)

        # ---- MTTC: save snapshot every tau epochs ----
        if mttc_mode and epoch % args.mttc_tau == 0:
            mttc_snapshot = {}
            with torch.no_grad():
                for p in classifier_params:
                    mttc_snapshot[p] = p.data.clone()

        # ---- 评估 ----
        model.eval()
        test_correct = 0
        test_total = 0
        vote_correct = 0  # CI voting
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data_batch, label_batch in (test_loader if do_eval else ()):
                data_batch = data_batch.to(device)
                label_batch = label_batch.to(device) - (0 if args.gpu_resident else 1)

                if _amp_eval_ctx is not None:
                    with _amp_eval_ctx:
                        logits, _ = model(data_batch)
                    logits = logits.float() if torch.is_tensor(logits) else logits
                else:
                    logits, _ = model(data_batch)

                if ci_mode:
                    # Voting: majority across K heads
                    preds_k = logits.argmax(dim=2)  # (K, N)
                    vote_pred = torch.mode(preds_k, dim=0).values  # (N,)
                    test_total += label_batch.size(0)
                    test_correct += vote_pred.eq(label_batch).sum().item()

                    # Also eval vote accuracy
                    vote_correct += vote_pred.eq(label_batch).sum().item()

                    all_preds.extend(vote_pred.cpu().numpy())
                    all_labels.extend(label_batch.cpu().numpy())
                else:
                    _, predicted = logits.max(1)
                    test_total += label_batch.size(0)
                    test_correct += predicted.eq(label_batch).sum().item()

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(label_batch.cpu().numpy())

        # [极限探针] 跳过评估时 test_total=0，沿用上一轮 OA（避免除零与 best_oa 误更新）
        oa = (100. * test_correct / test_total) if test_total > 0 else (oa_hist[-1] if oa_hist else 0.0)

        # ---- 源域留出集评估 (仅选参用, 不碰目标域) ----
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data_batch, label_batch in (val_loader if do_eval else ()):
                data_batch = data_batch.to(device)
                # [FIX 2026-10-08] val_loader 已 GPU 化（与 train/test 一致），
                #   label 在 GPUResidentLoader 里已经是 0-indexed ⇒ 不能再减 1。
                #   两种 loader 统一按「是否已减过」判断，避免静默错位。
                if getattr(val_loader, 'labels', None) is None:
                    label_batch = label_batch.to(device) - 1      # 普通 DataLoader: 1-indexed
                else:
                    label_batch = label_batch.to(device)          # GPUResidentLoader: 已 0-indexed
                if _amp_eval_ctx is not None:
                    with _amp_eval_ctx:
                        logits, _ = model(data_batch)
                    logits = logits.float() if torch.is_tensor(logits) else logits
                else:
                    logits, _ = model(data_batch)
                if ci_mode:
                    preds_k = logits.argmax(dim=2)
                    pred = torch.mode(preds_k, dim=0).values
                else:
                    _, pred = logits.max(1)
                val_total += label_batch.size(0)
                val_correct += pred.eq(label_batch).sum().item()
        val_oa = 100. * val_correct / val_total if val_total > 0 else 0.0

        # ---- [M2 2026-10-08] 幅度自适应: 按留出集 val_oa 线性映射到 [lo, hi] ----
        #   conf 模式: 用 val_oa 直接当"模型当前有多确定"的代理(越高越确定 ⇒ 可加大扰动)。
        #   下一 epoch 生效(本 epoch 已在前面用过 alpha)。
        if _alpha_adapt_on:
            _lo_a = float(getattr(args, 'sisa_alpha_adapt_lo', 0.5))
            _hi_a = float(getattr(args, 'sisa_alpha_adapt_hi', 2.0))
            _mode_a = str(getattr(args, 'sisa_alpha_adaptive', 'off'))
            if _mode_a == 'conf':
                _frac = max(0.0, min(1.0, (val_oa - 50.0) / 50.0))
            else:  # loss 模式: 用 (100 - val_oa) 当损失代理 ⇒ 越差扰动越小
                _frac = max(0.0, min(1.0, val_oa / 100.0))
            _a_new = _lo_a + (_hi_a - _lo_a) * _frac
            _adapt_alpha_state['v'] = float(_a_new)
            _adapt_alpha_state['hist'].append(
                {'epoch': int(epoch), 'val_oa': float(val_oa), 'alpha': float(_a_new)})
            print(f"[seed={seed}] [M2] epoch {epoch}: val_oa={val_oa:.2f} "
                  f"=> alpha({_mode_a})={_a_new:.4f}", flush=True)

        # Per-class AA
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        per_class = np.zeros(n_classes)
        for c in range(n_classes):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class[c] = 100. * (all_preds[mask] == c).sum() / mask.sum()
        aa = per_class.mean()

        if oa > best_oa:
            best_oa = oa
            best_epoch = epoch
            best_aa = aa
            best_val_oa = val_oa
            _m = model.base if ci_mode else model
            best_state = {k: v.detach().clone() for k, v in _m.state_dict().items()}

        oa_hist.append(oa)

        # === NSD: 后期权重平均（SWA）===
        if args.nsd and epoch >= args.nsd_start:
            _m = model.base if ci_mode else model
            sd = _m.state_dict()
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in sd.items()}
                swa_n = 1
            else:
                swa_n += 1
                for k in swa_state:
                    swa_state[k] = swa_state[k] + (sd[k].detach() - swa_state[k]) / swa_n

        # 日志
        if epoch == 1 or epoch % args.verbose_interval == 0 or epoch == args.epochs:
            elapsed = time.time() - t0
            eta = elapsed / epoch * (args.epochs - epoch) if epoch > 0 else 0
            r_str = f" r={r_t:.4f}" if gbtc_mode else ""
            ci_str = f" vote={100.*vote_correct/test_total:.2f}%" if ci_mode and vote_correct > 0 else ""
            print(f"ep {epoch:4d}/{args.epochs} | lr={current_lr:.4f} "
                  f"loss={train_loss/train_total:.4f} acc={train_acc:.2f}% | "
                  f"OA={oa:.2f}% AA={aa:.2f}%{ci_str} | "
                  f"best={best_oa:.2f}%@{best_epoch}{r_str} | "
                  f"{elapsed:.0f}s ETA={eta:.0f}s")

    total_time = time.time() - t0

    # ---- NSD: 评估（不依赖目标标签选点，仅测量）----
    if args.nsd and args.nsd_mode == 'swa' and swa_state is not None:
        _m = model.base if ci_mode else model
        _m.load_state_dict(swa_state)
        _m.eval()
        nc, nt, vp, ap, al = 0, 0, 0, [], []
        with torch.no_grad():
            for db, lb in test_loader:
                db = db.to(device)
                lb = lb.to(device) - (0 if args.gpu_resident else 1)
                if _amp_eval_ctx is not None:
                    with _amp_eval_ctx:
                        lg, _ = _m(db)
                    lg = lg.float() if torch.is_tensor(lg) else lg
                else:
                    lg, _ = _m(db)
                if ci_mode:
                    pr = torch.mode(lg.argmax(dim=2), dim=0).values
                    nc += pr.eq(lb).sum().item()
                else:
                    _, pr = lg.max(1)
                    nc += pr.eq(lb).sum().item()
                nt += lb.size(0)
                vp += pr.eq(lb).sum().item()
                ap.extend(pr.cpu().numpy())
                al.extend(lb.cpu().numpy())
        nsd_oa = 100. * nc / nt
        ap = np.array(ap); al = np.array(al)
        _pc = np.zeros(n_classes)
        for c in range(n_classes):
            mk = al == c
            if mk.sum() > 0:
                _pc[c] = 100. * (ap[mk] == c).sum() / mk.sum()
        nsd_aa = _pc.mean()
        print(f"[NSD] swa_n={swa_n} nsd_OA={nsd_oa:.2f}% nsd_AA={nsd_aa:.2f}% "
              f"(best={best_oa:.2f}%@{best_epoch} final from loop below)")

    # freeze / lrd: NSD 稳定模型 = 末轮模型（无需 oracle 选点）
    if args.nsd and args.nsd_mode in ('freeze', 'lrd'):
        nsd_oa = oa
        nsd_aa = aa
        print(f"[NSD:{args.nsd_mode}] nsd_OA={nsd_oa:.2f}% nsd_AA={nsd_aa:.2f}% "
              f"(best={best_oa:.2f}%@{best_epoch} final={oa:.2f}%)")

    # ---- 保存结果 ----
    os.makedirs(args.output_dir, exist_ok=True)

    # Build run_id with method tags
    tags = []
    # Projection identity
    proj_tag = args.proj_init if args.proj_init != 'VCA' else ''
    k_tag = f'k{args.pca_k if (args.proj_init in ("pca","PCA") and args.pca_k) else (args.vca_k or "default")}'
    if proj_tag:
        tags.append(f'{proj_tag}{k_tag}')
    elif args.proj_init == 'VCA':
        tags.append(f'VCA{k_tag}')
    # Regularization
    if ci_mode:
        tags.append(f'CI{args.ci_heads}')
    if gbtc_mode:
        tags.append(f'GBTC{args.gbtc_lambda}')
    if cp_mode:
        tags.append(f'CP{args.cp_epsilon}')
    if hgr_mode:
        tags.append(f'HGRm{args.hgr_fixed_m}' if args.hgr_fixed_m > 0
                    else f'HGRr{args.hgr_target}')
    if mttc_mode:
        tags.append(f'MTTC{args.mttc_beta}t{args.mttc_tau}')
    if acp_mode:
        tags.append(f'ACP{args.acp_h_target}l{args.acp_lambda}')
    if gd_mode:
        tags.append(f'GD{args.gd_alpha}')
    if args.nsd:
        tags.append(f'NSD-{args.nsd_mode}{args.nsd_start}')
    # Augmentation
    if hasattr(args, 'aug_type') and args.aug_type and args.aug_type != 'none':
        tags.append(f'{args.aug_type}{args.aug_param}')
    # Classifier LR (A2 marker)
    if args.classifier_lr is not None:
        tags.append(f'A2')
    # SISA / SDC (DSSP) markers — MUST be in run_id, else base/DSSP json overwrite each other
    if args.sisa_lambda > 0:
        tags.append(f'SISA{args.sisa_lambda}a{args.sisa_alpha}')
    if args.sdc_lambda > 0:
        tags.append(f'SDC{args.sdc_lambda}{args.sdc_dir}')
    # M1/LR-DRO marker — MUST be in run_id, else rho/λ 变体的 json 会互相覆盖
    if getattr(args, 'lrdro_lambda', 0.0) > 0:
        tags.append(f'LRDRO{args.lrdro_lambda}r{args.lrdro_rho}')
    if getattr(args, 'lrdro_feat_lambda', 0.0) > 0:
        tags.append(f'LRDROF{args.lrdro_feat_lambda}r{args.lrdro_feat_rho}')
    if args.sisa_mult > 0:
        tags.append(f'MULT{args.sisa_mult}')
    if args.sisa_path > 0:
        tags.append(f'PATH{args.sisa_path}')
    if args.sisa_calibrate:
        tags.append(f'CAL{args.sisa_target_rel}')
    if args.sisa_classwise:
        tags.append('CLS')
    if abs(float(getattr(args, 'sisa_sigma_gamma', 1.0)) - 1.0) >= 1e-12:
        tags.append('SG%g' % float(args.sisa_sigma_gamma))
    if getattr(args, 'e_true_as_sigma', False):
        tags.append('D1etrue')
    if getattr(args, 'sigma_override', '') and getattr(args, 'sigma_override_key', ''):
        tags.append('D3%s' % getattr(args, 'sigma_override_key', ''))
    if args.remove_shift:
        tags.append(f'RMVSHIFT')
    method_str = '_'.join(tags) if tags else 'VCA_Res2DCNN'
    run_id = f"{args.source}2{args.target}_{method_str}{'_' + args.run_tag if args.run_tag else ''}_s{seed}"

    model_p = model.base if ci_mode else model
    model_path = os.path.join(args.output_dir, f"{run_id}_best.pth")
    torch.save(best_state if best_state is not None else model_p.state_dict(), model_path)

    result = {
        "run_id": run_id,
        "model": "VCA_Res2DCNN",
        "mode": method_str,
        "config": {
            "source": args.source,
            "target": args.target,
            "params_M": round(n_params / 1e6, 6),
            "lr": args.lr,
            "classifier_lr": clr,
            "wd": args.wd,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "re_ratio": args.re_ratio,
            "patch_size": args.patch_size,
            "seed": seed,
            "bands": n_bands,
            "classes": n_classes,
            "k_endmembers": k_endmembers,
            "src_pixels": int(np.count_nonzero(gt_src)),
            "tgt_pixels": int(np.count_nonzero(gt_tgt)),
            "optimizer": "Adam (separate lr)" if clr != base_lr else "Adam",
            "lr_schedule": "flat" if args.no_lr_decay else "pow(1+10*(ep-1)/200, -0.75)",
            "ci_heads": args.ci_heads if ci_mode else 1,
            "gbtc_lambda": args.gbtc_lambda,
            "cp_epsilon": args.cp_epsilon,
            "mttc_beta": args.mttc_beta,
            "mttc_tau": args.mttc_tau,
            "acp_h_target": args.acp_h_target,
            "acp_lambda": args.acp_lambda,
            "gd_alpha": args.gd_alpha,
            # ---- 精度（2026-09-14 起记录，便于跨精度比较时甄别）----
            "precision": ("amp_full" if _amp_full else ("amp_fwd" if _amp else "default")),
            "amp_eval": bool(_amp_eval),
            "strict_fp32": bool(_strict_fp32),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            # ---- 不变性轴（2026-09-14 起记录：A3/A5 对照臂靠它甄别）----
            "sdc_dir": str(getattr(args, 'sdc_dir', 'shift')),
            # S2 剂量-反应（2026-09-15）：noclass 只投影掉 V_d 的前 k 维；None=全量
            "noclass_k": getattr(args, 'noclass_k', None),
            # 🔴 可审计性（2026-09-15 补）：这一条是"同 α = 同扰动预算"（R3）的**前提开关**。
            # 旧版本没记它 ⇒ 若哪天误开 True，跨轴逐峰比较会静默失效而无人察觉。
            "sisa_calibrate": bool(getattr(args, 'sisa_calibrate', False)),
            "sisa_classwise": bool(getattr(args, 'sisa_classwise', False)),
            "sisa_sigma_gamma": float(getattr(args, 'sisa_sigma_gamma', 1.0)),
            # [DIAG 2026-09-19] 解耦开关必须落盘，否则 classwise 与 classwise+global_delta
            # 两个臂在产物里无法区分（config 是事后归因的唯一依据）。
            "sisa_global_delta": bool(getattr(args, 'sisa_global_delta', False)),
            "sisa_alpha": args.sisa_alpha,
            "sisa_rank": args.sisa_rank,
            "sisa_blocks": args.sisa_blocks,
        },
        "best_OA": round(best_oa / 100., 4),
        "best_AA": round(best_aa / 100., 4),
        "src_val_OA": round(best_val_oa / 100., 4),
        "best_epoch": best_epoch,
        "mean_stable_OA": (round(float(np.mean(oa_hist[49:])) / 100., 4) if len(oa_hist) >= 50 else None),
        "m_last100_OA": (round(float(np.mean(oa_hist[-100:])) / 100., 4) if len(oa_hist) > 0 else None),
        "oa_hist": [round(float(v), 3) for v in oa_hist],
        "final_OA": round(oa / 100., 4),
        "final_AA": round(aa / 100., 4),
        "nsd_OA": round(nsd_oa / 100., 4) if nsd_oa is not None else None,
        "nsd_AA": round(nsd_aa / 100., 4) if nsd_aa is not None else None,
        "nsd_n": swa_n,
        "total_time_s": round(total_time, 2),
        "hgr": ({
            "enabled": True,
            "fixed_m": args.hgr_fixed_m if args.hgr_fixed_m > 0 else None,
            "r_target": hgr.r_target,
            "m_final": round(hgr.m, 6),
            "cls_lr_final": round(optimizer.param_groups[1]['lr'], 8),
            "r_ema_final": round(hgr.r_ema, 4) if hgr.r_ema else None,
            "r_mean": round(sum(hgr.r_hist) / max(len(hgr.r_hist), 1), 4),
            # 每 epoch 一个采样点，用于画 r 曲线（步数太多，降采样）
            "r_per_epoch": [round(v, 4) for v in
                            hgr.r_hist[::max(1, len(hgr.r_hist) // args.epochs)]][:args.epochs],
            "m_traj": hgr.m_hist[:200],
            "n_adjust": len(hgr.m_hist),
        } if hgr is not None else {"enabled": False}),
        "dssd": ({
            "enabled": True,
            "period": _dssd_period,
            "warmup": _dssd_warmup,
            "mix": _dssd_mix,
            "nmax": _dssd_nmax,
            "n_reestimate": len(_dssd_log),
            "drift_log": _dssd_log,
        } if _dssd_on else {"enabled": False}),
        # [M2 2026-10-08] 幅度自适应轨迹（C3 用）。关闭时只写 {"enabled": False}，
        # 保证历史 run 的 JSON 结构仅多一个恒等字段、不影响既有读数。
        "alpha_adapt": ({
            "enabled": True,
            "mode": args.sisa_alpha_adaptive,
            "lo": args.sisa_alpha_adapt_lo,
            "hi": args.sisa_alpha_adapt_hi,
            "alpha_final": _adapt_alpha_state['v'],
            "hist": _adapt_alpha_state['hist'],
        } if _alpha_adapt_on else {"enabled": False}),
        # [M3 2026-10-08] 逐方向幅度 σ 在线重估轨迹（C4 用）。
        "sigma_adapt": ({
            "enabled": True,
            "period": _sig_adapt_period,
            "warmup": _sig_adapt_warmup,
            "lambda": float(args.sisa_sigma_adapt_lambda),
            "floor": float(args.sisa_sigma_adapt_floor),
            "ceil": float(args.sisa_sigma_adapt_ceil),
            "factor_final": float(_SIG_ADAPT_STATE.factor),
            "n_reestimate": len(_SIG_ADAPT_STATE.hist),
            "hist": _SIG_ADAPT_STATE.hist,
        } if _sig_adapt_on else {"enabled": False}),
    }

    json_path = os.path.join(args.output_dir, f"{run_id}_results.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    if _dssd_on and _dssd_log:
        _ac = [d['mean_abs_cos'] for d in _dssd_log if d['mean_abs_cos'] == d['mean_abs_cos']]
        _an = [d['max_principal_angle_deg'] for d in _dssd_log
               if d['max_principal_angle_deg'] == d['max_principal_angle_deg']]
        _sh = [d['captured_energy_share'] for d in _dssd_log
               if d['captured_energy_share'] == d['captured_energy_share']]
        _side = dict(source=args.source, target=args.target, seed=int(seed),
                     period=_dssd_period, warmup=_dssd_warmup, mix=_dssd_mix,
                     n_reestimate=len(_dssd_log),
                     mean_abs_cos=float(np.mean(_ac)) if _ac else None,
                     max_principal_angle_deg=float(np.max(_an)) if _an else None,
                     captured_energy_share_min=float(np.min(_sh)) if _sh else None,
                     drift_log=_dssd_log)
        _side_path = os.path.join(args.output_dir, f"{run_id}_dssd_drift.json")
        with open(_side_path, 'w') as f:
            json.dump(_side, f, indent=2)
        print(f"[seed={seed}] [DSSP-D] 重估 {len(_dssd_log)} 次 | "
              f"mean|cos|={(_side['mean_abs_cos'] or float('nan')):.4f} "
              f"max_angle={(_side['max_principal_angle_deg'] or float('nan')):.2f}deg | "
              f"drift → {os.path.basename(_side_path)}")

    _ms = (float(np.mean(oa_hist[49:])) if len(oa_hist) >= 50 else float('nan'))
    _ns = (float(nsd_oa) if nsd_oa is not None else float('nan'))
    print(f"[seed={seed}] Done! best_OA={best_oa:.2f}% @ep{best_epoch}"
          f" | final={oa:.2f}% | mean_stable={_ms:.2f}% | nsd={_ns:.2f}%"
          f" | time={total_time:.0f}s")
    print(f"[seed={seed}] Results saved to {json_path}")
    return result


# ============================================================
# CLI
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description='VCA_Res2DCNN Training (M1/A2/CI/CP/GBTC/MTTC)')

    parser.add_argument('--source', type=str, required=True,
                        help='Source domain (e.g. Houston13, paviaU)')
    parser.add_argument('--target', type=str, required=True,
                        help='Target domain (e.g. Houston18, paviaC)')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Path to dataset .mat files')

    # Training
    parser.add_argument('--lr', type=float, default=0.01, help='Backbone learning rate')
    parser.add_argument('--classifier_lr', type=float, default=None,
                        help='Classifier head lr (None = same as --lr)')
    parser.add_argument('--wd', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size for eval/test/val passes. None = same as '
                             '--batch_size. RESULTS ARE IDENTICAL for any value: '
                             'in eval mode BatchNorm uses running statistics and '
                             'dropout is off, so predictions are per-sample '
                             'independent. Only throughput changes.')
    parser.add_argument('--patch_size', type=int, default=13)
    parser.add_argument('--num_workers', type=int, default=14)
    parser.add_argument('--re_ratio', type=int, default=5)
    parser.add_argument('--no_lr_decay', action='store_true', default=False,
                        help='Use flat LR instead of decay schedule')

    # Seeds
    parser.add_argument('--seed', type=int, default=3667, help='Single seed')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds')

    # VCA
    parser.add_argument('--vca_k', type=int, default=None,
                        help='VCA endmember count (default: auto based on n_bands)')
    parser.add_argument('--subsample', type=int, default=None,
                        help='Subsample N source pixels for low-data diagnosis')
    parser.add_argument('--pca_k', type=int, default=None,
                        help='Number of PCA components (for mix projection)')
    parser.add_argument('--no_vca', action='store_true', default=False,
                        help='Disable VCA endmember init, use Kaiming normal instead')
    parser.add_argument('--proj_init', type=str, default='VCA',
                        choices=['VCA', 'PCA', 'PCA_bottom', 'VCA_norm', 'VCA_orth', 'RandomOrth', 'Random', 'Identity', 'Unified', 'centroid', 'SDP'],
                        help='Projection init method')
    # === SDP: Shift-suppressed Discriminative Projection ===
    parser.add_argument('--sdp_tau', type=float, default=1.0,
                        help='SDP: weight on the source-estimated shift scatter in the denominator')
    parser.add_argument('--sdp_gamma', type=float, default=1.0,
                        help='SDP: weight on the within-class scatter in the denominator')
    parser.add_argument('--sdp_blocks', type=int, default=8,
                        help='SDP: spatial block grid used to estimate the shift scatter')
    parser.add_argument('--sdp_delta', type=float, default=0.05,
                        help='SDP: weight of total scatter in the numerator (fills directions beyond rank(S_b)=C-1)')
    parser.add_argument('--sdp_no_ortho', action='store_true', default=False,
                        help='SDP: skip QR orthonormalisation of the projection rows')
    parser.add_argument('--proj_frozen', action='store_true', default=True,
                        help='Freeze projection layer weights (default: True)')
    parser.add_argument('--proj_learnable', dest='proj_frozen', action='store_false',
                        help='Make projection layer learnable')
    parser.add_argument('--proj_ortho_lambda', type=float, default=0.0,
                        help='Soft orthogonal regularization on projection rows: lambda*||W W^T - I||_F')
    parser.add_argument('--unified_r0', type=float, default=None,
                        help='Unified projection R0 calibration (enables Unified init, forces frozen)')
    parser.add_argument('--res_channels', type=str, default='32,64,128',
                        help='ResBlock channels, comma-separated (default: 32,64,128)')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results/')
    parser.add_argument('--verbose_interval', type=int, default=5)

    # === New methods ===
    # CI: Classifier Integration
    parser.add_argument('--ci_heads', type=int, default=1,
                        help='Number of classifier heads (1 = disabled, 3 or 5 for CI)')

    # CP: Classifier Perturbation
    parser.add_argument('--cp_epsilon', type=float, default=0.0,
                        help='Classifier perturbation epsilon (0 = disabled)')

    # HGR: Head-Backbone Gradient Ratio 闭环控制
    parser.add_argument('--hgr_target', type=float, default=0.0,
                        help='HGR: target per-parameter RMS gradient ratio rho_bb/rho_cls. '
                             '>0 enables closed-loop controller. 1.0 = A2-observed optimum.')
    parser.add_argument('--hgr_gain', type=float, default=0.3,
                        help='HGR: proportional gain of the multiplicative update')
    parser.add_argument('--hgr_every', type=int, default=25,
                        help='HGR: adjust classifier lr multiplier every N steps')
    parser.add_argument('--hgr_warmup', type=int, default=100,
                        help='HGR: do not adjust before this many steps')
    parser.add_argument('--hgr_fixed_m', type=float, default=0.0,
                        help='HGR: disable controller, fix classifier lr multiplier to this '
                             'value (P4 dose-response scan). 0 = use controller.')

    # GBTC: Gradient Balance
    parser.add_argument('--gbtc_lambda', type=float, default=0.0,
                        help='GBTC lambda (0 = disabled)')

    # MTTC: Momentum Temporal Target
    parser.add_argument('--mttc_beta', type=float, default=0.0,
                        help='MTTC repulsion beta (0 = disabled)')
    parser.add_argument('--mttc_tau', type=int, default=50,
                        help='MTTC snapshot interval in epochs')
    parser.add_argument('--gpu_aug', type=int, default=1,
                        help='GPU batch-level augmentation (1=enabled, 0=disabled)')
    parser.add_argument('--gpu_resident', type=int, default=0,
                        help='Preload all patches to GPU, bypass DataLoader')

    # ACP: Adaptive Confidence Penalty
    parser.add_argument('--acp_h_target', type=float, default=0.0,
                        help='ACP entropy target (0 = disabled)')
    parser.add_argument('--acp_lambda', type=float, default=0.0,
                        help='ACP penalty weight')

    # GD: Gradient Decorrelation
    parser.add_argument('--gd_alpha', type=float, default=0.0,
                        help='GD projection strength (0 = disabled)')
    # ---- NSD: Null-Space Drift suppression ----
    parser.add_argument('--nsd', action='store_true', default=False,
                        help='Enable NSD: post-saturation weight averaging (SWA) to '
                             'stabilize cross-domain OA without target-label epoch selection')
    parser.add_argument('--nsd_start', type=int, default=100,
                        help='NSD: start from this epoch (after source saturation)')
    parser.add_argument('--nsd_mode', type=str, default='swa',
                        choices=['swa', 'freeze', 'lrd'],
                        help='NSD stabilizer: swa=post-saturation weight avg (failed), '
                             'freeze=stop backbone updates after nsd_start, '
                             'lrd=decay backbone lr by nsd_decay after nsd_start')
    parser.add_argument('--nsd_decay', type=float, default=0.1,
                        help='NSD lrd mode: backbone lr *= this factor after nsd_start')
    parser.add_argument('--l_con', type=float, default=0.0,
                        help='Dual-view consistency loss weight (0 = disabled)')
    parser.add_argument('--con_aug', type=str, default='chdrop',
                        help='Perturbation for consistency view (chdrop/none)')
    parser.add_argument('--con_aug_param', type=float, default=0.1,
                        help='Param for consistency perturbation (e.g., chdrop p)')
    parser.add_argument('--aug_type', type=str, default='none',
                        choices=['none', 'shuffle', 'cutmix', 'erase', 'chdrop'],
                        help='Abundance map augmentation type')
    parser.add_argument('--aug_param', type=str, default=None,
                        help='Augmentation parameter')
    # === CDSP: Cross-Domain Spectral Projection consistency ===
    parser.add_argument('--cdsp_lambda', type=float, default=0.0,
                        help='CDSP: weight for cross-domain spectral-shift consistency classification loss (0=disabled)')
    parser.add_argument('--cdsp_strength', type=float, default=0.1,
                        help='CDSP: spectral shift augmentation strength')
    parser.add_argument('--cdsp_shift', type=str, default='spec',
                        choices=['spec', 'none'],
                        help='CDSP: spectral shift operator set (spec = radiometric+band-shift+noise)')
    # === SISA: Source-Internal Shift-subspace Augmentation ===
    parser.add_argument('--remove_shift', action='store_true', default=False,
                        help='A4 baseline: confine the spectral projection to the ORTHOGONAL COMPLEMENT of the estimated shift subspace S and freeze it (removes S, like domain-alignment methods do)')
    parser.add_argument('--sisa_lambda', type=float, default=0.5,
                        help='SISA: weight of the shift-subspace consistency classification loss (0=disabled)')
    parser.add_argument('--sisa_alpha', type=float, default=1.0,
                        help='SISA: scale multiplier on the estimated shift magnitude sigma')
    parser.add_argument('--sisa_mult', type=float, default=0.0,
                        help='U0: multiplicative (illumination/geometry) shift strength; 0 = legacy additive')
    parser.add_argument('--xg_mode', type=str, default='off',
                        choices=['off', 'coupled', 'perm', 'anti'],
                        help='[XG] 偏移子空间 x 图像(光谱)空间 实例级交互指导')
    parser.add_argument('--xg_tau', type=float, default=1.0,
                        help='[XG] 相关性锐度: g = |p|^tau 后归一化到均值1')
    parser.add_argument('--xg_beta', type=float, default=1.0,
                        help='[XG] 与平坦 g=1 的混合强度(混合后重新归一化, 保持总能量)')
    parser.add_argument('--sisa_path', type=float, default=0.0,
                        help='U0: additive path-radiation (atmospheric) strength; 0 = disabled')
    parser.add_argument('--sisa_rank', type=int, default=5,
                        help='SISA: number of shift directions kept from the source-internal SVD')
    parser.add_argument('--sisa_blocks', type=int, default=8,
                        help='SISA: spatial block grid size used to estimate the shift subspace')
    parser.add_argument('--sisa_min_px', type=int, default=10,
                        help='SISA: minimum number of pixels of a class inside a block (m in the paper)')
    parser.add_argument('--sisa_hetero', action='store_true', default=False,
                        help='SISA: heterogeneous (per-block) shift directions instead of one global subspace (for Pavia-style heterogeneous shift)')
    parser.add_argument('--vuln_probe_ckpt', type=str, default=None,
                        help='[STRETCH] 脆弱性探针：加载该 ckpt，算 w_i=E|v_i^T grad_x L| 后存 json 并退出（不训练）。')
    parser.add_argument('--vuln_probe_out', type=str, default=None,
                        help='[STRETCH] 脆弱性探针输出 json 路径。')
    parser.add_argument('--sisa_sigma_gamma', type=float, default=1.0,
                        help='[STRETCH] 重塑 SISA 度量形状: sigma_i -> sigma_i^gamma, '
                             '并重新标定使 sum(sigma^2) 不变（预算匹配）。'
                             '1.0=现役(恒等, 默认), 0.0=集合内各向同性(最大熵先验)。')
    parser.add_argument('--e_true_as_sigma', action='store_true', default=False,
                        help='[STRETCH-D1] 用 oracle 真实方向 e_true（来自 tools/_e1_stretch_shape.json）'
                             '作为 SISA Λ 的形状，预算匹配(Σσ² 不变)使 α 不变。D1 收口实验:'
                             '若"对准真实方向"能赢，则方向有意义；否则方向可有可无。')
    parser.add_argument('--sigma_override', type=str, default='',
                        help='[STRETCH-D3] 任意信号作 Λ 形状: 指向 tools/_d3_signals.json 这类文件, '
                             '内含 {tag:{seeds:{key:[5-vector]}}}; 用其中某一信号(见 --sigma_override_key) '
                             '重塑 sigma 形状, 预算匹配(Σσ² 不变)使 α 不变。用于对比「偏移信号来自哪里」。')
    parser.add_argument('--sigma_override_key', type=str, default='',
                        help='[STRETCH-D3] 选用信号: e_true | w | s | sigma_ref')
    parser.add_argument('--sigma_override_seed', type=str, default='333',
                        help='[STRETCH-D3] json 内 seed 键: 333 | 777')
    parser.add_argument('--sisa_classwise', action='store_true', default=False,
                        help='SISA: class-conditional shift directions (one per class) instead of one '
                             'global subspace. Motivated by measured inter-class direction disagreement '
                             '(pairwise |cos| 0.47-0.78 vs 1.0). More sample-efficient than --sisa_hetero: '
                             'a class has thousands of pixels, a spatial block only hundreds.')
    # === S4/C1 跨城可移植性（2026-09-19 新增）===
    parser.add_argument('--sisa_shift_pkl', type=str, default=None,
                        help='[S4] 从 pkl 载入 (V, sigma) 替代 estimate_shift_subspace 的输出，'
                             '用于检验「用别的城估的 S 训本城」是否还灵。缺省 None ⇒ 与历史逐位一致。'
                             'pkl 需含键 V (r,B) 与 sigma (r,)；波段/正交性/形状 fail-closed 校验。'
                             '与 --sisa_hetero / --sisa_classwise 互斥。')
    # === 解耦诊断开关（2026-09-19 新增，仅用于归因，不进论文方法）===
    parser.add_argument('--sisa_global_delta', action='store_true', default=False,
                        help='[DIAG] 诊断专用：与 --sisa_classwise 同用时，子空间仍逐类（V_cls 照常估计），'
                             '但增广的 delta 采样退回【全局】sigma，即抹掉"类 c 所有 patch 共享同一 delta 的'
                             '幅度尺度"这一副作用。用于把 P1\' (逐类 82.72) 相对全局臂的差距拆成'
                             '"逐类子空间"vs"delta 类绑定"两部分。默认 False ⇒ 与历史读数逐位可比。')
    # === DSSP-D (2026-09-21): 动态偏移子空间 ===
    # 动机：偏移子空间目前训练前估一次即冻结（estimate_shift_subspace 从原始像素异质性出发）。
    # 但"哪个方向对当前模型的决策最敏感"是随训练变化的 —— 方向本身不可知，
    # 我们能优化的只有"子空间"这一件事。DSSP-D 在源域上按当前模型的输入梯度二阶矩
    # 周期重估 V（方向不可知 / 不碰目标域），σ 与 α 保持冻结 ⇒ 扰动预算逐位不变（R3）。
    parser.add_argument('--dssd_period', type=int, default=0,
                        help='DSSP-D: 每 N 个 epoch 用当前模型的决策敏感度重估偏移子空间；'
                             '0=关闭（默认，行为与历史逐位一致）。')
    parser.add_argument('--dssd_warmup', type=int, default=0,
                        help='DSSP-D: 在此 epoch 之前不重估（等模型先获得基本判别力）。')
    parser.add_argument('--dssd_mix', type=float, default=1.0,
                        help='DSSP-D: 新旧子空间插值系数，1.0=完全替换（默认），'
                             '0.0=冻结基线，中间=加权 Gram 的 top-r 主子空间（仍行正交归一）。')
    parser.add_argument('--dssd_nmax', type=int, default=2048,
                        help='DSSP-D: 估计敏感度子空间时最多用多少个源域 patch（开销闸）。')
    # === [M3 2026-10-08] 逐方向幅度 σ 在线重估（补充实验 C4）===
    # 与 --dssd_period 的区别：那个换【方向 V】(取向, σ/α 冻结, 预算逐位不变)；
    # 这个换【逐方向幅度乘子 σ】(V 不动, 预算随之改变)。
    # 按源域留出集特征沿每个 V 行的"自然残余能量"决定补多少：
    #   ratio_j = clip(1 - lambda * self_j / mean(self), floor, ceil)
    parser.add_argument('--sisa_sigma_adaptive', action='store_true', default=False,
                        help='[M3] 开启逐方向幅度 σ 在线重估（默认 False ⇒ 与历史逐位一致）。')
    parser.add_argument('--sisa_sigma_adapt_period', type=int, default=20,
                        help='[M3] 每 N 个 epoch 重估一次 σ 幅度。')
    parser.add_argument('--sisa_sigma_adapt_warmup', type=int, default=0,
                        help='[M3] 在此 epoch 之前不重估。')
    parser.add_argument('--sisa_sigma_adapt_lambda', type=float, default=1.0,
                        help='[M3] "减掉自然残余"的强度；0=不修正(等价关闭)，1=全额减。可正可负。')
    parser.add_argument('--sisa_sigma_adapt_floor', type=float, default=0.25,
                        help='[M3] 逐方向修正因子的下界（防某个方向被压到 0）。')
    parser.add_argument('--sisa_sigma_adapt_ceil', type=float, default=2.0,
                        help='[M3] 逐方向修正因子的上界。')
    parser.add_argument('--sisa_sigma_adapt_nmax', type=int, default=2048,
                        help='[M3] 估计自然残余能量时最多用多少个源域 patch（开销闸）。')
    # === DSSP: Discriminative Shift-Subspace Projection ===
    parser.add_argument('--dssp', action='store_true', default=False,
                        help='DSSP: enable discriminative Fisher regularization on the spectral projection')
    parser.add_argument('--sdc_lambda', type=float, default=1.0,
                        help='SDC: weight of the supervised shift-contrastive loss (0=disabled)')
    parser.add_argument('--sdc_temp', type=float, default=0.1,
                        help='SDC: contrastive temperature')
    parser.add_argument('--sdc_max_n', type=int, default=1024,
                        help='SDC: subsample cap on anchors (memory guard for large batches)')
    parser.add_argument('--sisa_mask_mode', type=str, default='none',
                        choices=['none', 'center', 'ring', 'mix', 'incons'],
                        help='[空间掩码 2026-09-21] none=生产口径(偏移空间一致); '
                             'center=只扰中心区; ring=只扰外环; '
                             'mix=逐样本随机取{全图,中心,外环}之一; '
                             'incons=全图覆盖但中心/外环各抽一次独立 delta(空间不一致)')
    parser.add_argument('--sisa_mask_radius', type=int, default=2,
                        help='中心区切比雪夫半径: radius=2 -> 中心 5x5, 其余为外环')
    parser.add_argument('--sisa_mask_energy', type=int, default=1,
                        help='1=按 1/sqrt(覆盖比例) 缩放，使扰动 Frobenius 能量与全图一致 (R3)')
    parser.add_argument('--sdc_dir', type=str, default='shift',
                        choices=['shift', 'random', 'smooth', 'pca', 'cap', 'noclass', 'ms1',
                                 'intra', 'disc', 'orth'],
                        help='SDC/A3-A5: shift=estimated source-internal shift subspace; '
                             'random=RANDOM orthonormal basis of same rank (white-noise control); '
                             'smooth=spectrally-smoothed random basis (realistic but data-free); '
                             'pca=top-r PCA of labelled source pixels (leading-data-subspace control); '
                             'cap=classwise only, shuffle the class->direction assignment; '
                             'noclass=S2 CANDIDATE AXIS (not a control): S with its '
                             'class-discriminative component (source class-mean row space) '
                             'projected out; rank/shape kept, sigma NOT recalibrated; '
                             'disc=E1 CANDIDATE AXIS (dual of noclass, not a control): S with '
                             'ONLY its class-discriminative component kept (P_Vd v per row); '
                             'rank/shape kept, sigma NOT recalibrated; '
                             'orth=E2 structured control: random orthonormal basis of same rank '
                             'constrained to be orthogonal to the estimated shift subspace '
                             'V-hat (the project-wide "orthogonal control" discipline)')
    parser.add_argument('--noclass_k', type=int, default=None,
                        help='S2 dose-response: with --sdc_dir noclass, project out only the '
                             'top-k directions of V_d (default None = remove all of V_d). '
                             'Irrelevant for every other --sdc_dir.')
    parser.add_argument('--disc_k', type=int, default=None,
                        help='E1 dose-response: with --sdc_dir disc, keep only the top-k '
                             'directions of V_d (default None = keep all of V_d). '
                             'Irrelevant for every other --sdc_dir.')
    parser.add_argument('--dssp_lambda', type=float, default=0.0,
                        help='DSSP: weight of the Fisher scatter term (0=disabled)')
    parser.add_argument('--sisa_calibrate', action='store_true', default=False,
                        help='DSSP/SISA: cap the shift-subspace perturbation relative magnitude at --sisa_target_rel')
    # === [M1 2026-10-08] 关增广但保留损失项（消融格子）===
    parser.add_argument('--sisa_no_augment', action='store_true', default=False,
                        help='[消融 M1] 保留 Ŝ 估计与 SISA/SDC 损失项，但把 sisa_augment 换成恒等映射'
                             '（x_sisa = x）。用于把「增广」与「损失项」解耦成独立格子。'
                             '默认 False ⇒ 与历史 150 runs 逐位一致。')
    # === [M2 2026-10-08] 幅度自适应（补充实验 C3）===
    parser.add_argument('--sisa_alpha_adaptive', type=str, default='off',
                        choices=['off', 'loss', 'conf'],
                        help='[M2] 幅度自适应: 每 epoch 末按源域留出集指标线性映射 alpha 到 '
                             '[--sisa_alpha_adapt_lo, --sisa_alpha_adapt_hi]。'
                             'loss=按 val_oa/100; conf=按 (val_oa-50)/50。'
                             '⚠️ 会改变扰动预算 ⇒ 每 epoch 的 alpha 轨迹存入结果 JSON。'
                             '默认 off ⇒ 与历史逐位一致。')
    parser.add_argument('--sisa_alpha_adapt_lo', type=float, default=0.5,
                        help='[M2] 幅度自适应的 alpha 下界。')
    parser.add_argument('--sisa_alpha_adapt_hi', type=float, default=2.0,
                        help='[M2] 幅度自适应的 alpha 上界。')
    # U1: Jacobian regularization
    parser.add_argument('--u1_lambda', type=float, default=0.0,
                        help='U1: Jacobian shift regularization weight (0=disabled)')
    parser.add_argument('--u1_mode', type=str, default='shift',
                        choices=['shift', 'isotropic', 'random'],
                        help='U1: shift=tr(J^T J Sigma_S), isotropic=||J||_F^2, '
                             'random=tr(J^T J Sigma_rand) control')
    # M1 / LR-DRO: rank-constrained distributionally robust optimisation
    # 不确定性集 U_S(rho) = {delta = V z : z^T Lambda^{-1} z <= rho^2}, Lambda = diag(sigma^2)
    # 一阶展开下内层 max 有闭式: delta* = rho * V Lambda V^T g / ||Lambda^{1/2} V^T g||_2
    # 已由 tools/_m1_verify_closedform.py 数值验证(相对误差 2.4e-7, 暴力搜索 0 次超越)。
    parser.add_argument('--lrdro_lambda', type=float, default=0.0,
                        help='M1/LR-DRO: weight of the worst-case loss term (0=disabled)')
    parser.add_argument('--lrdro_rho', type=float, default=1.0,
                        help='M1/LR-DRO: robustness radius rho of the uncertainty set')
    parser.add_argument('--lrdro_grad_frac', type=float, default=1.0,
                        help='M1/LR-DRO: sub-batch fraction for per-sample gradient computation '
                             '(<=1.0; lower it if VRAM is tight)')
    # M1b / LR-DRO-F: 同一个最坏情况问题，但把不确定性集放在【特征空间】
    #   做法：用网络本身把光谱方向 v_i 推到特征空间（JV），正交化后作为特征侧支撑。
    #   度量 = 物理能量 σ_i ⊗ 网络敏感性 ||J v_i||（双重拉伸）。
    #   好处：梯度只需反传到 feat（不反传卷积层），显存/时间成本远低于光谱侧
    #   ——Pavia 上光谱侧实测 18.3→21.7GB、慢 27×（显存 thrashing），特征侧可规避。
    parser.add_argument('--lrdro_feat_lambda', type=float, default=0.0,
                        help='M1b/LR-DRO-F: worst-case perturbation in FEATURE space (0=disabled)')
    parser.add_argument('--lrdro_feat_rho', type=float, default=1.0,
                        help='M1b/LR-DRO-F: robust radius in feature space')
    parser.add_argument('--lrdro_feat_eps', type=float, default=1e-2,
                        help='M1b: finite-difference step for pushing V through the network (JV)')
    parser.add_argument('--lrdro_feat_push_n', type=int, default=2048,
                        help='M1b: how many source patches to use for the JV push (subsample)')

    # CORAL / DANN baselines
    parser.add_argument('--coral_lambda', type=float, default=0.0,
                        help='CORAL: covariance alignment weight (0=disabled)')
    parser.add_argument('--dann_lambda', type=float, default=0.0,
                        help='DANN: domain adversarial weight (0=disabled)')
    parser.add_argument('--dann_alpha', type=float, default=1.0,
                        help='DANN: GRL scaling factor (schedule: alpha * (2/(1+exp(-10*p))-1))')

    # U3: alpha marginalization
    parser.add_argument('--sisa_alpha_marginalize', action='store_true', default=False,
                        help='U3: sample alpha from LogUniform each batch instead of fixed value')
    parser.add_argument('--sisa_alpha_min', type=float, default=0.05,
                        help='U3: lower bound for LogUniform alpha sampling')
    parser.add_argument('--sisa_alpha_max', type=float, default=3.0,
                        help='U3: upper bound for LogUniform alpha sampling')

    # U4: KL-SISA
    parser.add_argument('--u4_kl_sisa', action='store_true', default=False,
                        help="U4: replace SISA CE loss with KL divergence between views")
    parser.add_argument('--u4_kl_temp', type=float, default=2.0,
                        help='U4: temperature for KL divergence softening')

    parser.add_argument('--run_tag', type=str, default='', help='extra tag appended to run_id/json filename to avoid cross-config collisions')
    parser.add_argument('--sisa_target_rel', type=float, default=0.15,
                        help='DSSP/SISA: target relative perturbation magnitude (sigma/|x|) when --sisa_calibrate')

    # === 极限性能探针（默认全关，保持复现红线） ===
    parser.add_argument('--torch_compile', action='store_true', default=False,
                        help='[极限探针] 用 torch.compile 融合模型内核、削减 Python/launch 开销。FP32 不改数值，但可能带来极小末位差异。默认关。')
    parser.add_argument('--compile_mode', type=str, default='default',
                        help='[极限探针] torch.compile 模式: default / reduce-overhead / max-autotune')
    # ⚠️ 精度红线：TF32 / AMP(FP16) / BF16 默认【关闭】。
    #    历史背景：用户曾明令禁止改精度（会严重影响结果）。5090D 的剩余提速空间全在
    #    TensorCore(TF32/FP16)。当前 FP32 已跑到峰值效率的 ~73%（见 PROJECT.md §8）。
    #    >>> 2026-09-14 更新：用户授权做【有条件的】精度放宽验证 ——
    #        "可以开 AMP 跑一个最佳模型的种子，如果指标没有浮动就允许开始 AMP"。
    #        故新增 --tf32 / --amp 开关（默认全关，显式指定才生效）。
    #        验证契约：先用同一 seed 与 FP32 baseline 对照，指标漂移落在 seed 间 σ 内才放行。
    parser.add_argument('--tf32', action='store_true', default=False,
                        help='[精度探针·用户授权] 额外打开 matmul 的 TF32（cudnn 的 TF32 本来就是框架默认 on）'
                             '→ 真正意义上的"全 TF32"。参数仍 FP32 存储、无需 GradScaler、无溢出风险。默认关。')
    parser.add_argument('--strict_fp32', action='store_true', default=False,
                        help='[精度探针·诊断] 把 matmul 与 cudnn 的 TF32 全部关掉 → 真正的纯 FP32。'
                             '会明显变慢（实测 ~4x），仅用于量化"框架默认 TF32"到底贡献了多少数值/速度差。默认关。')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='[精度探针·用户授权] 模型前向用 FP16 autocast，输出立即转回 FP32，其余 loss 计算保持 FP32，'
                             '配 GradScaler 做 backward/step。数值漂移中等。默认关。')
    parser.add_argument('--amp_full', action='store_true', default=False,
                        help='[精度探针·用户授权] 把 autocast 作用域扩到整个 loss 计算块（SISA/SDC 亦在 FP16 下算）。'
                             '加速最大但漂移风险最高，需单独验证。默认关。')
    parser.add_argument('--amp_eval', action='store_true', default=False,
                        help='[精度探针·用户授权] 评估前向也用 FP16 autocast（logits 转回 FP32 后再算指标）。默认关。')
    parser.add_argument('--eval_every', type=int, default=1,
                        help='[极限探针] 每 N 个 epoch 评估一次（默认1=每轮评估，与原始一致）。调大可砍评估开销，但会改变 best-epoch 选择。')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seeds is not None:
        seed_list = [int(s.strip()) for s in args.seeds.split(',')]
    else:
        seed_list = [args.seed]

    print(f"train_einit.py: {args.source}→{args.target}, {len(seed_list)} seeds: {seed_list}")
    print(f"  lr={args.lr}, classifier_lr={args.classifier_lr}, wd={args.wd}, "
          f"epochs={args.epochs}, batch_size={args.batch_size}, re_ratio={args.re_ratio}, "
          f"no_lr_decay={args.no_lr_decay}")
    print(f"  proj_init={args.proj_init}, proj_frozen={args.proj_frozen}, "
          f"proj_ortho_lambda={args.proj_ortho_lambda}")
    if args.ci_heads > 1:
        print(f"  CI mode: K={args.ci_heads}")
    if args.cp_epsilon > 0:
        print(f"  CP mode: epsilon={args.cp_epsilon}")
    if args.gbtc_lambda > 0:
        print(f"  GBTC mode: lambda={args.gbtc_lambda}")
    if args.mttc_beta > 0:
        print(f"  MTTC mode: beta={args.mttc_beta}, tau={args.mttc_tau}")
    if args.acp_lambda > 0:
        print(f"  ACP mode: H_target={args.acp_h_target}, lambda={args.acp_lambda}")
    if args.gd_alpha > 0:
        print(f"  GD mode: alpha={args.gd_alpha}")
    if getattr(args, 'u4_kl_sisa', False):
        print(f"  U4: KL-SISA enabled, T={args.u4_kl_temp}")
    if getattr(args, 'sisa_alpha_marginalize', False):
        print(f"  U3: alpha LogUniform [{args.sisa_alpha_min}, {args.sisa_alpha_max}]")
    if args.coral_lambda > 0:
        print(f"  CORAL: lambda={args.coral_lambda}")
    if args.dann_lambda > 0:
        print(f"  DANN: lambda={args.dann_lambda}, alpha={args.dann_alpha}")
    if args.u1_lambda > 0:
        print(f"  U1 Jacobian: lambda={args.u1_lambda}, mode={args.u1_mode}")

    all_results = []
    for seed in seed_list:
        result = train_one_seed(args, seed)
        if result is None:
            # 探针路径（如 --vuln_probe_ckpt）已自行存盘并退出，无训练结果可汇总
            print("[main] probe-only run (result=None), skipping aggregation.")
            return
        all_results.append(result)

    # 汇总
    oas = [r['best_OA'] for r in all_results]
    if len(oas) > 1:
        mean_oa = np.mean(oas)
        std_oa = np.std(oas, ddof=1)
        print(f"\n{'='*60}")
        print(f"SUMMARY ({len(oas)} seeds): OA mean={mean_oa*100:.2f}% ± {std_oa*100:.2f}%")
        print(f"Individual: {[f'{o*100:.2f}%' for o in oas]}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
