# -*- coding: utf-8 -*-
"""
VCA_Res2DCNN: M1 基准模型
- VCA 端元提取 + 端元投影层 + 2D 残差 CNN
- A2 策略 = 同一模型, classifier_lr=0.001 (backbone_lr=0.01)

VCA algorithm translated from the MATLAB VCA (Nascimento & Bioucas-Dias, 2005).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from aug_abundance import apply_aug


# ============================================================
# VCA 端元提取 (手动实现, 不用 pysptools)
# ============================================================
def vca_extract(data_2d, k, seed=3667):
    """
    Vertex Component Analysis - 从源域像素中提取 k 个端元.

    Args:
        data_2d: (N, B) 源域所有有效像素的光谱
        k: int, 端元数量
        seed: int, 随机种子 (控制端元选择的随机投影方向)

    Returns:
        endmembers: (k, B) 端元光谱
    """
    rng = np.random.RandomState(seed)

    N, B = data_2d.shape

    # Step 1: 均值中心化
    X = data_2d - data_2d.mean(axis=0, keepdims=True)

    # Step 2: SVD 降维到 (k-1) 维
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # 投影矩阵 U_reduce: (B, k-1)
    U_reduce = Vt[:k - 1, :].T  # top-(k-1) right singular vectors
    # X 投影到 (k-1) 维子空间
    X_proj = X @ U_reduce  # (N, k-1)

    # Step 3: 迭代投影寻端元
    indices = []
    e = np.zeros((k, k - 1))  # 已找到端元在子空间中的方向

    for i in range(k):
        # 生成随机方向
        w = rng.randn(k - 1).astype(np.float32)
        w /= np.linalg.norm(w)

        if i == 0:
            # 首个端元: 最大投影点
            proj = X_proj @ w
            idx = int(np.argmax(proj))
        else:
            # 后续端元: 在已找到端元的正交补空间中找最大投影
            # 正交化 w 对已有端元方向
            for j in range(i):
                w = w - np.dot(w, e[j]) * e[j]
            w /= np.linalg.norm(w)

            proj = np.abs(X_proj @ w)
            idx = int(np.argmax(proj))

        eu = X_proj[idx]  # 子空间中的端元方向
        eu /= np.linalg.norm(eu)
        e[i] = eu
        indices.append(idx)

    endmembers = data_2d[np.array(indices)]  # (k, B)
    return endmembers


def init_unified_projection(data_2d, k, R0, seed=3667):
    """
    Unified projection: W = P_PCA(E) + gamma * (E - P_PCA(E)).
    gamma = R / (R + R0), where R measures VCA endmember alignment with PCs.
    """
    from sklearn.decomposition import PCA
    
    N, B = data_2d.shape
    
    # 1. VCA endmembers
    E = vca_extract(data_2d, k, seed=seed)  # (k, B)
    
    # 2. PCA directions (k components)
    pca = PCA(n_components=k).fit(data_2d)
    V = pca.components_  # (k, B)
    
    # 3. Project VCA endmembers onto PCA subspace: E @ V.T @ V
    E_pca = E @ V.T @ V  # (k, B)
    
    # 4. Compute R = avg cos with high-variance PCs / avg cos with low-variance PCs
    X = data_2d.astype(np.float64) - data_2d.mean(axis=0, keepdims=True)
    _, _, Vt_full = np.linalg.svd(X, full_matrices=False)
    n_pc = min(15, Vt_full.shape[0])
    pc_full = Vt_full[:n_pc, :]
    
    E_norm = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-10)
    pc_norm = pc_full / (np.linalg.norm(pc_full, axis=1, keepdims=True) + 1e-10)
    cos_mat = np.abs(E_norm @ pc_norm.T)
    
    n_high = min(5, n_pc)
    n_low_start = max(n_pc - 5, n_high)
    cos_high = cos_mat[:, :n_high].mean()
    cos_low = cos_mat[:, n_low_start:].mean() if n_low_start < n_pc else cos_high
    R = cos_high / (cos_low + 1e-10)
    
    # 5. gamma = R / (R + R0)
    gamma = R / (R + R0)
    
    # 6. W = E_pca + gamma * (E - E_pca)
    W = E_pca + gamma * (E - E_pca)
    
    # NOTE: no normalization — VCA baseline uses raw pixel values
    
    print(f'[Unified] k={k} B={B} R={R:.2f} gamma={gamma:.4f} cos_high={cos_high:.4f} cos_low={cos_low:.4f}')
    return W.astype(np.float32)


# ============================================================
# ResBlock2D (3层 Conv2d 残差块)
# shortcut = conv1 的输出, 不是原始输入
# ============================================================
def estimate_shift_scatter(source_data, source_gt, n_bands, n_blocks=8, min_px=10):
    """Estimate the cross-domain shift scatter matrix S_shift from SOURCE data ONLY.

    Idea (turns the R14/R15 domain-shift characterization into a usable operator):
    a real cross-domain spectral shift is a *coherent, class-independent* offset of
    the spectra caused by changing acquisition conditions. The same kind of offset
    is already present *inside* the source scene as spatial heterogeneity across
    regions. We measure it there -- no target data, no target labels, DG-legal.

      1) tile the scene into an n_blocks x n_blocks grid,
      2) for each block b, d_b = mean_c [ mu(b,c) - mu(c) ]  (class-conditional, so
         class-composition differences cannot masquerade as a shift),
      3) S_shift = (1/G) * sum_b (d_b - d_bar)(d_b - d_bar)^T.

    Returns (S_shift (B,B) float64, n_blocks_used int)
    """
    H, W, B = source_data.shape
    X = source_data.reshape(-1, B).astype(np.float64)
    y = source_gt.reshape(-1)

    gmean = {}
    for c in np.unique(y):
        if c <= 0:
            continue
        m = X[y == c]
        if len(m) >= min_px:
            gmean[int(c)] = m.mean(axis=0)

    bh = max(1, H // n_blocks)
    bw = max(1, W // n_blocks)
    n_col = int(np.ceil(W / bw))
    ii, jj = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    bid = ((ii // bh) * n_col + (jj // bw)).reshape(-1)

    rows = []
    for b in np.unique(bid):
        sel = (bid == b)
        devs = [X[sel & (y == c)].mean(axis=0) - gm
                for c, gm in gmean.items() if (sel & (y == c)).sum() >= min_px]
        if devs:
            rows.append(np.mean(devs, axis=0))

    if len(rows) < 2:  # fallback: unconditional block means
        rows = [X[(bid == b) & (y > 0)].mean(axis=0)
                for b in np.unique(bid) if ((bid == b) & (y > 0)).sum() >= min_px]
    if len(rows) < 2:  # last resort: global covariance
        sel = y > 0
        D = X[sel] - X[sel].mean(axis=0, keepdims=True)
    else:
        D = np.stack(rows)
        D = D - D.mean(axis=0, keepdims=True)

    S = (D.T @ D) / max(1, D.shape[0])
    return S, len(rows)


def init_sdp_projection(source_data, source_gt, n_bands, k, n_blocks=8,
                        tau=1.0, gamma=1.0, ortho=True, min_px=10, delta=0.05):
    """SDP -- Shift-suppressed Discriminative Projection (analytic, deterministic).

    Directly implements "preserve and exploit the *discriminative* cross-domain
    information": find the k spectral directions maximising class separability
    while minimising energy inside the estimated domain-shift subspace.

        max_w  (w^T (S_b + delta*S_t) w) / (w^T (tau*S_shift + gamma*S_w + eps*I) w)

    S_b     : between-class scatter (discriminability to keep)
    S_t     : total scatter -- needed because rank(S_b) <= n_classes-1, so for
              k > n_classes-1 the extra directions would otherwise be arbitrary.
              With a small delta they become "highest-variance among the
              shift-robust directions", i.e. discriminative directions first,
              shift-robust spectral variance afterwards.
    S_w     : within-class scatter  (nuisance variability)
    S_shift : source-internally estimated cross-domain shift scatter (will NOT
              transfer -- we want the projection blind to it)

    Solved as a symmetric generalised eigenproblem via whitening (numpy only,
    deterministic -> no seed noise). Rows are QR-orthonormalised when ortho=True,
    since orthogonal spectral projections are the empirically robust regime.

    Returns W (k, n_bands) float32.
    """
    B = n_bands
    src_mask = source_gt > 0
    X = source_data.reshape(-1, B)[src_mask.reshape(-1)].astype(np.float64)
    y = source_gt[src_mask].astype(np.int64)

    mu = X.mean(axis=0)
    S_b = np.zeros((B, B))
    S_w = np.zeros((B, B))
    N = len(X)
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) < 2:
            continue
        mc = Xc.mean(axis=0)
        dm = (mc - mu).reshape(-1, 1)
        S_b += (len(Xc) / N) * (dm @ dm.T)
        Dc = Xc - mc
        S_w += (Dc.T @ Dc) / N

    S_shift, n_blk = estimate_shift_scatter(source_data, source_gt, B,
                                            n_blocks=n_blocks, min_px=min_px)

    # normalise nuisance terms to unit trace so tau/gamma are scene-independent
    M = tau * (S_shift / (np.trace(S_shift) + 1e-12)) \
        + gamma * (S_w / (np.trace(S_w) + 1e-12))
    M = M + 1e-8 * (np.trace(M) / B) * np.eye(B)

    ev, U = np.linalg.eigh(M)
    ev = np.maximum(ev, 1e-12)
    Mih = U @ np.diag(ev ** -0.5) @ U.T          # M^{-1/2}
    Xc = X - mu
    S_t = (Xc.T @ Xc) / N
    Num = S_b / (np.trace(S_b) + 1e-12) + delta * S_t / (np.trace(S_t) + 1e-12)
    A = Mih @ Num @ Mih
    A = 0.5 * (A + A.T)
    w_ev, w_U = np.linalg.eigh(A)
    idx = np.argsort(w_ev)[::-1][:k]
    Wraw = Mih @ w_U[:, idx]                     # (B, k)

    if ortho:
        Q, _ = np.linalg.qr(Wraw)
        Wout = Q[:, :k].T
    else:
        Wout = (Wraw / (np.linalg.norm(Wraw, axis=0, keepdims=True) + 1e-12)).T

    print(f"[SDP] blocks={n_blk}, k={k}/{B}, tau={tau}, gamma={gamma}, delta={delta}, "
          f"ortho={ortho}, n_cls={len(np.unique(y))}, "
          f"gen-eig top{k}={np.array2string(w_ev[idx], precision=4)}")
    return np.ascontiguousarray(Wout, dtype=np.float32)


def init_centroid_projection(data_2d, gt_flat, k, seed=None):
    """Class-Centroid Projection. Compute centroids -> SVD -> (opt) PCA residual fill."""
    from sklearn.decomposition import PCA
    import numpy as np
    mask = gt_flat > 0
    X = data_2d[mask]
    Y = gt_flat[mask]
    C = int(Y.max())
    n_bands = X.shape[1]
    n_disc = C - 1

    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-10)
    centroids = np.empty((C, n_bands), dtype=np.float64)
    for c in range(C):
        centroids[c] = X_norm[Y == c + 1].mean(axis=0)
    mu_g = X_norm.mean(axis=0)
    D = centroids - mu_g
    _, _, Vt = np.linalg.svd(D, full_matrices=False)
    U_disc = Vt[:n_disc].astype(np.float32)

    if k <= n_disc:
        W = U_disc[:k]
        print(f'[Centroid] C={C} n_disc={n_disc} pure_disc k={k}')
        return W

    n_res = k - n_disc
    X_proj = X_norm @ U_disc.T @ U_disc
    X_res = X_norm - X_proj
    pca_model = PCA(n_components=n_res).fit(X_res.astype(np.float64))
    U_res = pca_model.components_.astype(np.float32)
    W = np.concatenate([U_disc, U_res], axis=0)
    print(f'[Centroid] C={C} disc=6 n_res={n_res} k={k}')
    return W

class ResBlock2D(nn.Module):
    """3-layer Conv2D residual block for VCA_Res2DCNN."""

    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes, out_planes, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        # downsample shortcut if planes change
        self.downsample = None
        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x, tmp_aug=None):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)

        # shortcut = conv1 的输出 (x1), 不是原始输入 x
        shortcut = x1
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = x3 + shortcut
        out = F.relu(out)
        return out


# ============================================================
# VCA_Res2DCNN (M1 模型)
# ============================================================
class VCA_Res2DCNN(nn.Module):
    """
    M1 baseline model.

    Input: (N, B, 13, 13)
      → unsqueeze(1) → (N, 1, B, 13, 13)
      → VCA Conv3d(1→k, kernel=(B,1,1))  [端元投影]
      → squeeze(dim=2) → (N, k, 13, 13)
      → ResBlock2D(k→32) + MaxPool2d → (N,32,7,7)
      → ResBlock2D(32→64) + MaxPool2d → (N,64,4,4)
      → ResBlock2D(64→128)             → (N,128,4,4)
      → Conv2d(128→128,3)              → (N,128,2,2)
      → Flatten → FC(512→128) → Linear(512→7)

    Output: (logits, features)
    """

    def __init__(self, bands, classes, k_endmembers, proj_weights, res_channels=(32, 64, 128),
                 proj_init='VCA', proj_frozen=True, aug_type='none', aug_param=None,
                 proj_ortho_lambda=0.0):
        """
        Args:
            bands: int, number of spectral bands
            classes: int, number of classes
            k_endmembers: int, number of projection output channels
            proj_weights: (k_endmembers, bands) numpy array, projection matrix
            res_channels: ResBlock2D channel sequence
            proj_init: str, projection init type
            proj_frozen: bool, freeze projection layer
        """
        super().__init__()
        self.bands = bands
        self.classes = classes
        self.k = k_endmembers
        self.proj_init = proj_init
        self.proj_ortho_lambda = proj_ortho_lambda

        # VCA 端元投影层: Conv3d(1 → k, kernel=(bands, 1, 1), no bias)
        self.vca_proj = nn.Conv3d(1, k_endmembers, kernel_size=(bands, 1, 1), bias=False)

        # 用 VCA 端元初始化权重: shape (k, 1, B, 1, 1)
        weight = torch.from_numpy(proj_weights).float().view(k_endmembers, 1, bands, 1, 1)
        self.vca_proj.weight.data.copy_(weight)

        # 冻结控制
        self.proj_frozen = proj_frozen
        if proj_frozen:
            for p in self.vca_proj.parameters():
                p.requires_grad = False

        self.aug_type = aug_type
        self.aug_param = aug_param

        self.aug_type = aug_type
        self.aug_param = aug_param

        # ResBlock2D stages
        self.block1 = ResBlock2D(k_endmembers, res_channels[0])
        self.pool1 = nn.MaxPool2d(2, stride=2, padding=1)

        self.block2 = ResBlock2D(res_channels[0], res_channels[1])
        self.pool2 = nn.MaxPool2d(2, stride=2, padding=1)

        self.block3 = ResBlock2D(res_channels[1], res_channels[2])

        # Final conv (128→128, kernel=3, no bias)
        self.conv_out = nn.Conv2d(res_channels[2], res_channels[2], 3, bias=False)

        # Embedding + Classifier
        fc_dim = res_channels[2] * 2 * 2
        self.fc_embed = nn.Linear(fc_dim, 128, bias=False)
        self.classifier = nn.Linear(fc_dim, classes, bias=False)

    def forward_backbone(self, x, tmp_aug=None):
        """tmp_aug: optional (aug_type, aug_param) tuple for per-call override."""
        """Forward through backbone only, returns (N, 512) before fc_embed/classifier."""
        x = x.unsqueeze(1)
        x = self.vca_proj(x)
        x = x.squeeze(2)
        if tmp_aug is not None:
            x = apply_aug(x, tmp_aug[0], tmp_aug[1])
        else:
            x = apply_aug(x, self.aug_type, self.aug_param)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        x = self.block3(x)
        x = self.conv_out(x)
        return x.flatten(1)                      # (N, 512)

    def forward(self, x, tmp_aug=None):
        """
        Args:
            x: (N, B, H, W) input patches
        Returns:
            logits: (N, classes)
            features: (N, 128)
        """
        feat = self.forward_backbone(x, tmp_aug=tmp_aug)  # (N, 512)
        features = self.fc_embed(feat)           # (N, 128)
        logits = self.classifier(feat)           # (N, classes)
        return logits, features

    def ortho_penalty(self):
        """Soft orthogonal regularization on projection rows: λ * ||W W^T - I||_F.

        W has shape (k, bands). Encouraging orthonormal rows makes the latent basis
        shift-robust (condition number -> 1), which is the root cause of DG robustness
        (see results.md §8.7). Returns None when disabled or projection frozen (no grad).
        """
        if self.proj_ortho_lambda == 0.0 or self.proj_frozen:
            return None
        W = self.vca_proj.weight.view(self.k, self.bands)  # (k, bands)
        I = torch.eye(self.k, device=W.device, dtype=W.dtype)
        return self.proj_ortho_lambda * torch.norm(W @ W.T - I, p='fro')


# ============================================================
# 工厂函数
# ============================================================
def _prepare_projection(source_data, source_gt, n_bands, k_endmembers, proj_init, seed):
    """Prepare projection weight matrix (k_endmembers, n_bands)."""
    if proj_init == 'Identity':
        assert k_endmembers == n_bands, f'Identity requires k=n_bands'
        W = np.eye(n_bands, dtype=np.float32)
    elif proj_init == 'RandomOrth':
        R = torch.randn(k_endmembers, n_bands)
        U, _, V = torch.svd(R)
        W = (U @ V.T).numpy().astype(np.float32)
    elif proj_init == 'Random':
        W = np.random.randn(k_endmembers, n_bands).astype(np.float32)
    elif proj_init == 'PCA':
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        pca = PCA(n_components=k_endmembers).fit(src_pixels)
        W = pca.components_.astype(np.float32)
    elif proj_init == 'PCA_bottom':
        # 反方差 PCA：保留最低方差（域变异/噪声）方向，用于证伪 κ、坐实方向选择机制
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        pca_full = PCA(n_components=n_bands).fit(src_pixels)
        W = pca_full.components_[-k_endmembers:].astype(np.float32)  # 最低方差 k 个成分
    elif proj_init == 'VCA':
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        W = vca_extract(src_pixels, k_endmembers, seed=seed)
    elif proj_init == 'VCA_norm':
        # 行单位化：保留端元方向，统一尺度，改善投影矩阵条件数
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        W = vca_extract(src_pixels, k_endmembers, seed=seed)
        W = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-12)
    elif proj_init == 'VCA_orth':
        # QR 正交化：行正交且单位范数，代数性质最接近 PCA 主成分
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        W = vca_extract(src_pixels, k_endmembers, seed=seed)
        Q, _ = np.linalg.qr(W.T)   # (bands, k)
        W = Q.T.copy()             # (k, bands)
    elif proj_init == 'SDP':
        # Shift-suppressed Discriminative Projection (analytic, source-only)
        W = init_sdp_projection(
            source_data, source_gt, n_bands, k_endmembers,
            n_blocks=getattr(_prepare_projection, 'sdp_blocks', 8),
            tau=getattr(_prepare_projection, 'sdp_tau', 1.0),
            gamma=getattr(_prepare_projection, 'sdp_gamma', 1.0),
            delta=getattr(_prepare_projection, 'sdp_delta', 0.05),
            ortho=getattr(_prepare_projection, 'sdp_ortho', True))
    elif proj_init == 'centroid':
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        gt_flat = source_gt[src_mask]
        W = init_centroid_projection(src_pixels, gt_flat, k_endmembers, seed=seed)
    elif proj_init == 'mix':
        # VCA+PCA hybrid: vca_k endmembers + pca_k components
        # k_endmembers holds vca_k, pca_k is inferred
        assert hasattr(_prepare_projection, 'pca_k'), 'Set _prepare_projection.pca_k before calling with mix'
        pca_k = _prepare_projection.pca_k
        vca_k = k_endmembers
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        vca_w = vca_extract(src_pixels, vca_k, seed=seed)
        pca_model = PCA(n_components=pca_k).fit(src_pixels)
        pca_w = pca_model.components_.astype(np.float32)
        W = np.concatenate([vca_w, pca_w], axis=0).astype(np.float32)
        # Update k to total
        _prepare_projection.total_k = vca_k + pca_k
    elif proj_init == 'Unified':
        assert hasattr(_prepare_projection, 'unified_r0'), 'Set _prepare_projection.unified_r0 before Unified'
        R0 = _prepare_projection.unified_r0
        src_mask = source_gt > 0
        src_pixels = source_data.reshape(-1, n_bands)[src_mask.reshape(-1)]
        W = init_unified_projection(src_pixels, k_endmembers, R0, seed=seed)
    else:
        raise ValueError(f'Unknown proj_init: {proj_init}')
    return W


def build_einit_cnn(source_data, source_gt, n_bands, n_classes, seed, k_endmembers=None, res_channels=(32, 64, 128), no_vca=False, proj_init='VCA', proj_frozen=True, aug_type='none', aug_param=None, proj_ortho_lambda=0.0):
    """
    构建 VCA_Res2DCNN 模型 (v3: 可配置投影).

    Args:
        source_data: (H, W, B) 源域图像
        source_gt: (H, W) 源域标签
        n_bands: int
        n_classes: int
        seed: int, 控制随机投影
        k_endmembers: projection output channels
        no_vca: legacy, -> Random + learnable
        proj_init: 'VCA'|'PCA'|'RandomOrth'|'Random'|'Identity'
        proj_frozen: bool
    """
    # Legacy
    if no_vca:
        proj_init = 'Random'
        proj_frozen = False

    # Auto k
    if k_endmembers is None:
        if proj_init == 'Identity':
            k_endmembers = n_bands
        elif n_bands <= 50:
            k_endmembers = n_classes * 2
        else:
            k_endmembers = n_classes

    # Prepare projection weights
    proj_weights = _prepare_projection(source_data, source_gt, n_bands, k_endmembers, proj_init, seed)

    # Mix projection: total k = vca_k + pca_k
    if proj_init == 'mix' and hasattr(_prepare_projection, 'total_k'):
        k_endmembers = _prepare_projection.total_k
        delattr(_prepare_projection, 'total_k')

    model = VCA_Res2DCNN(n_bands, n_classes, k_endmembers, proj_weights,
                         res_channels=res_channels, proj_init=proj_init, proj_frozen=proj_frozen,
                         aug_type=aug_type, aug_param=aug_param, proj_ortho_lambda=proj_ortho_lambda)
    return model, k_endmembers


if __name__ == '__main__':
    # Quick test
    import sys
    sys.path.insert(0, '.')
    from datasets import get_dataset

    img_src, gt_src, _, _, _, _, _ = get_dataset('Houston13', target_folder='../data/')
    model, k = build_einit_cnn(img_src, gt_src, img_src.shape[2], 7, seed=3667)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"VCA_Res2DCNN built: k={k}, params={n_params/1e6:.3f}M")

    x = torch.randn(2, img_src.shape[2], 13, 13)
    logits, feat = model(x)
    print(f"Input: {x.shape} → logits: {logits.shape}, features: {feat.shape}")
