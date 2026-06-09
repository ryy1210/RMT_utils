import torch
import torch.nn as nn
import math
import numpy as np
import scipy.stats
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy.stats import norm
from scipy.optimize import minimize

def get_esd_metrics(model, pl_fitting='median', conv_norm=1.0, filter_zeros=True, bins=100):
    """
    モデル内の全線形層（Conv2d, Linear）に対してESD関連のメトリクスを計算する関数
    """
    results = {
        'name': [],
        'spectral_norm': [],
        'entropy': [],
        'stable_rank': [],
        'alphahat': [],
        'alpha': [],
        'eigs': [],
        'eigs_num': []
    }
    
    # 補助関数
    def safe_log10(x):
        return torch.log10(x + 1e-12)

    def matrix_entropy(eigs):
        p = eigs / torch.sum(eigs)
        return -torch.sum(p * torch.log(p + 1e-12))

    device = next(model.parameters()).device

    # レイヤーのループ
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # 重みの取得とデバイス合わせ
            matrix = m.weight.data.clone().to(device).to(torch.float)
            
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                # Conv2dの形状調整 (PyTorchのconv重みは (out, in, k, k))
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            
            # SVD計算
            # 固有値 λ = σ^2
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            eigs, _ = torch.sort(eigs, descending=False)
            
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            stable_rank = fnorm / (spectral_norm + 1e-8)
            entropy = matrix_entropy(torch.sqrt(eigs))
            
            # Zero filtering
            if filter_zeros:
                nz_eigs = eigs[eigs > 1e-8] # EVALS_THRESH の代用
                N = len(nz_eigs)
                if N == 0:
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                nz_eigs = eigs
                N = len(nz_eigs)
                
            log_nz_eigs = torch.log(nz_eigs)
            
            # Power Law fitting (alpha)
            if pl_fitting == 'median':
                i = int(len(nz_eigs) / 2) # 元コード xmin_pos=2 を想定
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n, device=device)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
            else:
                # fix-finger 相当の実装が必要な場合はここに追加
                # 今回は汎用的に median を使用
                final_alpha = torch.tensor(1.0)
                
            final_alpha_val = final_alpha.item()
            
            # AlphaHat計算
            final_alphahat = final_alpha_val * safe_log10(torch.tensor(spectral_norm)).item()
            final_alphahat = math.log(1.0 + math.exp(final_alphahat))
            
            # 結果の保存
            results['name'].append(name)
            results['spectral_norm'].append(spectral_norm)
            results['entropy'].append(entropy.detach().cpu().item())
            results['stable_rank'].append(stable_rank)
            results['alphahat'].append(final_alphahat)
            results['alpha'].append(final_alpha_val)
            results['eigs'].append(eigs.detach().cpu().numpy())
            results['eigs_num'].append(len(eigs))
            
    return results



def dyson_equalizer_algorithm1(Y):
    """
    Landa & Kluger (2024) - Algorithm 1: The Dyson Equalizer
    論文の数式と記法に完全に対応させた実装。

    Input:
        Y: Data matrix (m x n), m <= n
    Returns:
        Y_hat: Normalized data matrix
        x_hat: Row scaling vector
        y_hat: Column scaling vector
    """
    m, n = Y.shape
    if m > n:
        raise ValueError("Input matrix Y must have m <= n. Transpose Y if necessary.")

    # 1: Compute the SVD of Y
    # U: m x m, sigma: m, V_h: n x n
    U, sigma, V_h = np.linalg.svd(Y, full_matrices=True)
    V = V_h.T  # V \in R^{n x n} (右特異ベクトルを列に持つ行列)

    # 2: Set eta as the median singular value of Y
    eta = np.median(sigma)

    # 3: Compute the vectors g_hat^(1) and g_hat^(2)
    # 論文 (3) 式の計算（行列演算で高速化）
    term1 = eta / (sigma**2 + eta**2)
    term2 = term1 - (1 / eta)

    # U は m x m, sigma は要素数 m
    g1_hat = (U**2) @ term1

    # V は n x n. sum は k=1 から m までなので V の最初の m 列を使用
    g2_hat = (1 / eta) + (V[:, :m]**2) @ term2

    # 4: Compute the vectors x_hat and y_hat
    # L1ノルム ||g_hat^(1)||_1 と ||g_hat^(2)||_1 の計算
    g1_norm1 = np.sum(np.abs(g1_hat))
    g2_norm1 = np.sum(np.abs(g2_hat))

    # 論文 (4) 式の計算
    x_hat = (1 / np.sqrt(m - eta * g1_norm1)) * ((1 / g1_hat) - eta)
    y_hat = (1 / np.sqrt(n - eta * g2_norm1)) * ((1 / g2_hat) - eta)

    # 数値的安定性のための安全策（負値の平方根エラー回避）
    x_hat = np.maximum(1e-12, x_hat)
    y_hat = np.maximum(1e-12, y_hat)

    # 5: Form the normalized data matrix Y_hat
    # Y_hat = (D_{x_hat})^{-1/2} Y (D_{y_hat})^{-1/2}
    Y_hat = Y / (np.sqrt(x_hat[:, None]) * np.sqrt(y_hat[None, :]))

    return Y_hat, x_hat, y_hat



def bema_loss(sigma2_proposal, evals_emp, gamma, p, alpha):
        """
        BEMAの損失関数 (BEMA.Rの `loss` 関数に相当)
        提案された分散 sigma^2 に基づいてMP分布に従うランダム行列をシミュレートし、
        経験的固有値のバルク部分（分位数）との二乗誤差を計算します。
        """
        n = int(p / gamma)
        L = np.zeros((10, p))
        
        # 提案された分散でランダム行列を10回モンテカルロシミュレーション
        for i in range(10):
            Z_sim = np.random.randn(p, n) * np.sqrt(sigma2_proposal)
            if p <= n:
                S_sim = Z_sim @ Z_sim.T / n
            else:
                S_sim = Z_sim.T @ Z_sim / n
            L[i, :] = np.sort(np.linalg.eigvalsh(S_sim))[::-1]
            
        evals_sim_mean = np.mean(L, axis=0)
        
        # alpha に基づいて、分布の「端（スパイクや微小固有値）」を切り落とす
        # 例: alpha=0.2 の場合、上位20%と下位20%を無視し、中間の60%のバルクだけで比較する
        idx_start = int(min(p, n) * alpha)
        idx_end = int(min(p, n) * (1 - alpha))
        
        evals_emp_bulk = evals_emp[idx_start:idx_end]
        evals_sim_bulk = evals_sim_mean[idx_start:idx_end]
        
        # バルク部分の分位数の二乗誤差
        loss = np.sum((evals_emp_bulk - evals_sim_bulk)**2)
        return loss

def apply_bema(evals_emp, gamma, p, alpha=0.2):
    """
    BEMAアルゴリズムを実行し、真の分散 sigma^2 を推定します。
    """
    print("BEMAによる分散推定を実行中...")
    # scipy.optimize.minimize_scalar を用いて、損失関数を最小化する分散を探索
    res = opt.minimize_scalar(
        bema_loss, 
        args=(evals_emp, gamma, p, alpha), 
        bounds=(0.01, 10.0), 
        method='bounded'
    )
    return res.x


def tw1_quantile(beta=0.1):
    """
    Type-I Tracy-Widom 分布の (1-beta) 分位点を返す。

    scipy に tracywidom がある環境ではそれを使用。
    ない場合は代表的な近似値を使う。
    """
    try:
        from scipy.stats import tracywidom
        return tracywidom.ppf(1 - beta, beta=1)
    except Exception:
        # Type-I Tracy-Widom TW1 の代表的な分位点近似
        # beta は右側確率。つまり返すのは 1-beta quantile。
        table = {
            0.20: -0.165,
            0.10:  0.450,
            0.05:  0.979,
            0.025: 1.454,
            0.01:  2.023,
            0.001: 3.272,
        }

        if beta in table:
            return table[beta]

        # 近い値を線形補間
        betas = np.array(sorted(table.keys()))
        vals = np.array([table[b] for b in betas])

        if beta < betas.min():
            return vals[0]
        if beta > betas.max():
            return vals[-1]

        return np.interp(beta, betas, vals)


def mp_pdf_zero_excluded(x, gamma, sigma2=1.0):
    """
    zero-excluded Marchenko-Pastur density.

    gamma = p / n.
    sigma2 = 1 のとき標準MP分布。

    gamma > 1 の場合、p x p sample covariance にはゼロ固有値が出るので、
    非ゼロ固有値に条件づけた zero-excluded density を使う。
    """
    x = np.asarray(x)

    a = sigma2 * (1 - np.sqrt(gamma)) ** 2
    b = sigma2 * (1 + np.sqrt(gamma)) ** 2

    pdf = np.zeros_like(x, dtype=float)

    mask = (x > a) & (x < b)
    xm = x[mask]

    # classical MP density の正規化係数は 2*pi*gamma*sigma2*x
    # gamma > 1 では非ゼロ部分の質量が 1/gamma なので、
    # zero-excluded にするため gamma 倍する。
    denom_gamma = min(gamma, 1.0)

    pdf[mask] = (
        np.sqrt((b - xm) * (xm - a))
        / (2 * np.pi * denom_gamma * sigma2 * xm)
    )

    return pdf


def mp_upper_quantiles(gamma, p_tilde, k_indices, grid_size=200000):
    """
    sigma2=1 の zero-excluded MP 分布について、
    k/p_tilde upper-quantile q_k を返す。

    k_indices は 1始まりの index を想定。
    """
    a = (1 - np.sqrt(gamma)) ** 2
    b = (1 + np.sqrt(gamma)) ** 2

    eps = 1e-10
    x_grid = np.linspace(a + eps, b - eps, grid_size)

    pdf = mp_pdf_zero_excluded(x_grid, gamma, sigma2=1.0)

    # 数値誤差補正のため、台形積分でCDFを作って正規化
    dx = x_grid[1] - x_grid[0]
    cdf = np.cumsum(pdf) * dx
    cdf = cdf / cdf[-1]

    # upper tail probability y = k / p_tilde
    # F(q_k) = 1 - y
    y_upper = np.asarray(k_indices, dtype=float) / p_tilde
    cdf_targets = 1.0 - y_upper

    inv_cdf = interp.interp1d(
        cdf,
        x_grid,
        bounds_error=False,
        fill_value=(a, b)
    )

    q = inv_cdf(cdf_targets)
    return q


def bema_algorithm1_from_eigenvalues(evals, p, n, alpha=0.2, beta=0.1):
    """
    BEMA Algorithm 1 for the standard spiked covariance model.

    Parameters
    ----------
    evals : array-like
        sample covariance matrix の非ゼロ固有値。
        降順でなくてもよい。
        -> 訂正：もとの行列の特異値
    p : int
        次元。Y が p x n のデータ行列なら p = Y.shape[0]。
    n : int
        サンプルサイズ。Y が p x n のデータ行列なら n = Y.shape[1]。
    alpha : float
        bulk eigenvalues の中央部分を選ぶパラメータ。
        論文のデフォルトは 0.2。
    beta : float
        over-estimation probability を制御するパラメータ。
        論文の実用デフォルトは 0.1。

    Returns
    -------
    result : dict
        sigma2_hat, K_hat, threshold, q_bulk, evals_sorted など。
    """
    evals = np.asarray(evals, dtype=float)
    evals = evals[evals > 1e-14]
    evals_sorted = np.sort(evals)[::-1]

    p_tilde = min(p, n)

    if len(evals_sorted) != p_tilde:
        # 数値的にゼロ固有値を除いた数がずれる場合に合わせる
        p_tilde = len(evals_sorted)

    gamma = p / n

    # 論文の index は 1 <= k <= p_tilde
    k_start = int(np.ceil(alpha * p_tilde))
    k_end = int(np.floor((1 - alpha) * p_tilde))

    # 1始まり index
    k_indices = np.arange(k_start, k_end + 1)

    # Python の配列 index は 0始まりなので -1
    evals_bulk = evals_sorted[k_indices - 1]

    q_bulk = mp_upper_quantiles(
        gamma=gamma,
        p_tilde=p_tilde,
        k_indices=k_indices
    )

    sigma2_hat = np.sum(q_bulk * evals_bulk) / np.sum(q_bulk ** 2)

    t_tw = tw1_quantile(beta=beta)

    threshold = sigma2_hat * (
        (1 + np.sqrt(gamma)) ** 2
        + t_tw
        * n ** (-2 / 3)
        * gamma ** (-1 / 6)
        * (1 + np.sqrt(gamma)) ** (4 / 3)
    )

    s_hat = int(np.sum(evals_sorted > threshold))

    return {
        "s_hat": s_hat,
        "sigma2_hat": sigma2_hat,
        "threshold": threshold,
        "gamma": gamma,
        "p_tilde": p_tilde,
        "k_indices": k_indices,
        "q_bulk": q_bulk,
        "evals_bulk": evals_bulk,
        "evals_sorted": evals_sorted,
        "tw_quantile": t_tw,
    }


def bema_algorithm1_from_data(Y, alpha=0.2, beta=0.1, center=False):
    """
    データ行列 Y から Algorithm 1 を実行する。

    Y は p x n、つまり
        p = 変数数
        n = サンプル数
    として扱う。

    sample covariance は S = Y Y^T / n。
    """
    Y = np.asarray(Y, dtype=float)

    if center:
        Y = Y - Y.mean(axis=1, keepdims=True)

    p, n = Y.shape

    if p <= n:
        S = Y @ Y.T / n
        evals = np.linalg.eigvalsh(S)
    else:
        # 非ゼロ固有値だけなら Y^T Y / n の固有値を使えばよい
        S_small = Y.T @ Y / n
        evals = np.linalg.eigvalsh(S_small)

    return bema_algorithm1_from_eigenvalues(
        evals=evals,
        p=p,
        n=n,
        alpha=alpha,
        beta=beta
    )

def gaussian_broadening_fit(evals, gamma_ratio, a=10):
    """
    論文のセクション2.3に基づく Gaussian Broadening と最小二乗法による sigma^2 の推定
    """
    m = len(evals)
    evals_sorted = np.sort(evals)
    
    # 1. 局所標準偏差 sigma_k の計算
    sigma_k = np.zeros(m)
    for k in range(m):
        k_minus = max(0, k - a)
        k_plus = min(m - 1, k + a)
        # ウィンドウ幅 2a に基づく局所的な間隔
        sigma_k[k] = (evals_sorted[k_plus] - evals_sorted[k_minus]) / 2.0
        
        # 同値が連続した場合のゼロ除算エラーを防ぐ安全策
        if sigma_k[k] < 1e-8:
            sigma_k[k] = 1e-8 

    # 2. 平滑化された経験的密度 P(gamma) の定義
    def P_gamma(x):
        x = np.atleast_1d(x)
        # x と evals_sorted の全組み合わせの差分 (N, M) 行列を計算
        diff = x[:, None] - evals_sorted[None, :]
        exponent = - (diff ** 2) / (2 * sigma_k[None, :] ** 2)
        coef = 1.0 / (np.sqrt(2 * np.pi) * sigma_k[None, :])
        # 各 x について m 個のガウス関数の平均をとる
        return np.mean(coef * np.exp(exponent), axis=1)

    # 3. MP分布 g(gamma) の定義
    def mp_pdf(x, sigma2):
        x = np.atleast_1d(x)
        lambda_plus = sigma2 * (1 + np.sqrt(gamma_ratio))**2
        lambda_minus = sigma2 * (1 - np.sqrt(gamma_ratio))**2
        
        pdf = np.zeros_like(x)
        valid = (x > lambda_minus) & (x < lambda_plus)
        if np.any(valid):
            pdf[valid] = np.sqrt((lambda_plus - x[valid]) * (x[valid] - lambda_minus)) / (2 * np.pi * gamma_ratio * sigma2 * x[valid])
        return pdf

    # 4. 最小二乗法のための目的関数の定義
    # フィッティング範囲: スパイクの影響を排除するため、下位 90% のバルク領域でカーブを比較する
    limit_idx = int(m * 0.90)
    x_eval = np.linspace(max(1e-5, evals_sorted[0]), evals_sorted[limit_idx], 200)
    P_val = P_gamma(x_eval) # 平滑化された経験的密度

    def objective(sigma2_val):
        sigma2_val = sigma2_val[0]
        if sigma2_val <= 0:
            return np.inf
        g_val = mp_pdf(x_eval, sigma2_val)
        # [P(gamma_i) - g(gamma_i)]^2 の和
        return np.sum((P_val - g_val)**2)

    # 5. 最適化の実行
    # 初期値としてバルクの平均値を仮置き
    initial_sigma2 = np.mean(evals_sorted[:limit_idx])
    res = minimize(objective, x0=[initial_sigma2], bounds=[(1e-5, None)])
    
    sigma2_hat = res.x[0]
    
    return {
        "sigma2_hat": sigma2_hat,
        "P_gammna": P_gamma,
        "mp.pdf":mp_pdf,
        "x_eval":x_eval
    }