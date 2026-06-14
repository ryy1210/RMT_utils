import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import scipy.stats
import scipy.optimize as opt
import scipy.interpolate as interp
from scipy import integrate
from scipy.stats import norm
from scipy.optimize import minimize, root_scalar
from tqdm.auto import tqdm


def get_esd_metrics(model, pl_fitting='median', conv_norm=1.0, filter_zeros=True, bins=100):
    """
    モデル内の全線形層（Conv2d, Linear）に対してESD関連のメトリクスを計算する関数
    Args:
        model (torch.nn.Module): 解析対象のPyTorchモデル（LLaMAなどのLLMも可）
        pl_fitting (str): べき指数 alpha を計算・フィッティングする手法 
                          ('median', 'fix-finger', 'goodness-of-fit' )
        conv_norm (float): 畳み込み層の次元変換時に適用する正規化係数
        filter_zeros (bool): ゼロや極小な特異値（ノイズ）を事前計算から除外するかどうか
        bins (int): ヒストグラム計算や fix-finger で使用するビンの数

    Returns:
        dict: 以下のキーと対応する各層のメトリクス（リスト）を格納した辞書

        [基本指標 (HT-SR理論のShape / Scale Metrics)]
        - 'name': 解析対象のモジュール（層）名
        - 'spectral_norm': スペクトルノルム（最大固有値）．層が持つ最大シグナルの絶対的な強さ．
        - 'entropy': 行列エントロピー．低いほど一部の特異値に情報が集中している（低ランク性が高い）．
        - 'stable_rank': 安定ランク．フロベニウスノルム^2 / スペクトルノルム^2．
        - 'alphahat': アルファハット．alpha と spectral_norm を統合したモデル汎化性能の予測指標．scaleに依存する
        - 'alpha_method': alpha を計算した際の手法の記録．
        - 'tail_xmin': alpha計算の際にESDのtailが始まっていると判定した値
        - 'alpha': べき指数．特異値分布の「裾の重さ」．小さいほど有用な特徴を強く学習している．scale不変
        - 'eigs': 特異値の2乗（固有値）の配列．
        - 'eigs_num': 固有値の総数．

        [Phase 1: Dyson Equalizer 適用前 (preDE) のRMT指標]
        - 'sigma2_preDE': BEMAによって推定された，ノイズ成分の分散．
        - 's_hat_preDE': 閾値を超えた「純粋なシグナル」とみなされる特異値の数．
        - 's_hat_ratio_preDE': 全特異値数に対するシグナル数 (s_hat) の割合（情報密度）．
        - 'threshold_preDE': ノイズとシグナルを分離する境界閾値（Tracy-Widom補正込み）．
        - 'KS_preDE': 実データの経験的CDFと，理論的なMP分布のCDFとのKS距離．
        - 'mp_soft_rank_preDE': MP Soft Rank = threshold_preDE / eig_max

        [Phase 2: Dyson Equalizer 適用後 (postDE) のRMT指標]
        - 'sigma2_postDE': DE適用後の重み行列に対する，BEMA推定ノイズ分散．
        - 's_hat_postDE': DE適用後のシグナル特異値の数．
        - 's_hat_ratio_postDE': DE適用後のシグナル割合．低ランク近似(LRA)時のランク決定の根拠となる．
        - 'threshold_postDE': DE適用後のノイズ/シグナル境界閾値．
        - 'KS_postDE': DE適用後の経験的CDFと，sigma2_postDEを用いた理論的MP分布とのKS距離．
        - 'KS_postDE_1': DE適用後の経験的CDFと，分散を1.0に固定した理論的MP分布とのKS距離．
                         (DEによるノイズ分散の均一化が完全に機能したかを確認する指標)
    """
    results = {
        'name': [],
        'spectral_norm': [],
        'entropy': [],
        'stable_rank': [],
        'alphahat': [],
        'alpha_method':[],
        'alpha': [],
        'tail_xmin':[],
        'eigs': [],
        'eigs_num': [],
        'sigma2_preDE':[],
        's_hat_preDE':[],
        's_hat_ratio_preDE': [], 
        'threshold_preDE':[],
        'KS_preDE':[],
        'mp_soft_rank_preDE': [],
        'sigma2_postDE':[],
        's_hat_postDE':[],
        's_hat_ratio_postDE': [], 
        'threshold_postDE':[],
        'KS_postDE':[],
        'KS_postDE_1':[]
    }
    
    # 補助関数
    def safe_log10(x):
        return torch.log10(x + 1e-12)

    def matrix_entropy(eigs):
        p = eigs / torch.sum(eigs)
        return -torch.sum(p * torch.log(p + 1e-12))
    
    # 【高速化】MP分布の累積分布関数（CDF）を計算する内部関数
    def calc_mp_cdf_fast(evals_sorted, gamma, sigma2):
        lambda_minus = sigma2 * (1 - math.sqrt(gamma))**2
        lambda_plus = sigma2 * (1 + math.sqrt(gamma))**2
        
        tcdf = np.zeros_like(evals_sorted, dtype=float)
        current_cdf = 0.0
        last_x = lambda_minus
        
        for i, x in enumerate(evals_sorted):
            if x <= lambda_minus:
                tcdf[i] = 0.0
            elif x >= lambda_plus:
                tcdf[i] = 1.0
            else:
                # LLaMAのような巨大行列でも積分計算が重くならないよう，
                # 前の固有値から現在の固有値までの区間だけを積分して加算します
                val, _ = integrate.quad(mp_pdf_zero_excluded, last_x, x, args=(gamma, sigma2), limit=50)
                current_cdf += val
                tcdf[i] = current_cdf
                last_x = x
                
        return np.clip(tcdf, 0.0, 1.0) # 積分誤差による1.0超過を防ぐ

    # KSダイバージェンス距離を計算する内部関数
    def calc_ks_distance(evals_sorted, gamma, sigma2):
        # 経験的CDF (1/p, 2/p, ..., p/p)
        ecdf = np.arange(1, len(evals_sorted) + 1) / len(evals_sorted)
        # 理論的CDF
        tcdf = calc_mp_cdf_fast(evals_sorted, gamma, sigma2)
        # KS距離 (経験CDFと理論CDFの最大絶対誤差)
        return np.max(np.abs(ecdf - tcdf))

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad(): # VRAM節約のため必ず勾配計算をオフにする
        # レイヤーのループ
        for name, m in model.named_modules():
            if "lm_head" in name:
                print(f"Skipping extremely large layer: {name}")
                continue

            if isinstance(m, (nn.Conv2d, nn.Linear)):
                print(f"Analyzing layer: {name}")
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

                elif pl_fitting == 'fix-finger':
                    # --- Fix-finger 法 ---
                    # ESD(経験的スペクトル密度)のピークを視覚的・経験的に特定し，そこを xmin とする手法
                    # PyTorchのヒストグラム計算を利用してピークのビンを特定します
                    hist, bin_edges = torch.histogram(nz_eigs, bins=100)
                    peak_bin_idx = torch.argmax(hist)
                    
                    # ピークとなるビンの左端を閾値 xmin とみなす
                    xmin_val = bin_edges[peak_bin_idx]
                    
                    # nz_eigsは昇順なので，xmin_val以上の最初のインデックス i を取得
                    i = torch.searchsorted(nz_eigs, xmin_val).item()
                    
                    # 全てがノイズとして切り捨てられないよう，最低限の要素数を確保する安全弁
                    if i >= N - 2:
                        i = N - 3
                    
                    xmin = nz_eigs[i]
                    n = float(N - i)
                    final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])

                elif pl_fitting == 'goodness-of-fit':
                    # --- Goodness-of-fit (KS距離最小化) 法 ---
                    # Clausetら(2009)の厳密な手法．すべての xmin 候補に対してモデルと実際のデータの
                    # コルモゴロフ・スミルノフ(KS)距離を計算し，距離が最小となる xmin を採用します
                    
                    best_ks = float('inf')
                    best_i = int(N / 2)
                    
                    # 計算効率化のため，対数の累積和を事前に計算してループ内の合計計算を O(1) にする
                    cumsum_log = torch.cumsum(log_nz_eigs, dim=0)
                    total_log_sum = cumsum_log[-1]
                    
                    # 端すぎる値(テールの要素数が少なすぎる/多すぎる)を除外するため，
                    # 実用上は全体の 10% 〜 90% の範囲を探索するのが安定的かつ高速です
                    start_idx = int(N * 0.1)
                    end_idx = int(N * 0.9)
                    
                    for i in range(start_idx, end_idx):
                        n_i = float(N - i)
                        xmin_i = nz_eigs[i]
                        
                        # 分母の log_nz_eigs[i:] の合計を累積和から高速に取得
                        sum_log = total_log_sum - cumsum_log[i-1]
                        alpha_i = 1.0 + n_i / (sum_log - n_i * log_nz_eigs[i])
                        
                        # KS距離の計算
                        # 経験的CDF (データが小さい順に並んでいるため 1/n, 2/n ... n/n となる)
                        empirical_cdf = torch.arange(1, int(n_i) + 1, device=device) / n_i
                        
                        # 理論的CDF (パレート分布の累積分布関数: 1 - (x / xmin)^(-alpha + 1) )
                        exponent = -(alpha_i - 1.0)
                        theoretical_cdf = 1.0 - torch.pow(nz_eigs[i:] / xmin_i, exponent)
                        
                        # KS統計量: CDFの差の絶対値の最大値
                        ks_dist = torch.max(torch.abs(empirical_cdf - theoretical_cdf)).item()
                        
                        # 最もKS距離が小さい(適合度が高い)インデックスを記録
                        if ks_dist < best_ks:
                            best_ks = ks_dist
                            best_i = i
                    
                    # 最適なインデックスで最終的な alpha を計算
                    i = best_i
                    xmin = nz_eigs[i]
                    n = float(N - i)
                    sum_log = total_log_sum - cumsum_log[i-1]
                    final_alpha = 1.0 + n / (sum_log - n * log_nz_eigs[i])

                else:
                    print("method for alpha is not selected.")
                    final_alpha = torch.tensor(1.0)
                    
                final_alpha_val = final_alpha.item()
                
                # AlphaHat計算
                final_alphahat = final_alpha_val * safe_log10(torch.tensor(spectral_norm)).item()
                final_alphahat = math.log(1.0 + math.exp(final_alphahat))

                # Numpy配列への変換
                Y = matrix.cpu().numpy()
                
                # RMTの標準形式 (p <= n になるように転置)
                p, n = Y.shape
                if p > n:
                    Y = Y.T
                    p, n = Y.shape
                gamma = p / n

                 # ----------------------------------------------------
                # [Phase 1] preDE (Dyson Equalizer適用前) の解析
                # ----------------------------------------------------
                # BEMAによる推定 (辞書型から値を取り出す)
                bema_res_pre = bema_algorithm1_from_data(Y)
                sigma2_pre = bema_res_pre["sigma2_hat"]
                threshold_pre = bema_res_pre["threshold"]
                s_hat_pre = bema_res_pre["s_hat"]
                s_hat_ratio_pre = s_hat_pre / p
                
                # 相関行列の固有値 (X = Y Y^T / n) を計算し，昇順ソート
                X_pre = (Y @ Y.T) / n
                evals_pre = np.sort(np.linalg.eigvalsh(X_pre).real)
                
                # KS距離の計算
                ks_pre = calc_ks_distance(evals_pre, gamma, sigma2_pre)

                max_eig_pre = np.max(evals_pre) if len(evals_pre) > 0 else 0.0

                if max_eig_pre > 0:
                    mp_soft_rank_pre = threshold_pre / max_eig_pre
                else:
                    mp_soft_rank_pre = np.nan

                # ----------------------------------------------------
                # [Phase 2] postDE (Dyson Equalizer適用後) の解析
                # ----------------------------------------------------
                # DEの適用（Y_hatのみを受け取る）
                Y_post, _, _ = dyson_equalizer_algorithm1(Y)
                
                # BEMAによる推定 (辞書型から値を取り出す)
                bema_res_post = bema_algorithm1_from_data(Y_post)
                sigma2_post = bema_res_post["sigma2_hat"]
                threshold_post = bema_res_post["threshold"]
                s_hat_post = bema_res_post["s_hat"]
                s_hat_ratio_post = s_hat_post / p
                
                # 固有値の計算と昇順ソート
                X_post = (Y_post @ Y_post.T) / n
                evals_post = np.sort(np.linalg.eigvalsh(X_post).real)
                
                # KS距離の計算 (BEMA推定分散)
                ks_post = calc_ks_distance(evals_post, gamma, sigma2_post)
                # KS距離の計算 (分散=1.0固定)
                ks_post_1 = calc_ks_distance(evals_post, gamma, 1.0)

                
                # 結果の保存
                results['name'].append(name)
                results['spectral_norm'].append(spectral_norm)
                results['entropy'].append(entropy.detach().cpu().item())
                results['stable_rank'].append(stable_rank)
                results['alphahat'].append(final_alphahat)
                results['alpha_method'].append(pl_fitting)
                results['alpha'].append(final_alpha_val)
                results['tail_xmin'].append(xmin)
                results['eigs'].append(eigs.detach().cpu().numpy())
                results['eigs_num'].append(len(eigs))

                results['sigma2_preDE'].append(sigma2_pre)
                results['s_hat_preDE'].append(s_hat_pre)
                results['s_hat_ratio_preDE'].append(s_hat_ratio_pre)
                results['threshold_preDE'].append(threshold_pre)
                results['KS_preDE'].append(ks_pre)
                results['mp_soft_rank_preDE'].append(mp_soft_rank_pre)
                
                results['sigma2_postDE'].append(sigma2_post)
                results['s_hat_postDE'].append(s_hat_post)
                results['s_hat_ratio_postDE'].append(s_hat_ratio_post)
                results['threshold_postDE'].append(threshold_post)
                results['KS_postDE'].append(ks_post)
                results['KS_postDE_1'].append(ks_post_1)
            
    return pd.DataFrame(results)


def apply_lra(model, results, alpha_threshold=2.0, DE=True, fast_SVD=True):
    """
    model: LRAを対象にするモデル
    results: get_esd_metricsをmodelに適用した結果 pd.DataFrame
    alpha_threshold: 重み行列のalphaでLRAするかどうかの閾値
    DE: s_hatとしてpostDEを使う
    fast_SVD: LRAするために実質s_hat分だけSVDできればよい．GPUだとさらに高速
    """
    lra_layer_name = []
    lra_params = {}
    
    res_indexed = results.set_index('name')
    
    # 対象となる全 Linear 層をリストアップ (tqdmで回すため)
    target_modules = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear)]
    
    print(f"全 {len(target_modules)} 個の Linear 層をスキャン中...")
    
    # tqdm でプログレスバーを表示
    for name, module in tqdm(target_modules, desc="LRA Progress"):
            
        if name not in res_indexed.index:
            lra_params[name] = module.weight.numel()
            if module.bias is not None:
                lra_params[name] += module.bias.numel()
            continue

        row = res_indexed.loc[name]
        alpha = row['alpha']
        
        s_col = 's_hat_postDE' if DE else 's_hat_preDE'
        if s_col not in row or pd.isna(row[s_col]):
            lra_params[name] = module.weight.numel()
            continue
            
        s_hat = int(row[s_col])
        
        W_raw = module.weight.data
        m_orig, n_orig = W_raw.shape
        full_rank = min(m_orig, n_orig)
        
        # 圧縮条件の判定
        if alpha < alpha_threshold and 0 < s_hat < full_rank:
            lra_layer_name.append(name)
            
            device = W_raw.device
            dtype = W_raw.dtype
            
            W_np = W_raw.detach().cpu().to(torch.float32).numpy()

            # W_npにnanなどが入っているとSVDができない
            W_np = np.asarray(W_np)
            W_np = np.nan_to_num(W_np, nan=0.0, posinf=0.0, neginf=0.0)
            W_np = W_np.astype(np.float64, copy=False)
            W_np = np.ascontiguousarray(W_np)

            # これでもエラー出る場合用の確認
            # print("layer:", name)
            # print("shape:", W_np.shape)
            # print("dtype:", W_np.dtype)
            # print("finite:", np.isfinite(W_np).all())
            # print("min/max:", np.min(W_np), np.max(W_np))

            # --- 💡 縦長行列 (m > n) の自動転置 ---
            # DE関数は m <= n を要求するため、縦長なら転置して横長にする
            transposed = False
            if m_orig > n_orig:
                W_np = W_np.T
                transposed = True

            # この時点で W_np は必ず m <= n の形状になる
            m, n = W_np.shape

            
            if DE:
                # 1. Dyson Equalizer
                Y_hat, x_hat, y_hat = dyson_equalizer_algorithm1(W_np)
                Y_hat = np.nan_to_num(Y_hat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 2. SVD と 低ランク近似
                Y_hat_tensor = torch.tensor(Y_hat, device=device, dtype=torch.float32)

                if fast_SVD:
                    U_approx, S_approx, Vh_approx = torch.svd_lowrank(Y_hat_tensor, q=s_hat + 10)

                    U_hat = U_approx[:, :s_hat]
                    S_hat = torch.diag(S_approx[:s_hat])
                    Vh_hat = Vh_approx[:s_hat, :]
                else:
                    U, S, Vh = torch.linalg.svd(Y_hat_tensor, full_matrices=False)

                    
                    U_hat = U[:, :s_hat]
                    S_hat = torch.diag(S[:s_hat])
                    Vh_hat = Vh[:s_hat, :]

                W_tilde = U_hat @ S_hat @ Vh_hat # 形状: (m, n)
                
                # 3. Re-coloring (復元）
                x_hat_flat = np.asarray(x_hat).flatten()
                y_hat_flat = np.asarray(y_hat).flatten()
                
                eps = 1e-12
                # x_hat は長さ m (行)、y_hat は長さ n (列)
                D_x_sqrt = torch.tensor(1.0 / (x_hat_flat + eps), device=device, dtype=torch.float32)
                D_y_sqrt = torch.tensor(1.0 / (y_hat_flat + eps), device=device, dtype=torch.float32)
                
                # ブロードキャスト計算
                # print("W_tilde.shape:", W_tilde.shape)
                # print("D_x_sqrt.shape:", D_x_sqrt.shape)
                # print("D_y_sqrt.shape:", D_y_sqrt.shape)
                W_hat = W_tilde * D_x_sqrt.unsqueeze(1) * D_y_sqrt.unsqueeze(0)
                
            else:
                # 通常の SVD         
                U, S, Vh = torch.linalg.svd(
                    W_raw.to(torch.float32),
                    full_matrices=False
                )

                U_hat = U[:, :s_hat]
                S_hat = torch.diag(S[:s_hat])
                Vh_hat = Vh[:s_hat, :]

                W_hat = U_hat @ S_hat @ Vh_hat


            # --- 元の形状に戻す (転置していた場合) ---
            if transposed:
                W_hat = W_hat.T

            # 重みの更新
            module.weight.data = W_hat.to(dtype)
            
            # 実効パラメータ数の記録
            lra_params[name] = s_hat * (m_orig + n_orig + 1)
            if module.bias is not None:
                lra_params[name] += module.bias.numel()
                
        else:
            lra_params[name] = module.weight.numel()
            if module.bias is not None:
                lra_params[name] += module.bias.numel()

    print(f"\n✅ LRA 完了: 全 {len(res_indexed)} 対象層のうち、{len(lra_layer_name)} 層を圧縮しました。")
    
    return model, lra_layer_name, lra_params


def dyson_equalizer_algorithm1(Y, full_matrices = True):
    """
    Landa & Kluger (2024) - Algorithm 1: The Dyson Equalizer
    論文の数式と記法に完全に対応させた実装．

    Input:
        Y: Data matrix (m x n), m <= n
        full_matrices(bool): SVDを完全に行うか　メモリを効率的にしたいならFalse
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

    Y = np.asarray(Y)
    Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
    Y = Y.astype(np.float64, copy=False)
    Y = np.ascontiguousarray(Y)

    U, sigma, V_h = np.linalg.svd(Y, full_matrices=full_matrices)
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
    denom_x = np.maximum(m - eta * g1_norm1, 1e-12)
    denom_y = np.maximum(n - eta * g2_norm1, 1e-12)

    # g1_hat, g2_hat が 0 になることによるゼロ除算の防止
    g1_hat_safe = np.where(np.abs(g1_hat) < 1e-12, 1e-12, g1_hat)
    g2_hat_safe = np.where(np.abs(g2_hat) < 1e-12, 1e-12, g2_hat)

    x_hat = (1 / np.sqrt(denom_x)) * ((1 / g1_hat_safe) - eta)
    y_hat = (1 / np.sqrt(denom_y)) * ((1 / g2_hat_safe) - eta)

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
        提案された分散 sigma^2 に基づいてMP分布に従うランダム行列をシミュレートし，
        経験的固有値のバルク部分（分位数）との二乗誤差を計算します．
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
        
        # alpha に基づいて，分布の「端（スパイクや微小固有値）」を切り落とす
        # 例: alpha=0.2 の場合，上位20%と下位20%を無視し，中間の60%のバルクだけで比較する
        idx_start = int(min(p, n) * alpha)
        idx_end = int(min(p, n) * (1 - alpha))
        
        evals_emp_bulk = evals_emp[idx_start:idx_end]
        evals_sim_bulk = evals_sim_mean[idx_start:idx_end]
        
        # バルク部分の分位数の二乗誤差
        loss = np.sum((evals_emp_bulk - evals_sim_bulk)**2)
        return loss

def apply_bema(evals_emp, gamma, p, alpha=0.2):
    """
    BEMAアルゴリズムを実行し，真の分散 sigma^2 を推定します．
    """
    print("BEMAによる分散推定を実行中...")
    # scipy.optimize.minimize_scalar を用いて，損失関数を最小化する分散を探索
    res = opt.minimize_scalar(
        bema_loss, 
        args=(evals_emp, gamma, p, alpha), 
        bounds=(0.01, 10.0), 
        method='bounded'
    )
    return res.x


def tw1_quantile(beta=0.1):
    """
    Type-I Tracy-Widom 分布の (1-beta) 分位点を返す．

    scipy に tracywidom がある環境ではそれを使用．
    ない場合は代表的な近似値を使う．
    """
    try:
        from scipy.stats import tracywidom
        return tracywidom.ppf(1 - beta, beta=1)
    except Exception:
        # Type-I Tracy-Widom TW1 の代表的な分位点近似
        # beta は右側確率．つまり返すのは 1-beta quantile．
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
    sigma2 = 1 のとき標準MP分布．

    gamma > 1 の場合，p x p sample covariance にはゼロ固有値が出るので，
    非ゼロ固有値に条件づけた zero-excluded density を使う．
    """
    x = np.asarray(x)

    a = sigma2 * (1 - np.sqrt(gamma)) ** 2
    b = sigma2 * (1 + np.sqrt(gamma)) ** 2

    pdf = np.zeros_like(x, dtype=float)

    mask = (x > a) & (x < b)
    xm = x[mask]

    # classical MP density の正規化係数は 2*pi*gamma*sigma2*x
    # gamma > 1 では非ゼロ部分の質量が 1/gamma なので，
    # zero-excluded にするため gamma 倍する．
    denom_gamma = min(gamma, 1.0)

    pdf[mask] = (
        np.sqrt((b - xm) * (xm - a))
        / (2 * np.pi * denom_gamma * sigma2 * xm)
    )

    return pdf


def mp_upper_quantiles(gamma, p_tilde, k_indices, grid_size=200000):
    """
    sigma2=1 の zero-excluded MP 分布について，
    k/p_tilde upper-quantile q_k を返す．

    k_indices は 1始まりの index を想定．
    """
    a = (1 - np.sqrt(gamma)) ** 2
    b = (1 + np.sqrt(gamma)) ** 2

    eps = 1e-10
    x_grid = np.linspace(a + eps, b - eps, grid_size)

    pdf = mp_pdf_zero_excluded(x_grid, gamma, sigma2=1.0)

    # 数値誤差補正のため，台形積分でCDFを作って正規化
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

def qmp_stable(probs, ndf, pdim, var=1.0, grid_size=200000):
    """
    RMTstat::qmp(probs, ndf=ndf, pdim=pdim, var=var) 相当．
    lower.tail=TRUE の下側分位点を返す．

    ndf >= pdim を想定．
    gamma = pdim / ndf <= 1.
    gamma=1 の下端特異性を避けるため，theta 変換でCDFを作る．
    """
    probs = np.asarray(probs, dtype=float)

    if ndf < pdim:
        raise ValueError("qmp_stable assumes ndf >= pdim. Use ndf=max(p,n), pdim=min(p,n).")

    gamma = pdim / ndf

    a = var * (1.0 - np.sqrt(gamma)) ** 2
    b = var * (1.0 + np.sqrt(gamma)) ** 2

    # x = a + (b-a)(1-cos(theta))/2
    # gamma=1 の x=0 特異性を避ける
    eps = 1e-6
    theta = np.linspace(eps, np.pi - eps, grid_size)

    x = a + (b - a) * (1.0 - np.cos(theta)) / 2.0
    dx_dtheta = (b - a) * np.sin(theta) / 2.0

    denom = np.maximum(x, np.finfo(float).tiny)

    pdf = np.sqrt(np.maximum((b - x) * (x - a), 0.0)) / (
        2.0 * np.pi * gamma * var * denom
    )

    integrand = pdf * dx_dtheta

    cdf = integrate.cumulative_trapezoid(integrand, theta, initial=0.0)
    cdf = cdf / cdf[-1]

    # 正確な端点を追加
    cdf_all = np.r_[0.0, cdf, 1.0]
    x_all = np.r_[a, x, b]

    mask = np.isfinite(cdf_all) & np.isfinite(x_all)
    cdf_all = cdf_all[mask]
    x_all = x_all[mask]

    # np.interp 用に重複CDFを除く
    cdf_unique, idx = np.unique(cdf_all, return_index=True)
    x_unique = x_all[idx]

    probs = np.clip(probs, 0.0, 1.0)
    return np.interp(probs, cdf_unique, x_unique)

def bema_algorithm1_from_eigenvalues(evals, p, n, alpha=0.2, beta=0.1):
    """
    BEMA Algorithm 1.

    evals は sample covariance matrix の固有値:
        S = Y Y^T / n
    または非ゼロ固有値として
        S_small = Y^T Y / n
    の固有値を渡す．

    注意:
        元行列 W の特異値 s_i を使う場合は
            evals = s_i**2 / n
        とする．
    """
    evals = np.asarray(evals, dtype=float)
    evals = evals[np.isfinite(evals)]

    # 数値誤差による微小負固有値を 0 に丸める
    evals = np.maximum(evals, 0.0)

    # 降順
    evals_sorted = np.sort(evals)[::-1]

    min_pn = min(p, n)
    max_pn = max(p, n)
    gamma = min_pn / max_pn

    # R:
    # k = floor(min(p,n)*alpha):floor(min(p,n)*(1-alpha))
    k_start = int(np.floor(min_pn * alpha))
    k_end = int(np.floor(min_pn * (1.0 - alpha)))

    # R の index は 1 始まり
    k_start = max(1, k_start)
    k_end = min(min_pn, k_end)

    if k_start > k_end:
        raise ValueError("Invalid alpha: selected bulk index set is empty.")

    # 重要:
    # 非ゼロ固有値数が min_pn より少なくても，
    # k_end まで存在すれば BEMA 回帰は実行可能
    if len(evals_sorted) < k_end:
        raise ValueError(
            f"Need at least k_end={k_end} eigenvalues for BEMA regression, "
            f"but got {len(evals_sorted)}."
        )

    k_r = np.arange(k_start, k_end + 1)  # R-style 1-index

    # R:
    # predictor = qmp(k/min(p,n), max(n,p), min(n,p)) * max(p,n)/n
    predictor = qmp_stable(
        probs=k_r / min_pn,
        ndf=max_pn,
        pdim=min_pn,
        var=1.0,
    ) * (max_pn / n)

    # R:
    # sigma2hat = lm(rev(l[k]) ~ predictor - 1)$coef[[1]]
    l_k = evals_sorted[k_r - 1]
    y = l_k[::-1]
    x = predictor

    denom = np.dot(x, x)
    if denom <= 0 or not np.isfinite(denom):
        raise FloatingPointError(
            "Invalid predictor: denominator is zero or non-finite."
        )

    sigma2_hat = np.dot(x, y) / denom

    t_tw = tw1_quantile(beta=beta)

    # R:
    # cutoff = sigma2hat * (
    #   (1+sqrt(gamma))^2
    #   + qtw(0.9) * max(p,n)^(-2/3)
    #     * gamma^(-1/6)
    #     * (1+sqrt(gamma))^(4/3)
    # ) * max(p,n)/n
    threshold = sigma2_hat * (
        (1.0 + np.sqrt(gamma)) ** 2
        + t_tw
        * max_pn ** (-2.0 / 3.0)
        * gamma ** (-1.0 / 6.0)
        * (1.0 + np.sqrt(gamma)) ** (4.0 / 3.0)
    ) * (max_pn / n)

    s_hat = int(np.sum(evals_sorted > threshold))

    return {
        "s_hat": s_hat,
        "sigma2_hat": sigma2_hat,
        "threshold": threshold,
        "gamma": gamma,
        "k_indices_R_style": k_r,
        "evals_sorted": evals_sorted,
        "tw_quantile": t_tw,
        "predictor": predictor,
    }


def bema_algorithm1_from_data(Y, alpha=0.2, beta=0.1, center=False):
    """
    Y は p x n として扱う．
    sample covariance は S = Y Y^T / n．
    """
    Y = np.asarray(Y, dtype=float)

    if center:
        Y = Y - Y.mean(axis=1, keepdims=True)

    p, n = Y.shape

    if p <= n:
        S = Y @ Y.T / n
        evals = np.linalg.eigvalsh(S)
    else:
        # 非ゼロ固有値だけ使う
        S_small = Y.T @ Y / n
        evals = np.linalg.eigvalsh(S_small)

    return bema_algorithm1_from_eigenvalues(
        evals=evals,
        p=p,
        n=n,
        alpha=alpha,
        beta=beta,
    )

def gaussian_broadening_fit(evals, gamma_ratio, a=10):
    """
    論文のセクション2.3に基づく Gaussian Broadening と最小二乗法による sigma^2 の推定
    うまく実装できていないので要修正
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
    # フィッティング範囲: スパイクの影響を排除するため，下位 90% のバルク領域でカーブを比較する
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