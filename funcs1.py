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
import random
from functools import lru_cache
import warnings

# ※事前に dyson_equalizer_algorithm1 および evaluater.py の ppl_eval がインポート可能な前提です
from evaluater import ppl_eval

# ==========================================
# 補助関数 1: apply_lra_1
# 1つの重み行列(Tensor)に対してLRAを行い、新しい重み行列を返す純粋な数学関数
# ==========================================

@torch.no_grad()
def apply_lra_1(W_raw, s_hat, DE=True, fast_SVD=False):
    """Colab互換: 元のshape/device/dtypeで切断SVDの近似行列を返す。"""
    if W_raw.ndim != 2 or not W_raw.is_floating_point():
        raise ValueError("W_raw must be a real floating-point matrix")
    rank = min(W_raw.shape)
    if int(s_hat) != s_hat or not 0 <= s_hat <= rank:
        raise ValueError("s_hat must be an integer between 0 and min(shape)")
    s_hat = int(s_hat)
    if not torch.isfinite(W_raw).all():
        raise ValueError("W_raw contains NaN/Inf")
    # 修正: rank=0/fullでは分解不要。full rankは丸め誤差なしで元の重みを返す。
    if s_hat == 0:
        return torch.zeros_like(W_raw)
    if s_hat == rank:
        return W_raw.detach().clone()
    transposed = W_raw.shape[0] > W_raw.shape[1]
    W = W_raw.detach().T if transposed else W_raw.detach()
    if DE:
        Y, x, y = dyson_equalizer_algorithm1(W.float().cpu().numpy(), full_matrices=False)
        matrix = torch.as_tensor(Y, dtype=torch.float32, device=W.device)
    else:
        matrix = W.float()
    if fast_SVD:
        # 修正: oversamplingが行列サイズを超えないよう制限する。
        q = min(max(s_hat + 5, 2 * s_hat), rank)
        U, values, V = torch.svd_lowrank(matrix, q=q)
        Vh = V.T
    else:
        # DE側は既存のCPU float32 SVDを維持する。
        svd_matrix = matrix.cpu() if DE else matrix
        U, values, Vh = torch.linalg.svd(svd_matrix, full_matrices=False)
    # 高速化: diag(S)の確保と余分な行列積を列ごとの積に置換。
    result = ((U[:, :s_hat] * values[:s_hat]) @ Vh[:s_hat]).to(W.device)
    if DE:
        result *= torch.as_tensor(np.sqrt(x), dtype=result.dtype, device=result.device)[:, None]
        result *= torch.as_tensor(np.sqrt(y), dtype=result.dtype, device=result.device)[None, :]
    if transposed:
        result = result.T
    # 修正: 全dtypeをfp16上限でclipしていた。異常値を隠さず検出する。
    if not torch.isfinite(result).all() or result.abs().max() > torch.finfo(W_raw.dtype).max:
        raise FloatingPointError("LRA result cannot be represented in the weight dtype")
    return result.to(dtype=W_raw.dtype, device=W_raw.device)


def get_lra_model(model, layer_name, W_hat_new):
    """
    指定されたLinear層の重みを安全に更新する
    """
    for name, module in model.named_modules():
        if name == layer_name and isinstance(module, nn.Linear):

            # shapeチェック
            if module.weight.shape != W_hat_new.shape:
                raise ValueError(
                    f"Shape mismatch: {module.weight.shape} vs {W_hat_new.shape}"
                )

            # device / dtypeを合わせる
            W_hat_new = W_hat_new.to(module.weight.device).to(module.weight.dtype)

            # 安全にコピー
            with torch.no_grad():
                module.weight.copy_(W_hat_new)

            break

    return model

# ==========================================
# 補助関数 3: get_ppl
# モデルのPerplexityを計算するラッパー関数
# ==========================================
def get_ppl(model, tokenizer, dataset_name='wikitext2', seq_len=2048, batch_size=4):
    """
    evaluater.py の ppl_eval を用いてPPLを計算する。
    """
    print(f"  > 評価中 ({dataset_name})...", end="")
    # ppl_eval は辞書を返す仕様（例: {'wikitext2': 12.34}）
    ppl_dict = ppl_eval(model, tokenizer, datasets=[dataset_name], model_seq_len=seq_len, batch_size=batch_size)
    ppl_val = ppl_dict[dataset_name]
    print(f" PPL: {ppl_val:.4f}")
    return ppl_val

# ==========================================
# メイン関数: run_lra_experiment
# ==========================================
def run_lra_experiment(model, tokenizer, results_df, lra_list, max_lra_layers, dataset_name='wikitext2', DE=True,fast_SVD = False, PPLcalc = True,  seq_len=2048, batch_size=4):
    """
    リストの順序に従って1層ずつLRAを適用し、PPLの推移を記録する。

    Parameters:
        model: 対象のPyTorchモデル
        tokenizer: トークナイザー
        results_df: get_esd_metrics 等で計算した alpha と s_hat が含まれるDataFrame
        lra_list: 圧縮を行う順番に並んだレイヤー名のリスト (list of str)
        max_lra_layers: 実験を行う最大層数
        DE: Trueならs_hat_postDE, Falseならs_hat_preDE
        fast_SVD: 
        PPLcalc: 一層LoRAするごとにPPLを計算するかどうか
    Returns:
        history_df: 実験結果の推移をまとめたDataFrame
    """
    results_indexed = results_df.set_index('name', verify_integrity=True)
    history = []
    
    # 💡 [追加] モデル全体の元のパラメータ数を計算
    total_original_params = sum(p.numel() for p in model.parameters())
    current_total_params = total_original_params
    
    print(f"【Step 0】ベースライン (圧縮なし) のPPLを計算します... (総パラメータ数: {total_original_params:,})")
    baseline_ppl = get_ppl(model, tokenizer, dataset_name,seq_len, batch_size)
    history.append({
        "step": 0,
        "layer_compressed": "baseline",
        "alpha_of_layer": np.nan,
        "ppl": baseline_ppl,
        "reduction_ratio_percent": 0.0  # 💡 [追加] ベースラインは削減率 0%
    })
    
    # 実行するレイヤー数を制限
    if max_lra_layers < 0:
        raise ValueError("max_lra_layers must be nonnegative")
    target_layers = lra_list[:max_lra_layers]
    if len(set(target_layers)) != len(target_layers):
        raise ValueError("lra_list contains duplicate layers (would double-count savings)")
    # 高速化: 各stepで全モジュールを探索せず一度だけ辞書化。
    modules = dict(model.named_modules())
    
    for i, layer_name in enumerate(target_layers):
        step = i + 1
        print(f"\n【Step {step}/{max_lra_layers}】層を圧縮中: {layer_name}")
        
        # 1. 圧縮に必要なメタデータ(s_hat, alpha)を取得
        if layer_name not in results_indexed.index:
            print(f"  [警告] {layer_name} の解析結果が見つかりません。スキップします。")
            history.append({
                "step": step,
                "layer_compressed": f"{layer_name} (スキップ)",
                "alpha_of_layer": np.nan,
                "ppl": history[-1]['ppl'], # 前回のPPLを引き継ぎ
                "reduction_ratio_percent": history[-1]['reduction_ratio_percent']
            })
            continue
            
        row = results_indexed.loc[layer_name]
        alpha = row['alpha']
        s_col = 's_hat_postDE' if DE else 's_hat_preDE'
        s_hat = int(row[s_col])
        
        # 2. モデルから元の重みを取得
        target_module = modules.get(layer_name)
        if target_module is None:
            print(f"  [警告] モジュール {layer_name} がモデル内に見つかりません。")
            continue
            
        W_raw = target_module.weight.data
        m_orig, n_orig = W_raw.shape
        
        # 💡 [追加] パラメータ削減量の計算
        original_layer_params = m_orig * n_orig
        compressed_layer_params = s_hat * (m_orig + n_orig)
        params_saved = original_layer_params - compressed_layer_params
        
        if params_saved > 0:
            current_total_params -= params_saved
            
        current_reduction_ratio = (1.0 - current_total_params / total_original_params) * 100
        
        # --- LRAの適用とPPL計算 ---
        W_hat = apply_lra_1(W_raw, s_hat, DE=DE, fast_SVD=fast_SVD)
        # 高速化: 既に解決したmoduleへ直接copyし、全moduleの再探索を避ける。
        with torch.no_grad():
            target_module.weight.copy_(W_hat)
        if PPLcalc:
            current_ppl = get_ppl(model, tokenizer, dataset_name, seq_len, batch_size)
        else:
            current_ppl = None
            print('skip PPL calculation')

        
        # 結果を保存
        history.append({
            "step": step,
            "layer_compressed": layer_name,
            "alpha_of_layer": alpha,
            "ppl": current_ppl,
            "reduction_ratio_percent": current_reduction_ratio # 💡 [追加] 削減率を保存
        })
        
        print(f"  -> 現在の累積パラメータ削減率: {current_reduction_ratio:.2f}%")
        
    if PPLcalc == False:
        print("==最終的なPPL==")
        current_ppl = get_ppl(model, tokenizer, dataset_name, seq_len, batch_size)
        # 修正: 最終PPLを計算するだけで捨てていたため履歴にも保存する。
        history[-1]['ppl'] = current_ppl

    print("\n✅ 実験完了！")
    return pd.DataFrame(history)




from copy import deepcopy
import pandas as pd

from prune_utils import alpha_prune_llama, check_sparsity

def run_pruning_experiment(
    model,
    tokenizer,
    results_df,
    dataset_name='wikitext2',
    seq_len=2048,
    batch_size=4,
    sparsity=0.5,
    alpha_prune=True,
    prune_metric="magnitude",
    blockwise=False,
    alpha_reverse=False,
):
    """
    Alpha Pruning → PPL評価 をまとめて実行する。

    Parameters
    ----------
    model : nn.Module
        対象モデル
    tokenizer
        tokenizer
    results_df : pd.DataFrame
        get_esd_metrics() の結果
    dataset_name : str
    seq_len : int
    batch_size : int
    sparsity : float
    alpha_prune : bool
    prune_metric : str
        "magnitude" or "random"
    blockwise : bool

    Returns
    -------
    pd.DataFrame
        各 sparsity に対する PPL を記録した DataFrame
    1つの sparsity に対して、
    pruning -> log_info print -> PPL計算
    を実行する関数。

    注意:
        この関数は model を in-place に枝刈りする。
        複数 sparsity で独立実験したい場合は、
        外側の for ループで毎回 model をロードし直すこと。
    """

    print("\n" + "=" * 80)
    print(f"Target sparsity = {sparsity:.3f}")
    print("=" * 80)

    model.eval()

    # 枝刈り
    with torch.no_grad():
        pruned_model, log_info = alpha_prune_llama(
            model,
            results_df,
            sparsity=sparsity,
            alpha_prune=alpha_prune,
            prune_metric=prune_metric,
            blockwise=blockwise,
            alpha_reverse=alpha_reverse,
        )

    # 枝刈り完了時点で log_info を print
    print("\n[Pruning finished. log_info]")
    print(log_info)

    # 実際の sparsity を計算
    actual_sparsity = check_sparsity(pruned_model)

    print(f"Target sparsity : {sparsity:.6f}")
    print(f"Actual sparsity : {actual_sparsity:.6f}")

    # PPL 計算
    ppl = get_ppl(
        pruned_model,
        tokenizer,
        dataset_name=dataset_name,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    result = {
        "target_sparsity": sparsity,
        "actual_sparsity": actual_sparsity,
        "ppl": ppl,
        "alpha_prune": alpha_prune,
        "alpha_reverse": alpha_reverse,
        "prune_metric": prune_metric,
        "blockwise": blockwise,
    }

    return result



def _fit_power_law(eigs, method, filter_zeros, bins):
    # 修正: 対数は正の値だけで計算し、ゼロ行列・定数列は推定不能(NaN)とする。
    values = eigs[eigs > (1e-8 if filter_zeros else 0)]
    if len(values) < 2:
        return np.nan, np.nan
    logs = np.log(values)
    N = len(values)
    tails = np.cumsum(logs[::-1])[::-1]
    def estimate(i):
        denom = tails[i] - (N - i) * logs[i]
        return 1 + (N - i) / denom if denom > 1e-12 else np.nan
    if method == 'median':
        i = N // 2
    elif method == 'fix-finger':
        # 修正: CUDA非対応のtorch.histogramをCPUのNumPyへ移し、bins引数を反映。
        hist, edges = np.histogram(values, bins=bins)
        i = min(np.searchsorted(values, edges[np.argmax(hist)]), max(0, N - 3))
    else:
        best, i = np.inf, N // 2
        for j in range(int(N * .1), min(int(N * .9) + 1, N - 1)):
            alpha = estimate(j)
            if not np.isfinite(alpha):
                continue
            cdf = 1 - (values[j:] / values[j]) ** (1 - alpha)
            count = N - j
            # 修正: KSは経験CDFのジャンプの前後を両方評価する。
            distance = max(np.max(np.arange(1, count + 1) / count - cdf),
                           np.max(cdf - np.arange(count) / count))
            if distance < best:
                best, i = distance, j
    return estimate(i), float(values[i])


def _mp_ks(evals, gamma, sigma2):
    if sigma2 <= 0:
        return 0.0 if np.all(evals == 0) else np.nan
    x, cdf = _mp_grid(gamma)
    theory = np.interp(evals / sigma2, x, cdf, left=0, right=1)
    N = len(evals)
    # 修正: 旧実装の右側のみの比較ではKS距離が過小評価される。
    return float(max(np.max(np.arange(1, N + 1) / N - theory),
                     np.max(theory - np.arange(N) / N)))


@torch.no_grad()
def get_esd_metrics(model, pl_fitting='median', conv_norm=1.0, filter_zeros=True, bins=100):
    """Linear/Conv2dのESDを解析。既存の列名・DataFrame形式を維持。

    eigsとspectral_normは従来どおり非正規化の特異値二乗。
    BEMA/KSはY Y.T/nの固有値。Conv2dはout×(in*kh*kw)へ展開する。
    """
    if pl_fitting not in {'median', 'fix-finger', 'goodness-of-fit'}:
        raise ValueError("Unknown pl_fitting method")
    if bins < 1 or conv_norm <= 0:
        raise ValueError("bins and conv_norm must be positive")
    columns = ['name', 'spectral_norm', 'entropy', 'stable_rank', 'weighted_alpha',
               'alpha_method', 'alpha', 'tail_xmin', 'eigs', 'eigs_num',
               'sigma2_preDE', 's_hat_preDE', 's_hat_ratio_preDE', 'threshold_preDE',
               'KS_preDE', 'mp_soft_rank_preDE', 'sigma2_postDE', 's_hat_postDE',
               's_hat_ratio_postDE', 'threshold_postDE', 'KS_postDE', 'KS_postDE_1']
    rows = []
    # 修正: 重みの解析にeval()/model全体のdevice移動は不要。学習状態を変えない。
    for name, layer in model.named_modules():
        if 'lm_head' in name or not isinstance(layer, (nn.Linear, nn.Conv2d)):
            continue
        print(f"Analyzing layer: {name}")
        Y = layer.weight.detach().float().cpu().numpy().astype(np.float64)
        if isinstance(layer, nn.Conv2d):
            # 修正: 旧実装は3Dのままp,nに展開して例外になっていた。
            Y = Y.reshape(Y.shape[0], -1) * np.sqrt(conv_norm)
        if Y.shape[0] > Y.shape[1]:
            Y = Y.T
        if not np.isfinite(Y).all():
            raise ValueError(f"Non-finite weights: {name}")
        p, n = Y.shape
        # 高速化: preDEのSVDをDEにも再利用。Gram行列の重複固有値分解を削除。
        U, sigma, Vh = np.linalg.svd(Y, full_matrices=False)
        eigs = np.sort(sigma**2)
        pre_evals = eigs / n
        pre = bema_algorithm1_from_eigenvalues(pre_evals, p, n)
        post_Y, _, _ = _dyson_from_svd(Y, U, sigma, Vh)
        post_evals = np.sort(np.linalg.svd(post_Y, compute_uv=False)**2) / n
        post = bema_algorithm1_from_eigenvalues(post_evals, p, n)
        alpha, xmin = _fit_power_law(eigs, pl_fitting, filter_zeros, bins)
        norm = float(eigs[-1])
        total = sigma.sum()
        probs = sigma / total if total > 0 else np.zeros_like(sigma)
        positive = probs > 0
        row = dict(name=name, spectral_norm=norm,
                   entropy=float(-np.sum(probs[positive] * np.log(probs[positive]))),
                   stable_rank=float(eigs.sum() / (norm + 1e-8)),
                   # 修正: log(1+exp(x))のoverflowをlogaddexpで回避。
                   weighted_alpha=float(np.logaddexp(0, alpha * np.log10(norm + 1e-12))) if np.isfinite(alpha) else np.nan,
                   alpha_method=pl_fitting, alpha=alpha, tail_xmin=xmin,
                   eigs=eigs, eigs_num=len(eigs),
                   mp_soft_rank_preDE=pre['threshold'] / pre_evals[-1] if norm > 0 else np.nan)
        for label, result, values in [('preDE', pre, pre_evals), ('postDE', post, post_evals)]:
            row.update({f'sigma2_{label}': result['sigma2_hat'],
                        f's_hat_{label}': result['s_hat'],
                        f's_hat_ratio_{label}': result['s_hat'] / p,
                        f'threshold_{label}': result['threshold'],
                        f'KS_{label}': _mp_ks(values, p / n, result['sigma2_hat'])})
        row['KS_postDE_1'] = _mp_ks(post_evals, p / n, 1.0)
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


@torch.no_grad()
def apply_lra(model, results, alpha_threshold=2.0, DE=True, fast_SVD=True):
    """α > thresholdのLinearを近似。戻り値(model, names, param_dict)は維持。"""
    indexed = results.set_index('name', verify_integrity=True)
    names, params = [], {}
    for name, layer in model.named_modules():
        if not isinstance(layer, nn.Linear):
            continue
        bias_count = layer.bias.numel() if layer.bias is not None else 0
        params[name] = layer.weight.numel() + bias_count
        if name not in indexed.index:
            continue
        row = indexed.loc[name]
        rank = row.get('s_hat_postDE' if DE else 's_hat_preDE', np.nan)
        if pd.isna(rank):
            continue
        if int(rank) != rank or not 0 <= rank <= min(layer.weight.shape):
            raise ValueError(f"Invalid estimated rank for {name}: {rank}")
        if row['alpha'] > alpha_threshold and 0 < rank < min(layer.weight.shape):
            # 修正: 重複したSVD処理を共通化しParameter自体は置換しない。
            layer.weight.copy_(apply_lra_1(layer.weight, int(rank), DE, fast_SVD))
            names.append(name)
            # 互換性: この旧APIは独立した特異値ベクトルのr個も数える。
            params[name] = int(rank) * (sum(layer.weight.shape) + 1) + bias_count
    return model, names, params


def _dyson_from_svd(Y, U, sigma, Vh):
    """get_esd_metricsとDEで同じSVDを再利用する内部関数。"""
    m, n = Y.shape
    eta = np.median(sigma)
    if eta == 0:
        if not np.any(sigma):
            return Y.copy(), np.ones(m), np.ones(n)
        raise ValueError("DE requires a positive median singular value")
    term1 = eta / (sigma**2 + eta**2)
    g1 = (U**2) @ term1
    g2 = 1 / eta + (Vh.T**2) @ (term1 - 1 / eta)
    dx = max(m - eta * np.sum(np.abs(g1)), 1e-12)
    dy = max(n - eta * np.sum(np.abs(g2)), 1e-12)
    x = np.maximum((1 / np.maximum(g1, 1e-12) - eta) / np.sqrt(dx), 1e-12)
    y = np.maximum((1 / np.maximum(g2, 1e-12) - eta) / np.sqrt(dy), 1e-12)
    return Y / (np.sqrt(x[:, None]) * np.sqrt(y[None, :])), x, y


def dyson_equalizer_algorithm1(Y, full_matrices=True):
    """Dyson Equalizer。戻り値(Y_hat, x_hat, y_hat)。入力はm <= n。"""
    Y = np.asarray(Y, dtype=np.float64)
    if Y.ndim != 2 or min(Y.shape) == 0 or Y.shape[0] > Y.shape[1]:
        raise ValueError("Y must be a nonempty matrix with m <= n")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains NaN/Inf")
    # 高速化: 必要なのは右特異ベクトルの先頭m本だけ。n×nのVを作らない。
    # full_matrices引数は既存Colabとの互換性のため受け付けるが、常にthin SVD。
    U, sigma, Vh = np.linalg.svd(Y, full_matrices=False)
    return _dyson_from_svd(Y, U, sigma, Vh)


def bema_loss(sigma2_proposal, evals_emp, gamma, p, alpha):
        """
        BEMAの損失関数 (BEMA.Rの `loss` 関数に相当)
        提案された分散 sigma^2 に基づいてMP分布に従うランダム行列をシミュレートし，
        経験的固有値のバルク部分（分位数）との二乗誤差を計算します．
        """
        n = int(p / gamma)
        L = np.zeros((10, min(p, n)))
        
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
    warnings.warn('apply_bema is the legacy stochastic Monte Carlo fit; '
                  'use bema_algorithm1_from_eigenvalues for the Colab production path.',
                  RuntimeWarning, stacklevel=2)
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
    if gamma <= 0 or p_tilde <= 0:
        raise ValueError('gamma and p_tilde must be positive')
    # 修正: x軸の等間隔積分はgamma=1の端点特異性で精度が落ちる。
    # theta変換による共通のMP分位点を利用する（gamma>1は非ゼロ部分へ条件付け）。
    ratio = min(gamma, 1 / gamma)
    scale = max(gamma, 1.0)
    return qmp_stable(1 - np.asarray(k_indices, dtype=float) / p_tilde,
                      1.0, ratio, var=scale, grid_size=grid_size)

@lru_cache(maxsize=16)
def _mp_grid(gamma, grid_size=200000):
    # 高速化: 同じ行列形状で繰り返す20万点のMP積分を再利用。
    a = (1.0 - np.sqrt(gamma)) ** 2
    b = (1.0 + np.sqrt(gamma)) ** 2

    # x = a + (b-a)(1-cos(theta))/2
    # gamma=1 の x=0 特異性を避ける
    eps = 1e-6
    theta = np.linspace(eps, np.pi - eps, grid_size)

    x = a + (b - a) * (1.0 - np.cos(theta)) / 2.0
    dx_dtheta = (b - a) * np.sin(theta) / 2.0

    denom = np.maximum(x, np.finfo(float).tiny)

    pdf = np.sqrt(np.maximum((b - x) * (x - a), 0.0)) / (
        2.0 * np.pi * gamma * denom
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

    return x_unique, cdf_unique


def qmp_stable(probs, ndf, pdim, var=1.0, grid_size=200000):
    """標準MPの下側分位点。旧APIと積分精度を維持する。"""
    if not 0 < pdim <= ndf or var < 0 or grid_size < 3:
        raise ValueError("Require 0 < pdim <= ndf, var >= 0, grid_size >= 3")
    x, cdf = _mp_grid(pdim / ndf, grid_size)
    return var * np.interp(np.clip(np.asarray(probs, dtype=float), 0, 1), cdf, x)


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
    if p <= 0 or n <= 0 or not 0 <= alpha < 0.5 or not 0 < beta < 1:
        raise ValueError("Require positive dimensions, 0 <= alpha < .5, 0 < beta < 1")
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
    # 修正: 未検証の旧推定法を確定した結果と誤認しないよう明示する。
    warnings.warn('gaussian_broadening_fit is experimental and not statistically validated.',
                  RuntimeWarning, stacklevel=2)
    if len(evals) < 2 or gamma_ratio <= 0 or a < 1 or int(a) != a:
        raise ValueError('Need >=2 eigenvalues, positive gamma and integer a >= 1')
    a = int(a)
    m = len(evals)
    evals_sorted = np.sort(np.asarray(evals, dtype=float))
    if not np.isfinite(evals_sorted).all() or np.any(evals_sorted < 0):
        raise ValueError('Eigenvalues must be finite and nonnegative')
    
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