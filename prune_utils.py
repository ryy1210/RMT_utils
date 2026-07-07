import torch 
import torch.nn as nn 
import os
import numpy as np
import weightwatcher as ww
from layerwrapper import WrappedLayer 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    指定されたモジュール内から、nn.Linearなどの対象層を再帰的に抽出して辞書で返す関数
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    """
    モデル全体の実際のスパースティ（ゼロになっているパラメータの割合）を計算する関数
    """
    subset = find_layers(model, layers=[nn.Linear])
    zero_cnt = 0
    fc_params = 0
    for name in subset:
        W = subset[name].weight.data
        zero_cnt += (W == 0).sum().item()
        fc_params += W.numel()
    return float(zero_cnt) / (fc_params + 1e-8)

def compute_mask(W_metric, prune_granularity, sparsity):
    """
    計算された重要度スコア（W_metric）に基づいて、下位（sparsity）の要素をゼロにするためのマスクを生成する関数
    """
    if sparsity <= 0:
        return torch.zeros_like(W_metric, dtype=torch.bool)
    if sparsity >= 1:
        return torch.ones_like(W_metric, dtype=torch.bool)
        
    thres = torch.sort(W_metric.flatten().cuda())[0][int(W_metric.numel() * sparsity)].cpu()
    W_mask = (W_metric <= thres)
    return W_mask 

# =========================================================================
# 1. 一律枝刈り（Uniform Pruning）用関数
# =========================================================================
def prune_vit_for_vit_pytorch(args, model, calib_data, device):
    """
    vit_pytorchの実装構造に適合させた、一律(Uniform)に各層を同じ割合で枝刈りする関数．
    """
    print("Uniform Pruning (Wanda / Magnitude) begin!")
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])

    metric_stats = []
    for blk in model.transformer.layers:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    # パッチ埋め込み初期処理
    inps = model.to_patch_embedding(inps)
    cls_tokens = model.cls_token.expand(bs, -1, -1)
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embedding
    inps = model.dropout(inps)

    for block_id, blk in enumerate(model.transformer.layers):
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # --- 【核心修正】blk(inps) を vit_pytorch の内部構造に合わせて分解して実行 ---
            attn, ff = blk[0], blk[1]
            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    chunk = inps[i1:j1]
                    chunk = attn(chunk) + chunk
                    chunk = ff(chunk) + chunk
                    tmp_res.append(chunk)
                inps = torch.cat(tmp_res, dim=0)
            else:
                x = attn(inps) + inps
                inps = ff(x) + x

            for h in handles:
                h.remove()     
        else:
            # マグニチュード枝刈り時の順伝播分解
            attn, ff = blk[0], blk[1]
            x = attn(inps) + inps
            inps = ff(x) + x
        
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)
            subset[name].weight.data[W_mask] = 0
            
    print("Uniform Pruning completed successfully.")

# =========================================================================
# 2. RMT指標を用いた不均衡枝刈り（AlphaPruning / OWL）用関数
# =========================================================================
def prune_vit_ww_for_vit_pytorch(args, model, calib_data, device):
    """
    vit_pytorchの実装構造に完全に適合させた、WeightWatcher（RMT指標）ベースの不均衡枝刈り関数．
    """
    layers = find_layers(model.transformer.layers)
    prunables = []
    for name in layers:
        prunables.append(layers[name].weight.numel())
    prunables = torch.tensor(prunables)
    
    args.metric_cache = args.metric_cache + f"/{args.model}"
    if not os.path.exists(args.metric_cache):
        os.makedirs(args.metric_cache)
         
    cache_file = f"{args.metric_cache}/{args.WW_metric}.npy"
    if os.path.exists(cache_file):
        metrics = np.load(cache_file)
        print(f"Loaded RMT metrics ({args.WW_metric}) from cache.")
    else:
        print("WeightWatcher analysis begin!")
        # 【修正】ModuleListではなく親モジュールを渡して安全に内部スキャンさせます
        watcher = ww.WeightWatcher(model=model.transformer)
        details = watcher.analyze()
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        else:
            metrics = np.array(details.alpha)
        
        np.save(cache_file, metrics)

    scores = torch.tensor(metrics)
    
    # 【修正】Stable Rank の場合は、高い層ほど「削らない」ようにスコアの大小を反転させる
    if args.WW_metric == 'stable_rank':
        scores = torch.max(scores) - scores
        
    alpha_max = torch.max(scores)
    alpha_min = torch.min(scores)
    # 以降の layerwise_pruning_ratios の計算はそのまま
        
    alpha_max = torch.max(scores)
    alpha_min = torch.min(scores)
    
    layerwise_pruning_ratios = (((scores - alpha_min) / (alpha_max - alpha_min + 1e-8)) * (2 * args.epsilon) + (1 - args.epsilon))
    scaler = torch.sum(prunables) * args.sparsity / (torch.sum(prunables * layerwise_pruning_ratios) + 1e-8)  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print("層ごとの動的枝刈り目標レート:", [round(r, 4) for r in ratios])

    # 【インデックスズレ検証用デバッグコード】
    print("\n--- [DEBUG] Layer-wise Parameter Matching Check ---")
    print(f"{'Layer Name':<40} | {'WW Metric Score':<15} | {'Pruning Ratio':<15}")
    print("-" * 80)
    
    # 実際の適用ループと同じ順序で名前を模倣してシミュレート
    check_idx = 0
    for block_id, blk in enumerate(model.transformer.layers):
        subset = find_layers(blk)
        for name in subset:
            if check_idx < len(ratios):
                # 実際のブロック番号などを付与してわかりやすく表示
                full_layer_path = f"Block_{block_id}.{name}"
                print(f"{full_layer_path:<40} | {float(scores[check_idx]):.4f} | {ratios[check_idx]:.4f}")
            check_idx += 1
    print("-" * 80 + "\n")
    
    print("Non-uniform pruning begin!")
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda_ww"])

    metric_stats = []
    for blk in model.transformer.layers:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    inps = model.to_patch_embedding(inps)
    cls_tokens = model.cls_token.expand(bs, -1, -1)
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embedding
    inps = model.dropout(inps)

    i = 0
    for block_id, blk in enumerate(model.transformer.layers):
        subset = find_layers(blk)

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            # --- 【核心修正】blk(inps) を vit_pytorch の内部構造に合わせて分解して実行 ---
            attn, ff = blk[0], blk[1]
            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    chunk = inps[i1:j1]
                    chunk = attn(chunk) + chunk
                    chunk = ff(chunk) + chunk
                    tmp_res.append(chunk)
                inps = torch.cat(tmp_res, dim=0)
            else:
                x = attn(inps) + inps
                inps = ff(x) + x

            for h in handles:
                h.remove()     
        else:
            attn, ff = blk[0], blk[1]
            x = attn(inps) + inps
            inps = ff(x) + x
        
        for name in subset:
            if i >= len(ratios):
                break
                
            if args.prune_metric == "wanda_ww":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, ratios[i])
            i += 1
            subset[name].weight.data[W_mask] = 0
            
    print("Non-uniform Pruning completed successfully.")


# =========================================================================
# 3. RMT指標を用いたブロック単位の不均衡枝刈り（Block-wise Alpha / OWL）
# =========================================================================
def prune_vit_blockwise_for_vit_pytorch(args, model, calib_data, device):
    """
    vit_pytorchの実装構造に完全に適合させた、Block-wise（ブロック単位）のRMTベース不均衡枝刈り関数．
    同じトランスフォーマーブロック内の4つの全結合層に対して、共通の枝刈り率を割り当てて過剰破壊を防ぎます．
    """
    num_blocks = len(model.transformer.layers) # 6ブロック
    layers_per_block = 4                      # 1ブロックあたり4つのnn.Linear
    
    # 1. 各層のパラメータ数を取得
    all_layer_params = []
    for blk in model.transformer.layers:
        subset = find_layers(blk)
        for name in subset:
            all_layer_params.append(subset[name].weight.numel())
            
    args.metric_cache = args.metric_cache + f"/{args.model}"
    if not os.path.exists(args.metric_cache):
        os.makedirs(args.metric_cache)
         
    # キャッシュのロードまたはRMT解析
    cache_file = f"{args.metric_cache}/{args.WW_metric}.npy"
    if os.path.exists(cache_file):
        metrics = np.load(cache_file)
        print(f"Loaded RMT metrics ({args.WW_metric}) from cache.")
    else:
        print("WeightWatcher analysis begin!")
        watcher = ww.WeightWatcher(model=model.transformer)
        details = watcher.analyze()
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        else:
            metrics = np.array(details.alpha)
        np.save(cache_file, metrics)

    scores = torch.tensor(metrics) # 長さ 24
    
    # 2. 【核心】層ごとのスコアとパラメータ数を「ブロック単位」に集約（平均化）する
    block_scores = []
    block_prunables = []
    for b in range(num_blocks):
        # そのブロックに属する4層のスコアの平均値をとる
        b_scores = scores[b * layers_per_block : (b + 1) * layers_per_block]
        block_scores.append(torch.mean(b_scores))
        
        # そのブロックの総パラメータ数の合計をとる
        b_params = sum(all_layer_params[b * layers_per_block : (b + 1) * layers_per_block])
        block_prunables.append(b_params)
        
    block_scores = torch.tensor(block_scores)
    block_prunables = torch.tensor(block_prunables)
    
    # 指標に応じた反転ロジック（OWLの場合は高いブロックほど保護する）
    if args.WW_metric == 'stable_rank':
        block_scores = torch.max(block_scores) - block_scores
        
    b_max = torch.max(block_scores)
    b_min = torch.min(block_scores)
    
    # ブロック単位での不均衡配分比率の計算
    block_ratios = (((block_scores - b_min) / (b_max - b_min + 1e-8)) * (2 * args.epsilon) + (1 - args.epsilon))
    
    # 全体の総目標スパースティを満たすようにブロック予算をアライメント
    scaler = torch.sum(block_prunables) * args.sparsity / (torch.sum(block_prunables * block_ratios) + 1e-8)  
    block_ratios = block_ratios * scaler
    block_ratios_list = block_ratios.cpu().numpy().tolist()
    
    print("ブロックごとの動的枝刈り目標レート (計6ブロック):", [round(r, 4) for r in block_ratios_list])
    
    # --- デバッグ出力（どの層に何%が割り当てられたか可視化） ---
    print("\n--- [DEBUG] Block-wise Parameter Matching Check ---")
    print(f"{'Layer Name':<40} | {'Assigned Block Ratio':<20}")
    print("-" * 70)
    for block_id, blk in enumerate(model.transformer.layers):
        subset = find_layers(blk)
        for name in subset:
            print(f"Block_{block_id}.{name:<34} | {block_ratios_list[block_id]:.4f}")
    print("-" * 70 + "\n")
    
    print("Block-wise non-uniform pruning begin!")
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda_ww"])

    metric_stats = []
    for blk in model.transformer.layers:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    inps = model.to_patch_embedding(inps)
    cls_tokens = model.cls_token.expand(bs, -1, -1)
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embedding
    inps = model.dropout(inps)

    for block_id, blk in enumerate(model.transformer.layers):
        subset = find_layers(blk)
        # そのブロック共通の比率を適用
        current_block_ratio = block_ratios_list[block_id]

        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            attn, ff = blk[0], blk[1]
            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    chunk = inps[i1:j1]
                    chunk = attn(chunk) + chunk
                    chunk = ff(chunk) + chunk
                    tmp_res.append(chunk)
                inps = torch.cat(tmp_res, dim=0)
            else:
                x = attn(inps) + inps
                inps = ff(x) + x

            for h in handles:
                h.remove()     
        else:
            attn, ff = blk[0], blk[1]
            x = attn(inps) + inps
            inps = ff(x) + x
        
        # マスキングと重みのゼロ化
        for name in subset:
            if args.prune_metric == "wanda_ww":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            # 共通の block_ratio でマスクを生成
            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, current_block_ratio)
            subset[name].weight.data[W_mask] = 0
            
    print("Block-wise Pruning completed successfully.")