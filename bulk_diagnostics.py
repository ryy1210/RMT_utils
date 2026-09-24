"""保存済みESDを入力にしたbulk KS診断。モデルの取得はNotebook側で明示する。"""
import io
import pickle
import re
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from funcs1 import get_bulk_ks, tw1_quantile

SOURCE_RUNS = [
    'ryoya-zushi1210-tokyo-university-of-science/LlaMa-RMT-Analysis/jv0d6kiu',
    'ryoya-zushi1210-tokyo-university-of-science/LlaMa-RMT-Analysis/vyzq5p34',
]


def load_metrics(path):
    """自分が保存した信頼済みpickleのみ使用する。既存KSは名称を付けて保持。"""
    path = Path(path)
    if path.suffix == '.pkl':
        # 追加: GPUで保存されたTensorもCPUへ読み込む。信頼済みの自分のpickle専用。
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)
        with path.open('rb') as handle:
            frame = CPUUnpickler(handle).load()
    else:
        frame = pd.read_csv(path)
    frame = frame.copy()
    if 'name' not in frame and frame.index.name == 'name':
        frame = frame.reset_index()
    required = ['name', 'sigma2_preDE', 'sigma2_postDE', 'threshold_preDE', 'threshold_postDE',
                'KS_preDE', 'KS_postDE', 'KS_postDE_1']
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if frame.name.isna().any() or frame.name.duplicated().any():
        raise ValueError('Layer names must be present and unique')
    for col in required[1:5]:
        if not np.isfinite(frame[col].to_numpy(dtype=float)).all() or (frame[col] <= 0).any():
            raise ValueError(f'Invalid saved parameter: {col}')
    return frame


def spectra(weight, device='cpu'):
    """追加: DE前のthin SVDをDEに再利用。DE後は特異値だけを計算する。

    float64で診断し、CPU/GPUの相違はmetadataに記録する。
    読込時の重み精度そのものはfloat64への変換では復元されない。
    """
    with torch.no_grad():
        y = weight.detach().to(device=device, dtype=torch.float64)
        if y.ndim != 2 or not torch.isfinite(y).all():
            raise ValueError('Expected finite 2D weight')
        if y.shape[0] > y.shape[1]:
            y = y.T
        p, n = y.shape
        if p == 0:
            raise ValueError('Empty weight matrix')
        u, s, vh = torch.linalg.svd(y, full_matrices=False)
        eta = s.median() if len(s) % 2 else (s[len(s)//2-1] + s[len(s)//2])/2
        if eta <= 0:
            raise ValueError('DE requires a positive median singular value')
        term = eta / (s*s + eta*eta)
        g1 = (u*u) @ term
        g2 = 1/eta + (vh.T*vh.T) @ (term-1/eta)
        # funcs1._dyson_from_svdと同じ式。下の回帰テストで数値一致を確認する。
        dx = torch.clamp(p - eta*g1.abs().sum(), min=1e-12)
        dz = torch.clamp(n - eta*g2.abs().sum(), min=1e-12)
        x = torch.clamp((1/torch.clamp(g1, min=1e-12)-eta)/torch.sqrt(dx), min=1e-12)
        z = torch.clamp((1/torch.clamp(g2, min=1e-12)-eta)/torch.sqrt(dz), min=1e-12)
        if not torch.isfinite(x).all() or not torch.isfinite(z).all() or (x <= 0).any() or (z <= 0).any():
            raise ValueError('Invalid DE scaling')
        post = y / torch.sqrt(x[:, None]*z[None, :])
        post_s = torch.linalg.svdvals(post)
        return np.sort(s.cpu().numpy()**2/n), np.sort(post_s.cpu().numpy()**2/n), p, n


def diagnose_spectra(pre, post, p, n, saved, beta=0.1):
    if len(pre) != p or len(post) != p or not 0 < p <= n:
        raise ValueError('Spectrum length must equal p <= n')
    # 追加: 分散1には専用の閾値を使用。BEMA閾値で選んだ集合を流用しない。
    gamma = p/n
    threshold1 = ((1+np.sqrt(gamma))**2 + tw1_quantile(beta)*n**(-2/3)
                  *gamma**(-1/6)*(1+np.sqrt(gamma))**(4/3))
    result = {'name': saved['name'], 'p': p, 'n': n, 'gamma': gamma, 'beta_fixed1': beta}
    for col in ['KS_preDE', 'KS_postDE', 'KS_postDE_1']:
        result[col+'_saved'] = float(saved[col])
    for suffix, values, variance, threshold in [
        ('preDE', pre, saved['sigma2_preDE'], saved['threshold_preDE']),
        ('postDE', post, saved['sigma2_postDE'], saved['threshold_postDE']),
        ('DE_1', post, 1., threshold1),
    ]:
        d = get_bulk_ks(values, gamma, float(variance), float(threshold))
        full_name = 'KS_postDE_1' if suffix == 'DE_1' else 'KS_'+suffix
        result[full_name] = d['full_ks']
        result['bulkKS_'+suffix] = d['ks']
        for key in ['n_bulk', 'n_signal', 'bulk_ratio', 'threshold', 'sigma2', 'mp_edge', 'status']:
            result[key+'_'+suffix] = d[key]
        if suffix in ('preDE', 'postDE') and 's_hat_'+suffix in saved:
            result['s_hat_saved_'+suffix] = int(saved['s_hat_'+suffix])
            result['rank_matches_saved_'+suffix] = int(saved['s_hat_'+suffix]) == d['n_signal']
    result['bulkKS_delta_post_minus_pre'] = result['bulkKS_postDE']-result['bulkKS_preDE']
    return result


def atomic_csv(frame, path):
    path = Path(path)
    temp = path.with_suffix('.tmp.csv')
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run_diagnostics(model, saved, output_dir, metadata, device='cpu', beta=0.1, limit=None):
    """行列単位で保存・再開。異なる入力やコードの結果との混在を拒否する。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(metadata, svd_device=device, svd_dtype='float64', beta_fixed1=beta,
                    layers=saved.name.tolist(), limit=limit,
                    input_parameters_sha256=hashlib.sha256(saved[[c for c in saved.columns if c != 'eigs']].to_json().encode()).hexdigest())
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    meta_path = output/'metadata.json'
    if meta_path.exists() and json.loads(meta_path.read_text())['identity'] != identity:
        raise ValueError('Output belongs to a different configuration; choose a new directory')
    meta_path.write_text(json.dumps(dict(identity=identity, config=manifest), ensure_ascii=False, indent=2))
    path = output/'bulk_ks.csv'
    records = pd.read_csv(path, dtype={'name': str}).to_dict('records') if path.exists() else []
    done = {r['name'] for r in records}
    if len(done) != len(records) or not done.issubset(set(saved.name)):
        raise ValueError('Invalid checkpoint layer names')
    modules = dict(model.named_modules())
    targets = saved if limit is None else saved.iloc[:limit]
    missing = set(targets.name)-set(modules)
    if missing:
        raise ValueError(f'Exact module names missing: {sorted(missing)[:8]}')
    for i, (_, row) in enumerate(targets.iterrows(), 1):
        if row['name'] in done:
            continue
        print(f'[{i}/{len(targets)}] SVD / DE / bulkKS: {row["name"]}', flush=True)
        start = time.monotonic()
        layer = modules[row['name']]
        if not isinstance(layer, torch.nn.Linear):
            raise ValueError('This LLM experiment supports Linear weights only')
        pre, post, p, n = spectra(layer.weight, device)
        # 追加: 保存されたpreDEスペクトルがあれば、モデル・dtypeの取り違えを検出。
        if 'eigs' in row and isinstance(row['eigs'], (list, np.ndarray, torch.Tensor)):
            old = np.sort(np.asarray(row['eigs'], dtype=float))/n
            if old.shape != pre.shape or not np.allclose(old, pre, rtol=2e-4, atol=1e-10):
                raise ValueError(f'Saved spectrum differs for {row["name"]}; verify model revision/dtype')
        # 追加: 固有値は小さいため保存し、将来の診断でSVDを繰り返さない。
        spectrum_dir = output/'spectra'
        spectrum_dir.mkdir(exist_ok=True)
        spectrum_name = hashlib.sha256(row['name'].encode()).hexdigest()+'.npz'
        np.savez_compressed(spectrum_dir/spectrum_name, pre=pre, post=post, p=p, n=n,
                            layer_name=row['name'])
        record = diagnose_spectra(pre, post, p, n, row, beta)
        record['spectrum_file'] = 'spectra/'+spectrum_name
        record['elapsed_seconds'] = time.monotonic()-start
        record['module_type'] = row['name'].split('.')[-1]
        match = re.search(r'(?:^|\.)layers\.(\d+)\.', row['name'])
        record['block_index'] = int(match.group(1)) if match else -1
        records.append(record)
        atomic_csv(pd.DataFrame(records), path)
        print(f'  saved; bulkKS pre={record["bulkKS_preDE"]:.4f}, post={record["bulkKS_postDE"]:.4f}', flush=True)
    return pd.DataFrame(records)


def publish_wandb(frame, output_dir, metadata, source_runs, project='LlaMa-RMT-Analysis',
                  entity='ryoya-zushi1210-tokyo-university-of-science'):
    """新規runに保存。元runは上書きしない。認証はwandb.login()で別途行う。"""
    import wandb
    with wandb.init(entity=entity, project=project, job_type='bulk-ks-diagnostic',
                    config=dict(metadata, source_runs=source_runs)) as run:
        run.log({'bulk_ks': wandb.Table(dataframe=frame)})
        artifact = wandb.Artifact('bulk-ks-'+metadata['model_id'].split('/')[-1].lower(), type='diagnostics')
        artifact.add_dir(str(output_dir))
        run.log_artifact(artifact)
        return run.url


def resolve_analysis_model(model, layer_names):
    """Transformers版によるmodel/language_modelの階層差を正確な名前で照合。"""
    expected = set(layer_names)
    if not expected:
        raise ValueError('No source layers')
    queue, seen, matches = [model], set(), []
    while queue:
        candidate = queue.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if expected.issubset(dict(candidate.named_modules())):
            matches.append(candidate)
        for attr in ('model', 'language_model'):
            child = getattr(candidate, attr, None)
            if isinstance(child, torch.nn.Module):
                queue.append(child)
    if len(matches) != 1:
        raise ValueError('Source layer names do not uniquely match the model hierarchy')
    return matches[0]
