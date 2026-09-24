"""Fresh-process ESD / baseline PPL / LRA entry point (CUDA required)."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['llama', 'llama-instruct', 'gemma'], default='llama')
    parser.add_argument('--stage', choices=['esd', 'ppl', 'lra'], default='ppl')
    parser.add_argument('--metrics', type=Path, help='Trusted local pickle from this runner for LRA')
    parser.add_argument('--output', type=Path, required=True, help='New run directory (must not exist)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-layers', type=int, default=1)
    parser.add_argument('--sort-by', choices=['alpha', 'KS_postDE_1'], default='alpha')
    parser.add_argument('--descending', action='store_true')
    parser.add_argument('--no-de', action='store_true')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.stage == 'lra' and (args.metrics is None or not args.metrics.is_file()):
        parser.error('--stage lra requires an existing --metrics file')
    if args.max_layers < 1 or args.seq_len < 2 or args.batch_size < 1:
        parser.error('Require max-layers >= 1, seq-len >= 2 and batch-size >= 1')

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This full-model runner requires a CUDA GPU. Select a Colab GPU kernel. Use 00_environment_check.ipynb locally.')
    import pandas as pd
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import funcs1
    from notebook_runtime import setup_auth

    setup_auth()
    set_seed(args.seed)
    model_id = {'llama': 'meta-llama/Llama-3.2-3B',
                'llama-instruct': 'meta-llama/Llama-3.2-3B-Instruct',
                'gemma': 'google/gemma-3-4b-pt'}[args.model]
    args.output.mkdir(parents=True, exist_ok=False)
    info = {**vars(args), 'model_id': model_id, 'started_at': datetime.now(timezone.utc).isoformat(),
            'torch': torch.__version__, 'transformers': transformers.__version__,
            'gpu': torch.cuda.get_device_name(0), 'status': 'started',
            'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'git_dirty': bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())}
    manifest = args.output / 'run.json'
    manifest.write_text(json.dumps(info, default=str, indent=2))
    try:
        dtype = torch.bfloat16 if args.model == 'gemma' and torch.cuda.is_bf16_supported() else torch.float16
        if args.model == 'gemma' and dtype == torch.float16:
            raise RuntimeError('Gemma experiment requires a GPU supporting bfloat16 (e.g. L4/A100).')
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map='auto', low_cpu_mem_usage=True)
        info['model_revision'] = getattr(model.config, '_commit_hash', None)
        info['dtype'] = str(dtype)
        if args.stage == 'esd':
            # Full outer-model names are retained so LRA needs no Gemma prefix guessing.
            metrics = funcs1.get_esd_metrics(model, pl_fitting='fix-finger')
            metrics.to_pickle(args.output / 'esd_metrics.pkl')
            metrics.drop(columns=['eigs']).to_csv(args.output / 'esd_metrics.csv', index=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            if args.stage == 'ppl':
                info['ppl_wikitext2'] = funcs1.get_ppl(model, tokenizer, seq_len=args.seq_len, batch_size=args.batch_size)
            else:
                metrics_manifest = args.metrics.parent / 'run.json'
                if not metrics_manifest.is_file():
                    raise ValueError('Use ESD metrics produced by this runner (run.json is required).')
                source_info = json.loads(metrics_manifest.read_text())
                if source_info.get('model_id') != model_id or source_info.get('stage') != 'esd' or source_info.get('status') != 'complete':
                    raise ValueError('Metrics must come from a completed ESD run of the same model.')
                info['metrics_source'] = source_info
                metrics = pd.read_pickle(args.metrics)
                names = metrics.sort_values(args.sort_by, ascending=not args.descending)['name'].tolist()
                unknown = set(names[:args.max_layers]) - dict(model.named_modules()).keys()
                if unknown:
                    raise ValueError(f'Metrics have names not present in this model: {sorted(unknown)}')
                history = funcs1.run_lra_experiment(model, tokenizer, metrics, names, args.max_layers,
                    DE=not args.no_de, seq_len=args.seq_len, batch_size=args.batch_size)
                history.to_csv(args.output / 'lra_history.csv', index=False)
        info['status'] = 'complete'
    except Exception as exc:
        info['status'] = 'failed'
        info['error_type'] = type(exc).__name__
        raise
    finally:
        manifest.write_text(json.dumps(info, default=str, indent=2))


if __name__ == '__main__':
    main()
