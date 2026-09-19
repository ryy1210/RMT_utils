"""言語モデル評価。公開関数名は既存Colabと互換。"""
import math
import time
import itertools
import warnings
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import get_test_data


def _input_device(model, device, move=True):
    # 修正: accelerate/device_mapで配置済みのモデルを一括to()しない。
    dispatched = bool(getattr(model, 'hf_device_map', None))
    if move and not dispatched:
        target = torch.device(device)
        if target.type == 'cuda' and not torch.cuda.is_available():
            target = torch.device('cpu')
        model.to(target)
    try:
        embedding = model.get_input_embeddings()
        hook = getattr(embedding, '_hf_hook', None)
        execution = getattr(hook, 'execution_device', None)
        if execution is not None:
            return torch.device(execution)
        if embedding.weight.device.type != 'meta':
            return embedding.weight.device
    except (AttributeError, NotImplementedError):
        pass
    return next(model.parameters()).device


def _ppl(model, tokenizer, datasets, seq_len, batch_size, device, log_every=0, move=True):
    if seq_len < 2 or batch_size < 1:
        raise ValueError('Require seq_len >= 2 and batch_size >= 1')
    input_device = _input_device(model, device, move)
    # 修正: 呼出元のtrain/eval状態を、例外時も含めて復元する。
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    ppls = {}
    try:
        with torch.inference_mode():
            for name in datasets:
                loader = get_test_data(name, tokenizer, seq_len=seq_len, batch_size=batch_size)
                total, count = 0.0, 0
                bar = tqdm(loader, desc=f'Eval {name}')
                for step, batch in enumerate(bar, 1):
                    if isinstance(batch, dict):
                        ids = batch['input_ids'].to(input_device)
                        mask = batch.get('attention_mask')
                        kwargs = {'attention_mask': mask.to(input_device)} if mask is not None else {}
                    else:
                        ids, mask, kwargs = batch.to(input_device), None, {}
                    logits = model(input_ids=ids, use_cache=False, **kwargs).logits
                    # 修正: 不正バッチを除外して良いPPLを報告せず、失敗を明示する。
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError(f'Non-finite logits in {name}, batch {step}')
                    labels = ids[:, 1:].to(logits.device)
                    valid = torch.ones_like(labels, dtype=torch.bool)
                    if mask is not None:
                        mask = mask.to(logits.device).bool()
                        valid = mask[:, 1:] & mask[:, :-1]
                    # 高速化: 全lossや巨大なfloat32 logitsコピーを保持せず時系列chunkで加算。
                    for start in range(0, labels.shape[1], 128):
                        end = min(start + 128, labels.shape[1])
                        targets = labels[:, start:end].clone()
                        targets[~valid[:, start:end]] = -100
                        values = logits[:, start:end, :].float().reshape(-1, logits.shape[-1])
                        loss = F.cross_entropy(values, targets.reshape(-1), reduction='sum', ignore_index=-100)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Non-finite loss in {name}')
                        total += loss.item()
                        count += int(valid[:, start:end].sum())
                    del logits
                    if log_every and step % log_every == 0 and count:
                        bar.set_postfix(nll=total/count)
                if count == 0:
                    raise ValueError(f'No valid evaluation tokens in {name}')
                ppls[name] = math.exp(total/count) if total/count < 709 else float('inf')
    finally:
        for module, training in modes:
            module.training = training
    print('PPL after pruning:', ppls)
    if torch.cuda.is_available():
        print(f'Weight Memory: {torch.cuda.memory_allocated()/1024**2:.2f} MiB')
    return ppls


def ppl_eval(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device='cuda'):
    return _ppl(model, tokenizer, datasets, model_seq_len, batch_size, device)


def ppl_eval2(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device='cuda', log_every=5):
    return _ppl(model, tokenizer, datasets, model_seq_len, batch_size, device, log_every)


def ppl_eval3(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], model_seq_len=2048, batch_size=32, device='cuda', log_every=20):
    return _ppl(model, tokenizer, datasets, model_seq_len, batch_size, device, log_every, move=False)


def ppl_eval_large(model, tokenizer, datasets=['wikitext2', 'ptb', 'c4'], seq_len=2048, batch_size=32, device='cuda'):
    # 修正: 旧版は学習済みnormを捨て、重み1のnormで評価していた。
    # 手製layer replayを廃止し、device_mapで配置済みの実モデルを通す。
    warnings.warn('ppl_eval_large uses the model forward; load large models with device_map first.', stacklevel=2)
    return _ppl(model, tokenizer, datasets, seq_len, batch_size, device, move=False)


@torch.no_grad()
def eff_eval(model, tokenizer, dataset='wikitext2', original_len=4, generated_len=128, batch_size=1, device='cuda'):
    if generated_len < 1:
        raise ValueError('generated_len must be positive')
    input_device = _input_device(model, device)
    modes = [(m, m.training) for m in model.modules()]
    model.eval()
    elapsed, tokens, peak = 0.0, 0, 0
    is_cuda = input_device.type == 'cuda'
    weight_memory = torch.cuda.memory_allocated(input_device) if is_cuda else 0
    try:
        loader = get_test_data(dataset, tokenizer, seq_len=original_len, batch_size=batch_size)
        for batch in itertools.islice(loader, 10):
            batch = batch.to(input_device)
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(input_device)
                torch.cuda.synchronize(input_device)
            start = time.perf_counter()
            outputs = model.generate(input_ids=batch, pad_token_id=tokenizer.eos_token_id,
                                     do_sample=True, use_cache=True, top_k=50,
                                     max_new_tokens=generated_len, top_p=.95, temperature=1)
            if is_cuda:
                torch.cuda.synchronize(input_device)
                peak = max(peak, torch.cuda.max_memory_allocated(input_device))
            elapsed += time.perf_counter() - start
            # 修正: EOSで早期終了した長さを考慮し、要求長だけを数えない。
            for row in outputs[:, batch.shape[1]:]:
                eos = (row == tokenizer.eos_token_id).nonzero()
                tokens += int(eos[0, 0]) + 1 if len(eos) else row.numel()
    finally:
        for module, training in modes:
            module.training = training
    print(f'Total Memory: {peak/1024**3:.3f} GB (input device only)')
    print(f'Weight Memory: {weight_memory/1024**3:.3f} GB')
    print(f'Throughput: {tokens/elapsed if elapsed else 0:.3f} tokens/sec')
