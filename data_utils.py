import os
import random
import torch
import sys
from datasets import load_dataset
from torch.utils.data.dataset import Dataset

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

def get_calib_train_data(name, tokenizer, nsamples, seqlen=2048, seed=3, batch_size=1, dataset_cache_dir=None):
    """nsamples個のtoken列を生成し、最後の端数batchも返す。"""
    if nsamples < 0 or seqlen < 1 or batch_size < 1:
        raise ValueError("Invalid sampling dimensions")
    # 修正: 旧cacheはtokenizerがキーに含まれず、別モデルのtoken IDを再利用した。
    # 校正データは生成し直し、datasets側のダウンロードcacheのみ利用する。
    if name == 'wikitext2':
        data = _wikitext_split('train', dataset_cache_dir)
        text = data['text']
    elif name == 'ptb':
        data = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
        text = data['sentence']
    elif name == 'c4':
        text = load_dataset('json', data_files='utils/c4-train.json')['train']['text']
    else:
        raise ValueError(f"Unknown dataset: {name}")
    ids = tokenizer("\n\n".join(text), return_tensors='pt').input_ids[0]
    if ids.numel() < seqlen:
        raise ValueError("Not enough tokens for calibration")
    rng = random.Random(seed)
    result = []
    # 修正: s -= 1ではforループをやり直せないため、token列から直接切り出す。
    for offset in range(0, nsamples, batch_size):
        samples = []
        for _ in range(min(batch_size, nsamples-offset)):
            start = rng.randint(0, ids.numel()-seqlen)
            samples.append(ids[start:start+seqlen])
        batch = torch.stack(samples)
        result.append({'input_ids': batch, 'attention_mask': torch.ones_like(batch)})
    return result


def _wikitext_split(split, cache_dir=None):
    return load_dataset('parquet', data_files={split:
        f'hf://datasets/Salesforce/wikitext@~parquet/wikitext-2-raw-v1/{split}/0000.parquet'},
        split=split, cache_dir=cache_dir)


def get_wikitext2(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    # 修正: test読み込み時にtrainを上書きし、trainファイルへtest splitを要求していた。
    traindata = _wikitext_split('train', dataset_cache_dir)
    testdata = _wikitext_split('test', dataset_cache_dir)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
    rng = random.Random(seed)
    trainloader = []
    for _ in range(nsamples):
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', cache_dir=dataset_cache_dir)

    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(valdata['sentence']), return_tensors='pt')

    import random
    rng = random.Random(seed)
    trainloader = []
    for _ in range(nsamples):
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    rng = random.Random(seed)
    trainloader = []
    for _ in range(nsamples):
        # 修正: 長い文書がないデータで無限ループにならないよう上限を設ける。
        for attempt in range(1000):
            i = rng.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        else:
            raise ValueError('Could not find a document with enough tokens')
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    import random
    rng = random.Random(0)
    valenc = []
    for _ in range(256):
        # 修正: 長い文書がないデータで無限ループにならないよう上限を設ける。
        for attempt in range(1000):
            i = rng.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        else:
            raise ValueError('Could not find a document with enough tokens')
        i = rng.randint(0, tmp.input_ids.shape[1] - seqlen)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc 



def get_ptb_new(nsamples, seed, seqlen, tokenizer, dataset_cache_dir=None):
    from datasets import load_dataset
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', cache_dir=dataset_cache_dir)
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test', cache_dir=dataset_cache_dir)

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    import random
    rng = random.Random(seed)
    trainloader = []
    for _ in range(nsamples):
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_c4_new(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset("json", data_files="utils/c4-train.json")['train']
    valdata = load_dataset("json", data_files="utils/c4-validation.json")['train']

    import random
    rng = random.Random(seed)
    trainloader = []
    for _ in range(nsamples):
        # 修正: 長い文書がないデータで無限ループにならないよう上限を設ける。
        for attempt in range(1000):
            i = rng.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        else:
            raise ValueError('Could not find a document with enough tokens')
        i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if 'ptb' in name:
        if 'new' in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer)
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer)
        return get_c4(nsamples, seed, seqlen, tokenizer)
    raise ValueError(f'Unknown dataset: {name}')


# 高速化: 同一tokenizer・設定の評価token列をCPUに2件まで保持。
# 行列ごとのPPL評価でダウンロード確認/tokenizeを繰り返さない。
from collections import OrderedDict
_TEST_CACHE = OrderedDict()


def clear_test_data_cache():
    """tokenizerの語彙/normalizerやデータを変更した場合に呼ぶ。"""
    _TEST_CACHE.clear()


def get_test_data(name, tokenizer, seq_len=2048, batch_size=4):
    if seq_len < 1 or batch_size < 1:
        raise ValueError('seq_len and batch_size must be positive')
    try:
        size = len(tokenizer)
    except TypeError:
        size = None
    key = (name, id(tokenizer), seq_len, size,
           repr(getattr(tokenizer, 'special_tokens_map', None)),
           repr(getattr(tokenizer, 'init_kwargs', None)))
    if key in _TEST_CACHE:
        _TEST_CACHE.move_to_end(key)
        _, tensors = _TEST_CACHE[key]
    else:
        if 'wikitext2' in name:
            data, field = _wikitext_split('test'), 'text'
        elif 'ptb' in name:
            data = load_dataset('ptb_text_only', 'penn_treebank', split='test')
            field = 'sentence'
        elif 'c4' in name:
            data = load_dataset('json', data_files='utils/c4-validation.json')['train'][:2000]
            field = 'text'
        else:
            raise ValueError(f'Unknown dataset: {name}')
        ids = tokenizer("\n\n".join(data[field]), return_tensors='pt').input_ids[0].cpu()
        count = ids.numel() // seq_len
        if count == 0:
            raise ValueError(f'{name} has fewer than {seq_len} tokens')
        # 高速化: Pythonのlist + stackをviewに置換。端数切捨ては旧評価と同じ。
        tensors = ids[:count * seq_len].reshape(count, seq_len)
        _TEST_CACHE[key] = (tokenizer, tensors)
        while len(_TEST_CACHE) > 2:
            _TEST_CACHE.popitem(last=False)
    return torch.utils.data.DataLoader(tensors, batch_size=batch_size, shuffle=False)
