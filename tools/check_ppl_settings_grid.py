"""Offline tests for 11_1. No model download, Drive writes, or GPU execution."""
from pathlib import Path
import os,sys,json,hashlib,math,gc,time,re,platform,importlib.metadata,itertools,tempfile
os.environ['HF_HUB_OFFLINE']='1'
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']='/tmp/rmt-grid-matplotlib'
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
torch.set_num_threads(1)
import matplotlib.pyplot as plt
import nbformat
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import funcs1
from funcs1 import _dyson_from_svd
nb=nbformat.read(ROOT/'notebooks/11_1_gemma_qk_ppl_settings_grid.ipynb',4);nbformat.validate(nb)
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':compile(c.source,f'cell-{i+1}','exec')
    if c.cell_type=='code' and c.source.startswith('# 11_1'):exec(compile(c.source,'notebook','exec'))
# Exact token-boundary and denominator tests, including partial last block.
raw=np.arange(10,41)
for mode in ('none','first_only','each_block'):
    a,m=pack_tokens(raw,8,mode,2)
    assert a.shape==((3 if mode=='none' else 4),8)
    assert m['scored_tokens']==len(a)*7
    assert m['bos_count']=={'none':0,'first_only':1,'each_block':4}[mode]
    if mode=='each_block': assert np.array_equal(a[:,1:].ravel(),raw[:28])
    elif mode=='first_only':assert np.array_equal(a.ravel(),np.r_[2,raw])
    else:assert np.array_equal(a.ravel(),raw[:24])
from transformers import LlamaConfig,LlamaForCausalLM
torch.manual_seed(7)
model=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=24,intermediate_size=40,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)).eval()
tokens=np.random.default_rng(3).integers(0,64,(5,16),dtype=np.int64)
a,c,p=evaluate(model,tokens,1);b,d,q=evaluate(model,tokens,2)
assert np.allclose(a,b,rtol=1e-6,atol=1e-5) and np.array_equal(c,d)
with torch.inference_mode():
    ids=torch.tensor(tokens)
    direct=F.cross_entropy(model(ids).logits[:,:-1].float().transpose(1,2),ids[:,1:],reduction='none').double()
assert np.allclose(a,direct.sum(1).numpy(),atol=1e-5)
assert np.allclose(p,direct.sum(0).numpy(),atol=1e-5)
# Restore floating buffers after cast cycle (rotary accuracy must not be lost).
model=model.bfloat16()
model.register_buffer('audit_fp32_buffer',torch.tensor([.1234567],dtype=torch.float32))
buffers=snapshot_buffers(model)
cast_eval(model,'fp32',buffers);cast_eval(model,'bf16',buffers)
assert model.audit_fp32_buffer.dtype==torch.float32
assert torch.equal(model.audit_fp32_buffer,buffers['audit_fp32_buffer'])
# Full DE inversion and per-decile removal on rectangular matrices.
SVD_DEVICE='cpu';DEVICE='cpu';DECILES=list(range(1,11))
for shape in [(20,32),(32,20)]:
    w=torch.randn(*shape);parts=decompose(w)
    for de in (False,True):
        assert torch.allclose(reconstruct(parts[de],np.arange(20)),w,atol=5e-5,rtol=5e-5)
        groups=np.array_split(np.arange(19,-1,-1),10)
        assert np.array_equal(np.sort(np.concatenate(groups)),np.arange(20))
        for idx in groups:
            lhs=w-reconstruct(parts[de],idx)
            rhs=reconstruct(parts[de],np.setdiff1d(np.arange(20),idx))
            assert torch.allclose(lhs,rhs,atol=5e-5,rtol=5e-5)
MODELS={'gemma':'tiny','llama':'tiny'};REVISIONS={'gemma':'a'*40,'llama':'b'*40}
ENV={'test':True};ATTENTION='sdpa';IMPLEMENTATION=implementation_hash()
LENGTHS=[16];BOS_MODES=['none'];PRECISIONS=['bf16','fp32'];LOADERS=['conditional_text'];BATCH_SIZES=[1,2]
GROUPS=['Q','K'];DE_OPTIONS=[False,True]
with tempfile.TemporaryDirectory() as temp:
    OUTPUT=Path(temp);CACHE=OUTPUT/'cache'
    setting=dict(length=16,bos='none',precision='fp32',batch=2,loader='auto',load_precision='bf16')
    meta=dict(blocks=5,length=16)
    spec=make_spec('gemma','test',setting,dict(group='baseline',de=False,decile=0),meta,{'test':True})
    model.float()
    obj=evaluate_job(model,spec,tokens);assert read_result(spec)==obj
    stamp=(result_path(spec)/'summary.json').stat().st_mtime_ns
    evaluate_job(model,spec,tokens)
    assert stamp==(result_path(spec)/'summary.json').stat().st_mtime_ns
    altered={**spec,'setting':{**setting,'batch':1}}
    assert read_result(altered) is None
    # Integrity check must detect corruption.
    with (result_path(spec)/'losses.npz').open('ab') as f:f.write(b'corruption')
    try:read_result(spec);raise AssertionError('Corruption accepted')
    except ValueError:pass
    # Full run orchestration, real tiny SVD/DE + evaluation, two precision settings.
    # Downloads and credential logic are replaced; no real model inference is scheduled.
    PHASE='qk_grid';NISHIKAWA_BATCH=1;NISHIKAWA_LOAD_PRECISION='default';BASELINE_MODELS=['llama','gemma']
    real_loader=load_model;real_prepare=prepare_data
    def tiny_load(key,loader,precision):
        torch.manual_seed(7)
        m=LlamaForCausalLM(LlamaConfig(vocab_size=64,hidden_size=40,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)).eval()
        return (m if precision=='default' else m.bfloat16()),{'test':True}
    def tiny_data(key):
        # Baseline phase uses (2048, first_only) but tokens need actual length for coverage validation.
        return {(16,'none'):(tokens,dict(blocks=5,length=16))}
    load_model=tiny_load;prepare_data=tiny_data
    # Change cache-independent output for orchestrator tests, no special shortcuts to run_grid.
    OUTPUT=Path(temp)/'grid';CACHE=OUTPUT/'svd_cache'
    run_grid()
    summaries=list((OUTPUT/'jobs').glob('*/summary.json'))
    assert len(summaries)==4*41,(len(summaries),'Expected 164 tiny evaluations')
    before={p:p.stat().st_mtime_ns for p in summaries}
    def forbidden_load(*args):raise AssertionError('Completed run loaded a model')
    load_model=forbidden_load;run_grid()
    assert all(p.stat().st_mtime_ns==t for p,t in before.items())
    # Partial resume: one missing result uses cached SVD; all other summaries stay untouched.
    victim=summaries[-1]; victim.unlink()
    original_decompose=decompose
    def forbidden_svd(*args):raise AssertionError('Cached SVD recomputed')
    decompose=forbidden_svd;load_model=tiny_load
    run_grid()
    assert victim.exists()
    assert all(p==victim or p.stat().st_mtime_ns==t for p,t in before.items())
    decompose=original_decompose
    # Exercise plot/table pipeline with the genuine tiny results. Never publish them as Gemma evidence.
    display=lambda x:None
    result=analyse()
    assert len(result)==164 and result.baseline_ppl.notna().all()
    assert (OUTPUT/'analysis/one_factor_contrasts.csv').exists()
    assert len(list((OUTPUT/'analysis').glob('heatmap_*.png')))==4
    import shutil
    shutil.copyfile(next((OUTPUT/'analysis').glob('heatmap_*.png')),'/tmp/rmt-grid-preview.png')
    # Stage 1 uses both model keys, fp32, actual 2048-long token blocks and skips completed models.
    OUTPUT=Path(temp)/'baseline';BASELINE_MODELS=['llama','gemma']
    def baseline_data(key):
        arr=np.random.default_rng(9).integers(0,64,(1,2048),dtype=np.int64);arr[:,0]=2
        return {(2048,'first_only'):(arr,dict(blocks=1,length=2048))}
    prepare_data=baseline_data;load_model=tiny_load
    run_nishikawa()
    assert len(list((OUTPUT/'jobs').glob('*/summary.json')))==2
    load_model=forbidden_load;run_nishikawa()
    print('PASS stage 1 both model branches and skip; partial resume uses SVD cache')
    print('PASS grid 164 tiny jobs, complete skip before model/SVD, saved PPL pairing, analysis plots')
print('PASS packing, weighted PPL/batch invariance, position losses, float buffers, DE reconstruction, checksums, nbformat/syntax')
