"""Offline checks: no downloads or production model execution."""
import json,hashlib,math,gc,time,re,platform,importlib.metadata,types,os,sys,tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import nbformat
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
import funcs1
from funcs1 import _dyson_from_svd
n=nbformat.read(repo/'notebooks/11_2_staats_sliding_decile_ppl.ipynb',as_version=4);nbformat.validate(n)
for c in n.cells:
 if c.cell_type=='code':compile(c.source,'cell','exec')
exec(n.cells[4].source)
# Regression: newer loaders return set-valued loading_info.
with tempfile.TemporaryDirectory() as tmp:
    receipt=Path(tmp)/'loading.json'
    obj={'loading_info':{'missing_keys':set(), 'unexpected_keys':set(),
                         'details':{'b','a'}},'dtype':torch.bfloat16,'count':np.int64(3)}
    write_json(receipt,obj)
    saved=json.loads(receipt.read_text())
    assert saved['loading_info']['missing_keys']==[]
    assert saved['loading_info']['details']==['a','b']
    assert saved['count']==3 and saved['dtype']=='torch.bfloat16'
    try: write_json(receipt, {'invalid':object()})
    except TypeError: pass
    else: raise AssertionError('Unsupported metadata was silently stringified')
    assert json.loads(receipt.read_text())==saved
print('Set-valued loading metadata and atomic write: PASS')

from transformers import LlamaConfig,LlamaForCausalLM
model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)).eval()
ids=np.arange(37)%32
for window,stride in [(8,2),(16,7),(64,8)]:
 plan=list(windows(len(ids),window,stride)); scored=[b+j for b,e,f in plan for j in range(f,e-b)]
 assert scored==list(range(1,len(ids)))
 loss,count,bounds=evaluate_sliding(model,ids,window,stride,False)
 expected=[]
 with torch.no_grad():
  for b,e,f in plan:
   x=torch.tensor(ids[b:e])[None,:]; logits=model(x).logits
   expected.append(F.cross_entropy(logits[:,f-1:-1].float().transpose(1,2),x[:,f:],reduction='none').double().sum().item())
 np.testing.assert_allclose(loss,expected,rtol=1e-6)
 assert count.sum()==len(ids)-1
print('Coverage, tail, shifted targets and CE match: PASS')
GROUPS={'UP':'up_proj','DOWN':'down_proj','GATE':'gate_proj','Q':'q_proj','K':'k_proj','V':'v_proj','O':'o_proj'}
assert all(len(targets(model,g))==1 for g in GROUPS)
SVD_DEVICE='cpu'
for shape in [(12,20),(20,12),(12,12)]:
 w=torch.randn(shape)
 parts=decompose(w)
 for de,p in parts.items():
  full=reconstruct(p,np.arange(min(shape)))
  torch.testing.assert_close(full,w,atol=2e-4,rtol=2e-4)
  bins=np.array_split(np.arange(min(shape)-1,-1,-1),10)
  assert sorted(np.concatenate(bins).tolist())==list(range(min(shape)))
print('All seven targets, both SVD orientations, DE inverse, deciles: PASS')
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp)
 spec=dict(tokens={'length':len(ids)},window=8,stride=2,condition={'group':'baseline'})
 evaluate_job(model,spec,ids,[])
 assert read_result(spec)['scored_tokens']==36
 p=job_path(spec)/'losses.npz'; p.write_bytes(b'corrupted')
 try:read_result(spec)
 except ValueError:pass
 else:raise AssertionError('Corruption was accepted')
print('Persistence and corruption detection: PASS')
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp); MODEL_KEY='llama'; BASELINE_ONLY=False
 prepare_tokens=lambda: (ids,{})
 make_plan=lambda meta: [{'condition':{'group':'baseline'}}]
 read_result=lambda spec: {'status':'ok'}
 def forbidden(*a,**kw):raise AssertionError('Completed plan loaded a model')
 load_model=forbidden
 run_experiment()
print('Completed plan skips model load/SVD/DE: PASS')
