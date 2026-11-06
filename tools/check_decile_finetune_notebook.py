"""Offline CPU validation: tiny LoRA and synthetic workflow, no pretrained models."""
import os,sys,json,hashlib,math,gc,time,re,inspect,importlib.metadata,tempfile
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from funcs1 import _dyson_from_svd
import nbformat
nb=nbformat.read(REPO/'notebooks/12_de_decile_finetune_order.ipynb',4)
for cell in nb.cells:
 if cell.cell_type=='code' and (cell.source.startswith('def sha(') or cell.source.startswith('# 12専用:')):
  exec(compile(cell.source,'notebook_runtime','exec'))
from transformers import LlamaConfig,LlamaForCausalLM
SVD_DEVICE='cpu';DECILES=list(range(1,11));SEQ_LEN=16;BATCH_SIZE=1
smoke_test()
CONFIG_HASH='smoke';TRAIN_BF16=False
GROUPS={'UP':['up_proj'],'DOWN':['down_proj'],'GATE':['gate_proj'],'Q':['q_proj'],'K':['k_proj'],'V':['v_proj'],'O':['o_proj']}
TRAIN=dict(epochs=1,learning_rate=.001,batch=1,accumulation=1,rank=2,alpha=4,dropout=0.,warmup_ratio=0.,save_steps=1,logging_steps=1)
TRAIN_TOKENS={'tiny':np.random.default_rng(3).integers(0,32,(2,16))};TOKENS=TRAIN_TOKENS
cfg=LlamaConfig(vocab_size=32,hidden_size=24,intermediate_size=40,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
def make():
 torch.manual_seed(7);return LlamaForCausalLM(cfg)
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp);LOCAL_CACHE=OUTPUT/'cache';CACHE_ROOT=LOCAL_CACHE;DELETE_CACHE_AFTER_GROUP=True
 model=make();original=model.model.layers[0].mlp.gate_proj.weight.detach().clone()
 parts=decompose(original);pruned,_=remove_decile(original,parts[True],1)
 with torch.no_grad():model.model.layers[0].mlp.gate_proj.weight.copy_(pruned)
 before=evaluate(model,TOKENS['tiny'])[0]
 merged=train_or_merge(model,'tiny',42,OUTPUT/'train')
 assert training_receipt(OUTPUT/'train')['global_step']==2
 after=evaluate(merged,TOKENS['tiny'])[0]
 assert not np.allclose(before,after), 'Training had no effect'
 restored=make()
 with torch.no_grad():restored.model.layers[0].mlp.gate_proj.weight.copy_(pruned)
 restored=train_or_merge(restored,'tiny',42,OUTPUT/'train')
 assert np.allclose(after,evaluate(restored,TOKENS['tiny'])[0],atol=1e-5)
 checkpoint=resume_checkpoint(OUTPUT/'train');assert checkpoint
 print('PASS actual LoRA train/merge/save/reload, complete checkpoint:',Path(checkpoint).name)
 # Resume Trainer from checkpoint-1 using same pruned initialization and settings.
 resume=OUTPUT/'resume';(resume/'trainer').mkdir(parents=True)
 import shutil
 shutil.copytree(OUTPUT/'train/trainer/checkpoint-1',resume/'trainer/checkpoint-1')
 fresh=make()
 with torch.no_grad():fresh.model.layers[0].mlp.gate_proj.weight.copy_(pruned)
 resumed=train_or_merge(fresh,'tiny',42,resume)
 assert np.allclose(after,evaluate(resumed,TOKENS['tiny'])[0],atol=1e-5)
 print('PASS checkpoint restart and final weights match uninterrupted training')
 # Save/read/checksum guard.
 save_result(OUTPUT/'result','tiny',after,np.full(2,15),dict(kind='test'))
 assert read_result(OUTPUT/'result')['ppl']>0
 print('PASS PPL save/read integrity')
# 実行制御を1モジュール×全10群×2順序×DE2条件で検証。
# 学習そのものは上で実検証済み。ここでは軽量な決定的変更に置換する。
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp);LOCAL_CACHE=OUTPUT/'cache';CACHE_ROOT=LOCAL_CACHE
 GROUPS={'GATE':['gate_proj']};MODELS={'tiny':'tiny'};RUN_MODELS=['tiny'];SEEDS=[42]
 PHASES=['finetune_then_prune','prune_then_finetune'];REVISIONS={'tiny':'fixed'}
 calls={'load':0,'train':0}
 def model_load(key): calls['load']+=1;return make().eval()
 def train_or_merge(model,key,seed,path):
  calls['train']+=1
  with torch.no_grad():model.model.layers[0].mlp.gate_proj.weight.add_(.001)
  write_json(path/'adapter/receipt.json',dict(seed=seed,source='mock'))
  return model.eval()
 orig_cuda=torch.cuda.is_available;orig_bf16=torch.cuda.is_bf16_supported
 torch.cuda.is_available=lambda:True;torch.cuda.is_bf16_supported=lambda:True
 try:
  run12()
  assert ALL_REQUESTED_COMPLETED and calls=={'load':21,'train':21},calls
  df=collect_results();assert len(df)==40 and (df.status=='complete').all()
  snapshot=dict(calls);run12();assert calls==snapshot
  # 結果破損の検出。黙って完了扱いしない。
  path=job_path('tiny',42,PHASES[0],'GATE',False,1)
  (path/'losses.npz').write_bytes(b'corrupt')
  try:complete_job('tiny',42,PHASES[0],'GATE',False,1)
  except AssertionError:pass
  else:raise AssertionError('corruption not detected')
  print('PASS all 40 branch conditions, 21 independent training starts, complete-job skip, corrupt-result rejection')
 finally:
  torch.cuda.is_available=orig_cuda;torch.cuda.is_bf16_supported=orig_bf16
# モデル選択: 未選択モデルの入力がなくても選択モデルだけで準備できる。
import ast
for cell in nb.cells:
 if cell.cell_type=='code' and cell.source.startswith('def prepare_data('):
  node=next(n for n in ast.parse(cell.source).body if isinstance(n,ast.FunctionDef) and n.name=='prepare_data')
  exec(compile(ast.Module(body=[node],type_ignores=[]),'prepare_data','exec'))
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT_ROOT=Path(tmp);OUTPUT=OUTPUT_ROOT/'gemma';OUTPUT.mkdir()
 LEGACY_OUTPUT=OUTPUT_ROOT/'legacy';LEGACY_OUTPUT.mkdir()
 MODELS={'llama':'unused','gemma':'test-gemma'};RUN_MODELS=['gemma'];REVISIONS={'gemma':'fixed'}
 arr=np.ones((2,SEQ_LEN),dtype=np.int64)
 np.save(LEGACY_OUTPUT/'gemma_tokens.npy',arr)
 write_json(LEGACY_OUTPUT/'gemma_tokens.json',dict(sha256=sha(LEGACY_OUTPUT/'gemma_tokens.npy')))
 pin=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',revision='test')
 write_json(OUTPUT_ROOT/'dataset_revision.json',pin)
 p=OUTPUT/'inputs/gemma_train.npy';p.parent.mkdir();np.save(p,arr)
 identity=dict(model=MODELS['gemma'],revision='fixed',dataset=pin,split='train',seq_len=SEQ_LEN,
               packing='join newline newline; add_special_tokens=False; disjoint full blocks; drop remainder')
 write_json(p.with_suffix('.json'),dict(identity=identity,sha256=sha(p)))
 prepared=prepare_data()
 assert all(set(d)=={'gemma'} for d in prepared)
 print('PASS single-model input isolation: Gemma selected, no Llama files/network/model required')
