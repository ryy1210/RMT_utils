"""Offline checks: paper packing, saved plans/imports, actual tiny pruning and resume."""
import os,sys,json,hashlib,math,gc,time,re,inspect,importlib.metadata,tempfile,platform
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['WANDB_MODE']='disabled'
import nbformat,numpy as np,pandas as pd,torch
import torch.nn.functional as F
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
import funcs1
from funcs1 import _dyson_from_svd
for name in ['12_de_decile_finetune_order','13_paper_ppl_2048_bos']:
 nb=nbformat.read(REPO/'notebooks'/f'{name}.ipynb',4);nbformat.validate(nb)
 for i,c in enumerate(nb.cells):
  if c.cell_type=='code':compile(c.source,f'{name}:{i}','exec')
for i in [4,5,6]:exec(compile(nb.cells[i].source,f'13:{i}','exec'))
print('PASS notebook syntax/schema')
SVD_DEVICE='cpu';DECILES=list(range(1,11));BATCH_SIZE=1;EVAL_SEQ_LEN=16;EVAL_HASH='test';CONFIG_HASH='test'
ALL_GROUPS={'UP':['up_proj'],'DOWN':['down_proj'],'GATE':['gate_proj'],'Q':['q_proj'],'K':['k_proj'],'V':['v_proj'],'O':['o_proj']}
GROUPS=dict(ALL_GROUPS);SEEDS=[42]
packed,meta=pack_paper_tokens(np.arange(34),99,16)
assert packed.shape==(2,16) and np.all(packed[:,0]==99)
assert np.array_equal(packed[:,1:].reshape(-1),np.arange(30)) and meta['dropped_tokens']==4
DATA_ROOT=next((Path.home()/'Library/CloudStorage').glob('GoogleDrive-*/*/TUS/hashiguchi/data'))
for key in ('llama','gemma'):
 jobs=build_plan13(key,DATA_ROOT/'ibis2026_de_lra')
 assert len(jobs)==163 and len({j['id'] for j in jobs})==163
 assert len([j for j in jobs if j['kind']=='top50_lora'])==4
print('PASS BOS packing and both 163-condition source plans')
# Real saved sources, only write copied compatible results into temporary output.
MODEL_KEY='gemma';REVISIONS={'gemma':'cc012e0a6d0787b4adcc0fa2c4da74402494554d'}
PLAN=build_plan13('gemma',DATA_ROOT/'ibis2026_de_lra');IMPORT_11_1=True
old=DATA_ROOT/'ibis2026_de_lra/gemma-qk-ppl-settings-grid-v1'
setting=dict(batch=1,bos='each_block',length=2048,load_precision='bf16',loader='conditional_text',precision='bf16')
sources=[json.loads(p.read_text()) for p in (old/'jobs').glob('*/summary.json')]
source=next(o for o in sources if o['spec']['setting']==setting and o['spec']['condition']['group']=='Q')
s=source['spec'];saved_env=paper_environment
paper_environment=lambda:s['environment']
TOKEN_HASH={'gemma':s['tokens']['sha256']};TOKENS={'gemma':np.empty((142,0))};EVAL_SEQ_LEN=2048
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp)
 assert import_11_1()==41
 assert import_11_1()==0
 assert sum(complete13(j) is not None for j in PLAN)==41
 EVAL_HASH='wrong-tokens';TOKEN_HASH={'gemma':'wrong'}
 assert import_11_1()==0
paper_environment=saved_env
print('PASS 41 real source imports, duplicate skip, token mismatch rejection')
# Tiny actual model, direct reference CE, decile pruning and restoration.
from transformers import LlamaConfig,LlamaForCausalLM
cfg=LlamaConfig(vocab_size=32,hidden_size=24,intermediate_size=40,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64)
def make():
 torch.manual_seed(7);return LlamaForCausalLM(cfg).eval()
MODEL_KEY='tiny';MODELS={'tiny':'tiny'};REVISIONS={'tiny':'fixed'};EVAL_SEQ_LEN=16;EVAL_HASH='tiny-eval'
TOKENS={'tiny':np.random.default_rng(3).integers(0,32,(2,16))};TOKENS['tiny'][:,0]=1
model=make();loss,count=evaluate(model,TOKENS['tiny'])
with torch.no_grad():
 ids=torch.tensor(TOKENS['tiny']);logits=model(input_ids=ids).logits
 ref=F.cross_entropy(logits[:,:-1].transpose(1,2),ids[:,1:],reduction='none').double().sum(1).numpy()
assert np.allclose(loss,ref) and np.all(count==15)
PLAN=[dict(id='baseline',kind='baseline')]+[dict(id=f'd{d}_{de}',kind='decile',group='GATE',de=de,decile=d) for de in (False,True) for d in range(1,11)]
PHASES=['decile'];GROUPS={'GATE':['gate_proj']};calls=[]
def model_load(key):calls.append(key);return make()
display=lambda x:None
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp);LOCAL_CACHE=OUTPUT/'local';VERIFIED_CACHE=set()
 before={n:t.clone() for n,t in model.state_dict().items()}
 originals=apply_job13(model,PLAN[1]);restore_group(model,originals)
 assert all(torch.equal(t,before[n]) for n,t in model.state_dict().items())
 cuda=torch.cuda.is_available;bf16=torch.cuda.is_bf16_supported
 torch.cuda.is_available=lambda:True;torch.cuda.is_bf16_supported=lambda:True
 try:
  run13();assert len(calls)==1 and all(complete13(j) for j in PLAN)
  run13();assert len(calls)==1
  # Corrupted persisted result must not silently skip.
  p=evaluation_dir(path13(PLAN[0]))/'losses.npz';p.write_bytes(b'bad')
  try:complete13(PLAN[0])
  except AssertionError:pass
  else:raise AssertionError('Corrupt result accepted')
 finally:torch.cuda.is_available=cuda;torch.cuda.is_bf16_supported=bf16
print('PASS direct CE agreement, 20 actual decile conditions, restoration/cache and complete skip')
# Actual LRA -> actual LoRA -> evaluate; evaluation-only rerun uses existing adapters.
TRAIN_BF16=False;TRAIN_TARGET_MODULES=[v[0] for v in ALL_GROUPS.values()]
TRAIN=dict(epochs=1,learning_rate=.001,batch=1,accumulation=1,rank=2,alpha=4,dropout=0.,warmup_ratio=0.,save_steps=1,logging_steps=1)
TRAIN_TOKENS=TOKENS
PLAN=[]
for de in (False,True):
 source=dict(names=['model.layers.0.mlp.gate_proj'],ranks={'model.layers.0.mlp.gate_proj':12},params_saved=40*24-12*(40+24))
 PLAN.append(dict(id=f'lora_de{de}',kind='top50_lora',group='all',de=de,seed=42,source=source))
PHASES=['top50_lora'];calls=[]
with tempfile.TemporaryDirectory() as tmp:
 OUTPUT=Path(tmp);LOCAL_CACHE=OUTPUT/'local';VERIFIED_CACHE=set()
 cuda=torch.cuda.is_available;bf16=torch.cuda.is_bf16_supported
 # CPU training must see real CUDA availability; bypass only the run13 guard.
 source=inspect.getsource if False else nb.cells[6].source
 import ast
 node=next(x for x in ast.parse(source).body if isinstance(x,ast.FunctionDef) and x.name=='run13')
 node.body=[x for x in node.body if not (isinstance(x,ast.If) and 'BF16 CUDA GPU required' in ast.unparse(x))]
 exec(compile(ast.Module(body=[node],type_ignores=[]),'cpu_run13','exec'))
 run13();assert len(calls)==2 and all(complete13(j) for j in PLAN)
 receipts={j['id']:sha(path13(j)/'training/adapter/receipt.json') for j in PLAN}
 EVAL_HASH='new-evaluation'
 run13();assert len(calls)==4 and all(complete13(j) for j in PLAN)
 assert receipts=={j['id']:sha(path13(j)/'training/adapter/receipt.json') for j in PLAN}
 run13();assert len(calls)==4
print('PASS actual DE off/on LRA+LoRA and evaluation-only adapter reuse')
