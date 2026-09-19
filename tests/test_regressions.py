"""ネットワークやLLM重みを使わない、数値・Colab APIの回帰テスト。"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
import funcs1 as f
import data_utils as data
import evaluater as ev
import prune_utils as prune
from layerwrapper import WrappedLayer
from Prompter import Prompter, ZeroPrompter


def reference_de(Y):
    m, n = Y.shape
    U, s, Vh = np.linalg.svd(Y, full_matrices=True)
    eta = np.median(s)
    t = eta / (s*s+eta*eta)
    g1 = (U*U) @ t
    g2 = 1/eta + (Vh.T[:, :m]**2) @ (t-1/eta)
    x = (1/g1-eta) / np.sqrt(m-eta*np.sum(g1))
    y = (1/g2-eta) / np.sqrt(n-eta*np.sum(g2))
    return Y / np.sqrt(x[:, None]*y[None, :]), x, y


@pytest.mark.parametrize('shape', [(12,12), (8,40)])
def test_de_matches_full_svd(shape):
    Y = np.random.default_rng(9).normal(size=shape)
    old = reference_de(Y)
    for full in [True, False]:
        for actual, expected in zip(f.dyson_equalizer_algorithm1(Y, full), old):
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize('de', [True, False])
@pytest.mark.parametrize('shape', [(9,5),(5,9)])
def test_lra_reference(shape,de):
    W = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    r=2
    Y=W.numpy();transpose=shape[0]>shape[1]
    if transpose:Y=Y.T
    if de:Y,x,y=reference_de(Y.astype(float))
    U,s,Vh=np.linalg.svd(Y,full_matrices=False)
    expected=(U[:,:r]*s[:r])@Vh[:r]
    if de:expected*=np.sqrt(x[:,None]*y[None,:])
    if transpose:expected=expected.T
    out=f.apply_lra_1(W,r,DE=de)
    np.testing.assert_allclose(out.numpy(),expected,atol=2e-5,rtol=2e-5)
    assert out.dtype==W.dtype and out.device==W.device
    assert torch.equal(f.apply_lra_1(W,min(shape),DE=de),W)
    assert torch.count_nonzero(f.apply_lra_1(W,0,DE=de))==0


def test_bema_svd_matches_gram():
    Y=np.random.default_rng(1).normal(size=(40,80))
    a=f.bema_algorithm1_from_data(Y)
    b=f.bema_algorithm1_from_eigenvalues(np.linalg.svd(Y,compute_uv=False)**2/80,40,80)
    assert a['s_hat']==b['s_hat']
    assert a['sigma2_hat']==pytest.approx(b['sigma2_hat'],rel=1e-10)
    assert a['threshold']==pytest.approx(b['threshold'],rel=1e-10)


@pytest.mark.parametrize('gamma', [.25,1.0])
def test_mp_grid_agrees_with_quadrature(gamma):
    from scipy.integrate import quad
    probs=np.array([.1,.5,.9])
    quantiles=f.qmp_stable(probs,100,int(100*gamma))
    a=(1-np.sqrt(gamma))**2
    for prob,x in zip(probs,quantiles):
        cdf=quad(f.mp_pdf_zero_excluded,a,x,args=(gamma,1),epsabs=1e-9)[0]
        assert cdf==pytest.approx(prob,abs=2e-5)
    assert f.qmp_stable([.5],100,int(100*gamma),var=3)[0]==pytest.approx(3*quantiles[1])


@pytest.mark.parametrize('method', ['median','fix-finger','goodness-of-fit'])
def test_esd_schema_and_zero(method):
    model=nn.Sequential(nn.Linear(8,4,bias=False),nn.Linear(4,4,bias=False))
    with torch.no_grad():model[1].weight.zero_()
    result=f.get_esd_metrics(model,pl_fitting=method)
    assert len(result)==2 and model.training
    assert np.isnan(result.iloc[1]['alpha'])
    assert result.iloc[1]['s_hat_postDE']==0
    assert all(k in result for k in ['name','alpha','s_hat_preDE','KS_postDE_1','eigs','mp_soft_rank_preDE'])
    assert isinstance(result.iloc[0]['tail_xmin'],float)


def test_conv_esd():
    out=f.get_esd_metrics(nn.Sequential(nn.Conv2d(2,4,3)))
    assert out.iloc[0]['eigs_num']==4


def test_masks_exact_ties_and_endpoints():
    W=torch.ones(3,10)
    for sp in [0,.2,.5,1]:
        mask=prune.compute_mask_2(W,sp)
        assert (~mask).sum()==round(W.numel()*sp)
        assert torch.equal(~mask,prune.compute_mask(W,'unstructured',sp))


class Blocks(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Module()
        self.model.layers=nn.ModuleList([nn.ModuleDict({'q_proj':nn.Linear(4,2,bias=False)}),nn.ModuleDict({'q_proj':nn.Linear(4,6,bias=False)})])
        self.lm_head=nn.Linear(4,4)


def test_alpha_uses_block_specific_values_and_weighted_budget():
    model=Blocks();head=model.lm_head.weight.detach().clone()
    df=pd.DataFrame({'name':['model.layers.0.q_proj','model.layers.1.q_proj'],'alpha':[1.,3.]})
    _,log=prune.alpha_prune_llama(model,df,sparsity=.5)
    a,b=[log[n]['applied_sparsity'] for n in df.name]
    assert b/a==pytest.approx(3)
    assert (8*a+24*b)/32==pytest.approx(.5)
    assert torch.equal(head,model.lm_head.weight)
    assert sum((m.weight==0).sum() for n,m in model.named_modules() if isinstance(m,nn.Linear) and n!='lm_head')==16


def test_ambiguous_alpha_fails_before_mutation():
    model=Blocks();weights=[p.clone() for p in model.parameters()]
    with pytest.raises(ValueError,match='Missing or ambiguous'):
        prune.alpha_prune_llama(model,pd.DataFrame({'name':['q_proj'],'alpha':[1.]}))
    # q_proj-only suffix must not be accepted across multiple blocks.
    for actual,old in zip(model.parameters(),weights):assert torch.equal(actual,old)


class Tokenizer:
    special_tokens_map={}
    def __init__(self,offset=0):self.offset=offset;self.calls=0
    def __call__(self,text,return_tensors):
        self.calls+=1
        return SimpleNamespace(input_ids=torch.arange(32).reshape(1,-1)+self.offset)


def test_data_cache_and_partial_calibration(monkeypatch):
    monkeypatch.setattr(data,'load_dataset',lambda *a,**kw:{'text':['a','b']})
    data.clear_test_data_cache();tok=Tokenizer()
    a=list(data.get_test_data('wikitext2',tok,seq_len=8,batch_size=3))
    b=list(data.get_test_data('wikitext2',tok,seq_len=8,batch_size=2))
    assert tok.calls==1 and torch.equal(torch.cat(a),torch.cat(b))
    other=Tokenizer(100)
    assert list(data.get_test_data('wikitext2',other,seq_len=8))[0][0,0]==100
    cal=data.get_calib_train_data('wikitext2',tok,5,seqlen=8,batch_size=2)
    assert [len(x['input_ids']) for x in cal]==[2,2,1]


def test_wikitext_splits(monkeypatch):
    seen=[]
    def load(*a,**kw):
        seen.append(kw)
        return {'text':['train' if kw['split']=='train' else 'test']}
    monkeypatch.setattr(data,'load_dataset',load)
    data.get_wikitext2(1,0,8,Tokenizer())
    assert [x['split'] for x in seen]==['train','test']
    assert '/test/' in seen[1]['data_files']['test']


class ToyLM(nn.Module):
    def __init__(self,bad=False):
        super().__init__();self.embed=nn.Embedding(7,7);self.bad=bad
    def get_input_embeddings(self):return self.embed
    def forward(self,input_ids,**kw):
        logits=self.embed(input_ids)
        if self.bad:logits=logits*float('nan')
        return SimpleNamespace(logits=logits)


def test_ppl_matches_full_reference_and_restores_mode(monkeypatch):
    model=ToyLM();ids=torch.tensor([[1,2,3,4],[4,3,2,1],[1,1,2,2]])
    monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[ids[:2],ids[2:]])
    expected=torch.nn.functional.cross_entropy(model(ids).logits[:,:-1].reshape(-1,7),ids[:,1:].reshape(-1)).exp().item()
    out=ev.ppl_eval(model,None,['tiny'],4,2,'cpu')
    assert out['tiny']==pytest.approx(expected,rel=1e-6)
    assert model.training


def test_ppl_rejects_nonfinite_and_empty(monkeypatch):
    ids=torch.tensor([[1,2,3]])
    monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[ids])
    model=ToyLM(True)
    with pytest.raises(FloatingPointError):ev.ppl_eval(model,None,['tiny'],3,1,'cpu')
    assert model.training
    monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[])
    with pytest.raises(ValueError,match='No valid'):ev.ppl_eval(ToyLM(),None,['tiny'],3,1,'cpu')


def test_dispatched_model_not_moved(monkeypatch):
    model=ToyLM();model.hf_device_map={'embed':'cpu'}
    def fail(*a,**k):raise AssertionError('must not call model.to')
    model.to=fail
    monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[torch.tensor([[1,2,3]])])
    assert np.isfinite(ev.ppl_eval(model,None,['tiny'],3,1)['tiny'])


def test_lra_final_ppl_and_columns(monkeypatch):
    model=nn.Sequential(nn.Linear(8,4,bias=False))
    results=pd.DataFrame({'name':['0'],'alpha':[2.], 's_hat_preDE':[2], 's_hat_postDE':[2]})
    vals=iter([10.,12.]);monkeypatch.setattr(f,'get_ppl',lambda *a,**k:next(vals))
    history=f.run_lra_experiment(model,None,results,['0'],1,DE=False,PPLcalc=False)
    assert history.iloc[-1]['ppl']==12
    assert list(history)==['step','layer_compressed','alpha_of_layer','ppl','reduction_ratio_percent']
    assert history.iloc[-1]['reduction_ratio_percent']==pytest.approx(25)


def test_wrapper_batch_invariance_and_prune():
    layer=nn.Linear(3,2);inp=torch.randn(7,3)
    a,b=WrappedLayer(layer),WrappedLayer(layer)
    a.add_batch(inp,None);b.add_batch(inp[:2],None);b.add_batch(inp[2:],None)
    torch.testing.assert_close(a.scaler_row,b.scaler_row)
    before=layer(inp).mean(0)
    a.prune(torch.tensor([[True,False,True],[False,True,False]]))
    torch.testing.assert_close(layer(inp).mean(0),before)
    a.free()


def test_prompt_edge_cases():
    assert Prompter().get_response('plain')=='plain'
    assert Prompter().get_response('x### Response:a### Response:b')=='a### Response:b'
    assert isinstance(ZeroPrompter().generate_prompt(''),str)


def test_ppl_chunk_boundaries_and_padding(monkeypatch):
    model=ToyLM();ids=torch.arange(520).reshape(2,260)%7
    mask=torch.ones_like(ids);mask[0,:4]=0;mask[1,-3:]=0
    labels=ids[:,1:].clone();valid=mask[:,1:].bool() & mask[:,:-1].bool();labels[~valid]=-100
    expected=torch.nn.functional.cross_entropy(model(ids).logits[:,:-1].reshape(-1,7),labels.reshape(-1)).exp().item()
    monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[{'input_ids':ids,'attention_mask':mask}])
    assert ev.ppl_eval3(model,None,['tiny'],260,2,'cpu')['tiny']==pytest.approx(expected,rel=1e-6)


def test_tiny_llama_and_gemma_lora_without_download(monkeypatch):
    from transformers import LlamaConfig, LlamaForCausalLM, Gemma3TextConfig, Gemma3ForCausalLM
    from peft import LoraConfig, TaskType, get_peft_model
    configs=[(LlamaForCausalLM,LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1)),
             (Gemma3ForCausalLM,Gemma3TextConfig(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1,head_dim=8))]
    for cls,config in configs:
        model=cls(config)
        metrics=f.get_esd_metrics(model,pl_fitting='fix-finger')
        assert len(metrics)==7
        # ノートと同じAPIでLRA→LoRA→PPLを接続する。
        ids=torch.tensor([[1,2,3,4]])
        monkeypatch.setattr(ev,'get_test_data',lambda *a,**k:[ids])
        f.run_lra_experiment(model,None,metrics,metrics.name.tolist(),1,DE=True,PPLcalc=False,seq_len=4,batch_size=1)
        model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=2,lora_alpha=4,target_modules=['q_proj','v_proj']))
        model.train();loss=model(input_ids=ids,labels=ids).loss;loss.backward()
        assert torch.isfinite(loss)
        assert np.isfinite(f.get_ppl(model,None,seq_len=4,batch_size=1))


def test_lora_helper_training(tmp_path,monkeypatch):
    import LoRA
    from datasets import Dataset,DatasetDict
    from tokenizers import Tokenizer as Backend
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast,LlamaConfig,LlamaForCausalLM
    backend=Backend(WordLevel({'[UNK]':0,'[EOS]':1,'[PAD]':2,'hello':3,'answer':4,':':5},unk_token='[UNK]'))
    backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
    dataset=DatasetDict(train=Dataset.from_dict({'instruction':['hello']*6,'input':['']*6,'output':['answer']*6}))
    monkeypatch.setattr(LoRA,'load_dataset',lambda *a,**k:dataset)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=6,hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=1))
    result=LoRA.apply_lora(model,tok,batch_size=2,micro_batch_size=1,cutoff_len=16,val_set_size=2,num_epochs=1,output_dir=str(tmp_path),lora_r=2)
    assert result.peft_config['default'].r==2
    assert tok.pad_token_id==2


def test_original_public_parameter_names():
    import json,inspect,importlib
    from pathlib import Path
    api=json.loads((Path(__file__).parent/'public_api.json').read_text())
    for name,functions in api.items():
        module=importlib.import_module(name)
        for fname,params in functions.items():
            assert list(inspect.signature(getattr(module,fname)).parameters)==params


def test_vit_variable_layer_count_and_cache_path(tmp_path):
    class MiniViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer=nn.Module()
            self.transformer.layers=nn.ModuleList([
                nn.ModuleList([nn.Linear(4,4),nn.Linear(4,4)]),
                nn.ModuleList([nn.Linear(4,4),nn.Sequential(nn.Linear(4,6),nn.Linear(6,4))])])
            self.to_patch_embedding=nn.Identity()
            self.cls_token=nn.Parameter(torch.zeros(1,1,4))
            self.pos_embedding=nn.Parameter(torch.zeros(1,10,4))
            self.dropout=nn.Identity()
    cache=tmp_path/'mini';cache.mkdir();np.save(cache/'alpha.npy',np.arange(1,6,dtype=float))
    args=SimpleNamespace(model='mini',metric_cache=str(tmp_path),WW_metric='alpha',epsilon=.5,sparsity=.7,prune_metric='magnitude',prune_granularity='unstructured')
    model=MiniViT()
    prune.prune_vit_blockwise_for_vit_pytorch(args,model,torch.randn(2,3,4),'cpu')
    assert args.metric_cache==str(tmp_path)
    weights=[m.weight for m in model.transformer.modules() if isinstance(m,nn.Linear)]
    assert abs(sum(int((w==0).sum()) for w in weights)/sum(w.numel() for w in weights)-.7)<.02
