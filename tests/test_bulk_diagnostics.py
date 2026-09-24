import numpy as np
import pandas as pd
import pytest
import torch
from funcs1 import get_bulk_ks, _mp_grid, dyson_equalizer_algorithm1, get_esd_metrics
from bulk_diagnostics import spectra, diagnose_spectra, run_diagnostics


def test_full_equals_bulk_without_outliers():
    x, cdf = _mp_grid(.5)
    values = np.interp((np.arange(100)+.5)/100, cdf, x)
    d = get_bulk_ks(values[::-1], .5, 1, 4)
    assert d['ks'] == d['full_ks']
    assert d['ks'] < .011
    out = get_bulk_ks(np.r_[values, 10, 20], .5, 1, 4)
    assert out['ks'] == d['ks']
    assert out['n_signal'] == 2


def test_empty_and_invalid():
    assert np.isnan(get_bulk_ks([10], .5, 1, 4)['ks'])
    for values, gamma, var, threshold in [([np.nan], .5, 1, 4), ([1], 2, 1, 4),
                                          ([1], .5, 0, 4), ([1], .5, 1, 1)]:
        with pytest.raises(ValueError):
            get_bulk_ks(values, gamma, var, threshold)


@pytest.mark.parametrize('shape', [(8,12), (12,8), (8,8), (7,12)])
def test_torch_de_matches_existing(shape):
    y = np.random.default_rng(12).normal(size=shape)
    pre, post, p, n = spectra(torch.tensor(y))
    oriented = y if shape[0] <= shape[1] else y.T
    expected, _, _ = dyson_equalizer_algorithm1(oriented)
    np.testing.assert_allclose(post, np.sort(np.linalg.svd(expected, compute_uv=False)**2)/n, rtol=1e-9)
    np.testing.assert_allclose(pre, np.sort(np.linalg.svd(oriented, compute_uv=False)**2)/n, rtol=1e-9)


def test_resume_and_fixed_variance(tmp_path):
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Linear(12,8,bias=False))
    saved = get_esd_metrics(model)
    a = run_diagnostics(model, saved, tmp_path, {'test':1})
    b = run_diagnostics(model, saved, tmp_path, {'test':1})
    assert len(a) == len(b) == 1
    assert a.iloc[0]['sigma2_DE_1'] == 1
    assert a.iloc[0]['KS_preDE_saved'] == saved.iloc[0]['KS_preDE']
    with pytest.raises(ValueError):
        run_diagnostics(model, saved, tmp_path, {'test':2})


def test_threshold_inclusive_and_no_refit():
    d = get_bulk_ks([0, 1, 4, 5], 1, 1, 4)
    assert d['n_bulk'] == 3
    assert d['sigma2'] == 1 and d['threshold'] == 4
    assert d['bulk_ratio'] == .75


def test_load_pickle_and_changed_input_rejected(tmp_path):
    from bulk_diagnostics import load_metrics
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(12,8,bias=False))
    saved = get_esd_metrics(model)
    source = tmp_path/'input.pkl'
    saved.to_pickle(source)
    restored = load_metrics(source)
    assert restored.name.tolist() == saved.name.tolist()
    run_diagnostics(model, restored, tmp_path/'out', {})
    restored.loc[0, 'threshold_preDE'] *= 1.1
    with pytest.raises(ValueError, match='configuration'):
        run_diagnostics(model, restored, tmp_path/'out', {})


def test_wandb_publication_contract(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from bulk_diagnostics import publish_wandb
    calls = {}
    class Run:
        url = 'https://wandb.ai/test/run'
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def log(self, data): calls['table'] = data
        def log_artifact(self, artifact): calls['artifact'] = artifact
    class Artifact:
        def __init__(self, name, type): self.name = name
        def add_dir(self, path): calls['directory'] = path
    def init(**kwargs):
        calls['init'] = kwargs
        return Run()
    monkeypatch.setitem(sys.modules, 'wandb', SimpleNamespace(init=init, Artifact=Artifact,
                                                            Table=lambda dataframe: dataframe))
    result = publish_wandb(pd.DataFrame({'x':[1]}), tmp_path, {'model_id':'test/model'}, ['source/run'])
    assert result == Run.url
    assert calls['init']['config']['source_runs'] == ['source/run']
    assert 'id' not in calls['init']  # 元runをresume/上書きしない。
    assert calls['directory'] == str(tmp_path)


def test_nested_gemma_model_resolution():
    from bulk_diagnostics import resolve_analysis_model
    outer = torch.nn.Module()
    outer.model = torch.nn.Module()
    outer.model.language_model = torch.nn.Module()
    inner = outer.model.language_model.model = torch.nn.Module()
    inner.layers = torch.nn.ModuleList([torch.nn.Linear(4,3)])
    assert resolve_analysis_model(outer, ['layers.0']) is inner
    with pytest.raises(ValueError):
        resolve_analysis_model(outer, ['unknown'])
