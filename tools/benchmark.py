"""CPU microbenchmark. LLM全体の速度やGPU peak memoryを示すものではない。"""
import sys,json,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import funcs1 as f


def timed(fn,repeats=3):
    times=[]
    for _ in range(repeats):
        start=time.perf_counter();fn();times.append(time.perf_counter()-start)
    return float(np.median(times))


Y=np.random.default_rng(1).normal(size=(64,1024))
np.linalg.svd(Y,full_matrices=False)
full=timed(lambda:np.linalg.svd(Y,full_matrices=True))
thin=timed(lambda:np.linalg.svd(Y,full_matrices=False))
def uncached():
    for _ in range(20):
        f._mp_grid.cache_clear();f.qmp_stable([.2,.5,.8],1024,64)
def cached():
    for _ in range(20):f.qmp_stable([.2,.5,.8],1024,64)
f._mp_grid.cache_clear();cold=timed(uncached);warm=timed(cached)
print(json.dumps({'python':platform.python_version(),'platform':platform.platform(),
 'matrix_shape':list(Y.shape),'median_of':3,'full_svd_seconds':full,
 'thin_svd_seconds':thin,'svd_speedup':full/thin,
 'right_vector_bytes_full':1024*1024*8,'right_vector_bytes_thin':64*1024*8,
 'mp_20_calls_uncached_seconds':cold,'mp_20_calls_cached_seconds':warm,
 'mp_reuse_speedup':cold/warm},indent=2))
