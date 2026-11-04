import os
import sys
import gc
import numpy as np

def get_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ('cb', wintypes.DWORD),
                ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t),
                ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t),
                ('PrivateUsage', ctypes.c_size_t),
            ]
        GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
        GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
        GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        hProcess = ctypes.windll.kernel32.GetCurrentProcess()
        GetProcessMemoryInfo(hProcess, ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / (1024.0 * 1024.0)
    return 0.0

def load_sift(count: int = 100_000):
    sift_path = os.path.join(os.path.dirname(__file__), "data", "sift", "sift_base.fvecs")
    return np.fromfile(sift_path, dtype=np.float32, count=count * 129).reshape(count, 129)[:, 1:].copy()

def measure_faiss(vectors: np.ndarray):
    import faiss
    gc.collect()
    rss_before = get_rss_mb()
    idx = faiss.IndexHNSWFlat(128, 16, faiss.METRIC_L2)
    idx.hnsw.efConstruction = 100
    idx.add(vectors)
    gc.collect()
    rss_after = get_rss_mb()
    return rss_after - rss_before, rss_after

def measure_vsearch(vectors: np.ndarray):
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
    from vsearch.hnsw import HNSWIndex, HNSWConfig
    gc.collect()
    rss_before = get_rss_mb()
    cfg = HNSWConfig(M=16, ef_construction=100, heuristic=True)
    idx = HNSWIndex(config=cfg, metric="l2")
    idx.batch_insert(vectors)
    gc.collect()
    rss_after = get_rss_mb()
    return rss_after - rss_before, rss_after

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    print(f"Loading 100K SIFT vectors...")
    vecs = load_sift(100_000)
    print(f"Vectors loaded: shape {vecs.shape}, raw array size {vecs.nbytes / (1024*1024):.2f} MB")
    
    if mode in ("both", "faiss"):
        delta, total = measure_faiss(vecs)
        print(f"[FAISS] RSS Delta: {delta:.2f} MB | Total Process RSS: {total:.2f} MB")
        
    if mode in ("both", "vsearch"):
        delta, total = measure_vsearch(vecs)
        print(f"[vsearch] RSS Delta: {delta:.2f} MB | Total Process RSS: {total:.2f} MB")
