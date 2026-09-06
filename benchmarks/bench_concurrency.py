import time
import threading
import numpy as np
import tempfile
import shutil

from src.vsearch.hnsw import HNSWIndex, HNSWConfig

def benchmark_read_throughput(num_readers: int, index: HNSWIndex, dim: int, queries_per_thread: int = 100):
    start_time = time.time()
    
    def reader_worker():
        for _ in range(queries_per_thread):
            q = np.random.rand(dim).astype(np.float32)
            index.search(q, k=10)
            
    threads = []
    for _ in range(num_readers):
        t = threading.Thread(target=reader_worker)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    duration = time.time() - start_time
    total_queries = num_readers * queries_per_thread
    qps = total_queries / duration
    return qps

def benchmark_write_throughput(num_readers: int, index: HNSWIndex, dim: int, inserts_to_do: int, start_node_id: int):
    # One writer, N readers
    
    stop_readers = False
    def reader_worker():
        while not stop_readers:
            q = np.random.rand(dim).astype(np.float32)
            index.search(q, k=10)
            
    reader_threads = []
    for _ in range(num_readers):
        t = threading.Thread(target=reader_worker)
        reader_threads.append(t)
        t.start()
        
    start_time = time.time()
    
    for i in range(inserts_to_do):
        vec = np.random.rand(dim).astype(np.float32)
        index.insert(start_node_id + i, vec)
        
    duration = time.time() - start_time
    
    stop_readers = True
    for t in reader_threads:
        t.join()
        
    ips = inserts_to_do / duration
    return ips

if __name__ == "__main__":
    temp_dir = tempfile.mkdtemp()
    try:
        config = HNSWConfig(M=16, ef_construction=100, ef_search=50)
        index = HNSWIndex(config=config, persist_dir=temp_dir)
        dim = 128
        
        print("Populating initial index with 2000 vectors...")
        for i in range(2000):
            vec = np.random.rand(dim).astype(np.float32)
            index.insert(i, vec)
            
        print("\n--- Read Throughput (Search QPS) ---")
        qps_1 = benchmark_read_throughput(1, index, dim, 50)
        print(f"1 Thread: {qps_1:.2f} QPS")
        qps_4 = benchmark_read_throughput(4, index, dim, 50)
        print(f"4 Threads: {qps_4:.2f} QPS")
        qps_16 = benchmark_read_throughput(16, index, dim, 50)
        print(f"16 Threads: {qps_16:.2f} QPS")
        
        print("\n--- Write Throughput Degradation ---")
        # Baseline: 1 Writer, 0 Readers
        ips_baseline = benchmark_write_throughput(0, index, dim, 50, 2000)
        print(f"Baseline (1 Writer, 0 Readers): {ips_baseline:.2f} inserts/sec")
        
        # Concurrent: 1 Writer, 4 Readers
        ips_concurrent = benchmark_write_throughput(4, index, dim, 50, 2500)
        print(f"Concurrent (1 Writer, 4 Readers): {ips_concurrent:.2f} inserts/sec")
        
    finally:
        shutil.rmtree(temp_dir)
