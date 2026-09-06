from tests.test_persistence import test_crash_recovery_wal, test_snapshot_roundtrip_recall
from tests.test_hnsw_recall import test_recall, test_insert_delete_consistency, test_edge_cases
from tests.test_concurrency import test_concurrency_mixed_workload, test_concurrency_stress_overlap
import tempfile
import shutil

if __name__ == '__main__':
    print("Running test_crash_recovery_wal...")
    test_crash_recovery_wal()
    print("Running test_snapshot_roundtrip_recall...")
    test_snapshot_roundtrip_recall()
    print("Running test_recall...")
    test_recall()
    print("Running test_insert_delete_consistency...")
    test_insert_delete_consistency()
    print("Running test_edge_cases...")
    test_edge_cases()
    
    print("Running test_concurrency_mixed_workload...")
    tmp = tempfile.mkdtemp()
    try:
        test_concurrency_mixed_workload(tmp)
    finally:
        shutil.rmtree(tmp)
        
    print("Running test_concurrency_stress_overlap...")
    tmp2 = tempfile.mkdtemp()
    try:
        test_concurrency_stress_overlap(tmp2)
    finally:
        shutil.rmtree(tmp2)
        
    print("All tests passed successfully!")
