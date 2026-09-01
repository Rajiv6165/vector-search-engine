from tests.test_persistence import test_crash_recovery_wal, test_snapshot_roundtrip_recall
from tests.test_hnsw_recall import test_recall, test_insert_delete_consistency, test_edge_cases

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
    print("All tests passed successfully!")
