from tests.test_persistence import test_crash_recovery_wal, test_snapshot_roundtrip_recall

if __name__ == '__main__':
    test_crash_recovery_wal()
    test_snapshot_roundtrip_recall()
    print("All tests passed successfully!")
