"""
Test Runner

Discovers and runs all unit tests.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_all_tests():
    """Discover and run all tests"""

    print("="*70)
    print("DATAHAWK TEST SUITE")
    print("="*70)
    print()

    # Discover tests
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n[OK] All tests passed!")
        return 0
    else:
        print("\n[FAIL] Some tests failed")
        return 1


if __name__ == '__main__':
    sys.exit(run_all_tests())
