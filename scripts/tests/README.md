# Insight Testing Suite

Comprehensive testing tools to uncover hidden issues and ensure app reliability.

## Quick Start

```bash
# Quick smoke test (<30 seconds)
./scripts/tests/run_all_tests.sh --quick

# Run all tests (10-15 minutes)
./scripts/tests/run_all_tests.sh --all

# Run specific test suite
./scripts/tests/run_all_tests.sh --stress
./scripts/tests/run_all_tests.sh --concurrent
./scripts/tests/run_all_tests.sh --memory
./scripts/tests/run_all_tests.sh --edge-case
```

## Test Suites

### 1. Stress Test (`stress_test.sh`)
Tests the app under heavy load to uncover hidden issues.

**Tests:**
- Rapid document switching (50+ iterations)
- Concurrent file uploads (5+ simultaneous)
- Large document operations
- Search stress test (50+ searches)
- Memory pressure test

**Run time:** ~2-3 minutes

**When to use:**
- Before releasing new version
- After major code changes
- To find performance bottlenecks

### 2. Concurrent Operations Test (`concurrent_test.sh`)
Tests for race conditions, deadlocks, and concurrency issues.

**Tests:**
- Concurrent chat operations (10+ workers)
- Concurrent document access
- Mixed workload simulation
- Rapid start/stop cycles

**Run time:** ~3-4 minutes

**When to use:**
- Testing multi-threaded code paths
- After IPC/async changes
- To find race conditions

### 3. Memory Leak Test (`memory_leak_test.sh`)
Monitors memory usage over time to detect memory leaks.

**Tests:**
- Continuous memory monitoring (5 minutes)
- Memory-intensive operations
- File descriptor leak detection
- Growth trend analysis

**Run time:** 5-10 minutes (configurable)

**When to use:**
- After UI/component changes
- To investigate memory usage growth
- Before release (quality gate)

### 4. Edge Case Test (`edge_case_test.sh`)
Tests unusual scenarios that may break the app.

**Tests:**
- Empty/zero-byte files
- Very long filenames (255+ chars)
- Special characters in filenames
- Unicode/international filenames
- Binary files
- Deep directory structures
- Many files in single directory
- Invalid UTF-8
- Symbolic links
- File permissions

**Run time:** ~1 minute

**When to use:**
- Testing file upload handling
- After file processing changes
- To improve robustness

## Test Runner

The `run_all_tests.sh` script orchestrates all test suites:

```bash
# Auto-detect app and run all tests
./scripts/tests/run_all_tests.sh --all

# Specify app path
./scripts/tests/run_all_tests.sh /path/to/Insight.app --all

# Quick smoke test only
./scripts/tests/run_all_tests.sh --quick

# Run specific suites
./scripts/tests/run_all_tests.sh --stress --memory
```

## Interpreting Results

### Success Criteria

**All tests pass:**
```
✓ All tests passed!
Test Suites Run: 4
Passed: 4
Failed: 0
```

**App is ready for distribution.**

### Warnings

**Some tests have warnings but passed:**
- Review warnings in test report
- Usually non-critical (e.g., high memory usage)
- Monitor in production if concerning

### Failures

**Any test fails:**
1. Review detailed test logs in `/tmp/insight_*_test_*/`
2. Check app logs: `~/.insight/logs/`
3. Fix identified issues
4. Re-run tests

## Test Artifacts

All test logs and reports are preserved in:
```
/tmp/insight_stress_test_*/
/tmp/insight_concurrent_test_*/
/tmp/insight_memory_test_*/
/tmp/insight_edge_case_test_*/
```

Master reports are saved to:
```
/tmp/insight_test_report_YYYYMMDD_HHMMSS.txt
/Users/nelson/py/insight/insight_app/TEST_REPORT_YYYYMMDD_HHMMSS.txt
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build app
        run: ./scripts/build_onedir_prod.sh --no-sign
      - name: Run smoke test
        run: ./scripts/tests/run_all_tests.sh --quick
      - name: Run stress tests
        run: ./scripts/tests/run_all_tests.sh --stress
      - name: Upload test reports
        uses: actions/upload-artifact@v3
        with:
          name: test-reports
          path: TEST_REPORT_*.txt
```

## Pre-Release Checklist

Before releasing a new version:

- [ ] Run quick smoke test: `./scripts/tests/run_all_tests.sh --quick`
- [ ] Run all test suites: `./scripts/tests/run_all_tests.sh --all`
- [ ] Review test reports
- [ ] Fix any critical issues
- [ ] Test on fresh macOS installation
- [ ] Verify offline functionality
- [ ] Manual UI testing

## Troubleshooting

### "App not running" Error

Start the app before running tests:
```bash
open /path/to/Insight.app
```

### "Permission denied" Error

Make scripts executable:
```bash
chmod +x scripts/tests/*.sh
```

### Tests Timeout

- Increase timeout in test script
- Check for app hangs in logs
- Verify app is responsive

### Memory Test Shows Leak

1. Verify leak is genuine (not normal growth)
2. Check for event listener cleanup
3. Look for unclosed resources (files, connections)
4. Profile with Chrome DevTools or Instruments

## Customization

### Adjust Test Parameters

Edit test scripts to change:
- Iterations (default: 50)
- Concurrent operations (default: 10)
- Memory test duration (default: 300s)
- Sample intervals

Example:
```bash
# Run stress test with 100 iterations
./scripts/tests/stress_test.sh /path/to/Insight.app 100

# Run memory test for 10 minutes
./scripts/tests/memory_leak_test.sh /path/to/Insight.app 600
```

## Contributing

To add new test cases:

1. Create new test function in appropriate suite
2. Add to test runner in `run_all_tests.sh`
3. Update this README
4. Test thoroughly

## Support

For issues or questions:
- Check test logs: `/tmp/insight_*_test_*/`
- Check app logs: `~/.insight/logs/`
- Review test reports
- Open GitHub issue with logs attached

---

**Version:** 1.0.0
**Platform:** macOS 10.13+ (ARM64 and x86_64)
**Test Coverage:** Stress, Concurrency, Memory Leaks, Edge Cases
