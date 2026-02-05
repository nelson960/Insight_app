from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.services.mlops.contract import build_effective_contract_snapshot


class ContractSnapshotTests(unittest.TestCase):
    def test_snapshot_allows_missing_artifacts_in_ci_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            snapshot = build_effective_contract_snapshot(
                settings={},
                workspace=workspace,
                verify_hashes=False,
                allow_missing_artifacts=True,
                profile="dev",
            )
            self.assertTrue(snapshot["validation"]["ok"])
            self.assertTrue(snapshot["validation"]["warnings"])
            self.assertEqual(len(str(snapshot["contract_id"])), 64)

    def test_snapshot_fails_when_artifacts_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            snapshot = build_effective_contract_snapshot(
                settings={},
                workspace=workspace,
                verify_hashes=False,
                allow_missing_artifacts=False,
                profile="dev",
            )
            self.assertFalse(snapshot["validation"]["ok"])
            self.assertTrue(snapshot["validation"]["errors"])


if __name__ == "__main__":
    unittest.main()
