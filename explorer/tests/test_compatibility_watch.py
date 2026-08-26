from types import SimpleNamespace
import unittest

from repo_checker.summarize import _bridge_compatibility_tracker
from repo_checker.triage import score_github_item


class CompatibilityWatchTests(unittest.TestCase):
    def test_signed_vector_scalar_support_is_investigate(self):
        item = SimpleNamespace(
            number=1,
            title="VMI signed vmins and vmaxs support",
            body="",
            files=(),
        )

        result = score_github_item(item)

        self.assertEqual(result.classification, "Investigate")
        self.assertGreaterEqual(result.score, 8)

    def test_one_lane_broadcast_load_support_is_investigate(self):
        item = SimpleNamespace(
            number=2,
            title="BRC_B32 one-lane broadcast load support",
            body="",
            files=(),
        )

        result = score_github_item(item)

        self.assertEqual(result.classification, "Investigate")
        self.assertGreaterEqual(result.score, 8)

    def test_exact_one_point_store_support_is_investigate(self):
        item = SimpleNamespace(
            number=3,
            title="VMI ONEPT_B32 rank-zero store support",
            body="",
            files=(),
        )

        result = score_github_item(item)

        self.assertEqual(result.classification, "Investigate")
        self.assertGreaterEqual(result.score, 8)

    def test_reduction_result_shape_support_is_investigate(self):
        item = SimpleNamespace(
            number=4,
            title="VMI reduction result shape support",
            body="",
            files=(),
        )

        result = score_github_item(item)

        self.assertEqual(result.classification, "Investigate")
        self.assertGreaterEqual(result.score, 8)

    def test_tracker_is_available_to_the_summarizer(self):
        tracker = _bridge_compatibility_tracker()

        self.assertIn("AVE to PTOAS VMI Compatibility Tracker", tracker)
        self.assertIn("ONEPT_B32", tracker)


if __name__ == "__main__":
    unittest.main()
