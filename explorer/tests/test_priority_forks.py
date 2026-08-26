from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from repo_checker.config import CheckerConfig, GitHubForkWatch, RepoConfig
from repo_checker.github_fork_collect import _branches, _discover_forks
from repo_checker.report import _persistent_context, _readme
from repo_checker.summarize import DailySummary


def _config(root: Path, *, max_forks_to_scan: int = 120) -> CheckerConfig:
    return CheckerConfig(
        root=root,
        state_path=root / "state.json",
        reports_dir=root / "reports",
        daily_report_dir=root / "reports/daily",
        generated_readme=root / "reports/README.md",
        timezone="UTC",
        model="test-model",
        big_change_threshold=7,
        max_fork_depth=2,
        max_commits_per_branch=40,
        max_diff_files_per_branch=80,
        max_diff_chars_per_branch=12000,
        max_markdown_files_for_diff_excerpt=8,
        max_markdown_changed_lines_for_diff=1200,
        max_forks_to_scan=max_forks_to_scan,
        max_branches_per_fork=30,
        max_fork_branch_changes_per_run=60,
        initial_backfill=False,
        env_files=(),
        repos=(),
    )


def _repo(root: Path) -> RepoConfig:
    return RepoConfig(
        id="ptoas",
        name="PTOAS",
        kind="github",
        path=root,
        fetch_remotes=(),
        track_branches=False,
        track_github_issues=False,
        track_github_prs=False,
        track_github_forks=True,
        priority_forks=(
            GitHubForkWatch(
                full_name="TaoTao-real/PTOAS",
                branches=("feature-vmi",),
            ),
        ),
        upstream="hw-native-sys/PTOAS",
    )


class PriorityForkTests(unittest.TestCase):
    @patch("repo_checker.github_fork_collect._get_json")
    @patch("repo_checker.github_fork_collect._page")
    def test_required_branch_is_queried_beyond_bounded_listing(self, page, get_json):
        page.return_value = [
            {"name": "codex/first", "commit": {"sha": "1" * 40}},
        ]
        get_json.return_value = {
            "name": "feature-vmi",
            "commit": {"sha": "2" * 40},
        }

        branches = _branches(
            "TaoTao-real/PTOAS",
            1,
            [],
            required_branches=("feature-vmi",),
        )

        self.assertEqual(
            [branch.name for branch in branches],
            ["feature-vmi", "codex/first"],
        )
        get_json.assert_called_once_with(
            "https://api.github.com/repos/TaoTao-real/PTOAS/branches/feature-vmi"
        )

    @patch("repo_checker.github_fork_collect._repo_metadata")
    @patch("repo_checker.github_fork_collect._forks")
    def test_priority_fork_displaces_ordinary_fork_at_discovery_cap(
        self,
        forks,
        repo_metadata,
    ):
        forks.side_effect = lambda parent, _errors: (
            (
                {
                    "full_name": "ordinary/PTOAS",
                    "html_url": "https://github.com/ordinary/PTOAS",
                    "default_branch": "main",
                    "pushed_at": "2026-08-26T00:00:00Z",
                },
            )
            if parent == "hw-native-sys/PTOAS"
            else ()
        )
        repo_metadata.return_value = {
            "full_name": "TaoTao-real/PTOAS",
            "html_url": "https://github.com/TaoTao-real/PTOAS",
            "default_branch": "main",
            "pushed_at": "2026-08-25T00:00:00Z",
            "parent": {"full_name": "hw-native-sys/PTOAS"},
            "source": {"full_name": "hw-native-sys/PTOAS"},
        }

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            discovered = _discover_forks(
                _repo(root),
                _config(root, max_forks_to_scan=1),
                [],
            )

        self.assertEqual(
            [fork.full_name for fork in discovered],
            ["TaoTao-real/PTOAS"],
        )

    def test_persistent_watch_context_survives_readme_regeneration(self):
        with TemporaryDirectory() as tmp:
            readme = Path(tmp) / "README.md"
            readme.write_text(
                "# Old report\n\n"
                "## Persistent Watch Context\n\n"
                "### TaoTao-real/PTOAS:feature-vmi\n\n"
                "Keep this status.\n"
            )
            context = _persistent_context(readme)

        rendered = _readme(
            "2026-08-26T00:00:00+00:00",
            DailySummary(
                title="No changes",
                importance=0,
                ptoas_state="No changes.",
                daily_markdown="No changes.",
                should_write_daily_report=False,
            ),
            (),
            (),
            context,
        )
        self.assertIn("## Persistent Watch Context", rendered)
        self.assertIn("Keep this status.", rendered)


if __name__ == "__main__":
    unittest.main()
