from __future__ import annotations

from pathlib import Path


def test_pi0_launcher_uses_two_isolated_gpus_and_fail_closed_resume() -> None:
    launcher = Path("scripts/run_pi0_intervention_control.sh").read_text()

    assert 'coarse_gpu="${11}"' in launcher
    assert 'position_gpu="${12}"' in launcher
    assert '[[ "$coarse_gpu" == "$position_gpu" ]]' in launcher
    assert 'run_grid coarse "$coarse_gpu" &' in launcher
    assert 'run_grid population_positions "$position_gpu" &' in launcher
    assert 'wait "$coarse_pid"' in launcher
    assert 'wait "$position_pid"' in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert '[[ -e "$output" && ! -f "$output/code_commit.txt" ]]' in launcher


def test_required_remote_virtualenv_symlink_is_ignored() -> None:
    rules = Path(".gitignore").read_text().splitlines()

    assert ".venv" in rules
