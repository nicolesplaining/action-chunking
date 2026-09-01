from pathlib import Path


def test_utility_launcher_binds_clean_code_and_isolates_policy_server() -> None:
    launcher = Path("scripts/run_catalog_retarget_utility.sh").read_text()

    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert 'action_chunking_commit="$(git -C "$repo_root" rev-parse HEAD)"' in launcher
    assert '[[ "$catalog_commit" != "$action_chunking_commit" ]]' in launcher
    assert '[[ -e "$utility_output" && ! -f "$utility_output/code_commit.txt" ]]' in launcher
    assert 'nvidia-smi --id="$gpu" --query-compute-apps=pid' in launcher
    assert 'screen -dmS "$server_session"' in launcher
    assert '--candidate-index "$handoff/candidate_index.json"' in launcher
    assert '--action-chunking-commit "$action_chunking_commit"' in launcher
