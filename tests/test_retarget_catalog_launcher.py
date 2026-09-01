from pathlib import Path


def test_catalog_launcher_isolates_two_workers_and_binds_resume() -> None:
    launcher = Path("scripts/run_retarget_catalog_parallel.sh").read_text()

    assert '[[ "$gpu_a" == "$gpu_b" || "$port_a" == "$port_b" ]]' in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert '[[ -e "$output" && ! -f "$output/code_commit.txt" ]]' in launcher
    assert 'grep -qx "$plan_digest" "$output/plan.sha256"' in launcher
    assert 'start_server "$gpu_a" "$port_a" "$session_a"' in launcher
    assert 'start_server "$gpu_b" "$port_b" "$session_b"' in launcher
    assert '--workers "$gpu_a:$port_a,$gpu_b:$port_b"' in launcher
