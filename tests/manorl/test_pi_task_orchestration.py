import json
import os
from pathlib import Path
import shutil
import stat
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "start_pi_task.sh"


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=True)


def disposable_repo(
    tmp_path: Path, *, with_dev: bool = True, with_dexhand: bool = True
) -> Path:
    repo = tmp_path / "manoRL_mujoco"
    repo.mkdir()
    run(["git", "init", "--initial-branch=main", "--quiet"], cwd=repo)
    run(["git", "config", "user.name", "test"], cwd=repo)
    run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    (repo / "README.md").write_text("test repository\n", encoding="utf-8")
    run(["git", "add", "README.md"], cwd=repo)
    run(["git", "commit", "--quiet", "-m", "initial"], cwd=repo)
    if with_dev:
        run(["git", "branch", "dev"], cwd=repo)
        if with_dexhand:
            run(["git", "branch", "dexhand", "dev"], cwd=repo)

    script_path = repo / "scripts" / "start_pi_task.sh"
    script_path.parent.mkdir()
    shutil.copy2(LAUNCHER, script_path)
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return repo


def invoke(
    repo: Path, *args: str, cwd: Path | None = None, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / "scripts" / "start_pi_task.sh"), *args],
        cwd=cwd or repo,
        env=env,
        text=True,
        capture_output=True,
    )


def task_worktree(repo: Path, task_key: str) -> Path:
    return repo.parent / "manoRL_mujoco-worktrees" / task_key


def branch_exists(repo: Path, branch_name: str) -> bool:
    return (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
            cwd=repo,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )


def test_dry_run_does_not_create_branch_or_worktree(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    before_worktrees = run(["git", "worktree", "list", "--porcelain"], cwd=repo).stdout

    result = invoke(repo, "feat", "controller-sync", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Dry run: no branch or worktree was created." in result.stdout
    assert run(["git", "worktree", "list", "--porcelain"], cwd=repo).stdout == before_worktrees
    assert not branch_exists(repo, "feat/controller-sync")
    assert not task_worktree(repo, "feat-controller-sync").exists()


def test_missing_dev_is_rejected_without_creating_a_worktree(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path, with_dev=False)

    result = invoke(repo, "feat", "controller-sync", "--dry-run")

    assert result.returncode == 2
    assert "required base branch does not exist: dev" in result.stderr
    assert not branch_exists(repo, "feat/controller-sync")
    assert not task_worktree(repo, "feat-controller-sync").exists()


def test_feature_alias_derives_feature_branch_worktree_and_pi_name(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)

    result = invoke(repo, "feature", "controller-sync", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Branch: feat/controller-sync" in result.stdout
    assert f"Worktree: {task_worktree(repo, 'feat-controller-sync')}" in result.stdout
    assert "Pi session: manorl-feat-controller-sync" in result.stdout


def test_case_derives_branch_worktree_and_pi_name(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)

    result = invoke(repo, "case", "cube1", "contact-tuning", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "Branch: case/cube1/contact-tuning" in result.stdout
    assert f"Worktree: {task_worktree(repo, 'case-cube1--contact-tuning')}" in result.stdout
    assert "Pi session: manorl-case-cube1--contact-tuning" in result.stdout


def test_dexhand_feature_and_case_use_dexhand_base_and_names(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)

    feature = invoke(repo, "dexfeature", "controller-sync", "--dry-run")
    case = invoke(repo, "dexcase", "cube1", "contact-tuning", "--dry-run")

    assert feature.returncode == 0, feature.stderr
    assert "Branch: dexfeat/controller-sync" in feature.stdout
    assert "Base: dexhand" in feature.stdout
    assert f"Worktree: {task_worktree(repo, 'dexfeat-controller-sync')}" in feature.stdout
    assert "Pi session: dexhand-feat-controller-sync" in feature.stdout
    assert case.returncode == 0, case.stderr
    assert "Branch: dexcase/cube1/contact-tuning" in case.stdout
    assert "Base: dexhand" in case.stdout
    assert f"Worktree: {task_worktree(repo, 'dexcase-cube1--contact-tuning')}" in case.stdout
    assert "Pi session: dexhand-case-cube1--contact-tuning" in case.stdout


def test_missing_dexhand_is_rejected_without_creating_a_worktree(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path, with_dexhand=False)

    result = invoke(repo, "dexfeat", "controller-sync", "--dry-run")

    assert result.returncode == 2
    assert "required base branch does not exist: dexhand" in result.stderr
    assert not branch_exists(repo, "dexfeat/controller-sync")
    assert not task_worktree(repo, "dexfeat-controller-sync").exists()


def test_case_component_separator_keeps_names_injective(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    first = invoke(repo, "case", "a-b", "c", "--dry-run")
    second = invoke(repo, "case", "a", "b-c", "--dry-run")

    first_worktree = task_worktree(repo, "case-a-b--c")
    second_worktree = task_worktree(repo, "case-a--b-c")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert first_worktree != second_worktree
    assert f"Worktree: {first_worktree}" in first.stdout
    assert f"Worktree: {second_worktree}" in second.stdout
    assert "Pi session: manorl-case-a-b--c" in first.stdout
    assert "Pi session: manorl-case-a--b-c" in second.stdout


def test_no_launch_creates_feature_branch_and_linked_worktree(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    destination = task_worktree(repo, "feat-controller-sync")

    result = invoke(repo, "feat", "controller-sync", "--no-launch")

    assert result.returncode == 0, result.stderr
    assert destination.is_dir()
    assert run(["git", "branch", "--show-current"], cwd=destination).stdout.strip() == "feat/controller-sync"
    assert branch_exists(repo, "feat/controller-sync")
    assert "Pi launch skipped." in result.stdout


def test_pi_failure_preserves_created_branch_and_worktree(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    destination = task_worktree(repo, "feat-controller-sync")
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_pi = fake_bin / "pi"
    fake_pi.write_text("#!/usr/bin/env bash\nexit 23\n", encoding="ascii")
    fake_pi.chmod(fake_pi.stat().st_mode | stat.S_IXUSR)
    environment = dict(os.environ)
    environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

    result = invoke(repo, "feat", "controller-sync", env=environment)

    assert result.returncode == 23
    assert destination.is_dir()
    assert branch_exists(repo, "feat/controller-sync")
    assert "Pi exited with status 23; task state is preserved." in result.stderr
    assert f"Branch: feat/controller-sync" in result.stderr
    assert f"Worktree: {destination}" in result.stderr
    assert "Recovery: cd " in result.stderr
    assert "Cleanup: git -C " in result.stderr
    assert "worktree remove" in result.stderr
    assert "branch -d feat/controller-sync" in result.stderr


def test_invalid_slug_is_rejected_without_mutation(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)

    result = invoke(repo, "case", "cube1", "Contact-Tuning", "--dry-run")

    assert result.returncode == 2
    assert "topic must be a lowercase hyphenated slug" in result.stderr
    assert not task_worktree(repo, "case-cube1-Contact-Tuning").exists()


def test_existing_feature_branch_is_rejected_before_worktree_creation(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    run(["git", "branch", "feat/controller-sync", "dev"], cwd=repo)

    result = invoke(repo, "feat", "controller-sync", "--no-launch")

    assert result.returncode == 2
    assert "branch already exists: feat/controller-sync" in result.stderr
    assert not task_worktree(repo, "feat-controller-sync").exists()


def test_existing_destination_is_rejected(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    destination = task_worktree(repo, "feat-controller-sync")
    destination.mkdir(parents=True)

    result = invoke(repo, "feat", "controller-sync", "--no-launch")

    assert result.returncode == 2
    assert f"worktree destination already exists: {destination}" in result.stderr
    assert not branch_exists(repo, "feat/controller-sync")


def test_linked_worktree_invocation_is_rejected(tmp_path: Path) -> None:
    repo = disposable_repo(tmp_path)
    linked = tmp_path / "linked"
    run(["git", "worktree", "add", "-b", "linked-task", str(linked), "dev"], cwd=repo)
    linked_script = linked / "scripts" / "start_pi_task.sh"
    linked_script.parent.mkdir()
    shutil.copy2(LAUNCHER, linked_script)
    linked_script.chmod(linked_script.stat().st_mode | stat.S_IXUSR)

    result = subprocess.run(
        [str(linked_script), "feat", "controller-sync", "--dry-run"],
        cwd=linked,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "run this launcher from the primary worktree" in result.stderr
    assert not task_worktree(repo, "feat-controller-sync").exists()


def test_git_guard_policy_and_runtime_config_match_task_topology() -> None:
    policy = json.loads((REPO_ROOT / ".git-guard" / "policy.json").read_text(encoding="utf-8"))
    config = json.loads((REPO_ROOT / ".git-guard" / "config.json").read_text(encoding="utf-8"))
    legacy_branch = "feature/manorl-mujoco-migration"

    assert policy["branches"] == {
        "long_lived": ["main", "dev", "dexhand"],
        "families": ["feat/*", "case/*/*", "dexfeat/*", "dexcase/*/*"],
    }
    assert {(edge["source"], edge["target"]) for edge in policy["branch_from"]} == {
        ("main", "dev"),
        ("dev", "dexhand"),
        ("dev", "feat/*"),
        ("dev", "case/*/*"),
        ("dexhand", "dexfeat/*"),
        ("dexhand", "dexcase/*/*"),
    }
    rules = {(rule["source"], rule["target"]): rule for rule in policy["merge_rules"]}
    assert set(rules) == {
        ("case/*/*", "feat/*"),
        ("dev", "feat/*"),
        ("feat/*", "dev"),
        ("dexcase/*/*", "dexfeat/*"),
        ("dexhand", "dexfeat/*"),
        ("dexfeat/*", "dexhand"),
        ("dev", "main"),
    }
    assert rules[("dev", "feat/*")]["sync"] is True
    assert rules[("feat/*", "dev")]["sync_merge_required"] is True
    assert {item["name"] for item in policy["direct_commit_refs"]} == {
        "feat/*",
        "case/*/*",
        "dexfeat/*",
        "dexcase/*/*",
    }
    assert legacy_branch not in json.dumps(policy, sort_keys=True)
    assert policy["tag_rules"] == []
    assert all("tag_pattern" not in rule for rule in policy["merge_rules"])
    assert config["branch_logs"]["force_diff_required"] is False
    assert config["pre_push"]["auto_push_missing_tags"] is False
    assert config["submodules"]["main_guard"] is False
    assert config["protected_branches"]["enabled"] is True
    assert config["runtime"]["auto_sync"] is True
    assert config["worktree"]["reject_branch_creation_in_linked_worktree"] is True
