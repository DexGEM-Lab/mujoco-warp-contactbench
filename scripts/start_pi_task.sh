#!/usr/bin/env bash
set -euo pipefail

fail() {
  printf 'error: %s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  scripts/start_pi_task.sh feat <topic> [--dry-run|--no-launch]
  scripts/start_pi_task.sh feature <topic> [--dry-run|--no-launch]
  scripts/start_pi_task.sh case <context> <topic> [--dry-run|--no-launch]
  scripts/start_pi_task.sh dexfeat <topic> [--dry-run|--no-launch]
  scripts/start_pi_task.sh dexfeature <topic> [--dry-run|--no-launch]
  scripts/start_pi_task.sh dexcase <context> <topic> [--dry-run|--no-launch]
EOF
  exit 2
}

valid_slug() {
  [[ "$1" =~ ^[a-z0-9]+(-[a-z0-9]+)*$ ]]
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/.." rev-parse --show-toplevel 2>/dev/null)" \
  || fail "launcher must be located inside a Git worktree"
repo_root="$(cd "$repo_root" && pwd)"
git_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-dir)"
git_common_dir="$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)"

if [[ "$git_dir" != "$git_common_dir" ]]; then
  fail "run this launcher from the primary worktree; linked worktrees cannot create task branches"
fi

[[ $# -ge 2 ]] || usage
kind="$1"
shift

case "$kind" in
  feat|feature)
    topic="$1"
    shift
    valid_slug "$topic" || fail "topic must be a lowercase hyphenated slug"
    branch_name="feat/$topic"
    task_key="feat-$topic"
    pi_name="manorl-feat-$topic"
    base_branch="dev"
    ;;
  case)
    [[ $# -ge 2 ]] || usage
    context="$1"
    topic="$2"
    shift 2
    valid_slug "$context" || fail "context must be a lowercase hyphenated slug"
    valid_slug "$topic" || fail "topic must be a lowercase hyphenated slug"
    branch_name="case/$context/$topic"
    task_key="case-$context--$topic"
    pi_name="manorl-case-$context--$topic"
    base_branch="dev"
    ;;
  dexfeat|dexfeature)
    topic="$1"
    shift
    valid_slug "$topic" || fail "topic must be a lowercase hyphenated slug"
    branch_name="dexfeat/$topic"
    task_key="dexfeat-$topic"
    pi_name="dexhand-feat-$topic"
    base_branch="dexhand"
    ;;
  dexcase)
    [[ $# -ge 2 ]] || usage
    context="$1"
    topic="$2"
    shift 2
    valid_slug "$context" || fail "context must be a lowercase hyphenated slug"
    valid_slug "$topic" || fail "topic must be a lowercase hyphenated slug"
    branch_name="dexcase/$context/$topic"
    task_key="dexcase-$context--$topic"
    pi_name="dexhand-case-$context--$topic"
    base_branch="dexhand"
    ;;
  *)
    fail "task kind must be feat, feature, case, dexfeat, dexfeature, or dexcase"
    ;;
esac

mode="launch"
case "$#" in
  0) ;;
  1)
    case "$1" in
      --dry-run) mode="dry-run" ;;
      --no-launch) mode="no-launch" ;;
      *) usage ;;
    esac
    ;;
  *) usage ;;
esac

git -C "$repo_root" show-ref --verify --quiet "refs/heads/$base_branch" \
  || fail "required base branch does not exist: $base_branch"
git -C "$repo_root" show-ref --verify --quiet "refs/heads/$branch_name" \
  && fail "branch already exists: $branch_name"

worktree_parent="$(dirname "$repo_root")/manoRL_mujoco-worktrees"
worktree_path="$worktree_parent/$task_key"
if [[ -e "$worktree_path" || -L "$worktree_path" ]]; then
  fail "worktree destination already exists: $worktree_path"
fi

if [[ "$mode" == "launch" ]] && ! command -v pi >/dev/null 2>&1; then
  fail "pi command was not found on PATH"
fi

printf 'Branch: %s\n' "$branch_name"
printf 'Base: %s\n' "$base_branch"
printf 'Worktree: %s\n' "$worktree_path"
printf 'Pi session: %s\n' "$pi_name"

if [[ "$mode" == "dry-run" ]]; then
  printf 'Dry run: no branch or worktree was created.\n'
  exit 0
fi

git -C "$repo_root" worktree add -b "$branch_name" "$worktree_path" "$base_branch"
git -C "$worktree_path" branch --set-upstream-to "$base_branch" "$branch_name" >/dev/null
printf 'Created task worktree: %s\n' "$worktree_path"

if [[ "$mode" == "no-launch" ]]; then
  printf 'Pi launch skipped.\n'
  exit 0
fi

cd "$worktree_path"
if pi --name "$pi_name"; then
  exit 0
else
  pi_status=$?
fi

{
  printf 'Pi exited with status %s; task state is preserved.\n' "$pi_status"
  printf 'Branch: %s\n' "$branch_name"
  printf 'Worktree: %s\n' "$worktree_path"
  printf 'Recovery: cd %q && pi --name %q\n' "$worktree_path" "$pi_name"
  printf 'Cleanup: git -C %q worktree remove %q && git -C %q branch -d %q\n' \
    "$repo_root" "$worktree_path" "$repo_root" "$branch_name"
} >&2
exit "$pi_status"
