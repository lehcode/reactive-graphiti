#!/usr/bin/env bash
# Run markdownlint --fix and put the fixes into the commit:
#   pre-commit: the staged Markdown files; fixes are re-staged, so they land in the
#               commit being made.
#   pre-push:   the Markdown files changed in the pushed range (pre-commit exports
#               PRE_COMMIT_FROM_REF/PRE_COMMIT_TO_REF); fixes are amended into HEAD.
#               git has already chosen the commit to send, so the push is stopped and
#               must be re-run.
# Files with unstaged edits are only checked, never fixed.
set -u

# Leave rebases, merges and cherry-picks alone
for state in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD; do
    [ -e "$(git rev-parse --git-path "$state")" ] && exit 0
done

pre_push=false
if [ -n "${PRE_COMMIT_FROM_REF:-}" ] && [ -n "${PRE_COMMIT_TO_REF:-}" ]; then
    pre_push=true
    if [ "$(git rev-parse "$PRE_COMMIT_TO_REF")" != "$(git rev-parse HEAD)" ]; then
        echo "markdownlint: pushed ref is not HEAD; check out the pushed branch to fix and amend" >&2
        exit 1
    fi
fi

# pre-commit moves unstaged edits aside into patch<time>-<its pid> (our parent) before
# running hooks and re-applies them afterwards; a fix to such a file would make that
# re-apply fail, so those files are only checked.
store=${PRE_COMMIT_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/pre-commit}
stashed=$'\n'
for patch in "$store"/patch*-"$PPID"; do
    [ -e "$patch" ] || continue
    while IFS= read -r -d '' entry; do
        stashed+="${entry#*$'\t'*$'\t'}"$'\n'
    done < <(git apply --numstat -z "$patch")
done

if $pre_push; then
    list_cmd=(git diff -z --name-only --diff-filter=ACMR "$PRE_COMMIT_FROM_REF" "$PRE_COMMIT_TO_REF" -- '*.md')
else
    list_cmd=(git diff --cached -z --name-only --diff-filter=ACMR -- '*.md')
fi

fixable=()
check_only=()
while IFS= read -r -d '' file; do
    if [[ "$stashed" == *$'\n'"$file"$'\n'* ]] || ! git diff --quiet -- "$file"; then
        check_only+=("$file")
    else
        fixable+=("$file")
    fi
done < <("${list_cmd[@]}")

rc=0
if [ ${#check_only[@]} -gt 0 ]; then
    if ! markdownlint --config .markdownlint.yaml "${check_only[@]}"; then
        echo "markdownlint: not auto-fixed (unstaged edits): ${check_only[*]}" >&2
        rc=1
    fi
fi

[ ${#fixable[@]} -eq 0 ] && exit $rc
markdownlint --config .markdownlint.yaml --fix "${fixable[@]}" || rc=1
git diff --quiet -- "${fixable[@]}" && exit $rc

if ! $pre_push; then
    git add -- "${fixable[@]}"
    echo "markdownlint: re-staged fixes"
    exit $rc
fi

# --only: amend just the fixed files; anything else staged stays staged.
# The push refs must not leak into the hooks the amend runs.
if env -u PRE_COMMIT_FROM_REF -u PRE_COMMIT_TO_REF -u PRE_COMMIT_REMOTE_NAME \
    -u PRE_COMMIT_REMOTE_URL -u PRE_COMMIT_REMOTE_BRANCH -u PRE_COMMIT_LOCAL_BRANCH \
    git commit --amend --no-edit --quiet --only -- "${fixable[@]}"; then
    echo "markdownlint: amended fixes into $(git rev-parse --short HEAD); re-run git push" >&2
else
    echo "markdownlint: amend failed; fixes left unstaged in ${fixable[*]}" >&2
fi
exit 1
