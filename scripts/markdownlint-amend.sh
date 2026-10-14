#!/usr/bin/env bash
# pre-commit hook: lint the staged Markdown files and re-stage markdownlint's fixes, so
# they land in the commit being made. Only the changes being committed are checked.
# A file is fixed only when its working-tree copy is exactly the staged content;
# otherwise the staged content is linted read-only, so unstaged edits are never swept
# into the commit.
set -u -o pipefail

# Leave rebases, merges and cherry-picks alone
for state in rebase-merge rebase-apply MERGE_HEAD CHERRY_PICK_HEAD; do
    [ -e "$(git rev-parse --git-path "$state")" ] && exit 0
done

# pre-commit moves unstaged edits aside into patch<time>-<its pid> (our parent) before
# running hooks and re-applies them afterwards, so they are invisible to git diff here.
# Each `git apply --numstat -z` record is "<added>\t<deleted>\t<path>"; the expansion
# strips the two counts.
store=${PRE_COMMIT_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/pre-commit}
stashed=$'\n'
for patch in "$store"/patch*-"$PPID"; do
    [ -e "$patch" ] || continue
    while IFS= read -r -d '' entry; do
        stashed+="${entry#*$'\t'*$'\t'}"$'\n'
    done < <(git apply --numstat -z "$patch")
done

fixable=()
check_only=()
while IFS= read -r -d '' file; do
    if [[ "$stashed" != *$'\n'"$file"$'\n'* ]] && git diff --quiet -- "$file"; then
        fixable+=("$file")
    else
        check_only+=("$file")
    fi
done < <(git diff --cached -z --name-only --diff-filter=ACMR -- '*.md')

rc=0
for file in ${check_only[@]+"${check_only[@]}"}; do
    if ! git show ":$file" | markdownlint --config .markdownlint.yaml --stdin; then
        echo "markdownlint: $file has errors (not auto-fixed: it has unstaged edits)" >&2
        rc=1
    fi
done

[ ${#fixable[@]} -eq 0 ] && exit $rc
markdownlint --config .markdownlint.yaml --fix "${fixable[@]}" || rc=1
if ! git diff --quiet -- "${fixable[@]}"; then
    git add -- "${fixable[@]}"
    echo "markdownlint: re-staged fixes"
fi
exit $rc
