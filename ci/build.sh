#!/usr/bin/env bash
#
# Build the lecture demo code.
#
# Usage:
#   ci/build.sh
set -euo pipefail

# Root directory.
ROOT_DIR="$(dirname "${BASH_SOURCE[0]}")/.."

# Demo directories not built yet, relative to the root directory. An exercise skeleton only holds
# placeholders until its lecture has been given, so it can't compile before then. Remove an entry
# once the lecture's code is in place.
SKIPPED_DIRS=(
    "lectures/L04/exercises"
)

################################################################################
# Return success if the given demo directory is listed in SKIPPED_DIRS.
# Globals:
#   ROOT_DIR
#   SKIPPED_DIRS
# Arguments:
#   $1 - Path to a demo directory.
################################################################################
is_skipped() {
    local dir="$1"
    local skipped

    for skipped in "${SKIPPED_DIRS[@]}"
    do
        if [[ "$dir" == "$ROOT_DIR/$skipped" ]]
        then
            return 0
        fi
    done
    return 1
}

# Build every lecture demo and solution tree that has a Makefile. Test directories are pruned:
# their makefiles need a solution to build against, and are handled by ci/test.sh instead.
for root in "$ROOT_DIR/lectures" "$ROOT_DIR/solutions"
do
    # Skip the solution trees when none have been written yet.
    if [[ ! -d "$root" ]]
    then
        continue
    fi

    while IFS= read -r -d '' makefile; do
        demo_dir="$(dirname "$makefile")"

        if is_skipped "$demo_dir"
        then
            echo "Skipping $demo_dir (not built yet)"
            continue
        fi
        echo "Building $demo_dir"
        make -C "$demo_dir" build
    done < <(find "$root" -type d -name test -prune -o -name Makefile -print0 | sort -z)
done
